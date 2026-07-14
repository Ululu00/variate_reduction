from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.train_runtime_stats = {
            'train_iter_count': 0,
            'train_time_sec': 0.0,
            'avg_train_iter_time_sec': 0.0,
            'last_epoch_time_sec': 0.0,
            'avg_epoch_time_sec': 0.0,
            'throughput_samples_per_sec': 0.0,
            'cycle_loss': 0.0,
            'weighted_cycle_loss': 0.0,
            'cycle_current_weight': 0.0,
        }
        self.latest_train_loss_components = {}
        self.current_cycle_weight = 0.0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        base_learning_rate = float(self.args.learning_rate)
        mass_learning_rate = float(
            getattr(self.args, 'sender_mass_learning_rate', 0.0) or 0.0
        )
        sender_mass_mode = str(getattr(self.args, 'sender_mass_mode', 'fixed'))
        if not np.isfinite(mass_learning_rate) or mass_learning_rate < 0.0:
            raise ValueError('sender_mass_learning_rate must be finite and non-negative')

        # The default path deliberately remains the original one-group Adam
        # construction so existing runs have identical grouping and defaults.
        if mass_learning_rate == 0.0:
            return optim.Adam(self.model.parameters(), lr=base_learning_rate)

        if sender_mass_mode not in {'learned_global', 'learned_layer_head'}:
            raise ValueError(
                'sender_mass_learning_rate > 0 requires a learned sender_mass_mode'
            )
        if not np.isfinite(base_learning_rate) or base_learning_rate <= 0.0:
            raise ValueError(
                'positive sender_mass_learning_rate requires a finite positive learning_rate'
            )

        all_parameters = list(self.model.parameters())
        mass_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if name.rsplit('.', 1)[-1] == 'sender_mass_power_theta'
        ]
        if len(mass_parameters) != 1:
            raise ValueError(
                'learned sender_mass_mode with a dedicated learning rate requires exactly '
                'one parameter named sender_mass_power_theta; found {}'.format(
                    len(mass_parameters)
                )
            )
        mass_parameter_ids = {id(parameter) for parameter in mass_parameters}
        base_parameters = [
            parameter
            for parameter in all_parameters
            if id(parameter) not in mass_parameter_ids
        ]
        grouped_parameters = base_parameters + mass_parameters
        grouped_ids = [id(parameter) for parameter in grouped_parameters]
        all_ids = [id(parameter) for parameter in all_parameters]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise RuntimeError('optimizer parameter groups contain duplicate parameters')
        if set(grouped_ids) != set(all_ids):
            raise RuntimeError('optimizer parameter groups do not cover every model parameter')

        return optim.Adam(
            [
                {
                    'params': base_parameters,
                    'lr': base_learning_rate,
                    'group_name': 'base',
                },
                {
                    'params': mass_parameters,
                    'lr': mass_learning_rate,
                    'lr_scale': mass_learning_rate / base_learning_rate,
                    'group_name': 'sender_mass_power',
                },
            ],
            lr=base_learning_rate,
        )

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _base_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _auxiliary_loss(self, outputs):
        base_model = self._base_model()
        if hasattr(base_model, 'get_aux_loss'):
            return base_model.get_aux_loss()
        return outputs.new_tensor(0.0)

    def _auxiliary_losses(self, outputs):
        base_model = self._base_model()
        if hasattr(base_model, 'get_aux_losses'):
            return base_model.get_aux_losses()
        return {
            'orthogonal': self._auxiliary_loss(outputs),
            'reconstruction': outputs.new_tensor(0.0),
            'coverage': outputs.new_tensor(0.0),
            'assignment_entropy': outputs.new_tensor(0.0),
            'wcomp_entropy': outputs.new_tensor(0.0),
            'expansion_weight_l2': outputs.new_tensor(0.0),
            'group_attention_entropy': outputs.new_tensor(0.0),
        }

    def _training_loss(self, outputs, batch_y, criterion):
        mse_loss = criterion(outputs, batch_y)
        aux_losses = self._auxiliary_losses(outputs)
        forecast_loss_type = str(getattr(self.args, 'forecast_loss_type', 'mse') or 'mse').lower()
        if forecast_loss_type in {'smooth_l1', 'huber'}:
            beta = float(getattr(self.args, 'huber_delta', 1.0) or 1.0)
            pred_loss = F.smooth_l1_loss(outputs, batch_y, beta=beta)
        else:
            pred_loss = mse_loss
        loss = pred_loss
        components = {
            'total_loss': loss,
            'pred_loss': pred_loss,
            'forecast_mse': mse_loss,
        }
        if forecast_loss_type in {'smooth_l1', 'huber'}:
            components['forecast_smooth_l1'] = pred_loss
        mae_loss = torch.mean(torch.abs(outputs - batch_y))
        weighted_mae = getattr(self.args, 'mae_loss_weight', 0.0) * mae_loss
        loss = loss + weighted_mae
        components['forecast_mae'] = mae_loss
        components['weighted_mae_loss'] = weighted_mae
        weighted_orthogonal = getattr(self.args, 'orthogonal_loss_weight', 0.0) * aux_losses.get(
            'orthogonal', outputs.new_tensor(0.0)
        )
        loss = loss + weighted_orthogonal
        components['orthogonal_loss'] = aux_losses.get('orthogonal', outputs.new_tensor(0.0))
        components['weighted_orthogonal_loss'] = weighted_orthogonal
        weighted_reconstruction = getattr(self.args, 'reconstruction_loss_weight', 0.0) * aux_losses.get(
            'reconstruction', outputs.new_tensor(0.0)
        )
        loss = loss + weighted_reconstruction
        components['reconstruction_loss'] = aux_losses.get('reconstruction', outputs.new_tensor(0.0))
        components['weighted_reconstruction_loss'] = weighted_reconstruction
        weighted_coverage = getattr(self.args, 'coverage_loss_weight', 0.0) * aux_losses.get(
            'coverage', outputs.new_tensor(0.0)
        )
        loss = loss + weighted_coverage
        components['coverage_loss'] = aux_losses.get('coverage', outputs.new_tensor(0.0))
        components['weighted_coverage_loss'] = weighted_coverage
        weighted_entropy = getattr(self.args, 'assignment_entropy_loss_weight', 0.0) * aux_losses.get(
            'assignment_entropy', outputs.new_tensor(0.0)
        )
        loss = loss + weighted_entropy
        components['assignment_entropy_loss'] = aux_losses.get('assignment_entropy', outputs.new_tensor(0.0))
        components['weighted_assignment_entropy_loss'] = weighted_entropy
        weighted_wcomp_entropy = getattr(self.args, 'wcomp_entropy_loss_weight', 0.0) * aux_losses.get(
            'wcomp_entropy', outputs.new_tensor(0.0)
        )
        loss = loss + weighted_wcomp_entropy
        components['wcomp_entropy_loss'] = aux_losses.get('wcomp_entropy', outputs.new_tensor(0.0))
        components['weighted_wcomp_entropy_loss'] = weighted_wcomp_entropy
        weighted_expansion_weight_l2 = getattr(self.args, 'expansion_weight_l2_loss_weight', 0.0) * aux_losses.get(
            'expansion_weight_l2', outputs.new_tensor(0.0)
        )
        loss = loss + weighted_expansion_weight_l2
        components['expansion_weight_l2_loss'] = aux_losses.get('expansion_weight_l2', outputs.new_tensor(0.0))
        components['weighted_expansion_weight_l2_loss'] = weighted_expansion_weight_l2
        weighted_group_entropy = getattr(self.args, 'group_attention_entropy_loss_weight', 0.0) * aux_losses.get(
            'group_attention_entropy', outputs.new_tensor(0.0)
        )
        loss = loss + weighted_group_entropy
        components['group_attention_entropy_loss'] = aux_losses.get(
            'group_attention_entropy', outputs.new_tensor(0.0)
        )
        components['weighted_group_attention_entropy_loss'] = weighted_group_entropy
        cycle_loss = aux_losses.get('cycle_slot', outputs.new_tensor(0.0))
        cycle_weight = outputs.new_tensor(float(getattr(self, 'current_cycle_weight', 0.0)))
        weighted_cycle = cycle_weight * cycle_loss
        loss = loss + weighted_cycle
        components['cycle_loss'] = cycle_loss
        components['cycle_current_weight'] = cycle_weight
        components['weighted_cycle_loss'] = weighted_cycle
        components['total_loss'] = loss
        self.latest_train_loss_components = {
            name: value.detach().float().cpu().item()
            for name, value in components.items()
        }
        return loss

    def _cycle_weight_for_epoch(self, epoch):
        if not bool(getattr(self.args, 'use_cycle_slot_loss', False)):
            return 0.0
        weight = float(getattr(self.args, 'cycle_loss_weight', 0.0) or 0.0)
        if weight <= 0.0:
            return 0.0
        warmup_ratio = float(getattr(self.args, 'cycle_warmup_ratio', 0.0) or 0.0)
        progress = float(epoch) / float(max(int(getattr(self.args, 'train_epochs', 1)), 1))
        return 0.0 if progress < warmup_ratio else weight

    def _average_loss_components(self, component_history):
        if not component_history:
            return {}
        keys = component_history[0].keys()
        return {
            key: float(np.average([components[key] for components in component_history]))
            for key in keys
        }

    def _wandb_log(self, payload, step=None):
        if not getattr(self.args, 'use_wandb', False):
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        wandb.log(payload, step=step)

    def _sync_if_cuda(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = 0.0
        total_elements = 0
        max_eval_batches = int(getattr(self.args, 'max_eval_batches', 0) or 0)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                num_elements = pred.numel()
                total_loss += float(loss.item()) * num_elements
                total_elements += num_elements
                if max_eval_batches > 0 and (i + 1) >= max_eval_batches:
                    break
        if total_elements == 0:
            raise RuntimeError('validation loader produced no elements')
        total_loss = total_loss / total_elements
        self.model.train()
        return total_loss

    def train(self, setting):
        self.train_runtime_stats = {
            'train_iter_count': 0,
            'train_time_sec': 0.0,
            'avg_train_iter_time_sec': 0.0,
            'last_epoch_time_sec': 0.0,
            'avg_epoch_time_sec': 0.0,
            'throughput_samples_per_sec': 0.0,
            'cycle_loss': 0.0,
            'weighted_cycle_loss': 0.0,
            'cycle_current_weight': 0.0,
        }
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        skip_test_eval = bool(getattr(self.args, 'skip_test_eval', False))
        skip_epoch_test_eval = skip_test_eval or bool(getattr(self.args, 'skip_epoch_test_eval', False))
        if skip_epoch_test_eval:
            test_data, test_loader = None, None
        else:
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.current_cycle_weight = self._cycle_weight_for_epoch(epoch)
            iter_count = 0
            epoch_step_count = 0
            max_train_batches = int(getattr(self.args, 'max_train_batches', 0) or 0)
            train_loss = []
            loss_component_history = []

            self.model.train()
            self._sync_if_cuda()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                epoch_step_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self._training_loss(outputs, batch_y, criterion)
                    train_loss.append(loss.item())
                    loss_component_history.append(dict(self.latest_train_loss_components))
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self._training_loss(outputs, batch_y, criterion)
                    train_loss.append(loss.item())
                    loss_component_history.append(dict(self.latest_train_loss_components))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if max_train_batches > 0 and epoch_step_count >= max_train_batches:
                    break

            self._sync_if_cuda()
            epoch_train_time = time.time() - epoch_time
            self.train_runtime_stats['train_time_sec'] += epoch_train_time
            self.train_runtime_stats['last_epoch_time_sec'] = epoch_train_time
            self.train_runtime_stats['train_iter_count'] += epoch_step_count
            if self.train_runtime_stats['train_iter_count'] > 0:
                self.train_runtime_stats['avg_train_iter_time_sec'] = (
                    self.train_runtime_stats['train_time_sec'] /
                    self.train_runtime_stats['train_iter_count']
                )
            completed_epochs = epoch + 1
            self.train_runtime_stats['avg_epoch_time_sec'] = (
                self.train_runtime_stats['train_time_sec'] / float(completed_epochs)
            )
            if self.train_runtime_stats['train_time_sec'] > 0:
                self.train_runtime_stats['throughput_samples_per_sec'] = (
                    self.train_runtime_stats['train_iter_count'] * int(getattr(self.args, 'batch_size', 1)) /
                    self.train_runtime_stats['train_time_sec']
                )

            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_train_time))
            train_loss = np.average(train_loss)
            averaged_components = self._average_loss_components(loss_component_history)
            self.train_runtime_stats['cycle_loss'] = averaged_components.get('cycle_loss', 0.0)
            self.train_runtime_stats['weighted_cycle_loss'] = averaged_components.get('weighted_cycle_loss', 0.0)
            self.train_runtime_stats['cycle_current_weight'] = averaged_components.get(
                'cycle_current_weight',
                self.current_cycle_weight,
            )
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if skip_epoch_test_eval:
                test_loss = float('nan')
            else:
                test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            wandb_payload = {
                'epoch': epoch + 1,
                'train/loss': float(train_loss),
                'vali/mse': float(vali_loss),
                'train/epoch_time_sec': float(epoch_train_time),
                'train/avg_epoch_time_sec': float(self.train_runtime_stats['avg_epoch_time_sec']),
                'train/avg_iter_time_sec': float(self.train_runtime_stats['avg_train_iter_time_sec']),
                'train/throughput_samples_per_sec': float(self.train_runtime_stats['throughput_samples_per_sec']),
                'train/learning_rate': float(model_optim.param_groups[0]['lr']),
            }
            if not skip_epoch_test_eval:
                wandb_payload['test/mse'] = float(test_loss)
            for name, value in averaged_components.items():
                wandb_payload['train/' + name] = value
            self._wandb_log(wandb_payload, step=epoch + 1)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, flag='test', save_results=True):
        eval_data, eval_loader = self._get_data(flag=flag)
        max_eval_batches = int(getattr(self.args, 'max_eval_batches', 0) or 0)
        if test and flag == 'test':
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        save_test_arrays = getattr(self.args, 'save_test_arrays', False)
        save_test_visuals = getattr(self.args, 'save_test_visuals', False)
        preds = []
        trues = []
        metric_sums = {
            'abs': 0.0,
            'sq': 0.0,
            'ape': 0.0,
            'spe': 0.0,
            'count': 0,
        }
        folder_path = './test_results/' + setting + '/'
        if save_test_visuals and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(eval_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                fast_metric_only = (
                    not save_test_arrays
                    and not save_test_visuals
                    and not (eval_data.scale and self.args.inverse)
                )
                if fast_metric_only:
                    diff = (outputs - batch_y).double()
                    true = batch_y.double()
                    ratio = diff / true
                    metric_sums['abs'] += torch.sum(torch.abs(diff)).item()
                    metric_sums['sq'] += torch.sum(diff * diff).item()
                    metric_sums['ape'] += torch.sum(torch.abs(ratio)).item()
                    metric_sums['spe'] += torch.sum(ratio * ratio).item()
                    metric_sums['count'] += diff.numel()
                    if max_eval_batches > 0 and (i + 1) >= max_eval_batches:
                        break
                    continue

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if eval_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = eval_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = eval_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                diff = pred - true
                ratio = diff / true
                metric_sums['abs'] += np.sum(np.abs(diff), dtype=np.float64)
                metric_sums['sq'] += np.sum(diff ** 2, dtype=np.float64)
                metric_sums['ape'] += np.sum(np.abs(ratio), dtype=np.float64)
                metric_sums['spe'] += np.sum(ratio ** 2, dtype=np.float64)
                metric_sums['count'] += pred.size

                if save_test_arrays:
                    preds.append(pred)
                    trues.append(true)
                if save_test_visuals and i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if eval_data.scale and self.args.inverse:
                        shape = input.shape
                        input = eval_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                if max_eval_batches > 0 and (i + 1) >= max_eval_batches:
                    break

        if metric_sums['count'] == 0:
            raise RuntimeError('No {} predictions were generated.'.format(flag))
        mae = metric_sums['abs'] / metric_sums['count']
        mse = metric_sums['sq'] / metric_sums['count']
        rmse = np.sqrt(mse)
        mape = metric_sums['ape'] / metric_sums['count']
        mspe = metric_sums['spe'] / metric_sums['count']

        if save_test_arrays:
            preds = np.array(preds)
            trues = np.array(trues)
            print('{} shape:'.format(flag), preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('{} shape:'.format(flag), preds.shape, trues.shape)
        else:
            print('{} elements:'.format(flag), metric_sums['count'])

        # result save
        folder_path = './results/' + setting + '/'
        if save_results and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('mse:{}, mae:{}'.format(mse, mae))
        if save_results:
            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('{} mse:{}, mae:{}'.format(flag, mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            if save_test_arrays:
                np.save(folder_path + 'pred.npy', preds)
                np.save(folder_path + 'true.npy', trues)

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'result_dir': folder_path if save_results else '',
            'eval_split': flag,
        }


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
