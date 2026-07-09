import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np
import csv
import fcntl
import os
import subprocess
import time


CSV_COLUMNS = [
    'dataset', 'pred_len',
    'method_family', 'method_name',
    'selection_id', 'selected_recipe', 'manual_override',
    'variate_reduction_type', 'variate_expansion_type', 'expansion_weight_source',
    'variate_decode_stage', 'loss_variant',
    'forecast_loss_type', 'huber_delta',
    'compute_reducer_aux_losses',
    'variate_token_split_factor',
    'num_variates', 'target_variate_tokens', 'source_variate_tokens',
    'reduced_variate_k', 'reduced_variate_tokens',
    'k_ratio', 'num_latents_K',
    'target_k_ratio', 'actual_k_ratio', 'source_k_ratio',
    'logical_k', 'executed_k', 'logical_k_ratio', 'executed_k_ratio',
    'k_budget_max', 'k_budget_exception',
    'lowrank_reducer_rank', 'decoder_residual_gate_init',
    'backbone_residual_gate_init', 'backbone_residual_gate_type',
    'backbone_residual_gate', 'backbone_residual_gate_abs_mean', 'backbone_residual_gate_max',
    'local_temporal_rank',
    'zero_init_projector', 'skip_backbone',
    'k_selection_mode', 'k_ratio_denominator', 'target_k_value',
    'variate_anchor_map_path',
    'original_encoder_tokens', 'reduced_encoder_tokens', 'num_extra_tokens',
    'token_ratio', 'token_reduction_percent',
    'attention_score_ratio', 'attention_score_reduction_percent',
    'eval_split', 'skip_epoch_test_eval',
    'mse', 'mae', 'rmse', 'mape', 'mspe',
    'orthogonal_loss_weight', 'reconstruction_loss_weight',
    'coverage_loss_weight', 'assignment_entropy_loss_weight',
    'wcomp_entropy_loss_weight', 'group_attention_entropy_loss_weight',
    'group_attention_entropy_target', 'lambda_orth', 'lambda_entropy',
    'use_cycle_slot_loss', 'cycle_loss_weight', 'cycle_warmup_ratio',
    'cycle_topr', 'cycle_topr_multiplier', 'cycle_b_norm', 'cycle_eps',
    'export_slot_diagnostics',
    'cycle_loss', 'weighted_cycle_loss', 'cycle_current_weight',
    'mae_loss_weight',
    'train_time_sec', 'train_iter_count', 'avg_train_iter_time_sec', 'throughput_iter_per_sec',
    'last_epoch_time_sec', 'avg_epoch_time_sec', 'throughput_samples_per_sec',
    'elapsed_sec', 'parameter_count',
    'peak_gpu_allocated_mb', 'peak_gpu_reserved_mb', 'gpu_memory_footprint_mb',
    'expansion_temperature', 'expansion_topk',
    'normalization', 'entmax_alpha',
    'sparse_compress_topk', 'sparse_expand_topk',
    'avg_comp_entropy', 'std_comp_entropy',
    'avg_effective_support', 'std_effective_support',
    'avg_top1_mass', 'avg_top5_mass', 'avg_top10_mass',
    'avg_pairwise_overlap_top5', 'avg_pairwise_overlap_top10',
    'avg_pairwise_cosine_between_latents',
    'w_comp_density', 'w_exp_density', 'w_eff_density',
    'avg_w_eff_entropy', 'avg_w_eff_top5_mass',
    'cycle_topr_jaccard', 'cycle_a_effective_support', 'cycle_b_effective_support',
    'cycle_slot_overlap', 'cycle_topr_size',
    'residual_gate_mean', 'residual_gate_abs_mean', 'residual_gate_max', 'hybrid_linear_gate',
    'local_temporal_branch', 'local_temporal_init', 'local_temporal_gate_init', 'local_temporal_gate',
    'output_calibration', 'output_calibration_scale_abs_mean', 'output_calibration_bias_abs_mean',
    'status', 'timestamp', 'git_commit', 'git_dirty', 'error', 'command', 'returncode',
    'gpu', 'cuda_visible_devices', 'data', 'root_path', 'data_path', 'target', 'freq',
    'seed', 'itr', 'model', 'model_id', 'setting',
    'seq_len', 'label_len', 'e_layers', 'd_model', 'd_ff',
    'batch_size', 'learning_rate', 'train_epochs', 'patience', 'num_workers',
    'use_norm', 'checkpoint_dir', 'result_dir', 'weight_matrix_dir', 'slot_diagnostic_dir', 'plot_dir',
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def strict_k_budget(num_variates):
    k_max = int(np.ceil(0.30 * int(num_variates))) - 1
    if k_max >= 1:
        return k_max, False
    return 1, True


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def git_value(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ''


def git_dirty():
    try:
        output = subprocess.check_output(['git', 'status', '--short'], stderr=subprocess.DEVNULL).decode().strip()
        return 'true' if output else 'false'
    except Exception:
        return ''


def csv_fieldnames(path=None):
    fieldnames = list(CSV_COLUMNS)
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, newline='') as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
            return existing_header + [column for column in fieldnames if column not in existing_header]
        except Exception:
            return fieldnames
    return fieldnames


def ensure_csv_schema(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames or []
        fieldnames = existing_header + [column for column in CSV_COLUMNS if column not in existing_header]
        if existing_header == fieldnames:
            return
        rows = list(reader)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in rows:
            writer.writerow({column: existing_row.get(column, '') for column in fieldnames})
    os.replace(tmp_path, path)


def append_csv_row(path, row):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lock_path = path + '.lock'
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            ensure_csv_schema(path)
            file_exists = os.path.exists(path) and os.path.getsize(path) > 0
            fieldnames = csv_fieldnames(path)
            with open(path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({column: row.get(column, '') for column in fieldnames})
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def base_model(model):
    return model.module if hasattr(model, 'module') else model


def model_metadata(model, args):
    model = base_model(model)
    if hasattr(model, 'get_variate_reduction_metadata'):
        return model.get_variate_reduction_metadata()
    reduced_k = args.enc_in if args.variate_reduction_type == 'none' else args.reduced_variate_k
    split_factor = int(getattr(args, 'variate_token_split_factor', 1))
    source_variate_tokens = args.enc_in * split_factor
    return {
        'variate_token_split_factor': split_factor,
        'variate_decode_stage': getattr(args, 'variate_decode_stage', 'feature'),
        'original_variate_tokens': args.enc_in,
        'target_variate_tokens': args.enc_in,
        'source_variate_tokens': source_variate_tokens,
        'reduced_variate_tokens': reduced_k,
        'num_extra_tokens': 0,
        'original_encoder_tokens': source_variate_tokens,
        'reduced_encoder_tokens': reduced_k,
        'logical_k': reduced_k,
        'executed_k': reduced_k,
        'k_budget_max': strict_k_budget(args.enc_in)[0],
        'k_budget_exception': strict_k_budget(args.enc_in)[1],
        'lowrank_reducer_rank': getattr(args, 'lowrank_reducer_rank', 0),
        'sparse_compress_topk': getattr(args, 'sparse_compress_topk', 0),
        'sparse_expand_topk': getattr(args, 'sparse_expand_topk', 0),
        'use_cycle_slot_loss': getattr(args, 'use_cycle_slot_loss', False),
        'cycle_topr': getattr(args, 'cycle_topr', 0),
        'cycle_topr_multiplier': getattr(args, 'cycle_topr_multiplier', 1.0),
        'cycle_b_norm': getattr(args, 'cycle_b_norm', 'row_l1'),
        'cycle_eps': getattr(args, 'cycle_eps', 1e-8),
        'export_slot_diagnostics': getattr(args, 'export_slot_diagnostics', False),
        'cycle_topr_jaccard': 0.0,
        'cycle_a_effective_support': 0.0,
        'cycle_b_effective_support': 0.0,
        'cycle_slot_overlap': 0.0,
        'cycle_topr_size': 0,
        'w_eff_density': 0.0,
        'decoder_residual_gate_init': getattr(args, 'decoder_residual_gate_init', 0.0),
        'local_temporal_rank': getattr(args, 'local_temporal_rank', 0),
        'zero_init_projector': getattr(args, 'zero_init_projector', False),
        'skip_backbone': getattr(args, 'skip_backbone', False),
        'residual_gate_mean': 0.0,
        'residual_gate_abs_mean': 0.0,
        'residual_gate_max': 0.0,
        'hybrid_linear_gate': 0.0,
        'local_temporal_branch': getattr(args, 'local_temporal_branch', 'none'),
        'local_temporal_gate': 0.0,
    }


def model_weight_matrices(model):
    model = base_model(model)
    if hasattr(model, 'get_variate_reduction_weights'):
        return model.get_variate_reduction_weights()
    return {}


def model_slot_diagnostic_matrices(model):
    model = base_model(model)
    if hasattr(model, 'get_slot_diagnostic_matrices'):
        return model.get_slot_diagnostic_matrices()
    return {}


def expansion_weight_source(reduction_type, expansion_type):
    if reduction_type == 'none':
        return ''
    if reduction_type in {'mlp_static_combination', 'mlp_slot_attention', 'mlp_sparse_slot_attention'}:
        return 'assignment_transpose'
    if 'query_decoder' in str(expansion_type):
        return 'query_decoder'
    if str(expansion_type).startswith('fixed_'):
        return 'fixed_expand_weight'
    if str(expansion_type).startswith('masked_'):
        return 'masked_learned_expand_linear'
    return 'learned_expand_linear'


def _save_heatmap(matrix, path, title, cmap='coolwarm', symmetric=True,
                  xlabel='input/source index', ylabel='output/target index'):
    try:
        os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-itformer')
        os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print('Skipping weight heatmap {}: {}'.format(path, exc))
        return False

    rows, cols = matrix.shape
    width = min(14.0, max(7.0, cols / 70.0))
    height = min(8.0, max(4.0, rows / 35.0))
    fig, ax = plt.subplots(figsize=(width, height))
    kwargs = {'aspect': 'auto', 'cmap': cmap}
    if symmetric:
        vmax = float(np.max(np.abs(matrix))) if matrix.size else 0.0
        if vmax > 0.0:
            kwargs.update({'vmin': -vmax, 'vmax': vmax})
    im = ax.imshow(matrix, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _set_integer_matrix_ticks(ax, rows, cols)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _integer_tick_positions(length, max_ticks):
    if length <= 0:
        return []
    if length <= max_ticks:
        return list(range(length))
    ticks = np.linspace(0, length - 1, num=max_ticks)
    return sorted({int(round(tick)) for tick in ticks})


def _matrix_tick_budgets(ax, rows, cols, full_tick_limit=24):
    fig_width, fig_height = ax.figure.get_size_inches()
    max_xticks = cols if cols <= full_tick_limit else max(4, min(16, int(fig_width * 1.3)))
    max_yticks = rows if rows <= full_tick_limit else max(4, min(20, int(fig_height * 1.8)))
    return max_xticks, max_yticks


def _set_integer_matrix_ticks(ax, rows, cols):
    max_xticks, max_yticks = _matrix_tick_budgets(ax, rows, cols)
    xticks = _integer_tick_positions(cols, max_ticks=max_xticks)
    yticks = _integer_tick_positions(rows, max_ticks=max_yticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(tick) for tick in xticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(tick) for tick in yticks])
    if cols > 0:
        ax.set_xlim(-0.5, cols - 0.5)
    if rows > 0:
        ax.set_ylim(rows - 0.5, -0.5)


def _row_normalized_abs(matrix, eps=1e-8):
    matrix = np.abs(matrix)
    return matrix / (matrix.sum(axis=-1, keepdims=True) + eps)


def _row_cosine(matrix, eps=1e-8):
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / (denom + eps)
    return normalized @ normalized.T


def _weight_axis_labels(name):
    if name == 'compress_weight':
        return 'input/source variable index', 'latent token index'
    if name in ('expand_weight', 'slot_expand_weight'):
        return 'latent token index', 'output/target variable index'
    if name == 'static_assignment':
        return 'input/source variable index', 'latent token index'
    return 'input/source index', 'output/target index'


def _safe_dir_name(value):
    value = str(value)
    safe = []
    for char in value:
        if char.isalnum() or char in ('-', '_', '.', '='):
            safe.append(char)
        else:
            safe.append('_')
    return ''.join(safe).strip('_') or 'run'


def _split_source_labels(target_variate_tokens, split_factor):
    return [
        'var{}_seg{}'.format(variate_idx, segment_idx)
        for variate_idx in range(int(target_variate_tokens))
        for segment_idx in range(int(split_factor))
    ]


def _save_top_sources(matrix, path, labels=None, topk=10):
    rows, cols = matrix.shape
    labels = labels or ['source{}'.format(index) for index in range(cols)]
    topk = min(int(topk), cols)
    with open(path, 'w') as f:
        for row_idx in range(rows):
            order = np.argsort(np.abs(matrix[row_idx]))[::-1][:topk]
            parts = []
            for col_idx in order:
                label = labels[col_idx] if col_idx < len(labels) else 'source{}'.format(col_idx)
                parts.append('{}={:.6g}'.format(label, float(matrix[row_idx, col_idx])))
            f.write('latent{}: {}\n'.format(row_idx, ', '.join(parts)))


def save_weight_matrix_visuals(args, model, result_dir, setting):
    if not getattr(args, 'save_weight_heatmaps', True):
        return {'weight_matrix_dir': '', 'weight_heatmap_paths': []}
    matrices = model_weight_matrices(model)
    if not matrices:
        return {'weight_matrix_dir': '', 'weight_heatmap_paths': []}

    if not result_dir:
        result_dir = os.path.join('./results', setting)
    weight_dir = os.path.join(result_dir, 'weight_matrices')
    central_weight_dir = ''
    configured_central_dir = str(getattr(args, 'weight_heatmap_dir', '') or '').strip()
    if configured_central_dir and configured_central_dir.lower() not in ('none', 'false', '0'):
        central_weight_dir = os.path.join(configured_central_dir, _safe_dir_name(setting))
    output_dirs = [weight_dir]
    if central_weight_dir and os.path.abspath(central_weight_dir) != os.path.abspath(weight_dir):
        output_dirs.append(central_weight_dir)
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
    heatmap_paths = []
    metadata = model_metadata(model, args)
    split_factor = int(metadata.get('variate_token_split_factor', getattr(args, 'variate_token_split_factor', 1)) or 1)

    for name, matrix in matrices.items():
        matrix = matrix.detach().float().cpu().numpy()
        xlabel, ylabel = _weight_axis_labels(name)
        source_labels = None
        if name == 'compress_weight' and split_factor > 1:
            source_labels = _split_source_labels(metadata.get('target_variate_tokens', args.enc_in), split_factor)

        for output_dir in output_dirs:
            npy_path = os.path.join(output_dir, '{}.npy'.format(name))
            np.save(npy_path, matrix)

            if source_labels:
                labels_path = os.path.join(output_dir, '{}_source_labels.txt'.format(name))
                with open(labels_path, 'w') as f:
                    f.write('\n'.join(source_labels))
                    f.write('\n')

            raw_path = os.path.join(output_dir, '{}_heatmap.png'.format(name))
            if _save_heatmap(matrix, raw_path, '{} raw signed'.format(name),
                             cmap='coolwarm', symmetric=True, xlabel=xlabel, ylabel=ylabel):
                if output_dir == weight_dir:
                    heatmap_paths.append(raw_path)

            norm = _row_normalized_abs(matrix)
            norm_path = os.path.join(output_dir, '{}_abs_row_normalized_heatmap.png'.format(name))
            if _save_heatmap(norm, norm_path, '{} abs row-normalized'.format(name),
                             cmap='viridis', symmetric=False, xlabel=xlabel, ylabel=ylabel):
                if output_dir == weight_dir:
                    heatmap_paths.append(norm_path)

            if name == 'compress_weight':
                cosine_path = os.path.join(output_dir, '{}_row_cosine_heatmap.png'.format(name))
                if _save_heatmap(_row_cosine(matrix), cosine_path, '{} row cosine'.format(name),
                                 cmap='coolwarm', symmetric=True,
                                 xlabel='latent token index', ylabel='latent token index'):
                    if output_dir == weight_dir:
                        heatmap_paths.append(cosine_path)
                top_sources_path = os.path.join(output_dir, '{}_top_sources.txt'.format(name))
                _save_top_sources(matrix, top_sources_path, labels=source_labels, topk=10)

    return {
        'weight_matrix_dir': weight_dir,
        'central_weight_matrix_dir': central_weight_dir,
        'weight_heatmap_paths': heatmap_paths,
    }


def _save_matrix_csv(matrix, path):
    np.savetxt(path, matrix, delimiter=',')


def _top_entry_string(row, topk):
    if row.size == 0:
        return ''
    topk = min(int(topk), row.size)
    order = np.argsort(np.abs(row))[::-1][:topk]
    return ';'.join('{}:{:.8g}'.format(int(index), float(row[index])) for index in order)


def _effective_support_np(row, eps=1e-12):
    values = np.abs(row).astype(np.float64)
    denom = values.sum()
    if denom <= eps:
        return 0.0
    prob = values / denom
    return float(np.exp(-np.sum(prob * np.log(np.maximum(prob, eps)))))


def save_slot_diagnostics(args, model, result_dir, setting):
    if not getattr(args, 'export_slot_diagnostics', False):
        return {'slot_diagnostic_dir': ''}
    matrices = model_slot_diagnostic_matrices(model)
    required = ('A', 'B', 'W_eff')
    if not all(name in matrices for name in required):
        return {'slot_diagnostic_dir': ''}
    if not result_dir:
        result_dir = os.path.join('./results', setting)
    diag_dir = os.path.join(result_dir, 'slot_diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    arrays = {}
    for name in required:
        matrix = matrices[name].detach().float().cpu().numpy()
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        arrays[name] = matrix
        np.save(os.path.join(diag_dir, '{}.npy'.format(name)), matrix)
        _save_matrix_csv(matrix, os.path.join(diag_dir, '{}.csv'.format(name)))

    A = arrays['A']
    B = arrays['B']
    rows, cols = A.shape if A.ndim == 2 else (0, 0)
    metadata = model_metadata(model, args)
    top_r = int(metadata.get('cycle_topr_size', 0) or 0)
    if top_r <= 0 and cols > 0:
        top_r = min(10, cols)
    top_r = max(1, min(top_r, cols)) if cols > 0 else 0
    display_topk = min(10, cols) if cols > 0 else 0

    card_path = os.path.join(diag_dir, 'slot_card.csv')
    with open(card_path, 'w', newline='') as f:
        fieldnames = [
            'slot_id',
            'top_source_variables',
            'top_target_variables',
            'jaccard',
            'a_effective_support',
            'b_effective_support',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for slot_id in range(rows):
            a_row = A[slot_id]
            b_row = B[slot_id] if slot_id < B.shape[0] else np.zeros_like(a_row)
            a_top = set(np.argsort(np.abs(a_row))[::-1][:top_r].tolist()) if top_r > 0 else set()
            b_top = set(np.argsort(np.abs(b_row))[::-1][:top_r].tolist()) if top_r > 0 else set()
            union = a_top | b_top
            jaccard = float(len(a_top & b_top) / len(union)) if union else 0.0
            writer.writerow({
                'slot_id': slot_id,
                'top_source_variables': _top_entry_string(a_row, display_topk),
                'top_target_variables': _top_entry_string(b_row, display_topk),
                'jaccard': jaccard,
                'a_effective_support': _effective_support_np(a_row, eps=getattr(args, 'cycle_eps', 1e-8)),
                'b_effective_support': _effective_support_np(b_row, eps=getattr(args, 'cycle_eps', 1e-8)),
            })

    return {'slot_diagnostic_dir': diag_dir}


def reset_cuda_peak_memory(args):
    if not getattr(args, 'use_gpu', False) or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(args.gpu)
        torch.cuda.reset_peak_memory_stats(args.gpu)
    except Exception:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def cuda_memory_metrics(args):
    if not getattr(args, 'use_gpu', False) or not torch.cuda.is_available():
        return {
            'peak_gpu_allocated_mb': '',
            'peak_gpu_reserved_mb': '',
            'gpu_memory_footprint_mb': '',
        }
    try:
        torch.cuda.synchronize(args.gpu)
        allocated = torch.cuda.max_memory_allocated(args.gpu)
        reserved = torch.cuda.max_memory_reserved(args.gpu)
    except Exception:
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
    return {
        'peak_gpu_allocated_mb': allocated / (1024 ** 2),
        'peak_gpu_reserved_mb': reserved / (1024 ** 2),
        'gpu_memory_footprint_mb': allocated / (1024 ** 2),
    }


def append_success_row(args, setting, metrics, elapsed_sec, itr_index, model, train_runtime=None):
    train_runtime = train_runtime or {}
    metadata = model_metadata(model, args)
    memory = cuda_memory_metrics(args)
    reduced_k = metadata.get('reduced_variate_tokens', args.reduced_variate_k)
    actual_k_ratio = float(reduced_k) / float(args.enc_in) if args.enc_in else ''
    source_variate_tokens = metadata.get('source_variate_tokens', args.enc_in)
    source_k_ratio = float(reduced_k) / float(source_variate_tokens) if source_variate_tokens else ''
    logical_k = metadata.get('logical_k', reduced_k)
    executed_k = metadata.get('executed_k', reduced_k)
    logical_k_ratio = float(logical_k) / float(args.enc_in) if args.enc_in else ''
    executed_k_ratio = float(executed_k) / float(args.enc_in) if args.enc_in else ''
    original_encoder_tokens = metadata.get('original_encoder_tokens', '')
    reduced_encoder_tokens = metadata.get('reduced_encoder_tokens', '')
    token_ratio = ''
    token_reduction_percent = ''
    attention_score_ratio = ''
    attention_score_reduction_percent = ''
    if original_encoder_tokens not in ('', 0) and reduced_encoder_tokens != '':
        token_ratio = float(reduced_encoder_tokens) / float(original_encoder_tokens)
        token_reduction_percent = 100.0 * (1.0 - token_ratio)
        attention_score_ratio = token_ratio ** 2
        attention_score_reduction_percent = 100.0 * (1.0 - attention_score_ratio)
    is_baseline = args.variate_reduction_type == 'none'
    method_family = getattr(args, 'method_family', '') or ('iTransformer' if is_baseline else args.variate_reduction_type)
    method_name = getattr(args, 'method_name', '') or method_family
    avg_iter_time = train_runtime.get('avg_train_iter_time_sec', '')
    throughput_iter_per_sec = ''
    if avg_iter_time not in ('', 0):
        throughput_iter_per_sec = 1.0 / float(avg_iter_time)
    parameter_count = sum(p.numel() for p in model.parameters()) if model is not None else ''
    sparsity_keys = [
        'avg_comp_entropy', 'std_comp_entropy',
        'avg_effective_support', 'std_effective_support',
        'avg_top1_mass', 'avg_top5_mass', 'avg_top10_mass',
        'avg_pairwise_overlap_top5', 'avg_pairwise_overlap_top10',
        'avg_pairwise_cosine_between_latents',
        'w_comp_density', 'w_exp_density', 'w_eff_density',
        'avg_w_eff_entropy', 'avg_w_eff_top5_mass',
        'cycle_topr_jaccard', 'cycle_a_effective_support', 'cycle_b_effective_support',
        'cycle_slot_overlap', 'cycle_topr_size',
    ]
    row = {
        'dataset': getattr(args, 'experiment_tag', '') or args.model_id.split('_pl')[0],
        'pred_len': args.pred_len,
        'method_family': method_family,
        'method_name': method_name,
        'selection_id': getattr(args, 'selection_id', ''),
        'selected_recipe': getattr(args, 'selected_recipe', ''),
        'manual_override': getattr(args, 'manual_override', '0'),
        'variate_reduction_type': args.variate_reduction_type,
        'variate_expansion_type': args.variate_expansion_type,
        'expansion_weight_source': expansion_weight_source(
            args.variate_reduction_type, args.variate_expansion_type
        ),
        'variate_decode_stage': metadata.get(
            'variate_decode_stage', getattr(args, 'variate_decode_stage', 'feature')
        ),
        'compute_reducer_aux_losses': metadata.get('compute_reducer_aux_losses', False),
        'loss_variant': getattr(args, 'loss_variant', ''),
        'forecast_loss_type': getattr(args, 'forecast_loss_type', 'mse'),
        'huber_delta': getattr(args, 'huber_delta', 1.0),
        'variate_token_split_factor': metadata.get(
            'variate_token_split_factor', getattr(args, 'variate_token_split_factor', 1)
        ),
        'num_variates': args.enc_in,
        'target_variate_tokens': metadata.get('target_variate_tokens', args.enc_in),
        'source_variate_tokens': source_variate_tokens,
        'reduced_variate_k': reduced_k,
        'reduced_variate_tokens': metadata.get('reduced_variate_tokens', ''),
        'k_ratio': '' if is_baseline else args.target_k_ratio,
        'num_latents_K': '' if is_baseline else reduced_k,
        'target_k_ratio': args.target_k_ratio,
        'actual_k_ratio': actual_k_ratio,
        'source_k_ratio': source_k_ratio,
        'logical_k': logical_k,
        'executed_k': executed_k,
        'logical_k_ratio': logical_k_ratio,
        'executed_k_ratio': executed_k_ratio,
        'k_budget_max': metadata.get('k_budget_max', strict_k_budget(args.enc_in)[0]),
        'k_budget_exception': metadata.get('k_budget_exception', strict_k_budget(args.enc_in)[1]),
        'lowrank_reducer_rank': metadata.get('lowrank_reducer_rank', getattr(args, 'lowrank_reducer_rank', 0)),
        'decoder_residual_gate_init': metadata.get(
            'decoder_residual_gate_init', getattr(args, 'decoder_residual_gate_init', 0.0)
        ),
        'backbone_residual_gate_init': metadata.get(
            'backbone_residual_gate_init', getattr(args, 'backbone_residual_gate_init', 1.0)
        ),
        'backbone_residual_gate_type': metadata.get(
            'backbone_residual_gate_type', getattr(args, 'backbone_residual_gate_type', 'scalar')
        ),
        'backbone_residual_gate': metadata.get('backbone_residual_gate', 1.0),
        'backbone_residual_gate_abs_mean': metadata.get('backbone_residual_gate_abs_mean', 1.0),
        'backbone_residual_gate_max': metadata.get('backbone_residual_gate_max', 1.0),
        'local_temporal_rank': metadata.get('local_temporal_rank', getattr(args, 'local_temporal_rank', 0)),
        'zero_init_projector': metadata.get('zero_init_projector', getattr(args, 'zero_init_projector', False)),
        'skip_backbone': metadata.get('skip_backbone', getattr(args, 'skip_backbone', False)),
        'k_selection_mode': args.k_selection_mode,
        'k_ratio_denominator': args.k_ratio_denominator,
        'target_k_value': args.target_k_value,
        'variate_anchor_map_path': getattr(args, 'variate_anchor_map_path', ''),
        'original_encoder_tokens': metadata.get('original_encoder_tokens', ''),
        'reduced_encoder_tokens': metadata.get('reduced_encoder_tokens', ''),
        'num_extra_tokens': metadata.get('num_extra_tokens', ''),
        'token_ratio': token_ratio,
        'token_reduction_percent': token_reduction_percent,
        'attention_score_ratio': attention_score_ratio,
        'attention_score_reduction_percent': attention_score_reduction_percent,
        'eval_split': metrics.get('eval_split', 'test'),
        'skip_epoch_test_eval': getattr(args, 'skip_epoch_test_eval', False),
        'mse': metrics.get('mse', ''),
        'mae': metrics.get('mae', ''),
        'rmse': metrics.get('rmse', ''),
        'mape': metrics.get('mape', ''),
        'mspe': metrics.get('mspe', ''),
        'orthogonal_loss_weight': args.orthogonal_loss_weight,
        'reconstruction_loss_weight': args.reconstruction_loss_weight,
        'coverage_loss_weight': args.coverage_loss_weight,
        'assignment_entropy_loss_weight': args.assignment_entropy_loss_weight,
        'wcomp_entropy_loss_weight': getattr(args, 'wcomp_entropy_loss_weight', 0.0),
        'group_attention_entropy_loss_weight': '' if is_baseline else getattr(
            args, 'group_attention_entropy_loss_weight', 0.0
        ),
        'group_attention_entropy_target': '' if is_baseline else getattr(
            args, 'group_attention_entropy_target', metadata.get('group_attention_entropy_target', 0.35)
        ),
        'lambda_orth': args.orthogonal_loss_weight,
        'lambda_entropy': '' if is_baseline else getattr(args, 'wcomp_entropy_loss_weight', 0.0),
        'use_cycle_slot_loss': getattr(args, 'use_cycle_slot_loss', False),
        'cycle_loss_weight': getattr(args, 'cycle_loss_weight', 0.0),
        'cycle_warmup_ratio': getattr(args, 'cycle_warmup_ratio', 0.0),
        'cycle_topr': getattr(args, 'cycle_topr', 0),
        'cycle_topr_multiplier': getattr(args, 'cycle_topr_multiplier', 1.0),
        'cycle_b_norm': getattr(args, 'cycle_b_norm', 'row_l1'),
        'cycle_eps': getattr(args, 'cycle_eps', 1e-8),
        'export_slot_diagnostics': getattr(args, 'export_slot_diagnostics', False),
        'cycle_loss': train_runtime.get('cycle_loss', ''),
        'weighted_cycle_loss': train_runtime.get('weighted_cycle_loss', ''),
        'cycle_current_weight': train_runtime.get('cycle_current_weight', ''),
        'mae_loss_weight': args.mae_loss_weight,
        'train_time_sec': train_runtime.get('train_time_sec', ''),
        'train_iter_count': train_runtime.get('train_iter_count', ''),
        'avg_train_iter_time_sec': train_runtime.get('avg_train_iter_time_sec', ''),
        'throughput_iter_per_sec': throughput_iter_per_sec,
        'last_epoch_time_sec': train_runtime.get('last_epoch_time_sec', ''),
        'avg_epoch_time_sec': train_runtime.get('avg_epoch_time_sec', ''),
        'throughput_samples_per_sec': train_runtime.get('throughput_samples_per_sec', ''),
        'elapsed_sec': elapsed_sec,
        'parameter_count': parameter_count,
        'peak_gpu_allocated_mb': memory.get('peak_gpu_allocated_mb', ''),
        'peak_gpu_reserved_mb': memory.get('peak_gpu_reserved_mb', ''),
        'gpu_memory_footprint_mb': memory.get('gpu_memory_footprint_mb', ''),
        'expansion_temperature': args.expansion_temperature,
        'expansion_topk': args.expansion_topk,
        'normalization': '' if is_baseline else getattr(args, 'wcomp_normalization', ''),
        'entmax_alpha': '' if is_baseline else getattr(args, 'entmax_alpha', ''),
        'sparse_compress_topk': metadata.get('sparse_compress_topk', getattr(args, 'sparse_compress_topk', 0)),
        'sparse_expand_topk': metadata.get('sparse_expand_topk', getattr(args, 'sparse_expand_topk', 0)),
        'residual_gate_mean': metadata.get('residual_gate_mean', 0.0),
        'residual_gate_abs_mean': metadata.get('residual_gate_abs_mean', 0.0),
        'residual_gate_max': metadata.get('residual_gate_max', 0.0),
        'hybrid_linear_gate': metadata.get('hybrid_linear_gate', 0.0),
        'local_temporal_branch': getattr(args, 'local_temporal_branch', 'none'),
        'local_temporal_init': getattr(args, 'local_temporal_init', ''),
        'local_temporal_gate_init': getattr(args, 'local_temporal_gate_init', ''),
        'local_temporal_gate': metadata.get('local_temporal_gate', 0.0),
        'output_calibration': metadata.get('output_calibration', getattr(args, 'output_calibration', 'none')),
        'output_calibration_scale_abs_mean': metadata.get('output_calibration_scale_abs_mean', 1.0),
        'output_calibration_bias_abs_mean': metadata.get('output_calibration_bias_abs_mean', 0.0),
        'status': 'success',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_commit': git_value(['git', 'rev-parse', 'HEAD']),
        'git_dirty': git_dirty(),
        'error': '',
        'command': '',
        'returncode': 0,
        'gpu': args.gpu,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'data': args.data,
        'root_path': args.root_path,
        'data_path': args.data_path,
        'target': args.target,
        'freq': args.freq,
        'seed': args.seed,
        'itr': itr_index,
        'model': args.model,
        'model_id': args.model_id,
        'setting': setting,
        'seq_len': args.seq_len,
        'label_len': args.label_len,
        'e_layers': args.e_layers,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'train_epochs': args.train_epochs,
        'patience': args.patience,
        'num_workers': args.num_workers,
        'use_norm': args.use_norm,
        'checkpoint_dir': os.path.join(args.checkpoints, setting),
        'result_dir': metrics.get('result_dir', ''),
        'weight_matrix_dir': metrics.get('weight_matrix_dir', ''),
        'slot_diagnostic_dir': metrics.get('slot_diagnostic_dir', ''),
        'plot_dir': '' if is_baseline else metrics.get('central_weight_matrix_dir', metrics.get('weight_matrix_dir', '')),
    }
    for key in sparsity_keys:
        row[key] = '' if is_baseline else metadata.get(key, '')
    append_csv_row(args.result_csv, row)


def wandb_tags(tags):
    if not tags:
        return []
    return [tag.strip() for tag in str(tags).split(',') if tag.strip()]


def wandb_config(args, setting):
    config = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith('_') and isinstance(value, (str, int, float, bool, type(None)))
    }
    config.update({
        'setting': setting,
        'git_commit': git_value(['git', 'rev-parse', 'HEAD']),
        'git_dirty': git_dirty(),
    })
    return config


def init_wandb(args, setting):
    if not getattr(args, 'use_wandb', False):
        return None
    try:
        import wandb
    except ImportError:
        print('W&B logging requested, but wandb is not installed. Continuing without W&B.')
        args.use_wandb = False
        return None

    dataset = getattr(args, 'experiment_tag', '') or args.model_id.split('_pl')[0]
    group = args.wandb_group or '{}_{}_{}'.format(
        dataset,
        args.variate_reduction_type,
        args.variate_expansion_type,
    )
    name = args.wandb_run_name or args.model_id
    init_kwargs = {
        'project': args.wandb_project,
        'entity': args.wandb_entity or None,
        'mode': args.wandb_mode,
        'name': name,
        'group': group,
        'tags': wandb_tags(args.wandb_tags),
        'config': wandb_config(args, setting),
        'reinit': True,
    }
    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        if args.wandb_mode != 'offline':
            print('W&B init failed in mode={} ({}). Retrying with offline mode.'.format(
                args.wandb_mode,
                exc,
            ))
            init_kwargs['mode'] = 'offline'
            try:
                run = wandb.init(**init_kwargs)
            except Exception as offline_exc:
                print('W&B offline init also failed ({}). Continuing without W&B.'.format(offline_exc))
                args.use_wandb = False
                return None
        else:
            print('W&B init failed ({}). Continuing without W&B.'.format(exc))
            args.use_wandb = False
            return None
    setattr(args, '_wandb_run_id', getattr(run, 'id', ''))
    setattr(args, '_wandb_run_name', getattr(run, 'name', name))
    return run


def log_wandb_final(args, metrics, elapsed_sec, train_runtime, model):
    if not getattr(args, 'use_wandb', False):
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    metadata = model_metadata(model, args)
    reduced_tokens = metadata.get('reduced_variate_tokens', '')
    source_tokens = metadata.get('source_variate_tokens', '')
    source_k_ratio = ''
    if reduced_tokens != '' and source_tokens not in ('', 0):
        source_k_ratio = float(reduced_tokens) / float(source_tokens)
    payload = {
        'final/mse': metrics.get('mse', ''),
        'final/mae': metrics.get('mae', ''),
        'final/rmse': metrics.get('rmse', ''),
        'final/mape': metrics.get('mape', ''),
        'final/mspe': metrics.get('mspe', ''),
        'runtime/elapsed_sec': elapsed_sec,
        'runtime/train_time_sec': train_runtime.get('train_time_sec', ''),
        'runtime/train_iter_count': train_runtime.get('train_iter_count', ''),
        'runtime/avg_train_iter_time_sec': train_runtime.get('avg_train_iter_time_sec', ''),
        'runtime/last_epoch_time_sec': train_runtime.get('last_epoch_time_sec', ''),
        'runtime/avg_epoch_time_sec': train_runtime.get('avg_epoch_time_sec', ''),
        'runtime/throughput_samples_per_sec': train_runtime.get('throughput_samples_per_sec', ''),
        'tokens/variate_token_split_factor': metadata.get('variate_token_split_factor', ''),
        'tokens/compute_reducer_aux_losses': metadata.get('compute_reducer_aux_losses', False),
        'tokens/original_variate_tokens': metadata.get('original_variate_tokens', ''),
        'tokens/target_variate_tokens': metadata.get('target_variate_tokens', ''),
        'tokens/source_variate_tokens': metadata.get('source_variate_tokens', ''),
        'tokens/reduced_variate_tokens': metadata.get('reduced_variate_tokens', ''),
        'tokens/logical_k': metadata.get('logical_k', ''),
        'tokens/executed_k': metadata.get('executed_k', ''),
        'tokens/k_budget_max': metadata.get('k_budget_max', ''),
        'tokens/k_budget_exception': metadata.get('k_budget_exception', ''),
        'tokens/lowrank_reducer_rank': metadata.get('lowrank_reducer_rank', ''),
        'gates/decoder_residual_gate_init': metadata.get('decoder_residual_gate_init', ''),
        'local_temporal/rank': metadata.get('local_temporal_rank', ''),
        'model/zero_init_projector': metadata.get('zero_init_projector', False),
        'tokens/source_k_ratio': source_k_ratio,
        'tokens/original_encoder_tokens': metadata.get('original_encoder_tokens', ''),
        'tokens/reduced_encoder_tokens': metadata.get('reduced_encoder_tokens', ''),
        'tokens/num_extra_tokens': metadata.get('num_extra_tokens', ''),
        'gates/residual_gate_mean': metadata.get('residual_gate_mean', 0.0),
        'gates/residual_gate_abs_mean': metadata.get('residual_gate_abs_mean', 0.0),
        'gates/residual_gate_max': metadata.get('residual_gate_max', 0.0),
        'gates/hybrid_linear_gate': metadata.get('hybrid_linear_gate', 0.0),
        'local_temporal/branch': metadata.get('local_temporal_branch', ''),
        'local_temporal/gate': metadata.get('local_temporal_gate', 0.0),
        'cycle/use_cycle_slot_loss': getattr(args, 'use_cycle_slot_loss', False),
        'cycle/loss_weight': getattr(args, 'cycle_loss_weight', 0.0),
        'cycle/warmup_ratio': getattr(args, 'cycle_warmup_ratio', 0.0),
        'cycle/topr': getattr(args, 'cycle_topr', 0),
        'cycle/topr_multiplier': getattr(args, 'cycle_topr_multiplier', 1.0),
        'cycle/loss': train_runtime.get('cycle_loss', ''),
        'cycle/weighted_loss': train_runtime.get('weighted_cycle_loss', ''),
        'cycle/current_weight': train_runtime.get('cycle_current_weight', ''),
        'cycle/topr_size': metadata.get('cycle_topr_size', 0),
        'cycle/topr_jaccard': metadata.get('cycle_topr_jaccard', 0.0),
        'cycle/a_effective_support': metadata.get('cycle_a_effective_support', 0.0),
        'cycle/b_effective_support': metadata.get('cycle_b_effective_support', 0.0),
        'cycle/slot_overlap': metadata.get('cycle_slot_overlap', 0.0),
        'cycle/w_eff_density': metadata.get('w_eff_density', 0.0),
    }
    wandb.log(payload)
    heatmap_paths = metrics.get('weight_heatmap_paths', [])
    if heatmap_paths:
        images = {}
        for path in heatmap_paths:
            key = 'weights/' + os.path.splitext(os.path.basename(path))[0]
            images[key] = wandb.Image(path)
        wandb.log(images)
    wandb.run.summary.update(payload)


def finish_wandb(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--save_test_arrays', action='store_true',
                        help='save test pred.npy and true.npy arrays')
    parser.add_argument('--save_test_visuals', action='store_true',
                        help='save test visualization PDFs')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--max_train_batches', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--max_eval_batches', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--forecast_loss_type', type=str, default='mse', choices=['mse', 'smooth_l1', 'huber'])
    parser.add_argument('--huber_delta', type=float, default=1.0)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', type=str2bool, nargs='?', const=True,
                        help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
    parser.add_argument('--variate_reduction_type', type=str, default='none',
                        choices=[
                            'none',
                            'mlp_static_combination',
                            'learned_anchor_selection',
                            'sample_adaptive_selection',
                            'variate_anchor_selection',
                            'variate_group_adaptive_selection',
                            'variate_group_adaptive_id_selection',
                            'variate_anchor_residual_pool',
                            'variate_anchor_channel_residual_pool',
                            'variate_anchor_grouplinear_residual_pool',
                            'variate_anchor_grouplinear_residual_pool_id',
                            'variate_anchor_softmax_pool',
                            'variate_grouped_average',
                            'variate_grouped_average_id',
                            'variate_grouped_corr_pool',
                            'variate_grouped_softmax_pool',
                            'variate_grouped_softmax_pool_id',
                            'variate_grouped_linear_pool',
                            'variate_grouped_linear_pool_id',
                            'grouped_soft_representative',
                            'MLP_attention',
                            'latent_query_attention',
                            'mlp_slot_attention',
                            'mlp_sparse_slot_attention',
                            'mlp_sparse_slot_attention_id',
                            'mlp_linear_sparse_slot_attention_id',
                            'mlp_static_sparse_slot_attention_id',
                            'mlp_slot_attention_hybrid_linear',
                        ])
    parser.add_argument('--reduced_variate_k', type=int, default=0)
    parser.add_argument('--lowrank_reducer_rank', type=int, default=8)
    parser.add_argument('--variate_expansion_type', type=str, default='transpose',
                        choices=[
                            'transpose',
                            'column_normalized',
                            'sharpened_column_normalized',
                            'topk_column_normalized',
                            'fixed_assignment',
                            'fixed_assignment_scalar_residual',
                            'fixed_masked_linear',
                            'fixed_masked_scalar_residual',
                            'fixed_masked_variate_residual',
                            'masked_linear',
                            'masked_softmax',
                            'masked_softmax_scalar_residual',
                            'masked_linear_scalar_residual',
                            'masked_softmax_variate_residual',
                            'slot_learned_linear',
                            'id_query_decoder',
                            'id_query_scalar_residual',
                            'id_query_fixed_scalar_residual',
                            'id_topk_decoder',
                            'id_topk_scalar_residual',
                            'id_topk_fixed_scalar_residual',
                            'query_decoder',
                            'query_decoder_scalar_residual',
                            'query_decoder_variate_residual',
                            'token_query_decoder',
                        ])
    parser.add_argument('--variate_decode_stage', type=str, default='feature',
                        choices=['feature', 'forecast'])
    parser.add_argument('--expansion_temperature', type=float, default=1.0)
    parser.add_argument('--expansion_topk', type=int, default=0)
    parser.add_argument('--wcomp_normalization', type=str, default='softmax', choices=['softmax', 'entmax15'])
    parser.add_argument('--entmax_alpha', type=float, default=1.5)
    parser.add_argument('--method_family', type=str, default='')
    parser.add_argument('--method_name', type=str, default='')
    parser.add_argument('--sparse_compress_topk', type=int, default=0)
    parser.add_argument('--sparse_expand_topk', type=int, default=0)
    parser.add_argument('--decoder_residual_gate_init', type=float, default=0.0)
    parser.add_argument('--backbone_residual_gate_init', type=float, default=1.0)
    parser.add_argument('--backbone_residual_gate_type', type=str, default='scalar', choices=['scalar', 'variate'])
    parser.add_argument('--orthogonal_loss_weight', type=float, default=0.0)
    parser.add_argument('--reconstruction_loss_weight', type=float, default=0.0)
    parser.add_argument('--coverage_loss_weight', type=float, default=0.0)
    parser.add_argument('--assignment_entropy_loss_weight', type=float, default=0.0)
    parser.add_argument('--wcomp_entropy_loss_weight', type=float, default=0.0)
    parser.add_argument('--group_attention_entropy_loss_weight', type=float, default=0.0)
    parser.add_argument('--group_attention_entropy_target', type=float, default=0.35)
    parser.add_argument('--use_cycle_slot_loss', type=str2bool, default=False)
    parser.add_argument('--cycle_loss_weight', type=float, default=0.0)
    parser.add_argument('--cycle_warmup_ratio', type=float, default=0.0)
    parser.add_argument('--cycle_topr', type=int, default=0)
    parser.add_argument('--cycle_topr_multiplier', type=float, default=1.0)
    parser.add_argument('--cycle_b_norm', type=str, default='row_l1', choices=['row_l1'])
    parser.add_argument('--cycle_eps', type=float, default=1e-8)
    parser.add_argument('--export_slot_diagnostics', type=str2bool, default=False)
    # Deprecated generation-only loss flags are accepted as zero for old shell
    # commands, but non-zero values are rejected during argument validation.
    parser.add_argument('--linear_coverage_loss_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--biorthogonal_loss_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--linear_weight_l2_loss_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--decoder_init_l2_loss_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--linear_entropy_loss_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--linear_cosine_loss_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--linear_decoder_coverage_loss_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--support_overlap_loss_weight', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--mae_loss_weight', type=float, default=0.0)
    parser.add_argument('--variate_token_split_factor', type=int, default=1)
    parser.add_argument('--enforce_k_budget', type=str2bool, default=True)
    parser.add_argument('--local_temporal_branch', type=str, default='none', choices=['none', 'linear', 'nlinear', 'nlinear_affine', 'nlinear_lowrank', 'nlinear_lowrank_affine', 'nlinear_decomp', 'nlinear_decomp_affine', 'anchor_residual_nlinear', 'anchor_residual_nlinear_affine', 'anchor_delta_nlinear', 'anchor_delta_nlinear_affine', 'anchor_mask_delta_nlinear', 'anchor_mask_delta_nlinear_affine', 'anchor_group_nlinear', 'anchor_group_nlinear_affine', 'nlinear_group', 'nlinear_group_affine', 'lowrank_linear', 'persistence', 'persistence_gate'])
    parser.add_argument('--local_temporal_init', type=str, default='persistence', choices=['persistence', 'zero'])
    parser.add_argument('--local_temporal_gate_init', type=float, default=1.0)
    parser.add_argument('--local_temporal_rank', type=int, default=4)
    parser.add_argument('--zero_init_projector', type=str2bool, default=False)
    parser.add_argument('--skip_backbone', type=str2bool, default=False)
    parser.add_argument('--output_calibration', type=str, default='none', choices=['none', 'variate_affine'])
    parser.add_argument('--loss_variant', type=str, default='')
    parser.add_argument('--result_csv', type=str, default='./results/mlp_variate_reduction_results.csv')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--experiment_tag', type=str, default='')
    parser.add_argument('--selection_id', type=str, default='')
    parser.add_argument('--selected_recipe', type=str, default='')
    parser.add_argument('--manual_override', type=str, default='0')
    parser.add_argument('--target_k_ratio', type=float, default=-1.0)
    parser.add_argument('--target_k_value', type=int, default=0)
    parser.add_argument('--k_selection_mode', type=str, default='baseline',
                        choices=['baseline', 'ratio', 'absolute', 'fixed'])
    parser.add_argument('--k_ratio_denominator', type=str, default='')
    parser.add_argument('--variate_anchor_map_path', type=str, default='')
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    parser.add_argument('--wandb_project', type=str, default='iTransformer-variate-reduction')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_group', type=str, default='')
    parser.add_argument('--wandb_run_name', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, default='')
    parser.add_argument('--save_weight_heatmaps', type=str2bool, default=True)
    parser.add_argument('--weight_heatmap_dir', type=str, default='./results/weight_heatmaps')
    parser.add_argument('--skip_test_eval', type=str2bool, default=False)
    parser.add_argument(
        '--skip_epoch_test_eval',
        type=str2bool,
        default=False,
        help='Skip per-epoch test-set validation during training while preserving the final test evaluation.',
    )

    args = parser.parse_args()
    set_seed(args.seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.variate_reduction_type != 'none' and args.reduced_variate_k <= 0:
        raise ValueError('--reduced_variate_k must be positive when variate_reduction_type is not none')
    if args.cycle_loss_weight < 0.0:
        raise ValueError('--cycle_loss_weight must be non-negative')
    if args.cycle_warmup_ratio < 0.0 or args.cycle_warmup_ratio > 1.0:
        raise ValueError('--cycle_warmup_ratio must be in [0, 1]')
    if args.cycle_topr < 0:
        raise ValueError('--cycle_topr must be non-negative')
    if args.cycle_topr_multiplier <= 0.0:
        raise ValueError('--cycle_topr_multiplier must be positive')
    if args.cycle_eps <= 0.0:
        raise ValueError('--cycle_eps must be positive')
    if args.cycle_b_norm != 'row_l1':
        raise ValueError('--cycle_b_norm supports only row_l1 in this minimal implementation')
    if args.exp_name == 'partial_train' and (args.use_cycle_slot_loss or args.export_slot_diagnostics):
        raise ValueError('cycle slot loss/diagnostics are supported only for exp_name=MTSF')
    if args.variate_reduction_type != 'none' and args.enforce_k_budget:
        k_budget_max, k_budget_exception = strict_k_budget(args.enc_in)
        if args.reduced_variate_k > k_budget_max:
            raise ValueError(
                '--reduced_variate_k={} exceeds K_max={} for V={} under strict K/V<0.30 budget'.format(
                    args.reduced_variate_k,
                    k_budget_max,
                    args.enc_in,
                )
            )
    if args.variate_token_split_factor < 1:
        raise ValueError('--variate_token_split_factor must be at least 1')
    if args.seq_len % args.variate_token_split_factor != 0:
        raise ValueError('--seq_len must be divisible by --variate_token_split_factor')
    if args.variate_token_split_factor > 1:
        raise ValueError('--variate_token_split_factor > 1 was only used by removed generation reducers')
    removed_loss_weights = {
        '--linear_coverage_loss_weight': args.linear_coverage_loss_weight,
        '--biorthogonal_loss_weight': args.biorthogonal_loss_weight,
        '--linear_weight_l2_loss_weight': args.linear_weight_l2_loss_weight,
        '--decoder_init_l2_loss_weight': args.decoder_init_l2_loss_weight,
        '--linear_entropy_loss_weight': args.linear_entropy_loss_weight,
        '--linear_cosine_loss_weight': args.linear_cosine_loss_weight,
        '--linear_decoder_coverage_loss_weight': args.linear_decoder_coverage_loss_weight,
        '--support_overlap_loss_weight': args.support_overlap_loss_weight,
    }
    nonzero_removed_losses = [
        name for name, value in removed_loss_weights.items()
        if float(value) != 0.0
    ]
    if nonzero_removed_losses:
        raise ValueError(
            'generation-only auxiliary losses were removed; set these to 0: {}'.format(
                ', '.join(nonzero_removed_losses)
            )
        )
    if args.variate_reduction_type == 'none':
        args.reduced_variate_k = args.enc_in
        args.variate_decode_stage = 'feature'
        args.orthogonal_loss_weight = 0.0
        args.reconstruction_loss_weight = 0.0
        args.coverage_loss_weight = 0.0
        args.assignment_entropy_loss_weight = 0.0
        args.wcomp_entropy_loss_weight = 0.0
        args.group_attention_entropy_loss_weight = 0.0
        args.use_cycle_slot_loss = False
        args.cycle_loss_weight = 0.0
        args.cycle_warmup_ratio = 0.0
        args.cycle_topr = 0
        args.cycle_topr_multiplier = 1.0
        args.cycle_b_norm = 'row_l1'
        args.cycle_eps = 1e-8
        args.export_slot_diagnostics = False
        args.linear_coverage_loss_weight = 0.0
        args.biorthogonal_loss_weight = 0.0
        args.linear_weight_l2_loss_weight = 0.0
        args.decoder_init_l2_loss_weight = 0.0
        args.linear_entropy_loss_weight = 0.0
        args.linear_cosine_loss_weight = 0.0
        args.linear_decoder_coverage_loss_weight = 0.0
        args.support_overlap_loss_weight = 0.0
        args.mae_loss_weight = 0.0
    if args.use_multi_gpu and args.variate_reduction_type != 'none':
        raise ValueError('Variate reduction auxiliary loss is currently supported only for single-GPU runs.')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            wandb_run = init_wandb(args, setting)
            try:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                start_time = time.time()
                reset_cuda_peak_memory(args)
                exp.train(setting)
                train_runtime = getattr(exp, 'train_runtime_stats', {})

                if args.skip_test_eval:
                    print('>>>>>>>validation metrics : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    metrics = exp.test(setting, flag='val', save_results=False)
                    metrics['eval_split'] = 'val'
                    metrics.update(save_slot_diagnostics(args, exp.model, metrics.get('result_dir', ''), setting))
                else:
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    metrics = exp.test(setting)
                    metrics['eval_split'] = metrics.get('eval_split', 'test')
                    metrics.update(save_weight_matrix_visuals(args, exp.model, metrics.get('result_dir', ''), setting))
                    metrics.update(save_slot_diagnostics(args, exp.model, metrics.get('result_dir', ''), setting))
                elapsed_sec = time.time() - start_time
                append_success_row(args, setting, metrics, elapsed_sec, ii, exp.model, train_runtime)
                log_wandb_final(args, metrics, elapsed_sec, train_runtime, exp.model)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)
            finally:
                finish_wandb(wandb_run)
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        wandb_run = init_wandb(args, setting)
        try:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            start_time = time.time()
            reset_cuda_peak_memory(args)
            metrics = exp.test(setting, test=1)
            metrics.update(save_weight_matrix_visuals(args, exp.model, metrics.get('result_dir', ''), setting))
            metrics.update(save_slot_diagnostics(args, exp.model, metrics.get('result_dir', ''), setting))
            elapsed_sec = time.time() - start_time
            append_success_row(args, setting, metrics, elapsed_sec, ii, exp.model)
            log_wandb_final(args, metrics, elapsed_sec, {}, exp.model)
        finally:
            finish_wandb(wandb_run)
            torch.cuda.empty_cache()
