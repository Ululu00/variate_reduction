import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.VariateReduction import VariateReducer
import numpy as np


def strict_k_budget(num_variates):
    k_max = int(np.ceil(0.30 * int(num_variates))) - 1
    if k_max >= 1:
        return k_max, False
    return 1, True


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.original_variate_tokens = int(configs.enc_in)
        self.target_variate_tokens = self.original_variate_tokens
        self.variate_token_split_factor = int(getattr(configs, 'variate_token_split_factor', 1))
        if self.variate_token_split_factor < 1:
            raise ValueError('variate_token_split_factor must be at least 1')
        if self.seq_len % self.variate_token_split_factor != 0:
            raise ValueError('seq_len must be divisible by variate_token_split_factor')
        self.source_variate_tokens = self.original_variate_tokens * self.variate_token_split_factor
        self.variate_reduction_type = getattr(configs, 'variate_reduction_type', 'none')
        self.variate_expansion_type = getattr(configs, 'variate_expansion_type', 'transpose')
        self.variate_decode_stage = getattr(configs, 'variate_decode_stage', 'feature')
        if self.variate_decode_stage not in {'feature', 'forecast'}:
            raise ValueError('Unknown variate_decode_stage: {}'.format(self.variate_decode_stage))
        self.expansion_temperature = getattr(configs, 'expansion_temperature', 1.0)
        self.expansion_topk = getattr(configs, 'expansion_topk', 0)
        self.wcomp_normalization = getattr(configs, 'wcomp_normalization', 'softmax')
        self.entmax_alpha = float(getattr(configs, 'entmax_alpha', 1.5))
        self.decoder_residual_gate_init = float(getattr(configs, 'decoder_residual_gate_init', 0.0))
        self.backbone_residual_gate_init = float(getattr(configs, 'backbone_residual_gate_init', 1.0))
        self.backbone_residual_gate_type = getattr(configs, 'backbone_residual_gate_type', 'scalar')
        if self.backbone_residual_gate_type not in {'scalar', 'variate'}:
            raise ValueError('Unknown backbone_residual_gate_type: {}'.format(self.backbone_residual_gate_type))
        self.lowrank_reducer_rank = int(getattr(configs, 'lowrank_reducer_rank', 8))
        self.local_temporal_rank = int(getattr(configs, 'local_temporal_rank', 4))
        self.zero_init_projector = bool(getattr(configs, 'zero_init_projector', False))
        self.skip_backbone = bool(getattr(configs, 'skip_backbone', False))
        self.output_calibration = getattr(configs, 'output_calibration', 'none')
        if self.output_calibration not in {'none', 'variate_affine'}:
            raise ValueError('Unknown output_calibration: {}'.format(self.output_calibration))
        self.enforce_k_budget = bool(getattr(configs, 'enforce_k_budget', True))
        self.k_budget_max, self.k_budget_exception = strict_k_budget(self.original_variate_tokens)
        if self.variate_token_split_factor > 1:
            raise ValueError('variate_token_split_factor > 1 was only used by removed generation reducers')
        self.use_reconstruction_loss = getattr(configs, 'reconstruction_loss_weight', 0.0) > 0.0
        self.use_coverage_loss = getattr(configs, 'coverage_loss_weight', 0.0) > 0.0
        self.use_assignment_entropy_loss = getattr(configs, 'assignment_entropy_loss_weight', 0.0) > 0.0
        self.use_wcomp_entropy_loss = getattr(configs, 'wcomp_entropy_loss_weight', 0.0) > 0.0
        self.use_group_attention_entropy_loss = getattr(configs, 'group_attention_entropy_loss_weight', 0.0) > 0.0
        self.use_cycle_slot_loss = bool(getattr(configs, 'use_cycle_slot_loss', False))
        self.export_slot_diagnostics = bool(getattr(configs, 'export_slot_diagnostics', False))
        self.compute_reducer_aux_losses = any(
            float(getattr(configs, name, 0.0)) > 0.0
            for name in (
                'orthogonal_loss_weight',
                'coverage_loss_weight',
                'assignment_entropy_loss_weight',
                'wcomp_entropy_loss_weight',
                'group_attention_entropy_loss_weight',
                'cycle_loss_weight',
            )
        ) or self.use_cycle_slot_loss
        self.reduced_variate_tokens = self.original_variate_tokens
        if self.variate_reduction_type != 'none':
            self.reduced_variate_tokens = int(getattr(configs, 'reduced_variate_k', 0))
            if self.enforce_k_budget and self.reduced_variate_tokens > self.k_budget_max:
                raise ValueError(
                    'reduced_variate_k={} exceeds K_max={} for V={} under strict K/V<0.30 budget'.format(
                        self.reduced_variate_tokens,
                        self.k_budget_max,
                        self.original_variate_tokens,
                    )
                )
            self.reducer = VariateReducer(
                reduction_type=self.variate_reduction_type,
                num_variates=self.source_variate_tokens,
                reduced_k=self.reduced_variate_tokens,
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                expansion_type=self.variate_expansion_type,
                expansion_temperature=self.expansion_temperature,
                expansion_topk=self.expansion_topk,
                use_coverage_loss=self.use_coverage_loss,
                use_assignment_entropy_loss=self.use_assignment_entropy_loss,
                target_num_variates=self.target_variate_tokens,
                lowrank_reducer_rank=self.lowrank_reducer_rank,
                decoder_residual_gate_init=self.decoder_residual_gate_init,
                variate_anchor_map_path=getattr(configs, 'variate_anchor_map_path', ''),
                compute_aux_losses=self.compute_reducer_aux_losses,
                force_assignment_cache=self.use_reconstruction_loss,
                sparse_compress_topk=getattr(configs, 'sparse_compress_topk', 0),
                sparse_expand_topk=getattr(configs, 'sparse_expand_topk', 0),
                wcomp_normalization=self.wcomp_normalization,
                entmax_alpha=self.entmax_alpha,
                use_wcomp_entropy_loss=self.use_wcomp_entropy_loss,
                use_group_attention_entropy_loss=self.use_group_attention_entropy_loss,
                group_attention_entropy_target=getattr(configs, 'group_attention_entropy_target', 0.35),
                use_cycle_slot_loss=self.use_cycle_slot_loss,
                cycle_topr=getattr(configs, 'cycle_topr', 0),
                cycle_topr_multiplier=getattr(configs, 'cycle_topr_multiplier', 1.0),
                cycle_b_norm=getattr(configs, 'cycle_b_norm', 'row_l1'),
                cycle_eps=getattr(configs, 'cycle_eps', 1e-8),
                export_slot_diagnostics=self.export_slot_diagnostics,
            )
        else:
            self.reducer = None
        if self.reducer is None:
            self.variate_decode_stage = 'feature'
        elif self.variate_decode_stage == 'forecast':
            unsupported_forecast_decoders = {
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
                'fixed_assignment_scalar_residual',
                'fixed_masked_scalar_residual',
                'fixed_masked_variate_residual',
                'masked_linear_scalar_residual',
                'masked_softmax_scalar_residual',
                'masked_softmax_variate_residual',
            }
            if self.variate_expansion_type in unsupported_forecast_decoders:
                raise ValueError(
                    'variate_decode_stage=forecast supports non-attention, non-residual decoders only; '
                    'got variate_expansion_type={}'.format(self.variate_expansion_type)
                )
        self.local_temporal_branch = getattr(configs, 'local_temporal_branch', 'none')
        if self.skip_backbone and self.local_temporal_branch == 'none':
            raise ValueError('skip_backbone requires a local_temporal_branch')
        if self.local_temporal_branch not in {
            'none',
            'linear',
            'nlinear',
            'nlinear_affine',
            'nlinear_lowrank',
            'nlinear_lowrank_affine',
            'nlinear_decomp',
            'nlinear_decomp_affine',
            'anchor_residual_nlinear',
            'anchor_residual_nlinear_affine',
            'anchor_delta_nlinear',
            'anchor_delta_nlinear_affine',
            'anchor_mask_delta_nlinear',
            'anchor_mask_delta_nlinear_affine',
            'anchor_group_nlinear',
            'anchor_group_nlinear_affine',
            'nlinear_group',
            'nlinear_group_affine',
            'lowrank_linear',
            'persistence',
            'persistence_gate',
        }:
            raise ValueError('Unknown local_temporal_branch: {}'.format(self.local_temporal_branch))
        if self.local_temporal_branch in {
            'linear',
            'nlinear',
            'nlinear_affine',
            'nlinear_lowrank',
            'nlinear_lowrank_affine',
            'nlinear_decomp',
            'nlinear_decomp_affine',
            'anchor_residual_nlinear',
            'anchor_residual_nlinear_affine',
            'anchor_delta_nlinear',
            'anchor_delta_nlinear_affine',
            'anchor_mask_delta_nlinear',
            'anchor_mask_delta_nlinear_affine',
        }:
            self.local_temporal_projection = nn.Linear(self.seq_len, self.pred_len)
            if self.local_temporal_branch in {'nlinear_decomp', 'nlinear_decomp_affine'}:
                self.local_temporal_trend_projection = nn.Linear(self.seq_len, self.pred_len)
                self.local_temporal_decomp_kernel = int(
                    getattr(configs, 'local_temporal_decomp_kernel', getattr(configs, 'moving_avg', 25))
                )
                if self.local_temporal_decomp_kernel < 1:
                    raise ValueError('local_temporal_decomp_kernel must be at least 1')
            else:
                self.local_temporal_trend_projection = None
                self.local_temporal_decomp_kernel = 0
            local_init = getattr(configs, 'local_temporal_init', 'persistence')
            if local_init not in {'persistence', 'zero'}:
                raise ValueError('Unknown local_temporal_init: {}'.format(local_init))
            with torch.no_grad():
                self.local_temporal_projection.weight.zero_()
                self.local_temporal_projection.bias.zero_()
                if self.local_temporal_trend_projection is not None:
                    self.local_temporal_trend_projection.weight.zero_()
                    self.local_temporal_trend_projection.bias.zero_()
                if local_init == 'persistence' and self.local_temporal_branch == 'linear':
                    self.local_temporal_projection.weight[:, -1] = 1.0
            gate_init = float(getattr(configs, 'local_temporal_gate_init', 1.0))
            self.local_temporal_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
            if self.local_temporal_branch in {'nlinear_lowrank', 'nlinear_lowrank_affine'}:
                if self.local_temporal_rank < 1:
                    raise ValueError('local_temporal_rank must be at least 1')
                self.local_temporal_basis_weight = nn.Parameter(
                    torch.empty(self.local_temporal_rank, self.pred_len, self.seq_len)
                )
                self.local_temporal_basis_bias = nn.Parameter(
                    torch.zeros(self.local_temporal_rank, self.pred_len)
                )
                self.local_temporal_variate_mix = nn.Parameter(
                    torch.zeros(self.original_variate_tokens, self.local_temporal_rank)
                )
                with torch.no_grad():
                    self.local_temporal_basis_weight.normal_(mean=0.0, std=1e-3)
            else:
                self.local_temporal_basis_weight = None
                self.local_temporal_basis_bias = None
                self.local_temporal_variate_mix = None
            if self.local_temporal_branch in {
                'nlinear_affine',
                'nlinear_lowrank_affine',
                'nlinear_decomp_affine',
                'anchor_residual_nlinear_affine',
                'anchor_delta_nlinear_affine',
                'anchor_mask_delta_nlinear_affine',
            }:
                self.local_temporal_variate_scale = nn.Parameter(
                    torch.ones(1, 1, self.original_variate_tokens)
                )
                self.local_temporal_horizon_bias = nn.Parameter(
                    torch.zeros(1, self.pred_len, self.original_variate_tokens)
                )
            else:
                self.local_temporal_variate_scale = None
                self.local_temporal_horizon_bias = None
        elif self.local_temporal_branch in {'anchor_group_nlinear', 'anchor_group_nlinear_affine'}:
            if self.reducer is None or not hasattr(self.reducer, 'group_ids'):
                raise ValueError('anchor_group_nlinear requires a fixed anchor/group variate reducer')
            self.local_temporal_projection = None
            self.local_temporal_trend_projection = None
            self.local_temporal_decomp_kernel = 0
            self.local_temporal_basis_weight = nn.Parameter(
                torch.zeros(self.reduced_variate_tokens, self.pred_len, self.seq_len)
            )
            self.local_temporal_basis_bias = nn.Parameter(
                torch.zeros(self.reduced_variate_tokens, self.pred_len)
            )
            self.local_temporal_variate_mix = None
            gate_init = float(getattr(configs, 'local_temporal_gate_init', 1.0))
            self.local_temporal_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
            if self.local_temporal_branch == 'anchor_group_nlinear_affine':
                self.local_temporal_variate_scale = nn.Parameter(
                    torch.ones(1, 1, self.original_variate_tokens)
                )
                self.local_temporal_horizon_bias = nn.Parameter(
                    torch.zeros(1, self.pred_len, self.original_variate_tokens)
                )
            else:
                self.local_temporal_variate_scale = None
                self.local_temporal_horizon_bias = None
        elif self.local_temporal_branch in {'nlinear_group', 'nlinear_group_affine'}:
            if self.local_temporal_rank < 1:
                raise ValueError('local_temporal_rank must be at least 1')
            self.local_temporal_projection = None
            self.local_temporal_trend_projection = None
            self.local_temporal_decomp_kernel = 0
            self.local_temporal_basis_weight = nn.Parameter(
                torch.empty(self.local_temporal_rank, self.pred_len, self.seq_len)
            )
            self.local_temporal_basis_bias = nn.Parameter(
                torch.zeros(self.local_temporal_rank, self.pred_len)
            )
            self.local_temporal_variate_mix = nn.Parameter(
                torch.zeros(self.original_variate_tokens, self.local_temporal_rank)
            )
            with torch.no_grad():
                self.local_temporal_basis_weight.normal_(mean=0.0, std=1e-3)
            gate_init = float(getattr(configs, 'local_temporal_gate_init', 1.0))
            self.local_temporal_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
            if self.local_temporal_branch == 'nlinear_group_affine':
                self.local_temporal_variate_scale = nn.Parameter(
                    torch.ones(1, 1, self.original_variate_tokens)
                )
                self.local_temporal_horizon_bias = nn.Parameter(
                    torch.zeros(1, self.pred_len, self.original_variate_tokens)
                )
            else:
                self.local_temporal_variate_scale = None
                self.local_temporal_horizon_bias = None
        elif self.local_temporal_branch == 'lowrank_linear':
            if self.local_temporal_rank < 1:
                raise ValueError('local_temporal_rank must be at least 1')
            self.local_temporal_projection = None
            self.local_temporal_trend_projection = None
            self.local_temporal_decomp_kernel = 0
            self.local_temporal_basis_weight = nn.Parameter(
                torch.zeros(self.local_temporal_rank, self.pred_len, self.seq_len)
            )
            self.local_temporal_basis_bias = nn.Parameter(
                torch.zeros(self.local_temporal_rank, self.pred_len)
            )
            self.local_temporal_variate_mix = nn.Parameter(
                torch.zeros(self.original_variate_tokens, self.local_temporal_rank)
            )
            self.local_temporal_variate_scale = None
            self.local_temporal_horizon_bias = None
            local_init = getattr(configs, 'local_temporal_init', 'persistence')
            if local_init not in {'persistence', 'zero'}:
                raise ValueError('Unknown local_temporal_init: {}'.format(local_init))
            if local_init == 'persistence':
                with torch.no_grad():
                    self.local_temporal_basis_weight[0, :, -1] = 1.0
                    self.local_temporal_variate_mix[:, 0] = 1.0
            gate_init = float(getattr(configs, 'local_temporal_gate_init', 1.0))
            self.local_temporal_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        elif self.local_temporal_branch == 'persistence':
            self.local_temporal_projection = None
            self.local_temporal_trend_projection = None
            self.local_temporal_decomp_kernel = 0
            self.local_temporal_basis_weight = None
            self.local_temporal_basis_bias = None
            self.local_temporal_variate_mix = None
            self.local_temporal_variate_scale = None
            self.local_temporal_horizon_bias = None
            gate_init = float(getattr(configs, 'local_temporal_gate_init', 1.0))
            self.register_buffer('local_temporal_gate', torch.tensor(gate_init, dtype=torch.float32), persistent=False)
        elif self.local_temporal_branch == 'persistence_gate':
            self.local_temporal_projection = None
            self.local_temporal_trend_projection = None
            self.local_temporal_decomp_kernel = 0
            self.local_temporal_basis_weight = None
            self.local_temporal_basis_bias = None
            self.local_temporal_variate_mix = None
            self.local_temporal_variate_scale = None
            self.local_temporal_horizon_bias = None
            gate_init = float(getattr(configs, 'local_temporal_gate_init', 0.5))
            self.local_temporal_gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        else:
            self.local_temporal_projection = None
            self.local_temporal_trend_projection = None
            self.local_temporal_decomp_kernel = 0
            self.local_temporal_basis_weight = None
            self.local_temporal_basis_bias = None
            self.local_temporal_variate_mix = None
            self.local_temporal_variate_scale = None
            self.local_temporal_horizon_bias = None
            self.register_buffer('local_temporal_gate', torch.tensor(0.0), persistent=False)
        self.register_buffer('_zero_aux_loss', torch.tensor(0.0), persistent=False)
        self.latest_aux_loss = self._zero_aux_loss
        self.latest_aux_losses = {
            'orthogonal': self._zero_aux_loss,
            'reconstruction': self._zero_aux_loss,
            'coverage': self._zero_aux_loss,
            'assignment_entropy': self._zero_aux_loss,
            'wcomp_entropy': self._zero_aux_loss,
            'cycle_slot': self._zero_aux_loss,
            'group_attention_entropy': self._zero_aux_loss,
        }
        self.latest_num_extra_tokens = 0
        self.latest_original_encoder_tokens = self.source_variate_tokens
        self.latest_reduced_encoder_tokens = self.reduced_variate_tokens
        self.latest_original_variate_tokens = self.original_variate_tokens
        self.latest_target_variate_tokens = self.target_variate_tokens
        self.latest_source_variate_tokens = self.source_variate_tokens
        self.latest_reduced_variate_tokens = self.reduced_variate_tokens
        self.latest_executed_k = self.reduced_variate_tokens
        self.latest_logical_k = self.reduced_variate_tokens
        self.latest_local_temporal_gate = 0.0
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout,
                                                    split_factor=self.variate_token_split_factor)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.zero_init_projector:
            with torch.no_grad():
                self.projector.weight.zero_()
                self.projector.bias.zero_()
        if self.backbone_residual_gate_type == 'variate':
            self.backbone_residual_gate = nn.Parameter(
                torch.full((1, 1, self.original_variate_tokens), self.backbone_residual_gate_init)
            )
        elif abs(self.backbone_residual_gate_init - 1.0) < 1e-12:
            self.register_buffer(
                'backbone_residual_gate',
                torch.tensor(1.0, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.backbone_residual_gate = nn.Parameter(
                torch.tensor(self.backbone_residual_gate_init, dtype=torch.float32)
            )
        if self.output_calibration == 'variate_affine':
            self.output_calibration_scale = nn.Parameter(torch.ones(1, 1, self.original_variate_tokens))
            self.output_calibration_bias = nn.Parameter(torch.zeros(1, self.pred_len, self.original_variate_tokens))
        else:
            self.output_calibration_scale = None
            self.output_calibration_bias = None
        self.latest_backbone_residual_gate = self.backbone_residual_gate_init
        self.latest_backbone_residual_gate_abs_mean = abs(self.backbone_residual_gate_init)
        self.latest_backbone_residual_gate_max = abs(self.backbone_residual_gate_init)
        self.latest_output_calibration_scale_abs_mean = 1.0
        self.latest_output_calibration_bias_abs_mean = 0.0

    def _moving_average(self, x, kernel_size):
        if kernel_size <= 1:
            return x
        left = (kernel_size - 1) // 2
        right = kernel_size - 1 - left
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (left, right), mode='replicate')
        return F.avg_pool1d(x_pad, kernel_size=kernel_size, stride=1).transpose(1, 2)

    def _anchor_residual_input(self, x_enc):
        if self.reducer is None or not hasattr(self.reducer, 'anchor_indices') or not hasattr(self.reducer, 'group_ids'):
            raise ValueError('anchor_residual local branch requires a fixed anchor variate reducer')
        if self.variate_token_split_factor != 1:
            raise ValueError('anchor_residual local branch requires variate_token_split_factor=1')
        group_ids = self.reducer.group_ids[:self.original_variate_tokens].to(x_enc.device)
        anchor_indices = self.reducer.anchor_indices.to(x_enc.device)
        assigned_anchor_indices = anchor_indices[group_ids]
        anchor_series = torch.index_select(x_enc, dim=2, index=assigned_anchor_indices)
        return x_enc - anchor_series

    def _anchor_mask_residual_input(self, x_enc):
        if (
            self.reducer is None or
            not hasattr(self.reducer, 'anchor_indices') or
            not hasattr(self.reducer, 'fixed_expand_indices') or
            not hasattr(self.reducer, 'fixed_expand_active')
        ):
            raise ValueError('anchor_mask_delta local branch requires a fixed masked-anchor reducer')
        if self.variate_token_split_factor != 1:
            raise ValueError('anchor_mask_delta local branch requires variate_token_split_factor=1')
        anchor_indices = self.reducer.anchor_indices.to(x_enc.device)
        anchor_series = torch.index_select(x_enc, dim=2, index=anchor_indices)
        expand_indices = self.reducer.fixed_expand_indices[:self.original_variate_tokens].to(x_enc.device)
        expand_active = self.reducer.fixed_expand_active[:self.original_variate_tokens].to(
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        batch_size, seq_len, _ = anchor_series.shape
        num_targets, max_active = expand_indices.shape
        gathered = torch.index_select(
            anchor_series,
            dim=2,
            index=expand_indices.reshape(-1),
        ).view(batch_size, seq_len, num_targets, max_active)
        weights = expand_active / expand_active.sum(dim=-1, keepdim=True).clamp_min(1.0)
        anchor_reference = (gathered * weights.view(1, 1, num_targets, max_active)).sum(dim=-1)
        return x_enc - anchor_reference

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        if N != self.original_variate_tokens:
            raise ValueError('x_enc has {} variates, but configs.enc_in is {}'.format(
                N, self.original_variate_tokens))
        adaptive_selection_scores = None
        if self.variate_reduction_type == 'sample_adaptive_selection':
            if self.variate_token_split_factor != 1:
                raise ValueError('sample_adaptive_selection requires variate_token_split_factor=1')
            if x_enc.shape[1] > 1:
                delta_score = (x_enc[:, 1:, :] - x_enc[:, :-1, :]).abs().mean(dim=1)
            else:
                delta_score = x_enc.new_zeros(x_enc.shape[0], x_enc.shape[2])
            recent_center = (x_enc[:, -1:, :] - x_enc.mean(dim=1, keepdim=True)).abs().squeeze(1)
            adaptive_selection_scores = delta_score + 0.25 * recent_center
        elif self.variate_reduction_type in {'variate_group_adaptive_selection', 'variate_group_adaptive_id_selection'}:
            if self.variate_token_split_factor != 1:
                raise ValueError('{} requires variate_token_split_factor=1'.format(self.variate_reduction_type))
            if self.reducer is None or not hasattr(self.reducer, 'group_ids'):
                raise ValueError('{} requires fixed group_ids'.format(self.variate_reduction_type))
            group_ids = self.reducer.group_ids[:self.original_variate_tokens].to(x_enc.device)
            centered = x_enc - x_enc.mean(dim=1, keepdim=True)
            scale = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            shape = (centered / scale).detach()
            group_index = group_ids.view(1, 1, -1).expand(shape.shape[0], shape.shape[1], -1)
            group_sum = shape.new_zeros(shape.shape[0], shape.shape[1], self.reduced_variate_tokens)
            group_sum.scatter_add_(2, group_index, shape)
            group_counts = self.reducer.group_counts.to(device=x_enc.device, dtype=shape.dtype).view(1, 1, -1)
            group_mean = group_sum / group_counts.clamp_min(1.0)
            member_group_mean = torch.gather(group_mean, dim=2, index=group_index)
            adaptive_selection_scores = -((shape - member_group_mean) ** 2).mean(dim=1)
            if hasattr(self.reducer, 'anchor_indices'):
                anchor_bonus = adaptive_selection_scores.new_zeros(self.original_variate_tokens)
                anchor_bonus[self.reducer.anchor_indices.to(x_enc.device)] = 1e-4
                adaptive_selection_scores = adaptive_selection_scores + anchor_bonus.view(1, -1)
        local_temporal_forecast = None
        if self.local_temporal_projection is not None:
            if self.local_temporal_branch in {
                'nlinear',
                'nlinear_affine',
                'nlinear_lowrank',
                'nlinear_lowrank_affine',
                'nlinear_decomp',
                'nlinear_decomp_affine',
                'anchor_residual_nlinear',
                'anchor_residual_nlinear_affine',
                'anchor_delta_nlinear',
                'anchor_delta_nlinear_affine',
                'anchor_mask_delta_nlinear',
                'anchor_mask_delta_nlinear_affine',
            }:
                if self.local_temporal_branch in {
                    'anchor_residual_nlinear',
                    'anchor_residual_nlinear_affine',
                    'anchor_delta_nlinear',
                    'anchor_delta_nlinear_affine',
                }:
                    local_input = self._anchor_residual_input(x_enc)
                elif self.local_temporal_branch in {
                    'anchor_mask_delta_nlinear',
                    'anchor_mask_delta_nlinear_affine',
                }:
                    local_input = self._anchor_mask_residual_input(x_enc)
                else:
                    local_input = x_enc
                is_anchor_delta = self.local_temporal_branch in {
                    'anchor_delta_nlinear',
                    'anchor_delta_nlinear_affine',
                    'anchor_mask_delta_nlinear',
                    'anchor_mask_delta_nlinear_affine',
                }
                seq_last = local_input[:, -1:, :].detach()
                centered_x = local_input - seq_last
                if self.local_temporal_branch in {'nlinear_decomp', 'nlinear_decomp_affine'}:
                    trend = self._moving_average(centered_x, self.local_temporal_decomp_kernel)
                    seasonal = centered_x - trend
                    local_temporal_forecast = (
                        self.local_temporal_projection(seasonal.transpose(1, 2)).transpose(1, 2) +
                        self.local_temporal_trend_projection(trend.transpose(1, 2)).transpose(1, 2)
                    )
                else:
                    local_temporal_forecast = self.local_temporal_projection(
                        centered_x.transpose(1, 2)
                    ).transpose(1, 2)
                if self.local_temporal_branch in {'nlinear_lowrank', 'nlinear_lowrank_affine'}:
                    basis_forecast = torch.einsum(
                        'bln,rpl->brpn',
                        centered_x,
                        self.local_temporal_basis_weight,
                    )
                    basis_forecast = basis_forecast + self.local_temporal_basis_bias.view(
                        1, self.local_temporal_rank, self.pred_len, 1
                    )
                    local_temporal_forecast = local_temporal_forecast + torch.einsum(
                        'brpn,nr->bpn',
                        basis_forecast,
                        self.local_temporal_variate_mix,
                    )
                if not is_anchor_delta:
                    local_temporal_forecast = local_temporal_forecast + seq_last
                if self.local_temporal_branch in {
                    'nlinear_affine',
                    'nlinear_lowrank_affine',
                    'nlinear_decomp_affine',
                    'anchor_residual_nlinear_affine',
                    'anchor_delta_nlinear_affine',
                    'anchor_mask_delta_nlinear_affine',
                }:
                    local_temporal_forecast = (
                        local_temporal_forecast * self.local_temporal_variate_scale +
                        self.local_temporal_horizon_bias
                    )
            else:
                local_temporal_forecast = self.local_temporal_projection(
                    x_enc.transpose(1, 2)
                ).transpose(1, 2)
            self.latest_local_temporal_gate = float(self.local_temporal_gate.detach().cpu().item())
        elif self.local_temporal_branch in {'anchor_group_nlinear', 'anchor_group_nlinear_affine'}:
            seq_last = x_enc[:, -1:, :].detach()
            centered_x = x_enc - seq_last
            group_ids = self.reducer.group_ids[:self.original_variate_tokens].to(x_enc.device)
            group_weight = self.local_temporal_basis_weight[group_ids]
            local_temporal_forecast = torch.einsum(
                'bln,npl->bpn',
                centered_x,
                group_weight,
            )
            group_bias = self.local_temporal_basis_bias[group_ids].transpose(0, 1).unsqueeze(0)
            local_temporal_forecast = local_temporal_forecast + group_bias + seq_last
            if self.local_temporal_branch == 'anchor_group_nlinear_affine':
                local_temporal_forecast = (
                    local_temporal_forecast * self.local_temporal_variate_scale +
                    self.local_temporal_horizon_bias
                )
            self.latest_local_temporal_gate = float(self.local_temporal_gate.detach().cpu().item())
        elif self.local_temporal_branch in {'nlinear_group', 'nlinear_group_affine'}:
            seq_last = x_enc[:, -1:, :].detach()
            centered_x = x_enc - seq_last
            group_forecast = torch.einsum(
                'bln,gpl->bgpn',
                centered_x,
                self.local_temporal_basis_weight,
            )
            group_forecast = group_forecast + self.local_temporal_basis_bias.view(
                1, self.local_temporal_rank, self.pred_len, 1
            )
            group_mix = torch.softmax(self.local_temporal_variate_mix, dim=-1)
            local_temporal_forecast = torch.einsum(
                'bgpn,ng->bpn',
                group_forecast,
                group_mix,
            )
            local_temporal_forecast = local_temporal_forecast + seq_last
            if self.local_temporal_branch == 'nlinear_group_affine':
                local_temporal_forecast = (
                    local_temporal_forecast * self.local_temporal_variate_scale +
                    self.local_temporal_horizon_bias
                )
            self.latest_local_temporal_gate = float(self.local_temporal_gate.detach().cpu().item())
        elif self.local_temporal_branch == 'lowrank_linear':
            basis_forecast = torch.einsum(
                'bln,rpl->brpn',
                x_enc,
                self.local_temporal_basis_weight,
            )
            basis_forecast = basis_forecast + self.local_temporal_basis_bias.view(
                1, self.local_temporal_rank, self.pred_len, 1
            )
            local_temporal_forecast = torch.einsum(
                'brpn,nr->bpn',
                basis_forecast,
                self.local_temporal_variate_mix,
            )
            self.latest_local_temporal_gate = float(self.local_temporal_gate.detach().cpu().item())
        elif self.local_temporal_branch in {'persistence', 'persistence_gate'}:
            local_temporal_forecast = x_enc[:, -1:, :].expand(-1, self.pred_len, -1)
            self.latest_local_temporal_gate = float(self.local_temporal_gate.detach().cpu().item())
        else:
            self.latest_local_temporal_gate = 0.0

        if self.skip_backbone:
            if local_temporal_forecast is None:
                raise RuntimeError('skip_backbone requires a computed local temporal forecast')
            zero = self._zero_aux_loss.to(x_enc.device)
            self.latest_aux_loss = zero
            self.latest_aux_losses = {
                'orthogonal': zero,
                'reconstruction': zero,
                'coverage': zero,
                'assignment_entropy': zero,
                'wcomp_entropy': zero,
                'cycle_slot': zero,
                'group_attention_entropy': zero,
            }
            self.latest_original_variate_tokens = self.original_variate_tokens
            self.latest_target_variate_tokens = self.target_variate_tokens
            self.latest_source_variate_tokens = self.source_variate_tokens
            self.latest_reduced_variate_tokens = self.reduced_variate_tokens
            self.latest_num_extra_tokens = 0
            self.latest_original_encoder_tokens = self.source_variate_tokens
            self.latest_reduced_encoder_tokens = 0
            self.latest_logical_k = 0
            self.latest_executed_k = 0
            self.latest_backbone_residual_gate = 0.0
            self.latest_backbone_residual_gate_abs_mean = 0.0
            self.latest_backbone_residual_gate_max = 0.0
            dec_out = self.local_temporal_gate.to(local_temporal_forecast.dtype) * local_temporal_forecast
            if self.output_calibration == 'variate_affine':
                dec_out = dec_out * self.output_calibration_scale.to(dec_out.dtype)
                dec_out = dec_out + self.output_calibration_bias.to(dec_out.dtype)
                self.latest_output_calibration_scale_abs_mean = float(
                    self.output_calibration_scale.detach().abs().mean().cpu().item()
                )
                self.latest_output_calibration_bias_abs_mean = float(
                    self.output_calibration_bias.detach().abs().mean().cpu().item()
                )
            if self.use_norm:
                dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            return dec_out, None
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        var_tokens = enc_out[:, :self.source_variate_tokens, :]
        extra_tokens = enc_out[:, self.source_variate_tokens:, :]
        num_extra_tokens = extra_tokens.shape[1]
        self.latest_original_variate_tokens = self.original_variate_tokens
        self.latest_target_variate_tokens = self.target_variate_tokens
        self.latest_source_variate_tokens = self.source_variate_tokens
        self.latest_reduced_variate_tokens = self.reduced_variate_tokens
        self.latest_num_extra_tokens = num_extra_tokens
        self.latest_original_encoder_tokens = self.source_variate_tokens + num_extra_tokens
        forecast_stage_cache = None
        forecast_stage_latent_tokens = None

        if self.reducer is None:
            self.latest_aux_loss = self._zero_aux_loss.to(enc_out.device)
            self.latest_aux_losses = {
                'orthogonal': self.latest_aux_loss,
                'reconstruction': self.latest_aux_loss,
                'coverage': self.latest_aux_loss,
                'assignment_entropy': self.latest_aux_loss,
                'wcomp_entropy': self.latest_aux_loss,
                'cycle_slot': self.latest_aux_loss,
                'group_attention_entropy': self.latest_aux_loss,
            }
            self.latest_reduced_variate_tokens = self.original_variate_tokens
            self.latest_reduced_encoder_tokens = self.source_variate_tokens + num_extra_tokens
            self.latest_logical_k = self.original_variate_tokens
            self.latest_executed_k = self.source_variate_tokens
        else:
            compressed_tokens, cache, aux_loss = self.reducer.compress(
                var_tokens,
                selection_scores=adaptive_selection_scores,
            )
            aux_losses = self.reducer.get_aux_losses()
            if self.use_reconstruction_loss:
                reconstructed_var_tokens = self.reducer.reconstruct(compressed_tokens, cache=cache)
                aux_losses = dict(aux_losses)
                aux_losses['reconstruction'] = F.mse_loss(reconstructed_var_tokens, var_tokens.detach())
            else:
                aux_losses = dict(aux_losses)
                aux_losses['reconstruction'] = aux_loss.new_tensor(0.0)
            self.latest_aux_loss = aux_loss
            self.latest_aux_losses = aux_losses
            self.latest_reduced_variate_tokens = self.reduced_variate_tokens
            self.latest_reduced_encoder_tokens = self.reduced_variate_tokens + num_extra_tokens
            self.latest_logical_k = self.reduced_variate_tokens
            self.latest_executed_k = self.reduced_variate_tokens
            enc_out = torch.cat([compressed_tokens, extra_tokens], dim=1)
            forecast_stage_cache = cache
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        if self.reducer is not None:
            encoded_latent_tokens = enc_out[:, :self.reduced_variate_tokens, :]
            if self.variate_decode_stage == 'forecast':
                forecast_stage_latent_tokens = encoded_latent_tokens
            else:
                expanded_var_tokens = self.reducer.decode(
                    encoded_latent_tokens,
                    cache=forecast_stage_cache,
                    original_variates=var_tokens,
                    use_residual=True,
                )
                enc_out = expanded_var_tokens

        # B N E -> B N S -> B S N 
        if forecast_stage_latent_tokens is not None:
            latent_forecast = self.projector(forecast_stage_latent_tokens)
            decoded_forecast = self.reducer.decode(
                latent_forecast,
                cache=forecast_stage_cache,
                original_variates=None,
                use_residual=False,
            )
            dec_out = decoded_forecast.permute(0, 2, 1)[:, :, :N]
        else:
            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        dec_out = self.backbone_residual_gate.to(dec_out.dtype) * dec_out
        gate = self.backbone_residual_gate.detach()
        self.latest_backbone_residual_gate = float(gate.mean().cpu().item())
        self.latest_backbone_residual_gate_abs_mean = float(gate.abs().mean().cpu().item())
        self.latest_backbone_residual_gate_max = float(gate.abs().max().cpu().item())
        if local_temporal_forecast is not None:
            dec_out = dec_out + self.local_temporal_gate.to(dec_out.dtype) * local_temporal_forecast
        if self.output_calibration == 'variate_affine':
            dec_out = dec_out * self.output_calibration_scale.to(dec_out.dtype)
            dec_out = dec_out + self.output_calibration_bias.to(dec_out.dtype)
            self.latest_output_calibration_scale_abs_mean = float(
                self.output_calibration_scale.detach().abs().mean().cpu().item()
            )
            self.latest_output_calibration_bias_abs_mean = float(
                self.output_calibration_bias.detach().abs().mean().cpu().item()
            )

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def get_aux_loss(self):
        return self.latest_aux_loss

    def get_aux_losses(self):
        return self.latest_aux_losses

    def get_variate_reduction_metadata(self):
        metadata = {
            'variate_reduction_type': self.variate_reduction_type,
            'variate_expansion_type': self.variate_expansion_type,
            'compute_reducer_aux_losses': self.compute_reducer_aux_losses,
            'variate_token_split_factor': self.variate_token_split_factor,
            'variate_decode_stage': self.variate_decode_stage,
            'expansion_topk': self.expansion_topk,
            'wcomp_normalization': self.wcomp_normalization,
            'entmax_alpha': self.entmax_alpha,
            'group_attention_entropy_target': getattr(self.reducer, 'group_attention_entropy_target', 0.35)
            if self.reducer is not None else 0.35,
            'use_cycle_slot_loss': self.use_cycle_slot_loss,
            'export_slot_diagnostics': self.export_slot_diagnostics,
            'cycle_topr': getattr(self.reducer, 'cycle_topr', 0) if self.reducer is not None else 0,
            'cycle_topr_multiplier': getattr(self.reducer, 'cycle_topr_multiplier', 1.0) if self.reducer is not None else 1.0,
            'cycle_b_norm': getattr(self.reducer, 'cycle_b_norm', 'row_l1') if self.reducer is not None else 'row_l1',
            'cycle_eps': getattr(self.reducer, 'cycle_eps', 1e-8) if self.reducer is not None else 1e-8,
            'sparse_compress_topk': getattr(self.reducer, 'sparse_compress_topk', 0) if self.reducer is not None else 0,
            'sparse_expand_topk': getattr(self.reducer, 'sparse_expand_topk', 0) if self.reducer is not None else 0,
            'original_variate_tokens': self.latest_original_variate_tokens,
            'target_variate_tokens': self.latest_target_variate_tokens,
            'source_variate_tokens': self.latest_source_variate_tokens,
            'reduced_variate_tokens': self.latest_reduced_variate_tokens,
            'logical_k': self.latest_logical_k,
            'executed_k': self.latest_executed_k,
            'k_budget_max': self.k_budget_max,
            'k_budget_exception': self.k_budget_exception,
            'lowrank_reducer_rank': self.lowrank_reducer_rank,
            'decoder_residual_gate_init': self.decoder_residual_gate_init,
            'backbone_residual_gate_init': self.backbone_residual_gate_init,
            'backbone_residual_gate_type': self.backbone_residual_gate_type,
            'backbone_residual_gate': self.latest_backbone_residual_gate,
            'backbone_residual_gate_abs_mean': self.latest_backbone_residual_gate_abs_mean,
            'backbone_residual_gate_max': self.latest_backbone_residual_gate_max,
            'local_temporal_rank': self.local_temporal_rank,
            'zero_init_projector': self.zero_init_projector,
            'skip_backbone': self.skip_backbone,
            'num_extra_tokens': self.latest_num_extra_tokens,
            'original_encoder_tokens': self.latest_original_encoder_tokens,
            'reduced_encoder_tokens': self.latest_reduced_encoder_tokens,
            'local_temporal_branch': self.local_temporal_branch,
            'local_temporal_gate': self.latest_local_temporal_gate,
            'local_temporal_decomp_kernel': self.local_temporal_decomp_kernel,
            'output_calibration': self.output_calibration,
            'output_calibration_scale_abs_mean': self.latest_output_calibration_scale_abs_mean,
            'output_calibration_bias_abs_mean': self.latest_output_calibration_bias_abs_mean,
        }
        if self.reducer is not None and hasattr(self.reducer, 'get_residual_gate_stats'):
            metadata.update(self.reducer.get_residual_gate_stats())
        else:
            metadata.update({
                'residual_gate_mean': 0.0,
                'residual_gate_abs_mean': 0.0,
                'residual_gate_max': 0.0,
                'residual_gate_min': 0.0,
                'residual_gate_value_max': 0.0,
                'residual_gate_range': 0.0,
                'residual_gate_std': 0.0,
                'residual_gate_numel': 0,
                'residual_gate_is_variate': 0,
                'hybrid_linear_gate': 0.0,
            })
        if self.reducer is not None and hasattr(self.reducer, 'get_masked_expand_stats'):
            metadata.update(self.reducer.get_masked_expand_stats())
        else:
            metadata.update({
                'masked_expand_fan_in_mean': 0.0,
                'masked_expand_fan_in_max': 0,
            })
        if self.reducer is not None and hasattr(self.reducer, 'get_cycle_slot_stats'):
            metadata.update(self.reducer.get_cycle_slot_stats())
        else:
            metadata.update({
                'cycle_topr_jaccard': 0.0,
                'cycle_a_effective_support': 0.0,
                'cycle_b_effective_support': 0.0,
                'cycle_slot_overlap': 0.0,
                'cycle_topr_size': 0,
                'w_eff_density': 0.0,
            })
        if self.reducer is not None and hasattr(self.reducer, 'get_attention_wcomp_stats'):
            metadata.update(self.reducer.get_attention_wcomp_stats())
        return metadata

    def get_variate_reduction_weights(self):
        if self.reducer is None or not hasattr(self.reducer, 'get_weight_matrices'):
            return {}
        return self.reducer.get_weight_matrices()

    def get_slot_diagnostic_matrices(self):
        if self.reducer is None or not hasattr(self.reducer, 'get_slot_diagnostic_matrices'):
            return {}
        return self.reducer.get_slot_diagnostic_matrices()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
