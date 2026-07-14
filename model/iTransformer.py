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


def compute_sender_effective_mass(
    group_ids,
    group_counts,
    assigned_similarity=None,
    similarity_power=0.0,
):
    """Return similarity-aware MPAA multiplicities for fixed sender groups.

    ``power=0`` deliberately ignores ``assigned_similarity`` and returns the
    original counts exactly.  For positive powers, each provider contributes
    ``clip(similarity, 0, 1) ** power`` to its group's mass.
    """
    power = float(similarity_power)
    if not np.isfinite(power) or power < 0.0:
        raise ValueError('sender_mass_similarity_power must be finite and non-negative')
    if power == 0.0:
        return group_counts.clone()
    if assigned_similarity is None:
        raise ValueError(
            'sender_mass_similarity_power > 0 requires assigned_abs_corr in the anchor map'
        )
    similarity = torch.as_tensor(
        assigned_similarity,
        device=group_counts.device,
        dtype=group_counts.dtype,
    ).reshape(-1)
    group_ids = group_ids.to(device=group_counts.device, dtype=torch.long).reshape(-1)
    if similarity.numel() != group_ids.numel():
        raise ValueError(
            'assigned_abs_corr length {} does not match group_ids length {}'.format(
                similarity.numel(), group_ids.numel()
            )
        )
    if not torch.isfinite(similarity).all():
        raise ValueError('assigned_abs_corr must contain only finite values')
    effective_mass = group_counts.new_zeros(group_counts.numel())
    effective_mass.scatter_add_(
        0,
        group_ids,
        similarity.clamp(0.0, 1.0).pow(power),
    )
    if torch.any(effective_mass <= 0.0):
        raise ValueError(
            'similarity-aware sender mass must be positive for every group; '
            'check assigned_abs_corr and sender_mass_similarity_power'
        )
    return effective_mass


def compute_learned_sender_effective_mass(
    group_ids,
    group_counts,
    assigned_similarity,
    similarity_power,
    eps=1e-8,
    validate=True,
):
    """Differentiable layer/head-wise similarity-aware multiplicities.

    ``similarity_power`` may have any leading shape (for example ``[L, H]``).
    The returned tensor has that shape followed by the sender-group dimension.
    Similarities are fixed map statistics; gradients flow to the power tensor.
    """
    if assigned_similarity is None:
        raise ValueError(
            'learned sender mass requires assigned_abs_corr in the anchor map'
        )
    if not np.isfinite(eps) or eps <= 0.0 or eps > 1.0:
        raise ValueError('sender mass similarity epsilon must be in (0, 1]')

    power = torch.as_tensor(similarity_power)
    if not power.is_floating_point():
        power = power.float()
    if validate and (
        not torch.isfinite(power.detach()).all()
        or torch.any(power.detach() < 0.0)
    ):
        raise ValueError('learned sender mass power must be finite and non-negative')

    similarity = torch.as_tensor(
        assigned_similarity,
        device=power.device,
        dtype=power.dtype,
    ).reshape(-1)
    ids = torch.as_tensor(group_ids, device=power.device, dtype=torch.long).reshape(-1)
    num_groups = int(torch.as_tensor(group_counts).numel())
    if similarity.numel() != ids.numel():
        raise ValueError(
            'assigned_abs_corr length {} does not match group_ids length {}'.format(
                similarity.numel(), ids.numel()
            )
        )
    if validate and not torch.isfinite(similarity).all():
        raise ValueError('assigned_abs_corr must contain only finite values')
    if num_groups < 1 or ids.numel() < 1:
        raise ValueError('learned sender mass requires non-empty sender groups')
    if validate and (torch.any(ids < 0) or torch.any(ids >= num_groups)):
        raise ValueError('group_ids contain an out-of-range sender group')

    log_similarity = similarity.clamp(min=float(eps), max=1.0).log()
    contributions = torch.exp(power.unsqueeze(-1) * log_similarity)
    scatter_index = ids.view(*([1] * power.ndim), -1).expand_as(contributions)
    effective_mass = contributions.new_zeros(*power.shape, num_groups).scatter_add(
        -1,
        scatter_index,
        contributions,
    )
    if validate and torch.any(effective_mass.detach() <= 0.0):
        raise ValueError(
            'learned similarity-aware sender mass must be positive for every group'
        )
    return effective_mass


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
        self.variate_backbone_mode = getattr(configs, 'variate_backbone_mode', 'bottleneck')
        if self.variate_backbone_mode not in {'bottleneck', 'layerwise_sender'}:
            raise ValueError('Unknown variate_backbone_mode: {}'.format(self.variate_backbone_mode))
        self.sender_mass_correction = bool(getattr(configs, 'sender_mass_correction', False))
        self.sender_mass_mode = str(getattr(configs, 'sender_mass_mode', 'fixed'))
        if self.sender_mass_mode not in {
            'fixed',
            'learned_global',
            'learned_layer_head',
        }:
            raise ValueError('Unknown sender_mass_mode: {}'.format(self.sender_mass_mode))
        self.sender_mass_similarity_power = float(
            getattr(configs, 'sender_mass_similarity_power', 0.0)
        )
        if not np.isfinite(self.sender_mass_similarity_power) or self.sender_mass_similarity_power < 0.0:
            raise ValueError('sender_mass_similarity_power must be finite and non-negative')
        if self.sender_mass_similarity_power > 0.0 and not self.sender_mass_correction:
            raise ValueError(
                'sender_mass_similarity_power > 0 requires sender_mass_correction=True'
            )
        self.sender_mass_power_cap = float(
            getattr(configs, 'sender_mass_power_cap', 64.0)
        )
        self.sender_mass_power_init = float(
            getattr(configs, 'sender_mass_power_init', 16.0)
        )
        self.sender_mass_num_layers = int(configs.e_layers)
        self.sender_mass_num_heads = int(configs.n_heads)
        if self.sender_mass_mode != 'fixed':
            if not self.sender_mass_correction:
                raise ValueError(
                    'learned sender_mass_mode requires sender_mass_correction=True'
                )
            if self.sender_mass_similarity_power != 0.0:
                raise ValueError(
                    'learned sender_mass_mode cannot be combined with '
                    'sender_mass_similarity_power; use sender_mass_power_init instead'
                )
            if not np.isfinite(self.sender_mass_power_cap) or self.sender_mass_power_cap <= 0.0:
                raise ValueError('sender_mass_power_cap must be finite and positive')
            if (
                not np.isfinite(self.sender_mass_power_init)
                or self.sender_mass_power_init <= 0.0
                or self.sender_mass_power_init >= self.sender_mass_power_cap
            ):
                raise ValueError(
                    'sender_mass_power_init must be finite and strictly between 0 and '
                    'sender_mass_power_cap'
                )
        self.sender_mass_similarity_source = 'none'
        if self.variate_decode_stage not in {'feature', 'forecast'}:
            raise ValueError('Unknown variate_decode_stage: {}'.format(self.variate_decode_stage))
        self.expansion_temperature = getattr(configs, 'expansion_temperature', 1.0)
        self.expansion_topk = getattr(configs, 'expansion_topk', 0)
        self.wcomp_normalization = getattr(configs, 'wcomp_normalization', 'softmax')
        self.entmax_alpha = float(getattr(configs, 'entmax_alpha', 1.5))
        self.decoder_residual_gate_init = float(getattr(configs, 'decoder_residual_gate_init', 0.0))
        self.backbone_residual_gate_init = float(getattr(configs, 'backbone_residual_gate_init', 1.0))
        self.backbone_residual_gate_type = getattr(configs, 'backbone_residual_gate_type', 'scalar')
        if self.backbone_residual_gate_type not in {'scalar', 'variate', 'scalar_plus_variate'}:
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
        self.use_expansion_weight_l2_loss = getattr(configs, 'expansion_weight_l2_loss_weight', 0.0) > 0.0
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
                'expansion_weight_l2_loss_weight',
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
                use_expansion_weight_l2_loss=self.use_expansion_weight_l2_loss,
                use_group_attention_entropy_loss=self.use_group_attention_entropy_loss,
                group_attention_entropy_target=getattr(configs, 'group_attention_entropy_target', 0.35),
                use_cycle_slot_loss=self.use_cycle_slot_loss,
                cycle_topr=getattr(configs, 'cycle_topr', 0),
                cycle_topr_multiplier=getattr(configs, 'cycle_topr_multiplier', 1.0),
                cycle_b_norm=getattr(configs, 'cycle_b_norm', 'row_l1'),
                cycle_eps=getattr(configs, 'cycle_eps', 1e-8),
                export_slot_diagnostics=self.export_slot_diagnostics,
                within_group_residual=bool(getattr(configs, 'within_group_residual', False)),
                within_group_residual_gate_init=float(getattr(configs, 'within_group_residual_gate_init', 0.0)),
                within_group_residual_rank=int(getattr(configs, 'within_group_residual_rank', 0)),
                within_group_residual_mode=getattr(configs, 'within_group_residual_mode', 'projection'),
                within_group_residual_gate_mode=getattr(configs, 'within_group_residual_gate_mode', 'scalar'),
            )
        else:
            self.reducer = None
        if self.variate_backbone_mode == 'layerwise_sender':
            if self.variate_reduction_type != 'variate_anchor_selection':
                raise ValueError(
                    'layerwise_sender currently supports only variate_anchor_selection; '
                    'adaptive/pooling reducers would be silently bypassed'
                )
            if self.reducer is None or not hasattr(self.reducer, 'anchor_indices'):
                raise ValueError(
                    'layerwise_sender requires a fixed reducer with anchor_indices'
                )
            if self.variate_decode_stage != 'feature':
                raise ValueError('layerwise_sender supports only variate_decode_stage=feature')
            if self.skip_backbone:
                raise ValueError('layerwise_sender cannot be combined with skip_backbone')
            if self.use_reconstruction_loss or self.compute_reducer_aux_losses:
                raise ValueError(
                    'layerwise_sender does not compress tokens and therefore does not support reducer auxiliary losses'
                )
            sender_counts = self.reducer.group_counts
            if sender_counts.numel() != self.reduced_variate_tokens:
                raise ValueError('group_counts must have one entry per selected sender')
            if torch.any(sender_counts <= 0):
                raise ValueError('layerwise_sender requires every selected sender to represent a non-empty group')
            if int(sender_counts.sum().item()) != self.source_variate_tokens:
                raise ValueError('group_counts must partition every source variable exactly once')
            assigned_similarity = None
            needs_similarity = (
                self.sender_mass_similarity_power > 0.0
                or self.sender_mass_mode != 'fixed'
            )
            if needs_similarity:
                anchor_map_path = str(getattr(configs, 'variate_anchor_map_path', '') or '')
                if not anchor_map_path:
                    raise ValueError(
                        'similarity-aware sender mass requires variate_anchor_map_path '
                        'with assigned_abs_corr'
                    )
                with np.load(anchor_map_path, allow_pickle=False) as anchor_map:
                    if 'assigned_abs_corr' not in anchor_map:
                        raise ValueError(
                            'similarity-aware sender mass requires assigned_abs_corr '
                            'in the anchor map: {}'.format(anchor_map_path)
                        )
                    assigned_similarity = np.array(
                        anchor_map['assigned_abs_corr'], dtype=np.float32, copy=True
                    )
                if assigned_similarity.size != self.reducer.group_ids.numel():
                    raise ValueError(
                        'assigned_abs_corr length {} does not match group_ids length {}'.format(
                            assigned_similarity.size,
                            self.reducer.group_ids.numel(),
                        )
                    )
                if not np.isfinite(assigned_similarity).all():
                    raise ValueError('assigned_abs_corr must contain only finite values')
                self.sender_mass_similarity_source = 'assigned_abs_corr'
            elif self.sender_mass_correction:
                self.sender_mass_similarity_source = 'group_count'
            if self.sender_mass_mode == 'fixed' and self.sender_mass_correction:
                sender_effective_mass = compute_sender_effective_mass(
                    self.reducer.group_ids,
                    sender_counts,
                    assigned_similarity=assigned_similarity,
                    similarity_power=self.sender_mass_similarity_power,
                )
            elif self.sender_mass_mode == 'fixed':
                sender_effective_mass = torch.ones_like(sender_counts)
            sender_assigned_similarity = (
                torch.as_tensor(assigned_similarity, dtype=torch.float32).reshape(-1)
                if assigned_similarity is not None
                else torch.empty(0, dtype=torch.float32)
            )
            self.register_buffer(
                'sender_assigned_similarity', sender_assigned_similarity, persistent=False
            )
            if self.sender_mass_mode == 'fixed':
                self.register_buffer(
                    'sender_effective_mass', sender_effective_mass, persistent=False
                )
            else:
                ratio = self.sender_mass_power_init / self.sender_mass_power_cap
                initial_theta = float(np.log(ratio / (1.0 - ratio)))
                if self.sender_mass_mode == 'learned_global':
                    theta_shape = (1,)
                else:
                    theta_shape = (
                        self.sender_mass_num_layers,
                        self.sender_mass_num_heads,
                    )
                self.sender_mass_power_theta = nn.Parameter(
                    torch.full(theta_shape, initial_theta, dtype=torch.float32)
                )
        elif self.sender_mass_correction:
            raise ValueError('sender_mass_correction requires variate_backbone_mode=layerwise_sender')
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
                'masked_token_query_decoder',
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
        self.group_residual_context_norm = None
        self.group_residual_context = None
        self.group_residual_basis_weight = None
        self.group_residual_basis_bias = None
        self.group_residual_gate = None
        if self.skip_backbone and self.local_temporal_branch == 'none':
            raise ValueError('skip_backbone requires a local_temporal_branch')
        if self.local_temporal_branch not in {
            'none',
            'linear',
            'nlinear',
            'nlinear_affine',
            'nlinear_affine_group_residual',
            'nlinear_affine_anchor_group_residual',
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
            'nlinear_affine_group_residual',
            'nlinear_affine_anchor_group_residual',
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
                'nlinear_affine_group_residual',
                'nlinear_affine_anchor_group_residual',
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
            if self.local_temporal_branch in {
                'nlinear_affine_group_residual',
                'nlinear_affine_anchor_group_residual',
            }:
                if self.reducer is None or not hasattr(self.reducer, 'group_ids'):
                    raise ValueError('{} requires a fixed anchor/group reducer'.format(
                        self.local_temporal_branch
                    ))
                if (
                    self.local_temporal_branch == 'nlinear_affine_anchor_group_residual'
                    and not hasattr(self.reducer, 'anchor_indices')
                ):
                    raise ValueError('nlinear_affine_anchor_group_residual requires anchor_indices')
                if self.local_temporal_rank < 1:
                    raise ValueError('local_temporal_rank must be at least 1 for group residual decoding')
                rng_state = torch.get_rng_state()
                self.group_residual_context_norm = nn.LayerNorm(configs.d_model)
                self.group_residual_context = nn.Linear(configs.d_model, self.local_temporal_rank)
                self.group_residual_basis_weight = nn.Parameter(
                    torch.empty(self.local_temporal_rank, self.pred_len, self.seq_len)
                )
                self.group_residual_basis_bias = nn.Parameter(
                    torch.zeros(self.local_temporal_rank, self.pred_len)
                )
                self.group_residual_gate = nn.Parameter(torch.tensor(
                    float(getattr(configs, 'group_residual_gate_init', 1.0)),
                    dtype=torch.float32,
                ))
                # A-1 (periodic phase decomposition): when period p > 0, the group-relative
                # residual is split into a per-phase profile (tiled deterministically into
                # the future) and a deseasonalized remainder (handled by the basis path).
                self.group_residual_period = int(getattr(configs, 'group_residual_period', 0))
                if self.group_residual_period > 0:
                    if self.seq_len % self.group_residual_period != 0:
                        raise ValueError('group_residual_period must divide seq_len')
                    self.group_residual_season_gate = nn.Parameter(torch.tensor(
                        float(getattr(configs, 'group_residual_season_gate_init', 0.1)),
                        dtype=torch.float32,
                    ))
                else:
                    self.group_residual_season_gate = None
                with torch.no_grad():
                    self.group_residual_context.weight.zero_()
                    self.group_residual_context.bias.zero_()
                    self.group_residual_basis_weight.normal_(mean=0.0, std=1e-3)
                torch.set_rng_state(rng_state)
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
        # Neighbor-cross branch: 변수별 top-m 이웃의 raw window를 저랭크 선형으로 예측에 추가.
        # (fine-grained premium 진단의 Model S를 아키텍처로 옮긴 것 — 저랭크 병목이
        #  나르지 못하는 identity-level cross-variate 정보를 백본 밖에서 보충)
        self.neighbor_cross_branch = str(getattr(configs, 'neighbor_cross_branch', 'none'))
        if self.neighbor_cross_branch not in {'none', 'lowrank_nlinear'}:
            raise ValueError('Unknown neighbor_cross_branch: {}'.format(self.neighbor_cross_branch))
        if self.neighbor_cross_branch != 'none':
            if self.variate_token_split_factor != 1:
                raise ValueError('neighbor_cross_branch requires variate_token_split_factor=1')
            neighbor_map_path = str(getattr(configs, 'neighbor_map_path', '') or '')
            if not neighbor_map_path:
                raise ValueError('neighbor_cross_branch requires --neighbor_map_path')
            neighbor_indices = np.load(neighbor_map_path)['neighbor_indices']
            if neighbor_indices.shape[0] != self.original_variate_tokens:
                raise ValueError('neighbor map has {} variates, expected {}'.format(
                    neighbor_indices.shape[0], self.original_variate_tokens))
            self.register_buffer(
                'neighbor_indices',
                torch.as_tensor(neighbor_indices, dtype=torch.long),
                persistent=False,
            )
            neighbor_m = int(self.neighbor_indices.shape[1])
            neighbor_rank = int(getattr(configs, 'neighbor_cross_rank', 4))
            self.neighbor_cross_down = nn.Parameter(
                torch.randn(self.original_variate_tokens, neighbor_m, neighbor_rank, self.seq_len)
                * (self.seq_len ** -0.5)
            )
            # up을 zero-init → 학습 시작 시 branch 출력이 정확히 0 (기존 동작 무해 보장)
            self.neighbor_cross_up = nn.Parameter(
                torch.zeros(self.original_variate_tokens, neighbor_m, self.pred_len, neighbor_rank)
            )
            self.neighbor_cross_gate = nn.Parameter(
                torch.tensor(float(getattr(configs, 'neighbor_cross_gate_init', 1.0)), dtype=torch.float32)
            )
        else:
            self.neighbor_indices = None
            self.neighbor_cross_down = None
            self.neighbor_cross_up = None
            self.neighbor_cross_gate = None
        self.latest_neighbor_cross_gate = 0.0
        self.register_buffer('_zero_aux_loss', torch.tensor(0.0), persistent=False)
        self.latest_aux_loss = self._zero_aux_loss
        self.latest_aux_losses = {
            'orthogonal': self._zero_aux_loss,
            'reconstruction': self._zero_aux_loss,
            'coverage': self._zero_aux_loss,
            'assignment_entropy': self._zero_aux_loss,
            'wcomp_entropy': self._zero_aux_loss,
            'expansion_weight_l2': self._zero_aux_loss,
            'cycle_slot': self._zero_aux_loss,
            'group_attention_entropy': self._zero_aux_loss,
        }
        self.latest_num_extra_tokens = 0
        self.latest_original_encoder_tokens = self.source_variate_tokens
        self.latest_reduced_encoder_tokens = self.reduced_variate_tokens
        self.latest_attention_query_tokens = self.source_variate_tokens
        self.latest_attention_kv_tokens = self.reduced_variate_tokens
        self.latest_original_variate_tokens = self.original_variate_tokens
        self.latest_target_variate_tokens = self.target_variate_tokens
        self.latest_source_variate_tokens = self.source_variate_tokens
        self.latest_reduced_variate_tokens = self.reduced_variate_tokens
        self.latest_executed_k = self.reduced_variate_tokens
        self.latest_logical_k = self.reduced_variate_tokens
        self.latest_local_temporal_gate = 0.0
        self.latest_group_residual_gate = 0.0
        self.latest_group_residual_coeff_abs_mean = 0.0
        self.latest_group_residual_season_gate = 0.0
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
        elif self.backbone_residual_gate_type == 'scalar_plus_variate':
            self.backbone_residual_gate = nn.Parameter(
                torch.tensor(self.backbone_residual_gate_init, dtype=torch.float32)
            )
            self.backbone_residual_variate_gate = nn.Parameter(
                torch.zeros(1, 1, self.original_variate_tokens)
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

    def learned_sender_mass_power(self):
        if self.sender_mass_mode == 'fixed':
            return None
        return self.sender_mass_power_cap * torch.sigmoid(self.sender_mass_power_theta)

    def _sender_mass_power_matrix(self):
        power = self.learned_sender_mass_power()
        if power is None:
            return None
        if self.sender_mass_mode == 'learned_global':
            return power.reshape(1, 1).expand(
                self.sender_mass_num_layers,
                self.sender_mass_num_heads,
            )
        return power

    def current_sender_effective_mass(self):
        if self.sender_mass_mode == 'fixed':
            return self.sender_effective_mass
        return compute_learned_sender_effective_mass(
            self.reducer.group_ids,
            self.reducer.group_counts,
            self.sender_assigned_similarity,
            self._sender_mass_power_matrix(),
            validate=False,
        )

    def _effective_backbone_residual_gate(self):
        if self.backbone_residual_gate_type == 'scalar_plus_variate':
            return self.backbone_residual_gate + self.backbone_residual_variate_gate
        return self.backbone_residual_gate

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

    def _group_conditioned_residual_temporal_forecast(self, x_enc, encoded_latent_tokens):
        """Forecast group-relative residuals with latent-conditioned shared bases."""
        if self.group_residual_context is None or self.reducer is None:
            return None
        group_ids = self.reducer.group_ids[:self.original_variate_tokens].to(x_enc.device)
        if self.local_temporal_branch == 'nlinear_affine_anchor_group_residual':
            anchor_indices = self.reducer.anchor_indices.to(x_enc.device)
            member_anchor_indices = anchor_indices[group_ids]
            reference = torch.index_select(x_enc, dim=2, index=member_anchor_indices)
        else:
            group_index = group_ids.view(1, 1, -1).expand(x_enc.shape[0], x_enc.shape[1], -1)
            group_sum = x_enc.new_zeros(x_enc.shape[0], x_enc.shape[1], self.reduced_variate_tokens)
            group_sum.scatter_add_(2, group_index, x_enc)
            group_counts = self.reducer.group_counts.to(
                device=x_enc.device,
                dtype=x_enc.dtype,
            ).view(1, 1, -1)
            reference = torch.gather(
                group_sum / group_counts.clamp_min(1.0),
                dim=2,
                index=group_index,
            )
        residual = x_enc - reference
        seasonal_future = None
        period = int(getattr(self, 'group_residual_period', 0) or 0)
        if period > 0:
            batch, length, channels = residual.shape
            cycles = length // period
            # per-phase profile s_c(phi): mean of the group-relative residual at each phase
            profile = residual.view(batch, cycles, period, channels).mean(dim=1)  # [B, p, C]
            residual = residual - profile.repeat(1, cycles, 1)  # deseasonalized remainder
            future_phase = (length + torch.arange(self.pred_len, device=residual.device)) % period
            seasonal_future = profile[:, future_phase, :]  # deterministic tile into the horizon
        residual = residual - residual[:, -1:, :].detach()
        basis_forecast = torch.einsum(
            'bln,rpl->brpn',
            residual,
            self.group_residual_basis_weight,
        )
        basis_forecast = basis_forecast + self.group_residual_basis_bias.view(
            1, self.local_temporal_rank, self.pred_len, 1
        )
        group_coeff = self.group_residual_context(self.group_residual_context_norm(encoded_latent_tokens))
        member_coeff = group_coeff[:, group_ids, :]
        forecast = torch.einsum('brpn,bnr->bpn', basis_forecast, member_coeff)
        self.latest_group_residual_gate = float(self.group_residual_gate.detach().cpu().item())
        self.latest_group_residual_coeff_abs_mean = float(member_coeff.detach().abs().mean().cpu().item())
        out = self.group_residual_gate.to(forecast.dtype) * forecast
        if seasonal_future is not None:
            season_gate = torch.tanh(self.group_residual_season_gate)
            self.latest_group_residual_season_gate = float(season_gate.detach().cpu().item())
            out = out + season_gate.to(out.dtype) * seasonal_future
        return out

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
                'nlinear_affine_group_residual',
                'nlinear_affine_anchor_group_residual',
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
                    'nlinear_affine_group_residual',
                    'nlinear_affine_anchor_group_residual',
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

        neighbor_cross_forecast = None
        if self.neighbor_cross_branch != 'none':
            x_nb = x_enc[:, :, self.neighbor_indices]            # (B, L, N, m)
            centered_nb = x_nb - x_nb[:, -1:, :, :].detach()
            low = torch.einsum('blnm,nmrl->bnmr', centered_nb, self.neighbor_cross_down)
            neighbor_cross_forecast = torch.einsum('bnmr,nmpr->bpn', low, self.neighbor_cross_up)
            neighbor_cross_forecast = (
                self.neighbor_cross_gate.to(neighbor_cross_forecast.dtype) * neighbor_cross_forecast
            )
            self.latest_neighbor_cross_gate = float(self.neighbor_cross_gate.detach().cpu().item())
        else:
            self.latest_neighbor_cross_gate = 0.0

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
        encoded_latent_tokens = None
        context_indices = None
        sender_attn_bias = None

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
        elif self.variate_backbone_mode == 'layerwise_sender':
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
            anchor_indices = self.reducer.anchor_indices.to(enc_out.device)
            extra_indices = torch.arange(
                self.source_variate_tokens,
                self.source_variate_tokens + num_extra_tokens,
                device=enc_out.device,
                dtype=torch.long,
            )
            context_indices = torch.cat([anchor_indices, extra_indices], dim=0)
            self.latest_reduced_variate_tokens = self.reduced_variate_tokens
            self.latest_reduced_encoder_tokens = self.source_variate_tokens + num_extra_tokens
            self.latest_logical_k = self.reduced_variate_tokens
            self.latest_executed_k = self.reduced_variate_tokens
            if self.sender_mass_correction:
                provider_mass = self.current_sender_effective_mass().to(
                    device=enc_out.device,
                    dtype=enc_out.dtype,
                )
                if self.sender_mass_mode == 'fixed':
                    if num_extra_tokens:
                        provider_mass = torch.cat(
                            [provider_mass, provider_mass.new_ones(num_extra_tokens)],
                            dim=0,
                        )
                    sender_attn_bias = provider_mass.log().view(1, 1, 1, -1)
                else:
                    if num_extra_tokens:
                        provider_mass = torch.cat(
                            [
                                provider_mass,
                                provider_mass.new_ones(
                                    self.sender_mass_num_layers,
                                    self.sender_mass_num_heads,
                                    num_extra_tokens,
                                ),
                            ],
                            dim=-1,
                        )
                    # [layer, singleton batch, head, singleton query, provider]
                    sender_attn_bias = provider_mass.log()[:, None, :, None, :]
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

        self.latest_attention_query_tokens = int(enc_out.shape[1])
        self.latest_attention_kv_tokens = (
            int(context_indices.numel()) if context_indices is not None else int(enc_out.shape[1])
        )
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=None,
            context_indices=context_indices,
            attn_bias=sender_attn_bias,
        )

        if self.reducer is not None and self.variate_backbone_mode == 'bottleneck':
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
        elif self.variate_backbone_mode == 'layerwise_sender':
            encoded_latent_tokens = enc_out.index_select(
                1,
                self.reducer.anchor_indices.to(enc_out.device),
            )

        group_residual_forecast = None
        if self.local_temporal_branch in {
            'nlinear_affine_group_residual',
            'nlinear_affine_anchor_group_residual',
        }:
            group_residual_forecast = self._group_conditioned_residual_temporal_forecast(
                x_enc,
                encoded_latent_tokens,
            )

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
        effective_gate = self._effective_backbone_residual_gate()
        dec_out = effective_gate.to(dec_out.dtype) * dec_out
        gate = effective_gate.detach()
        self.latest_backbone_residual_gate = float(gate.mean().cpu().item())
        self.latest_backbone_residual_gate_abs_mean = float(gate.abs().mean().cpu().item())
        self.latest_backbone_residual_gate_max = float(gate.abs().max().cpu().item())
        if local_temporal_forecast is not None:
            dec_out = dec_out + self.local_temporal_gate.to(dec_out.dtype) * local_temporal_forecast
        if neighbor_cross_forecast is not None:
            dec_out = dec_out + neighbor_cross_forecast.to(dec_out.dtype)
        if group_residual_forecast is not None:
            dec_out = dec_out + group_residual_forecast.to(dec_out.dtype)
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
        sender_effective_mass = None
        if self.variate_backbone_mode == 'layerwise_sender':
            sender_effective_mass = self.current_sender_effective_mass().detach()
        learned_power = self.learned_sender_mass_power()
        learned_power = learned_power.detach() if learned_power is not None else None
        metadata = {
            'variate_reduction_type': self.variate_reduction_type,
            'variate_expansion_type': self.variate_expansion_type,
            'variate_backbone_mode': self.variate_backbone_mode,
            'sender_mass_correction': self.sender_mass_correction,
            'sender_mass_mode': self.sender_mass_mode,
            'sender_mass_similarity_power': self.sender_mass_similarity_power,
            'sender_mass_similarity_source': self.sender_mass_similarity_source,
            'sender_mass_power_cap': self.sender_mass_power_cap,
            'sender_mass_power_init': self.sender_mass_power_init,
            'sender_mass_learned_power_min': float(learned_power.min().cpu())
            if learned_power is not None else '',
            'sender_mass_learned_power_mean': float(learned_power.mean().cpu())
            if learned_power is not None else '',
            'sender_mass_learned_power_max': float(learned_power.max().cpu())
            if learned_power is not None else '',
            'sender_effective_mass_min': float(sender_effective_mass.min().cpu())
            if sender_effective_mass is not None else 1.0,
            'sender_effective_mass_mean': float(sender_effective_mass.mean().cpu())
            if sender_effective_mass is not None else 1.0,
            'sender_effective_mass_max': float(sender_effective_mass.max().cpu())
            if sender_effective_mass is not None else 1.0,
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
            'attention_query_tokens': self.latest_attention_query_tokens,
            'attention_kv_tokens': self.latest_attention_kv_tokens,
            'local_temporal_branch': self.local_temporal_branch,
            'local_temporal_gate': self.latest_local_temporal_gate,
            'neighbor_cross_gate': self.latest_neighbor_cross_gate,
            'group_residual_gate': self.latest_group_residual_gate,
            'group_residual_period': int(getattr(self, 'group_residual_period', 0) or 0),
            'group_residual_season_gate': self.latest_group_residual_season_gate,
            'group_residual_coeff_abs_mean': self.latest_group_residual_coeff_abs_mean,
            'within_group_residual_gate': float(
                getattr(self.reducer, 'latest_within_group_residual_gate', 0.0)
            ) if self.reducer is not None else 0.0,
            'within_group_residual_mode': getattr(
                self.reducer,
                'within_group_residual_mode',
                'projection',
            ) if self.reducer is not None else '',
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
