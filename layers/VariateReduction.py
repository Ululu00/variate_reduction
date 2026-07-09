import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class VariateReducer(nn.Module):
    def __init__(
        self,
        reduction_type,
        num_variates,
        reduced_k,
        d_model,
        n_heads=1,
        expansion_type='transpose',
        expansion_temperature=1.0,
        expansion_topk=0,
        use_coverage_loss=False,
        use_assignment_entropy_loss=False,
        target_num_variates=None,
        lowrank_reducer_rank=8,
        decoder_residual_gate_init=0.0,
        variate_anchor_map_path='',
        compute_aux_losses=True,
        force_assignment_cache=False,
        sparse_compress_topk=0,
        sparse_expand_topk=0,
        wcomp_normalization='softmax',
        entmax_alpha=1.5,
        use_wcomp_entropy_loss=False,
        use_group_attention_entropy_loss=False,
        group_attention_entropy_target=0.35,
        use_cycle_slot_loss=False,
        cycle_topr=0,
        cycle_topr_multiplier=1.0,
        cycle_b_norm='row_l1',
        cycle_eps=1e-8,
        export_slot_diagnostics=False,
    ):
        super().__init__()
        self.reduction_type = reduction_type
        self.num_variates = int(num_variates)
        self.target_num_variates = int(target_num_variates) if target_num_variates is not None else self.num_variates
        self.reduced_k = int(reduced_k)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.expansion_type = expansion_type
        self.expansion_temperature = float(expansion_temperature)
        self.expansion_topk = int(expansion_topk)
        self.use_coverage_loss = use_coverage_loss
        self.use_assignment_entropy_loss = use_assignment_entropy_loss
        self.lowrank_reducer_rank = int(lowrank_reducer_rank)
        self.decoder_residual_gate_init = float(decoder_residual_gate_init)
        self.variate_anchor_map_path = str(variate_anchor_map_path or '')
        self.compute_aux_losses = bool(compute_aux_losses)
        self.force_assignment_cache = bool(force_assignment_cache)
        self.sparse_compress_topk = int(sparse_compress_topk)
        self.sparse_expand_topk = int(sparse_expand_topk)
        self.wcomp_normalization = str(wcomp_normalization or 'softmax')
        self.entmax_alpha = float(entmax_alpha)
        self.use_wcomp_entropy_loss = bool(use_wcomp_entropy_loss)
        self.use_group_attention_entropy_loss = bool(use_group_attention_entropy_loss)
        self.group_attention_entropy_target = float(group_attention_entropy_target)
        self.use_cycle_slot_loss = bool(use_cycle_slot_loss)
        self.cycle_topr = int(cycle_topr)
        self.cycle_topr_multiplier = float(cycle_topr_multiplier)
        self.cycle_b_norm = str(cycle_b_norm or 'row_l1')
        self.cycle_eps = float(cycle_eps)
        self.export_slot_diagnostics = bool(export_slot_diagnostics)
        self.latest_aux_loss = None
        self.latest_aux_losses = None
        self.latest_selected_indices = None
        self.latest_attention_matrix = None
        self.latest_cycle_diagnostics = {}
        self.latest_cycle_matrices = {}

        if self.reduction_type == 'none':
            return
        removed_generation_reducers = {
            'mlp_generation',
            'mlp_generation_id',
            'mlp_lowrank_generation',
            'mlp_sparse_representative',
        }
        if self.reduction_type in removed_generation_reducers:
            raise ValueError(
                '{} was removed from the active code path; use attention/slot/anchor reducers instead'.format(
                    self.reduction_type
                )
            )

        valid_expansion_types = {
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
        }
        if self.expansion_type not in valid_expansion_types:
            raise ValueError('Unknown variate expansion type: {}'.format(self.expansion_type))
        if self.expansion_temperature <= 0:
            raise ValueError('expansion_temperature must be positive')
        if self.wcomp_normalization not in {'softmax', 'entmax15'}:
            raise ValueError('Unknown wcomp_normalization: {}'.format(self.wcomp_normalization))
        if self.cycle_b_norm != 'row_l1':
            raise ValueError('Unknown cycle_b_norm: {}'.format(self.cycle_b_norm))
        if self.cycle_topr < 0:
            raise ValueError('cycle_topr must be non-negative')
        if self.cycle_topr_multiplier <= 0:
            raise ValueError('cycle_topr_multiplier must be positive')
        if self.cycle_eps <= 0:
            raise ValueError('cycle_eps must be positive')
        if self.group_attention_entropy_target < 0.0 or self.group_attention_entropy_target > 1.0:
            raise ValueError('group_attention_entropy_target must be in [0, 1]')
        if self.expansion_type == 'topk_column_normalized' and self.expansion_topk <= 0:
            raise ValueError('expansion_topk must be positive for topk_column_normalized expansion')

        if self.reduced_k <= 0:
            raise ValueError('reduced_k must be positive when variate reduction is enabled')
        if self.reduced_k > self.num_variates:
            raise ValueError('reduced_k cannot exceed num_variates')
        if self.target_num_variates != self.num_variates:
            raise ValueError('target_num_variates can differ from num_variates only for supported split reducers')
        if self._uses_id_query_decoder() or self._uses_id_topk_decoder():
            self.register_buffer(
                'target_id_embedding',
                self._build_source_id_embedding(self.target_num_variates, self.d_model),
                persistent=False,
            )
            if self._uses_id_topk_decoder():
                self.register_buffer(
                    'latent_id_embedding',
                    self._build_source_id_embedding(self.reduced_k, self.d_model),
                    persistent=False,
                )
                self._init_id_topk_decoder()
            if self.expansion_type in {'id_query_scalar_residual', 'id_topk_scalar_residual'}:
                self.residual_gate = nn.Parameter(torch.tensor(self.decoder_residual_gate_init, dtype=torch.float32))
            elif self.expansion_type in {'id_query_fixed_scalar_residual', 'id_topk_fixed_scalar_residual'}:
                self.register_buffer(
                    'residual_gate',
                    torch.tensor(self.decoder_residual_gate_init, dtype=torch.float32),
                    persistent=False,
                )
        elif self._uses_attention_decoder():
            if self.d_model % self.n_heads != 0:
                raise ValueError('d_model must be divisible by n_heads for query decoder expansion')
            if self._uses_learned_query_decoder():
                self.variate_queries = nn.Parameter(torch.empty(self.target_num_variates, self.d_model))
                nn.init.xavier_uniform_(self.variate_queries)
            self.query_decoder_attn = nn.MultiheadAttention(
                self.d_model,
                self.n_heads,
                batch_first=True,
            )
            if self.expansion_type == 'query_decoder_scalar_residual':
                self.residual_gate = nn.Parameter(torch.tensor(self.decoder_residual_gate_init, dtype=torch.float32))
            elif self.expansion_type == 'query_decoder_variate_residual':
                self.residual_gate = nn.Parameter(
                    torch.full((1, self.target_num_variates, 1), self.decoder_residual_gate_init)
                )
        elif self.expansion_type in {
            'fixed_assignment_scalar_residual',
            'fixed_masked_scalar_residual',
            'masked_linear_scalar_residual',
            'masked_softmax_scalar_residual',
        }:
            self.residual_gate = nn.Parameter(torch.tensor(self.decoder_residual_gate_init, dtype=torch.float32))
        elif self.expansion_type in {
            'fixed_masked_variate_residual',
            'masked_softmax_variate_residual',
        }:
            self.residual_gate = nn.Parameter(
                torch.full((1, self.target_num_variates, 1), self.decoder_residual_gate_init)
            )

        if self.reduction_type == 'mlp_static_combination':
            self.compression_logits = nn.Parameter(torch.zeros(self.reduced_k, self.num_variates))
            nn.init.xavier_uniform_(self.compression_logits)
        elif self.reduction_type in {'MLP_attention', 'latent_query_attention'}:
            if self.expansion_type != 'slot_learned_linear':
                raise ValueError(
                    '{} requires slot_learned_linear expansion so W_exp is a learned K->C linear map'.format(
                        self.reduction_type
                    )
                )
            if self.reduction_type == 'MLP_attention':
                self.score_mlp = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.reduced_k),
                )
            else:
                self.latent_queries = nn.Parameter(torch.empty(self.reduced_k, self.d_model))
                nn.init.xavier_uniform_(self.latent_queries)
                self.key_proj = nn.Linear(self.d_model, self.d_model, bias=False)
                self.value_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.slot_expand_linear = nn.Linear(self.reduced_k, self.target_num_variates, bias=False)
        elif self.reduction_type == 'learned_anchor_selection':
            if self.expansion_type in {
                'fixed_assignment',
                'fixed_masked_linear',
                'masked_linear',
                'masked_softmax',
                'masked_linear_scalar_residual',
                'masked_softmax_scalar_residual',
                'masked_softmax_variate_residual',
                'slot_learned_linear',
            }:
                raise ValueError(
                    '{} expansion requires fixed anchor neighborhoods; use a dense or query decoder for '
                    'learned_anchor_selection'.format(self.expansion_type)
                )
            self.selector_logits = nn.Parameter(torch.zeros(self.reduced_k, self.num_variates))
            self._init_selector_logits()
            if not self._uses_attention_decoder():
                self.expand_linear = nn.Linear(self.reduced_k, self.target_num_variates, bias=False)
        elif self.reduction_type == 'sample_adaptive_selection':
            if not self._uses_token_query_decoder():
                raise ValueError(
                    'sample_adaptive_selection requires token_query_decoder so original variable identity '
                    'is used when decoding dynamic representatives'
                )
        elif self.reduction_type in {
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
        }:
            if self.expansion_type not in {
                'fixed_assignment',
                'fixed_assignment_scalar_residual',
                'fixed_masked_linear',
                'fixed_masked_scalar_residual',
            } and not self._uses_attention_decoder() and not self._uses_sparse_parameter_decoder():
                self.expand_linear = nn.Linear(self.reduced_k, self.target_num_variates, bias=False)
            if self.reduction_type == 'variate_anchor_residual_pool':
                self.anchor_residual_gate = nn.Parameter(
                    torch.tensor(self.decoder_residual_gate_init, dtype=torch.float32)
                )
            if self.reduction_type == 'variate_anchor_channel_residual_pool':
                self.anchor_channel_residual_gate = nn.Parameter(
                    torch.full((self.reduced_k, self.d_model), self.decoder_residual_gate_init)
                )
            if self.reduction_type in {
                'variate_anchor_grouplinear_residual_pool',
                'variate_anchor_grouplinear_residual_pool_id',
            }:
                self.anchor_residual_gate = nn.Parameter(
                    torch.tensor(self.decoder_residual_gate_init, dtype=torch.float32)
                )
            if self.reduction_type == 'variate_anchor_grouplinear_residual_pool_id':
                self.register_buffer(
                    'source_id_embedding',
                    self._build_source_id_embedding(self.num_variates, self.d_model),
                    persistent=False,
                )
                self.source_id_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            if self.reduction_type == 'variate_group_adaptive_id_selection':
                self.selected_variate_embedding = nn.Parameter(torch.empty(self.num_variates, self.d_model))
                nn.init.normal_(self.selected_variate_embedding, std=0.02)
            if self.reduction_type in {
                'variate_grouped_average_id',
                'variate_grouped_softmax_pool_id',
                'variate_grouped_linear_pool_id',
            }:
                self.register_buffer(
                    'source_id_embedding',
                    self._build_source_id_embedding(self.num_variates, self.d_model),
                    persistent=False,
                )
                self.source_id_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self._init_fixed_reduction()
            if self._uses_sparse_parameter_decoder():
                self._init_sparse_parameter_decoder()
        elif self.reduction_type in {
            'mlp_slot_attention',
            'mlp_sparse_slot_attention',
            'mlp_sparse_slot_attention_id',
            'mlp_linear_sparse_slot_attention_id',
            'mlp_static_sparse_slot_attention_id',
            'mlp_slot_attention_hybrid_linear',
        }:
            if self.reduction_type == 'mlp_static_sparse_slot_attention_id':
                self.register_buffer(
                    'static_slot_queries',
                    F.normalize(self._build_source_id_embedding(self.reduced_k, self.d_model), p=2, dim=-1),
                    persistent=False,
                )
            elif self.reduction_type == 'mlp_linear_sparse_slot_attention_id':
                self.score_mlp = nn.Linear(self.d_model, self.reduced_k)
            else:
                self.score_mlp = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.reduced_k),
                )
            if self.reduction_type in {
                'mlp_sparse_slot_attention_id',
                'mlp_linear_sparse_slot_attention_id',
                'mlp_static_sparse_slot_attention_id',
            }:
                self.register_buffer(
                    'source_id_embedding',
                    self._build_source_id_embedding(self.num_variates, self.d_model),
                    persistent=False,
                )
                if self.reduction_type == 'mlp_static_sparse_slot_attention_id':
                    self.register_buffer('source_id_gate', torch.tensor(0.1, dtype=torch.float32), persistent=False)
                else:
                    self.source_id_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            if self.reduction_type == 'mlp_slot_attention_hybrid_linear':
                self.compress_linear = nn.Linear(self.num_variates, self.reduced_k, bias=False)
                self.expand_linear = nn.Linear(self.reduced_k, self.num_variates, bias=False)
                self.hybrid_linear_gate = nn.Parameter(torch.zeros(()))
            if (
                self.expansion_type == 'slot_learned_linear'
                and self.reduction_type
                in {
                    'mlp_slot_attention',
                    'mlp_sparse_slot_attention',
                    'mlp_sparse_slot_attention_id',
                    'mlp_linear_sparse_slot_attention_id',
                    'mlp_static_sparse_slot_attention_id',
                }
            ):
                self.slot_expand_linear = nn.Linear(self.reduced_k, self.num_variates, bias=False)
        else:
            raise ValueError('Unknown variate reduction type: {}'.format(self.reduction_type))

    def _init_selector_logits(self):
        anchor_map = self._load_anchor_map()
        if anchor_map is not None and 'anchor_indices' in anchor_map:
            anchor_indices = self._validate_anchor_indices(anchor_map['anchor_indices'])
        else:
            anchor_indices = torch.linspace(0, self.num_variates - 1, self.reduced_k).round().long()
        with torch.no_grad():
            self.selector_logits.zero_()
            self.selector_logits[torch.arange(self.reduced_k), anchor_indices] = 2.0
        self.register_buffer('selector_init_indices', anchor_indices, persistent=True)

    def _load_anchor_map(self):
        if not self.variate_anchor_map_path:
            return None
        data = np.load(self.variate_anchor_map_path)
        result = {}
        if 'anchor_indices' in data:
            result['anchor_indices'] = torch.as_tensor(data['anchor_indices'], dtype=torch.long)
        if 'group_ids' in data:
            result['group_ids'] = torch.as_tensor(data['group_ids'], dtype=torch.long)
        if 'expand_mask' in data:
            result['expand_mask'] = torch.as_tensor(data['expand_mask'], dtype=torch.float32)
        if 'expand_init_weight' in data:
            result['expand_init_weight'] = torch.as_tensor(data['expand_init_weight'], dtype=torch.float32)
        if 'compress_init_weight' in data:
            result['compress_init_weight'] = torch.as_tensor(data['compress_init_weight'], dtype=torch.float32)
        return result

    def _validate_group_ids_for_length(self, group_ids, length, label):
        if group_ids.numel() < length:
            raise ValueError(
                '{} group_ids length {} is smaller than expected length {}'.format(
                    label, group_ids.numel(), length
                )
            )
        group_ids = group_ids[:length].long()
        if group_ids.min().item() < 0 or group_ids.max().item() >= self.reduced_k:
            raise ValueError('{} group_ids out of range for reduced_k={}'.format(label, self.reduced_k))
        counts = torch.bincount(group_ids, minlength=self.reduced_k).float()
        if torch.any(counts <= 0):
            raise ValueError('{} group_ids has empty groups; every latent token needs input'.format(label))
        return group_ids, counts

    def _validate_anchor_indices(self, anchor_indices):
        if anchor_indices.numel() != self.reduced_k:
            raise ValueError(
                'anchor map has {} anchors, expected reduced_k={}'.format(
                    anchor_indices.numel(), self.reduced_k
                )
            )
        if anchor_indices.min().item() < 0 or anchor_indices.max().item() >= self.num_variates:
            raise ValueError('anchor_indices out of range for num_variates={}'.format(self.num_variates))
        return anchor_indices.long()

    def _validate_group_ids(self, group_ids):
        group_ids, _ = self._validate_group_ids_for_length(
            group_ids,
            self.target_num_variates,
            'anchor map',
        )
        return group_ids

    def _compress_init_from_anchor_map(self, anchor_map):
        if anchor_map is None:
            return None
        if 'compress_init_weight' in anchor_map:
            weight = anchor_map['compress_init_weight'][:self.reduced_k, :self.num_variates].float()
            if tuple(weight.shape) != (self.reduced_k, self.num_variates):
                raise ValueError(
                    'compress_init_weight shape {} does not match ({}, {})'.format(
                        tuple(weight.shape), self.reduced_k, self.num_variates
                    )
                )
            return weight
        if 'group_ids' not in anchor_map:
            return None
        group_ids, counts = self._validate_group_ids_for_length(
            anchor_map['group_ids'],
            self.num_variates,
            'linear bottleneck compress init',
        )
        weight = torch.zeros(self.reduced_k, self.num_variates)
        source_ids = torch.arange(self.num_variates)
        weight[group_ids, source_ids] = 1.0 / counts[group_ids]
        return weight

    def _expand_init_from_anchor_map(self, anchor_map):
        if anchor_map is None:
            return None
        if 'expand_init_weight' in anchor_map:
            weight = anchor_map['expand_init_weight'][:self.target_num_variates, :self.reduced_k].float()
            if tuple(weight.shape) != (self.target_num_variates, self.reduced_k):
                raise ValueError(
                    'expand_init_weight shape {} does not match ({}, {})'.format(
                        tuple(weight.shape), self.target_num_variates, self.reduced_k
                    )
                )
            return weight
        if 'group_ids' not in anchor_map:
            return None
        group_ids, _ = self._validate_group_ids_for_length(
            anchor_map['group_ids'],
            self.target_num_variates,
            'linear bottleneck expand init',
        )
        weight = torch.zeros(self.target_num_variates, self.reduced_k)
        target_ids = torch.arange(self.target_num_variates)
        weight[target_ids, group_ids] = 1.0
        return weight

    def _init_dense_generation_from_anchor_map(self):
        anchor_map = self._load_anchor_map()
        if anchor_map is None:
            return
        compress = self._compress_init_from_anchor_map(anchor_map)
        expand = self._expand_init_from_anchor_map(anchor_map)
        with torch.no_grad():
            if compress is not None:
                self.compress_linear.weight.copy_(compress.to(self.compress_linear.weight))
            if expand is not None and hasattr(self, 'expand_linear'):
                self.expand_linear.weight.copy_(expand.to(self.expand_linear.weight))
            if hasattr(self, 'reconstruct_linear'):
                reconstruct = self._compress_reconstruction_init(anchor_map)
                if reconstruct is not None:
                    self.reconstruct_linear.weight.copy_(reconstruct.to(self.reconstruct_linear.weight))

    def _compress_reconstruction_init(self, anchor_map):
        if anchor_map is None:
            return None
        if 'group_ids' not in anchor_map:
            return None
        group_ids, _ = self._validate_group_ids_for_length(
            anchor_map['group_ids'],
            self.num_variates,
            'linear bottleneck reconstruction init',
        )
        weight = torch.zeros(self.num_variates, self.reduced_k)
        source_ids = torch.arange(self.num_variates)
        weight[source_ids, group_ids] = 1.0
        return weight

    @staticmethod
    def _copy_svd_factorization(weight, basis_linear, mix_linear):
        rank = min(basis_linear.weight.shape[0], weight.shape[0], weight.shape[1])
        if rank <= 0:
            return
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        with torch.no_grad():
            basis_linear.weight.zero_()
            mix_linear.weight.zero_()
            basis_linear.weight[:rank, :].copy_(vh[:rank, :].to(basis_linear.weight))
            mix_linear.weight[:, :rank].copy_((u[:, :rank] * s[:rank].unsqueeze(0)).to(mix_linear.weight))

    def _init_lowrank_generation_from_anchor_map(self):
        anchor_map = self._load_anchor_map()
        if anchor_map is None:
            return
        compress = self._compress_init_from_anchor_map(anchor_map)
        expand = self._expand_init_from_anchor_map(anchor_map)
        if compress is not None:
            self._copy_svd_factorization(
                compress,
                self.compress_source_basis,
                self.compress_latent_mix,
            )
        if expand is not None:
            self._copy_svd_factorization(
                expand,
                self.expand_latent_basis,
                self.expand_target_mix,
            )

    def _init_fixed_reduction(self):
        anchor_map = self._load_anchor_map()
        if self.reduction_type in {
            'variate_anchor_selection',
            'variate_group_adaptive_selection',
            'variate_group_adaptive_id_selection',
            'variate_anchor_residual_pool',
            'variate_anchor_channel_residual_pool',
            'variate_anchor_grouplinear_residual_pool',
            'variate_anchor_grouplinear_residual_pool_id',
            'variate_anchor_softmax_pool',
        }:
            if anchor_map is not None and 'anchor_indices' in anchor_map:
                anchor_indices = self._validate_anchor_indices(anchor_map['anchor_indices'])
            else:
                anchor_indices = torch.linspace(0, self.num_variates - 1, self.reduced_k).round().long()
            self.register_buffer('anchor_indices', anchor_indices, persistent=True)
            if anchor_map is not None and 'group_ids' in anchor_map:
                group_ids = self._validate_group_ids(anchor_map['group_ids'])
            else:
                group_ids = self._nearest_anchor_ids(anchor_indices)
            counts = torch.bincount(group_ids, minlength=self.reduced_k).float()
            max_group_size = int(counts.max().item())
            group_member_indices = torch.zeros(self.reduced_k, max_group_size, dtype=torch.long)
            group_member_mask = torch.zeros(self.reduced_k, max_group_size, dtype=torch.bool)
            for group_idx in range(self.reduced_k):
                members = torch.nonzero(group_ids == group_idx, as_tuple=False).flatten()
                group_member_indices[group_idx, :members.numel()] = members
                group_member_mask[group_idx, :members.numel()] = True
            self.register_buffer('group_member_indices', group_member_indices, persistent=False)
            self.register_buffer('group_member_mask', group_member_mask, persistent=False)
            compress_weight = torch.zeros(self.reduced_k, self.num_variates)
            compress_weight[torch.arange(self.reduced_k), anchor_indices] = 1.0
            group_mean_weight = torch.zeros(self.reduced_k, self.num_variates)
            source_ids = torch.arange(self.num_variates)
            group_mean_weight[group_ids, source_ids] = 1.0 / counts[group_ids]
        elif self.reduction_type in {
            'variate_grouped_average',
            'variate_grouped_average_id',
            'variate_grouped_corr_pool',
            'variate_grouped_softmax_pool',
            'variate_grouped_softmax_pool_id',
            'variate_grouped_linear_pool',
            'variate_grouped_linear_pool_id',
            'grouped_soft_representative',
        }:
            source_ids = torch.arange(self.num_variates)
            if anchor_map is not None and 'group_ids' in anchor_map:
                group_ids = self._validate_group_ids(anchor_map['group_ids'])
            else:
                group_ids = torch.div(source_ids * self.reduced_k, self.num_variates, rounding_mode='floor')
                group_ids = torch.clamp(group_ids, max=self.reduced_k - 1).long()
            counts = torch.bincount(group_ids, minlength=self.reduced_k).float()
            compress_weight = torch.zeros(self.reduced_k, self.num_variates)
            if self.reduction_type in {
                'variate_grouped_corr_pool',
                'variate_grouped_softmax_pool',
                'variate_grouped_softmax_pool_id',
                'variate_grouped_linear_pool',
                'variate_grouped_linear_pool_id',
            } and anchor_map is not None and 'compress_init_weight' in anchor_map:
                compress_init = anchor_map['compress_init_weight'][:, :self.num_variates].float()
                if tuple(compress_init.shape) != (self.reduced_k, self.num_variates):
                    raise ValueError(
                        'compress_init_weight shape {} does not match ({}, {})'.format(
                            tuple(compress_init.shape), self.reduced_k, self.num_variates
                        )
                    )
                group_mask = torch.zeros_like(compress_init)
                group_mask[group_ids, source_ids] = 1.0
                compress_weight = compress_init * group_mask
                denom = compress_weight.abs().sum(dim=-1, keepdim=True)
                valid_rows = denom.squeeze(-1) > 1e-8
                compress_weight = torch.where(
                    denom > 1e-8,
                    compress_weight / denom.clamp_min(1e-8),
                    compress_weight,
                )
                if torch.any(~valid_rows):
                    empty_groups = torch.nonzero(~valid_rows, as_tuple=False).flatten()
                    for group_idx in empty_groups.tolist():
                        members = group_ids == int(group_idx)
                        compress_weight[int(group_idx), members] = 1.0 / counts[int(group_idx)]
            elif self.reduction_type in {
                'variate_grouped_corr_pool',
                'variate_grouped_softmax_pool',
                'variate_grouped_softmax_pool_id',
                'variate_grouped_linear_pool',
                'variate_grouped_linear_pool_id',
            } and anchor_map is not None and 'expand_init_weight' in anchor_map:
                expand_init = anchor_map['expand_init_weight'][:self.target_num_variates, :self.reduced_k].float()
                if tuple(expand_init.shape) != (self.target_num_variates, self.reduced_k):
                    raise ValueError(
                        'expand_init_weight shape {} does not match ({}, {})'.format(
                            tuple(expand_init.shape), self.target_num_variates, self.reduced_k
                        )
                    )
                source_scores = expand_init[source_ids, group_ids].clamp_min(0.0)
                denom = torch.zeros(self.reduced_k)
                denom.scatter_add_(0, group_ids, source_scores)
                valid = denom[group_ids] > 1e-8
                source_weights = torch.where(valid, source_scores / denom[group_ids], 1.0 / counts[group_ids])
                compress_weight[group_ids, source_ids] = source_weights
            else:
                compress_weight[group_ids, source_ids] = 1.0 / counts[group_ids]
            group_mean_weight = compress_weight.clone()
        else:
            raise ValueError('fixed reduction init called for {}'.format(self.reduction_type))

        self.register_buffer('group_ids', group_ids, persistent=True)
        self.register_buffer('group_counts', counts, persistent=True)
        self.register_buffer('fixed_compress_weight', compress_weight, persistent=True)
        self.register_buffer('fixed_group_mean_weight', group_mean_weight, persistent=True)
        if not hasattr(self, 'group_member_indices'):
            max_group_size = int(counts.max().item())
            group_member_indices = torch.zeros(self.reduced_k, max_group_size, dtype=torch.long)
            group_member_mask = torch.zeros(self.reduced_k, max_group_size, dtype=torch.bool)
            for group_idx in range(self.reduced_k):
                members = torch.nonzero(group_ids == group_idx, as_tuple=False).flatten()
                group_member_indices[group_idx, :members.numel()] = members
                group_member_mask[group_idx, :members.numel()] = True
            self.register_buffer('group_member_indices', group_member_indices, persistent=False)
            self.register_buffer('group_member_mask', group_member_mask, persistent=False)
        if self.reduction_type == 'variate_anchor_softmax_pool':
            logits = torch.zeros(self.num_variates)
            logits[anchor_indices] = 6.0
            self.group_pool_logits = nn.Parameter(logits)
        if self.reduction_type in {
            'variate_anchor_grouplinear_residual_pool',
            'variate_anchor_grouplinear_residual_pool_id',
        }:
            source_init = group_mean_weight.sum(dim=0)
            self.group_pool_weight = nn.Parameter(source_init.clone())
        if self.reduction_type in {'variate_grouped_softmax_pool', 'variate_grouped_softmax_pool_id'}:
            source_init = compress_weight.sum(dim=0).clamp_min(1e-8)
            self.group_pool_logits = nn.Parameter(torch.log(source_init))
        if self.reduction_type in {'variate_grouped_linear_pool', 'variate_grouped_linear_pool_id'}:
            source_init = compress_weight.sum(dim=0)
            self.group_pool_weight = nn.Parameter(source_init.clone())
        if self.reduction_type == 'grouped_soft_representative':
            self.group_rep_score_mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, 1),
            )
            source_init = compress_weight.sum(dim=0).clamp_min(1e-8)
            self.group_rep_source_logits = nn.Parameter(torch.log(source_init))
        expand_weight = torch.zeros(self.target_num_variates, self.reduced_k)
        target_ids = torch.arange(self.target_num_variates)
        expand_weight[target_ids, group_ids[:self.target_num_variates]] = 1.0
        if anchor_map is not None and 'expand_mask' in anchor_map:
            expand_mask = anchor_map['expand_mask'][:self.target_num_variates, :self.reduced_k].float()
            if tuple(expand_mask.shape) != (self.target_num_variates, self.reduced_k):
                raise ValueError(
                    'expand_mask shape {} does not match ({}, {})'.format(
                        tuple(expand_mask.shape), self.target_num_variates, self.reduced_k
                    )
                )
            if torch.any(expand_mask.sum(dim=-1) <= 0):
                raise ValueError('expand_mask must allow at least one anchor per target variable')
            expand_mask = torch.maximum(expand_mask, expand_weight)
        else:
            expand_mask = (expand_weight > 0).float()
        expand_init_weight = expand_weight
        if anchor_map is not None and 'expand_init_weight' in anchor_map:
            expand_init_weight = anchor_map['expand_init_weight'][:self.target_num_variates, :self.reduced_k].float()
            if tuple(expand_init_weight.shape) != (self.target_num_variates, self.reduced_k):
                raise ValueError(
                    'expand_init_weight shape {} does not match ({}, {})'.format(
                        tuple(expand_init_weight.shape), self.target_num_variates, self.reduced_k
                    )
                )
            expand_init_weight = expand_init_weight * expand_mask
            if torch.any(expand_init_weight.abs().sum(dim=-1) <= 0):
                raise ValueError('expand_init_weight must keep at least one non-zero active anchor per target')
        self.register_buffer('fixed_expand_weight', expand_weight, persistent=True)
        self.register_buffer('fixed_expand_mask', expand_mask, persistent=True)
        self.register_buffer('fixed_expand_init_weight', expand_init_weight, persistent=True)
        active_counts = expand_mask.sum(dim=-1).long()
        max_active = int(active_counts.max().item())
        expand_indices = torch.zeros(self.target_num_variates, max_active, dtype=torch.long)
        expand_active = torch.zeros(self.target_num_variates, max_active, dtype=torch.float32)
        for target_idx in range(self.target_num_variates):
            active = torch.nonzero(expand_mask[target_idx] > 0, as_tuple=False).flatten().long()
            expand_indices[target_idx, :active.numel()] = active
            expand_active[target_idx, :active.numel()] = 1.0
        self.register_buffer('fixed_expand_indices', expand_indices, persistent=True)
        self.register_buffer('fixed_expand_active', expand_active, persistent=True)
        if hasattr(self, 'expand_linear'):
            with torch.no_grad():
                if self.expansion_type in {'masked_softmax', 'masked_softmax_scalar_residual'}:
                    logits = torch.full_like(expand_weight, -4.0)
                    logits = logits.masked_fill(expand_mask <= 0, -20.0)
                    logits = logits.masked_fill(expand_weight > 0, 4.0)
                    self.expand_linear.weight.copy_(logits)
                else:
                    self.expand_linear.weight.copy_(expand_init_weight)

    def _init_sparse_parameter_decoder(self):
        if not hasattr(self, 'fixed_expand_indices') or not hasattr(self, 'fixed_expand_active'):
            raise ValueError('sparse parameter decoder requires fixed anchor mask initialization')
        active = self.fixed_expand_active.float()
        # Keep the stable one-hot assigned-latent initialization used by the
        # dense masked-softmax decoder, while learning only active top-m logits.
        init_weight = self.fixed_expand_weight.float()
        selected_init = torch.gather(init_weight, dim=1, index=self.fixed_expand_indices).clamp_min(0.0)
        selected_init = selected_init * active
        denom = selected_init.sum(dim=-1, keepdim=True)
        uniform = active / active.sum(dim=-1, keepdim=True).clamp_min(1.0)
        weights = torch.where(denom > 1e-8, selected_init / denom.clamp_min(1e-8), uniform)
        logits = torch.log(weights.clamp_min(1e-8)).masked_fill(active <= 0, -20.0)
        self.sparse_expand_logits = nn.Parameter(logits)

    def _nearest_anchor_ids(self, anchor_indices):
        target_ids = torch.arange(self.target_num_variates)
        distances = torch.abs(target_ids.unsqueeze(1) - anchor_indices.unsqueeze(0))
        return torch.argmin(distances, dim=1).long()

    def _group_softmax_weights(self):
        logits = self.group_pool_logits
        group_ids = self.group_ids.to(logits.device)
        group_max = logits.new_full((self.reduced_k,), -float('inf'))
        group_max.scatter_reduce_(0, group_ids, logits, reduce='amax', include_self=True)
        exp_logits = torch.exp(logits - group_max[group_ids])
        denom = logits.new_zeros(self.reduced_k)
        denom.scatter_add_(0, group_ids, exp_logits)
        return exp_logits / (denom[group_ids] + 1e-8)

    def _group_softmax_compress_weight(self):
        source_weights = self._group_softmax_weights()
        weight = source_weights.new_zeros(self.reduced_k, self.num_variates)
        source_ids = torch.arange(self.num_variates, device=source_weights.device)
        weight[self.group_ids.to(source_weights.device), source_ids] = source_weights
        return weight

    def _normalize_group_member_scores(self, scores, member_mask):
        temperature = max(float(self.expansion_temperature), 1e-6)
        scores = scores / temperature
        mask = member_mask.view(1, self.reduced_k, -1).to(device=scores.device, dtype=torch.bool)
        if self.wcomp_normalization == 'softmax':
            weights = torch.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
        elif self.wcomp_normalization == 'entmax15':
            weights = self._entmax15(scores.masked_fill(~mask, -1e9), dim=-1)
        else:
            raise ValueError('Unknown wcomp_normalization: {}'.format(self.wcomp_normalization))
        weights = weights * mask.to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _materialize_group_assignment(self, group_weights, member_indices, member_mask):
        batch_size = group_weights.shape[0]
        assignment = group_weights.new_zeros(batch_size, self.reduced_k, self.num_variates)
        indices = member_indices.to(group_weights.device).view(1, self.reduced_k, -1).expand(batch_size, -1, -1)
        values = group_weights * member_mask.to(device=group_weights.device, dtype=group_weights.dtype).view(1, self.reduced_k, -1)
        assignment.scatter_add_(2, indices, values)
        return assignment

    def _group_linear_source_weights(self):
        raw = self.group_pool_weight
        group_ids = self.group_ids.to(raw.device)
        denom = raw.new_zeros(self.reduced_k)
        denom.scatter_add_(0, group_ids, raw.abs())
        fallback_weight = self.fixed_group_mean_weight if hasattr(self, 'fixed_group_mean_weight') else self.fixed_compress_weight
        fixed_source = fallback_weight.to(device=raw.device, dtype=raw.dtype).sum(dim=0)
        denom_per_source = denom[group_ids]
        return torch.where(
            denom_per_source > 1e-8,
            raw / denom_per_source.clamp_min(1e-8),
            fixed_source,
        )

    def _group_linear_compress_weight(self):
        source_weights = self._group_linear_source_weights()
        weight = source_weights.new_zeros(self.reduced_k, self.num_variates)
        source_ids = torch.arange(self.num_variates, device=source_weights.device)
        weight[self.group_ids.to(source_weights.device), source_ids] = source_weights
        return weight

    @staticmethod
    def _zero_like_tokens(var_tokens):
        return var_tokens.new_tensor(0.0)

    def _set_zero_aux_losses(self, var_tokens):
        aux_loss = self._zero_like_tokens(var_tokens)
        self.latest_aux_losses = self._aux_loss_dict(aux_loss)
        return aux_loss

    @staticmethod
    def _build_source_id_embedding(num_variates, d_model):
        position = torch.arange(0, int(num_variates), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, int(d_model), 2, dtype=torch.float32)
            * (-(math.log(10000.0) / float(d_model)))
        )
        embedding = torch.zeros(int(num_variates), int(d_model), dtype=torch.float32)
        embedding[:, 0::2] = torch.sin(position * div_term)
        if int(d_model) > 1:
            embedding[:, 1::2] = torch.cos(position * div_term[:embedding[:, 1::2].shape[1]])
        return embedding

    @staticmethod
    def _zero_like_param(module):
        try:
            param = next(module.parameters())
            return param.new_tensor(0.0)
        except StopIteration:
            return torch.tensor(0.0)

    @staticmethod
    def _orthogonal_loss(matrix, eps=1e-8):
        matrix_norm = F.normalize(matrix, p=2, dim=-1, eps=eps)
        gram = torch.matmul(matrix_norm, matrix_norm.transpose(-1, -2))
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        if gram.dim() == 3:
            eye = eye.unsqueeze(0)
        return torch.mean((gram - eye) ** 2)

    def _coverage_loss(self, matrix):
        if not self.use_coverage_loss:
            return matrix.new_tensor(0.0)
        coverage = matrix.sum(dim=-2)
        target = float(self.reduced_k) / float(self.num_variates)
        return torch.mean((coverage - target) ** 2)

    def _assignment_entropy_loss(self, matrix, eps=1e-8):
        if not self.use_assignment_entropy_loss:
            return matrix.new_tensor(0.0)
        entropy = -(matrix * torch.log(matrix + eps)).sum(dim=-1)
        normalizer = torch.log(matrix.new_tensor(float(self.num_variates)))
        return torch.mean(entropy / (normalizer + eps))

    def _wcomp_entropy_loss(self, matrix, eps=1e-8):
        if not self.use_wcomp_entropy_loss:
            return matrix.new_tensor(0.0)
        entropy = -(matrix * torch.log(matrix + eps)).sum(dim=-1)
        normalizer = torch.log(matrix.new_tensor(float(self.num_variates)))
        return torch.mean(entropy / (normalizer + eps))

    def _group_attention_entropy_loss(self, group_weights, group_mask, eps=1e-8):
        if not self.use_group_attention_entropy_loss:
            return group_weights.new_tensor(0.0)
        mask = group_mask.to(device=group_weights.device, dtype=group_weights.dtype).view(1, self.reduced_k, -1)
        weights = group_weights * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        entropy = -(weights * torch.log(weights.clamp_min(eps))).sum(dim=-1)
        counts = mask.sum(dim=-1).clamp_min(1.0)
        normalizer = torch.log(counts.clamp_min(2.0))
        normalized_entropy = torch.where(counts > 1.0, entropy / normalizer.clamp_min(eps), entropy.new_zeros(entropy.shape))
        target = entropy.new_tensor(float(self.group_attention_entropy_target))
        valid = (counts > 1.0).to(entropy.dtype)
        denom = valid.sum().clamp_min(1.0)
        return (((normalized_entropy - target) ** 2) * valid).sum() / denom

    def _zero_cycle_diagnostics(self, reference):
        self.latest_cycle_diagnostics = {
            'cycle_topr_jaccard': 0.0,
            'cycle_a_effective_support': 0.0,
            'cycle_b_effective_support': 0.0,
            'cycle_slot_overlap': 0.0,
            'cycle_topr_size': 0,
            'w_eff_density': 0.0,
        }
        self.latest_cycle_matrices = {}
        return reference.new_tensor(0.0)

    def _slot_cycle_supported(self):
        return (
            self.reduction_type in {
                'mlp_slot_attention',
                'mlp_sparse_slot_attention',
                'mlp_sparse_slot_attention_id',
                'mlp_linear_sparse_slot_attention_id',
                'mlp_static_sparse_slot_attention_id',
            }
            and self.expansion_type == 'slot_learned_linear'
            and hasattr(self, 'slot_expand_linear')
        )

    def _cycle_topr_size(self, num_sources):
        if self.cycle_topr > 0:
            top_r = self.cycle_topr
        else:
            top_r = int(math.ceil((float(num_sources) / float(self.reduced_k)) * self.cycle_topr_multiplier))
        return max(1, min(int(top_r), int(num_sources)))

    def _cycle_b_matrix(self, device, dtype):
        if self.cycle_b_norm != 'row_l1':
            raise ValueError('Unknown cycle_b_norm: {}'.format(self.cycle_b_norm))
        B = self.slot_expand_linear.weight.transpose(0, 1).to(device=device, dtype=dtype).abs()
        B = B + float(self.cycle_eps)
        return B / B.sum(dim=-1, keepdim=True).clamp_min(float(self.cycle_eps))

    @staticmethod
    def _effective_support(probabilities, eps=1e-8):
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(eps))).sum(dim=-1)
        return torch.exp(entropy)

    def _cycle_slot_loss(self, A):
        zero = A.new_tensor(0.0)
        if not (self.use_cycle_slot_loss or self.export_slot_diagnostics):
            return self._zero_cycle_diagnostics(A)
        if A is None or A.dim() != 3 or not self._slot_cycle_supported():
            return self._zero_cycle_diagnostics(A if A is not None else zero)

        eps = float(self.cycle_eps)
        batch_size, num_slots, num_sources = A.shape
        if num_slots != self.reduced_k:
            return self._zero_cycle_diagnostics(A)

        A_prob = A / A.sum(dim=-1, keepdim=True).clamp_min(eps)
        B_prob = self._cycle_b_matrix(A_prob.device, A_prob.dtype)
        if B_prob.shape != (num_slots, num_sources):
            return self._zero_cycle_diagnostics(A)

        top_r = self._cycle_topr_size(num_sources)
        a_indices = torch.topk(A_prob.detach(), top_r, dim=-1).indices
        b_indices = torch.topk(B_prob.detach(), top_r, dim=-1).indices
        a_mask = torch.zeros_like(A_prob, dtype=torch.bool)
        a_mask.scatter_(-1, a_indices, True)
        b_mask_2d = torch.zeros_like(B_prob, dtype=torch.bool)
        b_mask_2d.scatter_(-1, b_indices, True)
        b_mask = b_mask_2d.unsqueeze(0).expand(batch_size, -1, -1)
        union_mask = a_mask | b_mask

        mask = union_mask.to(A_prob.dtype)
        B_expanded = B_prob.unsqueeze(0).expand_as(A_prob)
        A_masked = A_prob * mask
        B_masked = B_expanded * mask
        A_masked = A_masked / A_masked.sum(dim=-1, keepdim=True).clamp_min(eps)
        B_masked = B_masked / B_masked.sum(dim=-1, keepdim=True).clamp_min(eps)
        cycle_loss = (A_masked - B_masked).abs().sum(dim=-1).mean()
        cycle_loss = torch.nan_to_num(cycle_loss, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            intersection = (a_mask & b_mask).float().sum(dim=-1)
            union = (a_mask | b_mask).float().sum(dim=-1).clamp_min(1.0)
            A_support = self._effective_support(A_prob.detach().float(), eps=eps)
            B_support = self._effective_support(B_prob.detach().float(), eps=eps)
            if num_slots > 1:
                slot_overlap = torch.einsum(
                    'bkc,blc->bkl',
                    a_mask.float(),
                    a_mask.float(),
                ) / float(top_r)
                slot_overlap_value = float(self._offdiag_mean(slot_overlap).item())
            else:
                slot_overlap_value = 0.0
            W_eff = torch.einsum(
                'ck,bkv->bcv',
                self.slot_expand_linear.weight.detach().float(),
                A_prob.detach().float(),
            )
            A_mean = A_prob.detach().float().mean(dim=0)
            B_float = B_prob.detach().float()
            W_eff_mean = torch.matmul(self.slot_expand_linear.weight.detach().float(), A_mean)
            self.latest_cycle_diagnostics = {
                'cycle_topr_jaccard': float((intersection / union).mean().item()),
                'cycle_a_effective_support': float(A_support.mean().item()),
                'cycle_b_effective_support': float(B_support.mean().item()),
                'cycle_slot_overlap': slot_overlap_value,
                'cycle_topr_size': int(top_r),
                'w_eff_density': float((W_eff.abs() > eps).float().mean().item()),
            }
            self.latest_cycle_matrices = {
                'A': A_mean.cpu(),
                'B': B_float.cpu(),
                'W_eff': W_eff_mean.cpu(),
            }

        if not self.use_cycle_slot_loss:
            return zero
        return cycle_loss

    def _entmax15(self, scores, dim=-1, n_iter=50, eps=1e-8):
        if abs(float(self.entmax_alpha) - 1.5) > 1e-6:
            raise ValueError('Only entmax_alpha=1.5 is implemented for this experiment')
        scores = scores - scores.max(dim=dim, keepdim=True).values
        tau_lo = scores.max(dim=dim, keepdim=True).values - 2.0
        tau_hi = scores.max(dim=dim, keepdim=True).values
        for _ in range(int(n_iter)):
            tau_mid = (tau_lo + tau_hi) / 2.0
            probs = torch.clamp(0.5 * (scores - tau_mid), min=0.0).pow(2)
            sums = probs.sum(dim=dim, keepdim=True)
            tau_lo = torch.where(sums >= 1.0, tau_mid, tau_lo)
            tau_hi = torch.where(sums < 1.0, tau_mid, tau_hi)
        probs = torch.clamp(0.5 * (scores - tau_lo), min=0.0).pow(2)
        return probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)

    def _normalize_wcomp_scores(self, scores):
        temperature = max(float(self.expansion_temperature), 1e-6)
        scores = scores / temperature
        if self.wcomp_normalization == 'softmax':
            return F.softmax(scores, dim=-1)
        if self.wcomp_normalization == 'entmax15':
            return self._entmax15(scores, dim=-1)
        raise ValueError('Unknown wcomp_normalization: {}'.format(self.wcomp_normalization))

    def _decoder_weight_matrix(self):
        if hasattr(self, 'expand_linear'):
            if self.expansion_type in {
                'masked_linear',
                'masked_softmax',
                'masked_linear_scalar_residual',
                'masked_softmax_scalar_residual',
                'masked_softmax_variate_residual',
            } and hasattr(self, 'fixed_expand_mask'):
                if self.expansion_type in {'masked_softmax', 'masked_softmax_scalar_residual'}:
                    return self._masked_softmax_weight_matrix()
                return self.expand_linear.weight * self.fixed_expand_mask.to(
                    device=self.expand_linear.weight.device,
                    dtype=self.expand_linear.weight.dtype,
                )
            return self.expand_linear.weight
        if hasattr(self, 'reconstruct_linear'):
            return self.reconstruct_linear.weight
        if hasattr(self, 'sparse_expand_logits'):
            return self._sparse_softmax_weight_matrix()
        if hasattr(self, 'slot_expand_linear'):
            return self.slot_expand_linear.weight
        return None

    def _linear_weight_matrices(self):
        matrices = []
        if hasattr(self, 'compress_linear'):
            matrices.append(self.compress_linear.weight)
        decoder = self._decoder_weight_matrix()
        if decoder is not None:
            matrices.append(decoder)
        return matrices

    def _topk_softmax_from_logits(self, logits, topk, dim=-1):
        temperature = max(float(self.expansion_temperature), 1e-6)
        if int(topk) <= 0 or int(topk) >= logits.shape[dim]:
            weight = F.softmax(logits / temperature, dim=dim)
            return weight, None
        values, indices = torch.topk(logits, int(topk), dim=dim)
        weights = F.softmax(values / temperature, dim=dim)
        matrix = logits.new_zeros(logits.shape)
        matrix.scatter_(dim, indices, weights)
        return matrix, indices

    def _selector_soft_matrix(self):
        temperature = max(float(self.expansion_temperature), 1e-6)
        return F.softmax(self.selector_logits / temperature, dim=-1)

    def _selector_hard_matrix(self):
        soft = self._selector_soft_matrix()
        indices = torch.argmax(soft, dim=-1)
        return F.one_hot(indices, num_classes=self.num_variates).to(dtype=soft.dtype)

    def _aux_loss_dict(
        self,
        orthogonal,
        reference_matrix=None,
        wcomp_matrix=None,
        cycle_matrix=None,
        group_attention_weights=None,
        group_attention_mask=None,
    ):
        if reference_matrix is None:
            coverage = orthogonal.new_tensor(0.0)
            entropy = orthogonal.new_tensor(0.0)
        else:
            coverage = self._coverage_loss(reference_matrix)
            entropy = self._assignment_entropy_loss(reference_matrix)
        if wcomp_matrix is None:
            wcomp_entropy = orthogonal.new_tensor(0.0)
        else:
            wcomp_entropy = self._wcomp_entropy_loss(wcomp_matrix)
        if cycle_matrix is None:
            cycle_slot = self._zero_cycle_diagnostics(orthogonal)
        else:
            cycle_slot = self._cycle_slot_loss(cycle_matrix)
        if group_attention_weights is None or group_attention_mask is None:
            group_attention_entropy = orthogonal.new_tensor(0.0)
        else:
            group_attention_entropy = self._group_attention_entropy_loss(
                group_attention_weights,
                group_attention_mask,
            )
        return {
            'orthogonal': orthogonal,
            'coverage': coverage,
            'assignment_entropy': entropy,
            'wcomp_entropy': wcomp_entropy,
            'cycle_slot': cycle_slot,
            'group_attention_entropy': group_attention_entropy,
        }

    def _uses_learned_query_decoder(self):
        return self.expansion_type in {
            'query_decoder',
            'query_decoder_scalar_residual',
            'query_decoder_variate_residual',
        }

    def _uses_id_query_decoder(self):
        return self.expansion_type in {
            'id_query_decoder',
            'id_query_scalar_residual',
            'id_query_fixed_scalar_residual',
        }

    def _uses_id_topk_decoder(self):
        return self.expansion_type in {
            'id_topk_decoder',
            'id_topk_scalar_residual',
            'id_topk_fixed_scalar_residual',
        }

    def _uses_token_query_decoder(self):
        return self.expansion_type == 'token_query_decoder'

    def _uses_attention_decoder(self):
        return self._uses_learned_query_decoder() or self._uses_token_query_decoder()

    def _uses_query_decoder(self):
        return self._uses_attention_decoder() or self._uses_id_query_decoder() or self._uses_id_topk_decoder()

    def _uses_fixed_decoder(self):
        return self.expansion_type in {
            'fixed_assignment',
            'fixed_assignment_scalar_residual',
            'fixed_masked_linear',
            'fixed_masked_scalar_residual',
            'fixed_masked_variate_residual',
        }

    def _uses_sparse_parameter_decoder(self):
        return self.expansion_type in {
            'masked_softmax',
            'masked_softmax_scalar_residual',
            'masked_softmax_variate_residual',
        }

    def _uses_anchor_masked_decoder(self):
        return self.expansion_type in {
            'fixed_assignment',
            'fixed_assignment_scalar_residual',
            'fixed_masked_linear',
            'fixed_masked_scalar_residual',
            'fixed_masked_variate_residual',
            'masked_linear',
            'masked_softmax',
            'masked_softmax_scalar_residual',
            'masked_softmax_variate_residual',
            'masked_linear_scalar_residual',
        }

    def _slot_decoder_needs_assignment_cache(self):
        if self.reduction_type not in {
            'mlp_slot_attention',
            'mlp_sparse_slot_attention',
            'mlp_sparse_slot_attention_id',
            'mlp_linear_sparse_slot_attention_id',
            'mlp_static_sparse_slot_attention_id',
        }:
            return False
        if self.expansion_type == 'slot_learned_linear':
            return False
        if self._uses_query_decoder():
            return False
        return True

    def _should_materialize_slot_assignment(self):
        return (
            self.compute_aux_losses
            or self.force_assignment_cache
            or self.use_cycle_slot_loss
            or self.export_slot_diagnostics
            or self._slot_decoder_needs_assignment_cache()
        )

    def _uses_residual_decoder(self):
        return self.expansion_type in {
            'fixed_assignment_scalar_residual',
            'fixed_masked_scalar_residual',
            'fixed_masked_variate_residual',
            'id_query_scalar_residual',
            'id_query_fixed_scalar_residual',
            'id_topk_scalar_residual',
            'id_topk_fixed_scalar_residual',
            'query_decoder_scalar_residual',
            'query_decoder_variate_residual',
            'masked_linear_scalar_residual',
            'masked_softmax_scalar_residual',
            'masked_softmax_variate_residual',
        }

    def get_residual_gate_stats(self):
        stats = {
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
        }
        if not hasattr(self, 'residual_gate'):
            if hasattr(self, 'hybrid_linear_gate'):
                stats['hybrid_linear_gate'] = torch.tanh(self.hybrid_linear_gate.detach()).item()
            return stats
        gate = torch.tanh(self.residual_gate.detach())
        gate_min = gate.min().item()
        gate_max = gate.max().item()
        stats.update({
            'residual_gate_mean': gate.mean().item(),
            'residual_gate_abs_mean': gate.abs().mean().item(),
            'residual_gate_max': gate.abs().max().item(),
            'residual_gate_min': gate_min,
            'residual_gate_value_max': gate_max,
            'residual_gate_range': gate_max - gate_min,
            'residual_gate_std': gate.std(unbiased=False).item() if gate.numel() > 1 else 0.0,
            'residual_gate_numel': int(gate.numel()),
            'residual_gate_is_variate': int(gate.numel() > 1),
        })
        if hasattr(self, 'hybrid_linear_gate'):
            stats['hybrid_linear_gate'] = torch.tanh(self.hybrid_linear_gate.detach()).item()
        return stats

    def get_masked_expand_stats(self):
        if not hasattr(self, 'fixed_expand_active'):
            return {
                'masked_expand_fan_in_mean': 0.0,
                'masked_expand_fan_in_max': 0,
            }
        fan_in = self.fixed_expand_active.detach().sum(dim=-1)
        return {
            'masked_expand_fan_in_mean': float(fan_in.float().mean().item()),
            'masked_expand_fan_in_max': int(fan_in.max().item()),
        }

    @staticmethod
    def _offdiag_mean(matrix):
        if matrix.shape[-1] <= 1:
            return matrix.new_tensor(0.0)
        mask = ~torch.eye(matrix.shape[-1], dtype=torch.bool, device=matrix.device)
        if matrix.dim() == 3:
            return matrix[:, mask].mean()
        return matrix[mask].mean()

    @staticmethod
    def _pairwise_topk_overlap(assignments, topk):
        if assignments.shape[1] <= 1:
            return assignments.new_tensor(0.0)
        topk = min(int(topk), assignments.shape[-1])
        indices = torch.topk(assignments, topk, dim=-1).indices
        selected = torch.zeros_like(assignments, dtype=torch.bool)
        selected.scatter_(-1, indices, True)
        overlap = torch.einsum('bkc,blc->bkl', selected.float(), selected.float()) / float(topk)
        return VariateReducer._offdiag_mean(overlap)

    @staticmethod
    def _row_normalized_abs(matrix, eps=1e-8):
        matrix = matrix.abs()
        return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(eps)

    def get_attention_wcomp_stats(self, eps=1e-8):
        if self.reduction_type not in {'MLP_attention', 'latent_query_attention', 'grouped_soft_representative'}:
            return {}
        if self.latest_attention_matrix is None:
            return {}
        A = self.latest_attention_matrix.detach().float()
        entropy = -(A * torch.log(A + eps)).sum(dim=-1)
        effective_support = torch.exp(entropy)
        sorted_mass = torch.sort(A, dim=-1, descending=True).values
        top1 = sorted_mass[..., :1].sum(dim=-1)
        top5 = sorted_mass[..., :min(5, sorted_mass.shape[-1])].sum(dim=-1)
        top10 = sorted_mass[..., :min(10, sorted_mass.shape[-1])].sum(dim=-1)
        normalized = F.normalize(A, p=2, dim=-1, eps=eps)
        cosine = torch.bmm(normalized, normalized.transpose(1, 2))
        stats = {
            'avg_comp_entropy': float(entropy.mean().item()),
            'std_comp_entropy': float(entropy.std(unbiased=False).item()),
            'avg_effective_support': float(effective_support.mean().item()),
            'std_effective_support': float(effective_support.std(unbiased=False).item()),
            'avg_top1_mass': float(top1.mean().item()),
            'avg_top5_mass': float(top5.mean().item()),
            'avg_top10_mass': float(top10.mean().item()),
            'avg_pairwise_overlap_top5': float(self._pairwise_topk_overlap(A, 5).item()),
            'avg_pairwise_overlap_top10': float(self._pairwise_topk_overlap(A, 10).item()),
            'avg_pairwise_cosine_between_latents': float(self._offdiag_mean(cosine).item()),
            'w_comp_density': float((A.abs() > eps).float().mean().item()),
        }
        if hasattr(self, 'slot_expand_linear'):
            W_exp = self.slot_expand_linear.weight.detach().float()
            stats['w_exp_density'] = float((W_exp.abs() > eps).float().mean().item())
            W_eff = torch.einsum('ck,bkv->bcv', W_exp, A)
            stats['w_eff_density'] = float((W_eff.abs() > eps).float().mean().item())
            W_eff_prob = self._row_normalized_abs(W_eff, eps=eps)
            W_eff_entropy = -(W_eff_prob * torch.log(W_eff_prob + eps)).sum(dim=-1)
            W_eff_sorted = torch.sort(W_eff_prob, dim=-1, descending=True).values
            stats['avg_w_eff_entropy'] = float(W_eff_entropy.mean().item())
            stats['avg_w_eff_top5_mass'] = float(
                W_eff_sorted[..., :min(5, W_eff_sorted.shape[-1])].sum(dim=-1).mean().item()
            )
        else:
            stats.update({
                'w_exp_density': 0.0,
                'w_eff_density': 0.0,
                'avg_w_eff_entropy': 0.0,
                'avg_w_eff_top5_mass': 0.0,
            })
        return stats

    def get_cycle_slot_stats(self):
        defaults = {
            'cycle_topr_jaccard': 0.0,
            'cycle_a_effective_support': 0.0,
            'cycle_b_effective_support': 0.0,
            'cycle_slot_overlap': 0.0,
            'cycle_topr_size': 0,
            'w_eff_density': 0.0,
        }
        if not self.latest_cycle_diagnostics:
            return defaults
        defaults.update(self.latest_cycle_diagnostics)
        return defaults

    def get_slot_diagnostic_matrices(self):
        if not self.latest_cycle_matrices:
            return {}
        return {
            name: matrix.detach().float()
            for name, matrix in self.latest_cycle_matrices.items()
        }

    def _apply_source_id_descriptor(self, compressed_tokens, assignment):
        if self.reduction_type not in {
            'mlp_sparse_slot_attention_id',
            'mlp_linear_sparse_slot_attention_id',
            'mlp_static_sparse_slot_attention_id',
        }:
            return compressed_tokens
        source_id = self.source_id_embedding.to(
            device=compressed_tokens.device,
            dtype=compressed_tokens.dtype,
        )
        id_tokens = torch.matmul(assignment.to(compressed_tokens.dtype), source_id)
        return compressed_tokens + torch.tanh(self.source_id_gate).to(compressed_tokens.dtype) * id_tokens

    def _apply_source_id_descriptor_from_topk(self, compressed_tokens, indices, weights):
        if self.reduction_type not in {
            'mlp_sparse_slot_attention_id',
            'mlp_linear_sparse_slot_attention_id',
            'mlp_static_sparse_slot_attention_id',
        }:
            return compressed_tokens
        source_id = self.source_id_embedding.to(
            device=compressed_tokens.device,
            dtype=compressed_tokens.dtype,
        )
        selected_id = source_id.index_select(0, indices.reshape(-1)).view(
            indices.shape[0],
            indices.shape[1],
            indices.shape[2],
            self.d_model,
        )
        id_tokens = torch.sum(selected_id * weights.to(compressed_tokens.dtype).unsqueeze(-1), dim=2)
        return compressed_tokens + torch.tanh(self.source_id_gate).to(compressed_tokens.dtype) * id_tokens

    def _apply_group_source_id_descriptor(self, compressed_tokens, source_weights):
        if self.reduction_type not in {
            'variate_anchor_grouplinear_residual_pool_id',
            'variate_grouped_average_id',
            'variate_grouped_softmax_pool_id',
            'variate_grouped_linear_pool_id',
        }:
            return compressed_tokens
        group_ids = self.group_ids.to(source_weights.device)
        source_id = self.source_id_embedding.to(device=compressed_tokens.device, dtype=compressed_tokens.dtype)
        id_tokens = compressed_tokens.new_zeros(self.reduced_k, self.d_model)
        id_tokens.scatter_add_(
            0,
            group_ids.view(-1, 1).expand(-1, self.d_model),
            source_id * source_weights.to(compressed_tokens.dtype).view(-1, 1),
        )
        return compressed_tokens + torch.tanh(self.source_id_gate).to(compressed_tokens.dtype) * id_tokens.unsqueeze(0)

    def _static_slot_scores(self, var_tokens):
        queries = self.static_slot_queries.to(device=var_tokens.device, dtype=var_tokens.dtype)
        return torch.einsum('bvd,kd->bkv', var_tokens, queries) / math.sqrt(float(self.d_model))

    def _init_id_topk_decoder(self):
        temperature = max(float(self.expansion_temperature), 1e-6)
        target = self.target_id_embedding
        latent = self.latent_id_embedding
        logits = torch.matmul(target, latent.transpose(0, 1)) / math.sqrt(float(self.d_model))
        topk = int(self.expansion_topk)
        if topk > 0 and topk < self.reduced_k:
            values, indices = torch.topk(logits, topk, dim=-1)
            anchor_map = self._load_anchor_map()
            if anchor_map is not None and 'group_ids' in anchor_map:
                group_ids, _ = self._validate_group_ids_for_length(
                    anchor_map['group_ids'],
                    self.target_num_variates,
                    'id top-k decoder anchor map',
                )
                primary = group_ids.view(-1, 1)
                if topk == 1:
                    indices = primary
                else:
                    has_primary = torch.any(indices == primary, dim=-1, keepdim=True)
                    indices = torch.where(has_primary, indices, indices.clone())
                    indices[:, -1:] = torch.where(has_primary, indices[:, -1:], primary)
                values = torch.gather(logits, dim=-1, index=indices)
            weights = F.softmax(values / temperature, dim=-1)
            self.register_buffer('id_topk_indices', indices.long(), persistent=False)
            self.register_buffer('id_topk_weights', weights.float(), persistent=False)
        else:
            weights = F.softmax(logits / temperature, dim=-1)
            self.register_buffer('id_topk_dense_weights', weights.float(), persistent=False)

    def get_weight_matrices(self):
        matrices = {}
        if self.reduction_type in {
            'MLP_attention',
            'latent_query_attention',
            'grouped_soft_representative',
        } and self.latest_attention_matrix is not None:
            compress = self.latest_attention_matrix.detach().float().mean(dim=0)
            matrices['compress_weight'] = compress
            if hasattr(self, 'slot_expand_linear'):
                expand = self.slot_expand_linear.weight.detach().float()
                matrices['expand_weight'] = expand
                matrices['effective_weight'] = torch.matmul(expand, compress)
        if hasattr(self, 'compression_logits'):
            matrices['static_assignment'] = F.softmax(self.compression_logits.detach(), dim=-1)
        if hasattr(self, 'compress_linear'):
            matrices['compress_weight'] = self.compress_linear.weight.detach()
        if hasattr(self, 'selector_logits'):
            matrices['compress_weight'] = self._selector_hard_matrix().detach()
            matrices['selector_soft_weight'] = self._selector_soft_matrix().detach()
        if hasattr(self, 'expand_linear'):
            if self.expansion_type in {
                'masked_linear',
                'masked_softmax',
                'masked_linear_scalar_residual',
                'masked_softmax_scalar_residual',
                'masked_softmax_variate_residual',
            } and hasattr(self, 'fixed_expand_mask'):
                matrices['expand_weight'] = (self.expand_linear.weight * self.fixed_expand_mask).detach()
                if self.expansion_type in {'masked_softmax', 'masked_softmax_scalar_residual'}:
                    matrices['expand_weight'] = self._masked_softmax_weight_matrix().detach()
            else:
                matrices['expand_weight'] = self.expand_linear.weight.detach()
        if hasattr(self, 'sparse_expand_logits'):
            matrices['expand_weight'] = self._sparse_softmax_weight_matrix().detach()
        if hasattr(self, 'fixed_expand_mask'):
            matrices['expand_mask'] = self.fixed_expand_mask.detach()
        if hasattr(self, 'fixed_expand_init_weight'):
            matrices['expand_init_weight'] = self.fixed_expand_init_weight.detach()
        if (
            hasattr(self, 'fixed_expand_weight')
            and not hasattr(self, 'expand_linear')
            and not hasattr(self, 'sparse_expand_logits')
        ):
            if (
                self.expansion_type in {
                    'fixed_masked_linear',
                    'fixed_masked_scalar_residual',
                    'fixed_masked_variate_residual',
                }
                and hasattr(self, 'fixed_expand_init_weight')
            ):
                matrices['expand_weight'] = self.fixed_expand_init_weight.detach()
            else:
                matrices['expand_weight'] = self.fixed_expand_weight.detach()
        if hasattr(self, 'reconstruct_linear'):
            matrices['reconstruct_weight'] = self.reconstruct_linear.weight.detach()
        if hasattr(self, 'slot_expand_linear'):
            matrices['slot_expand_weight'] = self.slot_expand_linear.weight.detach()
        if hasattr(self, 'id_topk_dense_weights'):
            matrices['id_topk_weight'] = self.id_topk_dense_weights.detach()
        elif hasattr(self, 'id_topk_indices') and hasattr(self, 'id_topk_weights'):
            weight = self.id_topk_weights.new_zeros(self.target_num_variates, self.reduced_k)
            weight.scatter_(1, self.id_topk_indices, self.id_topk_weights)
            matrices['id_topk_weight'] = weight.detach()
        if hasattr(self, 'fixed_compress_weight'):
            if self.reduction_type == 'variate_anchor_softmax_pool' and hasattr(self, 'group_pool_logits'):
                matrices['compress_weight'] = self._group_softmax_compress_weight().detach()
                matrices['anchor_weight'] = self.fixed_compress_weight.detach()
                matrices['group_mean_weight'] = self.fixed_group_mean_weight.detach()
            elif (
                self.reduction_type in {'variate_grouped_softmax_pool', 'variate_grouped_softmax_pool_id'}
                and hasattr(self, 'group_pool_logits')
            ):
                matrices['compress_weight'] = self._group_softmax_compress_weight().detach()
                matrices['group_init_weight'] = self.fixed_compress_weight.detach()
                if self.reduction_type == 'variate_grouped_softmax_pool_id':
                    matrices['source_id_assignment'] = matrices['compress_weight']
            elif self.reduction_type == 'variate_group_adaptive_selection':
                matrices['compress_weight'] = self.fixed_compress_weight.detach()
                matrices['anchor_weight'] = self.fixed_compress_weight.detach()
            elif self.reduction_type == 'variate_group_adaptive_id_selection':
                matrices['compress_weight'] = self.fixed_compress_weight.detach()
                matrices['anchor_weight'] = self.fixed_compress_weight.detach()
                matrices['selected_variate_embedding'] = self.selected_variate_embedding.detach()
            elif self.reduction_type == 'variate_grouped_average_id':
                matrices['compress_weight'] = self.fixed_compress_weight.detach()
                matrices['group_init_weight'] = self.fixed_compress_weight.detach()
                matrices['source_id_assignment'] = self.fixed_compress_weight.detach()
            elif self.reduction_type in {'variate_grouped_linear_pool', 'variate_grouped_linear_pool_id'} and hasattr(self, 'group_pool_weight'):
                matrices['compress_weight'] = self._group_linear_compress_weight().detach()
                matrices['group_init_weight'] = self.fixed_compress_weight.detach()
                if self.reduction_type == 'variate_grouped_linear_pool_id':
                    matrices['source_id_assignment'] = matrices['compress_weight']
            elif self.reduction_type == 'grouped_soft_representative':
                if self.latest_attention_matrix is not None:
                    matrices['compress_weight'] = self.latest_attention_matrix.detach().float().mean(dim=0)
                matrices['group_init_weight'] = self.fixed_compress_weight.detach()
            elif self.reduction_type == 'variate_anchor_residual_pool' and hasattr(self, 'anchor_residual_gate'):
                gate = torch.tanh(self.anchor_residual_gate.detach())
                matrices['compress_weight'] = (
                    self.fixed_compress_weight + gate * (self.fixed_group_mean_weight - self.fixed_compress_weight)
                ).detach()
                matrices['anchor_weight'] = self.fixed_compress_weight.detach()
                matrices['group_mean_weight'] = self.fixed_group_mean_weight.detach()
            elif (
                self.reduction_type in {
                    'variate_anchor_grouplinear_residual_pool',
                    'variate_anchor_grouplinear_residual_pool_id',
                }
                and hasattr(self, 'anchor_residual_gate')
                and hasattr(self, 'group_pool_weight')
            ):
                gate = torch.tanh(self.anchor_residual_gate.detach())
                group_linear_weight = self._group_linear_compress_weight()
                matrices['compress_weight'] = (
                    self.fixed_compress_weight + gate * (group_linear_weight - self.fixed_compress_weight)
                ).detach()
                matrices['anchor_weight'] = self.fixed_compress_weight.detach()
                matrices['group_linear_weight'] = group_linear_weight.detach()
                matrices['group_mean_weight'] = self.fixed_group_mean_weight.detach()
                if self.reduction_type == 'variate_anchor_grouplinear_residual_pool_id':
                    matrices['source_id_assignment'] = group_linear_weight.detach()
            elif (
                self.reduction_type == 'variate_anchor_channel_residual_pool'
                and hasattr(self, 'anchor_channel_residual_gate')
            ):
                gate = torch.tanh(self.anchor_channel_residual_gate.detach()).mean(dim=-1, keepdim=True)
                matrices['compress_weight'] = (
                    self.fixed_compress_weight + gate * (self.fixed_group_mean_weight - self.fixed_compress_weight)
                ).detach()
                matrices['anchor_weight'] = self.fixed_compress_weight.detach()
                matrices['group_mean_weight'] = self.fixed_group_mean_weight.detach()
                matrices['channel_residual_gate'] = torch.tanh(self.anchor_channel_residual_gate.detach())
            else:
                matrices['compress_weight'] = self.fixed_compress_weight.detach()
        if 'compress_weight' in matrices and 'expand_weight' in matrices and 'effective_weight' not in matrices:
            compress = matrices['compress_weight'].detach().float()
            expand = matrices['expand_weight'].detach().float()
            if compress.dim() == 2 and expand.dim() == 2 and expand.shape[1] == compress.shape[0]:
                matrices['effective_weight'] = torch.matmul(expand, compress)
        return matrices

    def _normalize_expansion(self, expansion, eps=1e-8):
        if self.expansion_type == 'transpose':
            return expansion
        if self.expansion_type == 'sharpened_column_normalized':
            expansion = (expansion + eps).pow(1.0 / self.expansion_temperature)
        elif self.expansion_type == 'topk_column_normalized' and self.expansion_temperature != 1.0:
            expansion = (expansion + eps).pow(1.0 / self.expansion_temperature)
        return expansion / (expansion.sum(dim=-1, keepdim=True) + eps)

    def _apply_expansion(self, expansion, encoded_latent_tokens):
        if self.expansion_type != 'topk_column_normalized':
            if expansion.dim() == 2:
                return torch.einsum('nk,bkd->bnd', expansion, encoded_latent_tokens)
            return torch.bmm(expansion, encoded_latent_tokens)

        topk = min(self.expansion_topk, expansion.shape[-1])
        if topk >= expansion.shape[-1]:
            if expansion.dim() == 2:
                return torch.einsum('nk,bkd->bnd', expansion, encoded_latent_tokens)
            return torch.bmm(expansion, encoded_latent_tokens)

        if expansion.dim() == 2:
            values, indices = torch.topk(expansion, topk, dim=-1)
            values = values / (values.sum(dim=-1, keepdim=True) + 1e-8)
            batch_size, _, d_model = encoded_latent_tokens.shape
            num_variates = expansion.shape[0]
            gather_index = indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, num_variates, topk, d_model)
            latent = encoded_latent_tokens.unsqueeze(1).expand(batch_size, num_variates, -1, d_model)
            selected = torch.gather(latent, dim=2, index=gather_index)
            return torch.sum(selected * values.unsqueeze(0).unsqueeze(-1), dim=2)

        values, indices = torch.topk(expansion, topk, dim=-1)
        values = values / (values.sum(dim=-1, keepdim=True) + 1e-8)
        batch_size, num_variates, _ = expansion.shape
        d_model = encoded_latent_tokens.shape[-1]
        gather_index = indices.unsqueeze(-1).expand(batch_size, num_variates, topk, d_model)
        latent = encoded_latent_tokens.unsqueeze(1).expand(batch_size, num_variates, -1, d_model)
        selected = torch.gather(latent, dim=2, index=gather_index)
        return torch.sum(selected * values.unsqueeze(-1), dim=2)

    def compress(self, var_tokens, selection_scores=None):
        if self.reduction_type == 'none':
            aux_loss = self._zero_like_tokens(var_tokens)
            self.latest_aux_losses = self._aux_loss_dict(aux_loss)
            self.latest_aux_loss = aux_loss
            return var_tokens, None, aux_loss

        if self.reduction_type == 'mlp_static_combination':
            A = F.softmax(self.compression_logits, dim=-1)
            compressed_tokens = torch.einsum('kn,bnd->bkd', A, var_tokens)
            if self.compute_aux_losses:
                aux_loss = self._orthogonal_loss(A)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, A)
            else:
                aux_loss = self._set_zero_aux_losses(var_tokens)
            cache = {'A': A}
        elif self.reduction_type == 'learned_anchor_selection':
            selector_soft = self._selector_soft_matrix().to(dtype=var_tokens.dtype)
            selector_hard = self._selector_hard_matrix().to(dtype=var_tokens.dtype)
            if self.training:
                selector = selector_hard + selector_soft - selector_soft.detach()
                compressed_tokens = torch.einsum('kv,bvd->bkd', selector, var_tokens)
            else:
                anchor_indices = torch.argmax(selector_hard, dim=-1).to(var_tokens.device)
                compressed_tokens = var_tokens.index_select(1, anchor_indices)
                selector = selector_hard
            if self.compute_aux_losses:
                aux_loss = self._orthogonal_loss(selector_soft)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, selector_soft)
            else:
                aux_loss = self._set_zero_aux_losses(var_tokens)
            cache = {'selector': selector, 'selector_soft': selector_soft}
        elif self.reduction_type == 'sample_adaptive_selection':
            if selection_scores is None:
                selection_scores = var_tokens.detach().pow(2).mean(dim=-1)
            selection_scores = selection_scores[:, :self.num_variates]
            if selection_scores.shape != (var_tokens.shape[0], self.num_variates):
                raise ValueError(
                    'selection_scores shape {} does not match ({}, {})'.format(
                        tuple(selection_scores.shape),
                        var_tokens.shape[0],
                        self.num_variates,
                    )
                )
            selected_indices = torch.topk(selection_scores, self.reduced_k, dim=1, largest=True).indices
            gather_index = selected_indices.unsqueeze(-1).expand(-1, -1, var_tokens.shape[-1])
            compressed_tokens = torch.gather(var_tokens, dim=1, index=gather_index)
            self.latest_selected_indices = selected_indices.detach()
            aux_loss = var_tokens.new_tensor(0.0)
            self.latest_aux_losses = self._aux_loss_dict(aux_loss)
            cache = {'selected_indices': selected_indices}
        elif self.reduction_type in {'variate_group_adaptive_selection', 'variate_group_adaptive_id_selection'}:
            if selection_scores is None:
                selection_scores = var_tokens.detach().pow(2).mean(dim=-1)
            selection_scores = selection_scores[:, :self.num_variates]
            if selection_scores.shape != (var_tokens.shape[0], self.num_variates):
                raise ValueError(
                    'selection_scores shape {} does not match ({}, {})'.format(
                        tuple(selection_scores.shape),
                        var_tokens.shape[0],
                        self.num_variates,
                    )
                )
            member_indices = self.group_member_indices.to(selection_scores.device)
            member_mask = self.group_member_mask.to(selection_scores.device)
            batch_size = selection_scores.shape[0]
            flat_member_indices = member_indices.reshape(-1)
            member_scores = selection_scores.index_select(1, flat_member_indices)
            member_scores = member_scores.view(batch_size, self.reduced_k, member_indices.shape[1])
            member_scores = member_scores.masked_fill(~member_mask.view(1, self.reduced_k, -1), float('-inf'))
            best_member_pos = torch.argmax(member_scores, dim=2)
            selected_indices = member_indices.unsqueeze(0).expand(batch_size, -1, -1).gather(
                2,
                best_member_pos.unsqueeze(-1),
            ).squeeze(-1)
            gather_index = selected_indices.unsqueeze(-1).expand(-1, -1, var_tokens.shape[-1])
            compressed_tokens = torch.gather(var_tokens, dim=1, index=gather_index)
            if self.reduction_type == 'variate_group_adaptive_id_selection':
                id_embedding = self.selected_variate_embedding.index_select(
                    0,
                    selected_indices.reshape(-1),
                ).view(selected_indices.shape[0], selected_indices.shape[1], self.d_model)
                compressed_tokens = compressed_tokens + id_embedding.to(compressed_tokens.dtype)
            self.latest_selected_indices = selected_indices.detach()
            aux_loss = var_tokens.new_tensor(0.0)
            self.latest_aux_losses = self._aux_loss_dict(aux_loss)
            cache = {'selected_indices': selected_indices}
        elif self.reduction_type == 'variate_anchor_selection':
            compressed_tokens = var_tokens.index_select(1, self.anchor_indices)
            aux_loss = var_tokens.new_tensor(0.0)
            self.latest_aux_losses = self._aux_loss_dict(aux_loss)
            cache = None
        elif self.reduction_type == 'variate_anchor_residual_pool':
            batch_size, _, d_model = var_tokens.shape
            anchor_tokens = var_tokens.index_select(1, self.anchor_indices)
            group_index = self.group_ids.view(1, -1, 1).expand(batch_size, -1, d_model)
            group_mean_tokens = var_tokens.new_zeros(batch_size, self.reduced_k, d_model)
            group_mean_tokens.scatter_add_(1, group_index, var_tokens)
            group_mean_tokens = group_mean_tokens / self.group_counts.view(1, -1, 1).to(var_tokens.dtype)
            gate = torch.tanh(self.anchor_residual_gate)
            compressed_tokens = anchor_tokens + gate * (group_mean_tokens - anchor_tokens)
            aux_loss = var_tokens.new_tensor(0.0)
            self.latest_aux_losses = self._aux_loss_dict(aux_loss)
            cache = None
        elif self.reduction_type == 'variate_anchor_channel_residual_pool':
            batch_size, _, d_model = var_tokens.shape
            anchor_tokens = var_tokens.index_select(1, self.anchor_indices)
            group_index = self.group_ids.view(1, -1, 1).expand(batch_size, -1, d_model)
            group_mean_tokens = var_tokens.new_zeros(batch_size, self.reduced_k, d_model)
            group_mean_tokens.scatter_add_(1, group_index, var_tokens)
            group_mean_tokens = group_mean_tokens / self.group_counts.view(1, -1, 1).to(var_tokens.dtype)
            gate = torch.tanh(self.anchor_channel_residual_gate).to(var_tokens.dtype).unsqueeze(0)
            compressed_tokens = anchor_tokens + gate * (group_mean_tokens - anchor_tokens)
            aux_loss = var_tokens.new_tensor(0.0)
            self.latest_aux_losses = self._aux_loss_dict(aux_loss)
            cache = None
        elif self.reduction_type in {
            'variate_anchor_grouplinear_residual_pool',
            'variate_anchor_grouplinear_residual_pool_id',
        }:
            batch_size, _, d_model = var_tokens.shape
            anchor_tokens = var_tokens.index_select(1, self.anchor_indices)
            group_index = self.group_ids.view(1, -1, 1).expand(batch_size, -1, d_model)
            source_weights = self._group_linear_source_weights().to(var_tokens.dtype)
            group_linear_tokens = var_tokens.new_zeros(batch_size, self.reduced_k, d_model)
            group_linear_tokens.scatter_add_(1, group_index, var_tokens * source_weights.view(1, -1, 1))
            gate = torch.tanh(self.anchor_residual_gate).to(var_tokens.dtype)
            compressed_tokens = anchor_tokens + gate * (group_linear_tokens - anchor_tokens)
            compressed_tokens = self._apply_group_source_id_descriptor(compressed_tokens, source_weights)
            aux_loss = var_tokens.new_tensor(0.0)
            self.latest_aux_losses = self._aux_loss_dict(aux_loss)
            cache = None
        elif self.reduction_type in {
            'variate_anchor_softmax_pool',
            'variate_grouped_softmax_pool',
            'variate_grouped_softmax_pool_id',
        }:
            batch_size, _, d_model = var_tokens.shape
            source_weights = self._group_softmax_weights().to(var_tokens.dtype)
            group_index = self.group_ids.view(1, -1, 1).expand(batch_size, -1, d_model)
            compressed_tokens = var_tokens.new_zeros(batch_size, self.reduced_k, d_model)
            compressed_tokens.scatter_add_(1, group_index, var_tokens * source_weights.view(1, -1, 1))
            compressed_tokens = self._apply_group_source_id_descriptor(compressed_tokens, source_weights)
            aux_loss = var_tokens.new_tensor(0.0)
            self.latest_aux_losses = self._aux_loss_dict(aux_loss)
            cache = None
        elif self.reduction_type in {
            'variate_grouped_average',
            'variate_grouped_average_id',
            'variate_grouped_corr_pool',
            'variate_grouped_linear_pool',
            'variate_grouped_linear_pool_id',
            'grouped_soft_representative',
        }:
            batch_size, _, d_model = var_tokens.shape
            group_index = self.group_ids.view(1, -1, 1).expand(batch_size, -1, d_model)
            compressed_tokens = var_tokens.new_zeros(batch_size, self.reduced_k, d_model)
            if self.reduction_type == 'variate_grouped_corr_pool':
                source_weights = self.fixed_compress_weight.sum(dim=0).to(var_tokens.dtype)
                compressed_tokens.scatter_add_(1, group_index, var_tokens * source_weights.view(1, -1, 1))
            elif self.reduction_type in {'variate_grouped_linear_pool', 'variate_grouped_linear_pool_id'}:
                source_weights = self._group_linear_source_weights().to(var_tokens.dtype)
                compressed_tokens.scatter_add_(1, group_index, var_tokens * source_weights.view(1, -1, 1))
                compressed_tokens = self._apply_group_source_id_descriptor(compressed_tokens, source_weights)
            elif self.reduction_type == 'grouped_soft_representative':
                member_indices = self.group_member_indices.to(var_tokens.device)
                member_mask = self.group_member_mask.to(var_tokens.device)
                flat_indices = member_indices.reshape(-1)
                member_tokens = var_tokens.index_select(1, flat_indices).view(
                    batch_size,
                    self.reduced_k,
                    member_indices.shape[1],
                    d_model,
                )
                content_scores = self.group_rep_score_mlp(member_tokens).squeeze(-1)
                source_bias = self.group_rep_source_logits.to(device=var_tokens.device, dtype=var_tokens.dtype)
                source_bias = source_bias.index_select(0, flat_indices).view(1, self.reduced_k, member_indices.shape[1])
                group_scores = content_scores + source_bias
                group_weights = self._normalize_group_member_scores(group_scores, member_mask)
                compressed_tokens = torch.sum(member_tokens * group_weights.unsqueeze(-1), dim=2)
                A = self._materialize_group_assignment(group_weights, member_indices, member_mask)
                self.latest_attention_matrix = A.detach()
                if self.compute_aux_losses:
                    aux_loss = self._orthogonal_loss(A)
                    self.latest_aux_losses = self._aux_loss_dict(
                        aux_loss,
                        A,
                        A,
                        group_attention_weights=group_weights,
                        group_attention_mask=member_mask,
                    )
                else:
                    aux_loss = self._set_zero_aux_losses(var_tokens)
                    self.latest_aux_losses = self._aux_loss_dict(
                        aux_loss,
                        wcomp_matrix=A,
                        group_attention_weights=group_weights,
                        group_attention_mask=member_mask,
                    )
                cache = {'A': A, 'group_weights': group_weights}
                self.latest_aux_loss = aux_loss
                return compressed_tokens, cache, aux_loss
            else:
                compressed_tokens.scatter_add_(1, group_index, var_tokens)
                compressed_tokens = compressed_tokens / self.group_counts.view(1, -1, 1).to(var_tokens.dtype)
                if self.reduction_type == 'variate_grouped_average_id':
                    source_weights = self.fixed_compress_weight.sum(dim=0).to(var_tokens.dtype)
                    compressed_tokens = self._apply_group_source_id_descriptor(compressed_tokens, source_weights)
            aux_loss = var_tokens.new_tensor(0.0)
            self.latest_aux_losses = self._aux_loss_dict(aux_loss)
            cache = None
        elif self.reduction_type == 'MLP_attention':
            scores = self.score_mlp(var_tokens).transpose(1, 2)
            A = self._normalize_wcomp_scores(scores)
            compressed_tokens = torch.bmm(A, var_tokens)
            self.latest_attention_matrix = A.detach()
            if self.compute_aux_losses:
                aux_loss = self._orthogonal_loss(A)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, A, A)
            else:
                aux_loss = self._set_zero_aux_losses(var_tokens)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, wcomp_matrix=A)
            cache = {'A': A}
        elif self.reduction_type == 'latent_query_attention':
            batch_size = var_tokens.shape[0]
            queries = self.latent_queries.to(device=var_tokens.device, dtype=var_tokens.dtype)
            queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
            keys = self.key_proj(var_tokens)
            values = self.value_proj(var_tokens)
            scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(float(self.d_model))
            A = self._normalize_wcomp_scores(scores)
            compressed_tokens = torch.bmm(A, values)
            self.latest_attention_matrix = A.detach()
            if self.compute_aux_losses:
                aux_loss = self._orthogonal_loss(A)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, A, A)
            else:
                aux_loss = self._set_zero_aux_losses(var_tokens)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, wcomp_matrix=A)
            cache = {'A': A}
        elif self.reduction_type == 'mlp_slot_attention':
            scores = self.score_mlp(var_tokens).transpose(1, 2)
            A = F.softmax(scores, dim=-1)
            compressed_tokens = torch.bmm(A, var_tokens)
            self.latest_attention_matrix = A.detach()
            if self.compute_aux_losses:
                aux_loss = self._orthogonal_loss(A)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, A, cycle_matrix=A)
            else:
                aux_loss = var_tokens.new_tensor(0.0)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, cycle_matrix=A)
            cache = {'A': A}
        elif self.reduction_type in {
            'mlp_sparse_slot_attention',
            'mlp_sparse_slot_attention_id',
            'mlp_linear_sparse_slot_attention_id',
            'mlp_static_sparse_slot_attention_id',
        }:
            if self.reduction_type == 'mlp_static_sparse_slot_attention_id':
                scores = self._static_slot_scores(var_tokens)
            else:
                scores = self.score_mlp(var_tokens).transpose(1, 2)
            temperature = max(float(self.expansion_temperature), 1e-6)
            topk = int(self.expansion_topk)
            materialize_assignment = self._should_materialize_slot_assignment()
            if topk <= 0 or topk >= scores.shape[-1]:
                A = F.softmax(scores / temperature, dim=-1)
                compressed_tokens = torch.bmm(A, var_tokens)
                compressed_tokens = self._apply_source_id_descriptor(compressed_tokens, A)
            else:
                topk = min(topk, scores.shape[-1])
                values, indices = torch.topk(scores, topk, dim=-1)
                weights = F.softmax(values / temperature, dim=-1)
                batch_size, _, d_model = var_tokens.shape
                gather_index = indices.unsqueeze(-1).expand(batch_size, self.reduced_k, topk, d_model)
                source = var_tokens.unsqueeze(1).expand(batch_size, self.reduced_k, -1, d_model)
                selected = torch.gather(source, dim=2, index=gather_index)
                compressed_tokens = torch.sum(selected * weights.unsqueeze(-1), dim=2)
                A = None
                if materialize_assignment:
                    A = scores.new_zeros(scores.shape)
                    A.scatter_(-1, indices, weights)
                compressed_tokens = self._apply_source_id_descriptor_from_topk(
                    compressed_tokens,
                    indices,
                    weights,
                )
            self.latest_attention_matrix = A.detach() if A is not None else None
            if self.compute_aux_losses:
                aux_loss = self._orthogonal_loss(A)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, A, cycle_matrix=A)
            else:
                aux_loss = var_tokens.new_tensor(0.0)
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, cycle_matrix=A)
            cache = {'A': A} if A is not None else None
        elif self.reduction_type == 'mlp_slot_attention_hybrid_linear':
            scores = self.score_mlp(var_tokens).transpose(1, 2)
            A = F.softmax(scores, dim=-1)
            slot_tokens = torch.bmm(A, var_tokens)
            linear_tokens = self.compress_linear(var_tokens.transpose(1, 2)).transpose(1, 2)
            compressed_tokens = slot_tokens + torch.tanh(self.hybrid_linear_gate) * linear_tokens
            if self.compute_aux_losses:
                aux_loss = 0.5 * (self._orthogonal_loss(A) + self._orthogonal_loss(self.compress_linear.weight))
                self.latest_aux_losses = self._aux_loss_dict(aux_loss, A)
            else:
                aux_loss = self._set_zero_aux_losses(var_tokens)
            cache = {'A': A}
        else:
            raise ValueError('Unknown variate reduction type: {}'.format(self.reduction_type))

        self.latest_aux_loss = aux_loss
        return compressed_tokens, cache, aux_loss

    def _id_query_decode(self, latent_tokens, original_variates=None, use_residual=True):
        query = self.target_id_embedding.to(device=latent_tokens.device, dtype=latent_tokens.dtype)
        logits = torch.einsum('vd,bkd->bvk', query, latent_tokens)
        logits = logits / math.sqrt(float(latent_tokens.shape[-1]))
        weights = F.softmax(logits, dim=-1)
        decoded = torch.bmm(weights, latent_tokens)
        if self.expansion_type in {'id_query_scalar_residual', 'id_query_fixed_scalar_residual'} and use_residual:
            if original_variates is None:
                raise ValueError('original_variates must be provided when id query residual expansion is enabled')
            decoded = decoded + torch.tanh(self.residual_gate).to(decoded.dtype) * original_variates
        return decoded

    def _id_topk_decode(self, latent_tokens, original_variates=None, use_residual=True):
        if hasattr(self, 'id_topk_indices'):
            indices = self.id_topk_indices.to(device=latent_tokens.device)
            weights = self.id_topk_weights.to(device=latent_tokens.device, dtype=latent_tokens.dtype)
            batch_size, _, d_model = latent_tokens.shape
            selected = latent_tokens.index_select(1, indices.reshape(-1))
            selected = selected.view(batch_size, self.target_num_variates, indices.shape[-1], d_model)
            decoded = torch.sum(selected * weights.unsqueeze(0).unsqueeze(-1), dim=2)
        else:
            weights = self.id_topk_dense_weights.to(device=latent_tokens.device, dtype=latent_tokens.dtype)
            decoded = torch.einsum('vk,bkd->bvd', weights, latent_tokens)
        if self.expansion_type in {'id_topk_scalar_residual', 'id_topk_fixed_scalar_residual'} and use_residual:
            if original_variates is None:
                raise ValueError('original_variates must be provided when id top-k residual expansion is enabled')
            decoded = decoded + torch.tanh(self.residual_gate).to(decoded.dtype) * original_variates
        return decoded

    def _query_decode(self, latent_tokens, original_variates=None, use_residual=True):
        batch_size = latent_tokens.shape[0]
        if self._uses_token_query_decoder():
            if original_variates is None:
                raise ValueError('original_variates must be provided when token_query_decoder is enabled')
            query = original_variates
        else:
            query = self.variate_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded, _ = self.query_decoder_attn(
            query=query,
            key=latent_tokens,
            value=latent_tokens,
            need_weights=False,
        )
        if self._uses_residual_decoder() and use_residual:
            if original_variates is None:
                raise ValueError('original_variates must be provided when residual query decoder is enabled')
            decoded = decoded + torch.tanh(self.residual_gate) * original_variates
        return decoded

    def _apply_linear_residual(self, decoded, original_variates=None, use_residual=True):
        if self.expansion_type not in {
            'fixed_assignment_scalar_residual',
            'fixed_masked_scalar_residual',
            'fixed_masked_variate_residual',
            'masked_linear_scalar_residual',
            'masked_softmax_scalar_residual',
            'masked_softmax_variate_residual',
        } or not use_residual:
            return decoded
        if original_variates is None:
            raise ValueError('original_variates must be provided when linear residual expansion is enabled')
        return decoded + torch.tanh(self.residual_gate) * original_variates

    def _apply_masked_linear_expansion(self, latent_tokens):
        indices = self.fixed_expand_indices.to(latent_tokens.device)
        active = self.fixed_expand_active.to(device=latent_tokens.device, dtype=latent_tokens.dtype)
        selected_weight = torch.gather(
            self.expand_linear.weight,
            dim=1,
            index=indices,
        ).to(latent_tokens.dtype)
        selected_latents = latent_tokens[:, indices, :]
        return torch.sum(
            selected_latents * selected_weight.unsqueeze(0).unsqueeze(-1) * active.unsqueeze(0).unsqueeze(-1),
            dim=2,
        )

    def _apply_fixed_masked_linear_expansion(self, latent_tokens):
        indices = self.fixed_expand_indices.to(latent_tokens.device)
        active = self.fixed_expand_active.to(device=latent_tokens.device, dtype=latent_tokens.dtype)
        fixed_weight = self.fixed_expand_init_weight.to(device=latent_tokens.device, dtype=latent_tokens.dtype)
        selected_weight = torch.gather(
            fixed_weight,
            dim=1,
            index=indices,
        )
        selected_latents = latent_tokens[:, indices, :]
        return torch.sum(
            selected_latents * selected_weight.unsqueeze(0).unsqueeze(-1) * active.unsqueeze(0).unsqueeze(-1),
            dim=2,
        )

    def _masked_softmax_weight_matrix(self):
        mask = self.fixed_expand_mask.to(
            device=self.expand_linear.weight.device,
            dtype=torch.bool,
        )
        logits = self.expand_linear.weight.masked_fill(~mask, -1e9)
        return torch.softmax(logits, dim=-1) * mask.to(logits.dtype)

    def _apply_masked_softmax_expansion(self, latent_tokens):
        indices = self.fixed_expand_indices.to(latent_tokens.device)
        active = self.fixed_expand_active.to(device=latent_tokens.device, dtype=latent_tokens.dtype)
        selected_logits = torch.gather(
            self.expand_linear.weight,
            dim=1,
            index=indices,
        ).to(latent_tokens.dtype)
        selected_logits = selected_logits.masked_fill(active <= 0, -1e9)
        selected_weight = torch.softmax(selected_logits, dim=-1) * active
        selected_weight = selected_weight / selected_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        selected_latents = latent_tokens[:, indices, :]
        return torch.sum(
            selected_latents * selected_weight.unsqueeze(0).unsqueeze(-1),
            dim=2,
        )

    def _sparse_softmax_selected_weight(self, device=None, dtype=None):
        logits = self.sparse_expand_logits
        if device is not None or dtype is not None:
            logits = logits.to(device=device if device is not None else logits.device, dtype=dtype or logits.dtype)
        active = self.fixed_expand_active.to(device=logits.device, dtype=logits.dtype)
        logits = logits.masked_fill(active <= 0, -1e9)
        weight = torch.softmax(logits, dim=-1) * active
        return weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _sparse_softmax_weight_matrix(self):
        indices = self.fixed_expand_indices.to(self.sparse_expand_logits.device)
        selected_weight = self._sparse_softmax_selected_weight()
        weight = selected_weight.new_zeros(self.target_num_variates, self.reduced_k)
        weight.scatter_add_(1, indices, selected_weight)
        return weight

    def _apply_sparse_softmax_expansion(self, latent_tokens):
        indices = self.fixed_expand_indices.to(latent_tokens.device)
        selected_weight = self._sparse_softmax_selected_weight(
            device=latent_tokens.device,
            dtype=latent_tokens.dtype,
        )
        selected_latents = latent_tokens[:, indices, :]
        return torch.sum(selected_latents * selected_weight.unsqueeze(0).unsqueeze(-1), dim=2)

    def decode(self, latent_tokens, cache=None, original_variates=None, use_residual=True):
        if self._uses_id_query_decoder():
            return self._id_query_decode(latent_tokens, original_variates=original_variates, use_residual=use_residual)

        if self._uses_id_topk_decoder():
            return self._id_topk_decode(latent_tokens, original_variates=original_variates, use_residual=use_residual)

        if self._uses_attention_decoder():
            return self._query_decode(latent_tokens, original_variates=original_variates, use_residual=use_residual)

        if self.reduction_type == 'none':
            return latent_tokens

        if self.reduction_type == 'mlp_static_combination':
            A = cache['A']
            expansion = A.transpose(0, 1)
            expansion = self._normalize_expansion(expansion)
            return self._apply_expansion(expansion, latent_tokens)
        if self.reduction_type == 'learned_anchor_selection':
            decoded = self.expand_linear(latent_tokens.transpose(1, 2)).transpose(1, 2)
            return self._apply_linear_residual(decoded, original_variates=original_variates, use_residual=use_residual)
        if self.reduction_type in {'MLP_attention', 'latent_query_attention'}:
            decoded = self.slot_expand_linear(latent_tokens.transpose(1, 2)).transpose(1, 2)
            return self._apply_linear_residual(decoded, original_variates=original_variates, use_residual=use_residual)
        if self.reduction_type in {
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
        }:
            if self.expansion_type in {'fixed_assignment', 'fixed_assignment_scalar_residual'}:
                decoded = torch.einsum('nk,bkd->bnd', self.fixed_expand_weight, latent_tokens)
            elif self.expansion_type in {
                'fixed_masked_linear',
                'fixed_masked_scalar_residual',
                'fixed_masked_variate_residual',
            }:
                decoded = self._apply_fixed_masked_linear_expansion(latent_tokens)
            elif self.expansion_type in {'masked_linear', 'masked_linear_scalar_residual'}:
                decoded = self._apply_masked_linear_expansion(latent_tokens)
            elif self.expansion_type in {'masked_softmax', 'masked_softmax_scalar_residual', 'masked_softmax_variate_residual'}:
                decoded = self._apply_sparse_softmax_expansion(latent_tokens)
            else:
                decoded = self.expand_linear(latent_tokens.transpose(1, 2)).transpose(1, 2)
            return self._apply_linear_residual(decoded, original_variates=original_variates, use_residual=use_residual)
        if self.reduction_type == 'mlp_slot_attention_hybrid_linear':
            decoded = self.expand_linear(latent_tokens.transpose(1, 2)).transpose(1, 2)
            return self._apply_linear_residual(decoded, original_variates=original_variates, use_residual=use_residual)
        if self.reduction_type in {
            'mlp_slot_attention',
            'mlp_sparse_slot_attention',
            'mlp_sparse_slot_attention_id',
            'mlp_linear_sparse_slot_attention_id',
            'mlp_static_sparse_slot_attention_id',
        }:
            if self.expansion_type == 'slot_learned_linear':
                decoded = self.slot_expand_linear(latent_tokens.transpose(1, 2)).transpose(1, 2)
                return self._apply_linear_residual(decoded, original_variates=original_variates, use_residual=use_residual)
            A = cache['A']
            expansion = A.transpose(1, 2)
            expansion = self._normalize_expansion(expansion)
            decoded = self._apply_expansion(expansion, latent_tokens)
            return self._apply_linear_residual(decoded, original_variates=original_variates, use_residual=use_residual)

        raise ValueError('Unknown variate reduction type: {}'.format(self.reduction_type))

    def reconstruct(self, latent_tokens, cache=None):
        if hasattr(self, 'reconstruct_linear'):
            return self.reconstruct_linear(latent_tokens.transpose(1, 2)).transpose(1, 2)
        return self.decode(latent_tokens, cache=cache, original_variates=None, use_residual=False)

    def expand(self, encoded_latent_tokens, cache):
        return self.decode(encoded_latent_tokens, cache=cache, original_variates=None, use_residual=False)

    def get_aux_loss(self):
        if self.latest_aux_loss is None:
            device = next(self.parameters(), torch.empty(0)).device
            return torch.tensor(0.0, device=device)
        return self.latest_aux_loss

    def get_aux_losses(self):
        if self.latest_aux_losses is None:
            zero = self._zero_like_param(self)
            return self._aux_loss_dict(zero)
        return self.latest_aux_losses
