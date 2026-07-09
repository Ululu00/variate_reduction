import argparse
import csv
import hashlib
import math
import os
import shlex
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

CONFIG = {
    "DATASETS": ["Weather", "Electricity", "Traffic", "Solar"],
    "PRED_LENS": [96, 192, 336, 720],
    "K_RATIOS": [0.1, 0.2, 0.3],
    "METHODS": ["mlp_slot_attention"],
    "RUN_BASELINE": True,

    "VARIATE_TOKEN_SPLIT_FACTOR": 1,
    "ORTHOGONAL_LOSS_WEIGHT": 0.0,
    "WEIGHT_HEATMAP_DIR": "./results/attention_weight_heatmaps",
}
CONFIG_DEFAULTS = {
    "ETTH1_K_VALUES": [2, 4],
    "K_VALUES": None,
    "FIXED_K_VALUES": None,
    "INCLUDE_RATIO_K": True,
    "K_RATIO_DENOMINATOR": "target",
    "VARIATE_EXPANSION_TYPE": "transpose",
    "VARIATE_DECODE_STAGE": "feature",
    "EXPANSION_TEMPERATURE": 1.0,
    "EXPANSION_TOPK": 0,
    "VARIATE_TOKEN_SPLIT_FACTOR": 1,
    "ORTHOGONAL_LOSS_WEIGHT": 0.0,
    "RECONSTRUCTION_LOSS_WEIGHT": 0.0,
    "COVERAGE_LOSS_WEIGHT": 0.0,
    "ASSIGNMENT_ENTROPY_LOSS_WEIGHT": 0.0,
    "LINEAR_COVERAGE_LOSS_WEIGHT": 0.0,
    "BIORTHOGONAL_LOSS_WEIGHT": 0.0,
    "LINEAR_WEIGHT_L2_LOSS_WEIGHT": 0.0,
    "DECODER_INIT_L2_LOSS_WEIGHT": 0.0,
    "LINEAR_ENTROPY_LOSS_WEIGHT": 0.0,
    "LINEAR_COSINE_LOSS_WEIGHT": 0.0,
    "LINEAR_DECODER_COVERAGE_LOSS_WEIGHT": 0.0,
    "SUPPORT_OVERLAP_LOSS_WEIGHT": 0.0,
    "USE_CYCLE_SLOT_LOSS": False,
    "CYCLE_LOSS_WEIGHT": 0.0,
    "CYCLE_WARMUP_RATIO": 0.0,
    "CYCLE_TOPR": 0,
    "CYCLE_TOPR_MULTIPLIER": 1.0,
    "CYCLE_B_NORM": "row_l1",
    "CYCLE_EPS": 1e-8,
    "EXPORT_SLOT_DIAGNOSTICS": False,
    "SPARSE_COMPRESS_TOPK": 0,
    "SPARSE_EXPAND_TOPK": 0,
    "WCOMP_NORMALIZATION": "softmax",
    "ENTMAX_ALPHA": 1.5,
    "WCOMP_ENTROPY_LOSS_WEIGHT": 0.0,
    "GROUP_ATTENTION_ENTROPY_LOSS_WEIGHT": 0.0,
    "GROUP_ATTENTION_ENTROPY_TARGET": 0.35,
    "SEEDS": [2021],
    "TRAIN_EPOCHS": None,
    "PATIENCE": None,
    "NUM_WORKERS": 0,
    "USE_WANDB": True,
    "WANDB_PROJECT": "iTransformer-variate-reduction",
    "WANDB_ENTITY": "",
    "WANDB_MODE": "online",
    "WANDB_GROUP": "",
    "WANDB_TAGS": ["variate-reduction"],
    "WEIGHT_HEATMAP_DIR": "./results/weight_heatmaps",
    "ENFORCE_K_BUDGET": True,
    "LOCAL_TEMPORAL_BRANCH": "none",
    "LOCAL_TEMPORAL_INIT": "persistence",
    "LOCAL_TEMPORAL_GATE_INIT": 1.0,
    "LOCAL_TEMPORAL_RANK": 4,
    "ZERO_INIT_PROJECTOR": False,
    "SKIP_BACKBONE": False,
    "OUTPUT_CALIBRATION": "none",
    "SELECTION_ID": "",
    "SELECTED_RECIPE": "",
    "MANUAL_OVERRIDE": "0",
    "LOWRANK_REDUCER_RANK": 8,
    "DECODER_RESIDUAL_GATE_INIT": 0.0,
    "BACKBONE_RESIDUAL_GATE_INIT": 1.0,
    "BACKBONE_RESIDUAL_GATE_TYPE": "scalar",
    "VARIATE_ANCHOR_MAP_PATH": "",
    "VARIATE_ANCHOR_MAP_TEMPLATE": "",
    "SKIP_TEST_EVAL": False,
    "SKIP_EPOCH_TEST_EVAL": False,
    "FORECAST_LOSS_TYPE": "mse",
    "HUBER_DELTA": 1.0,
}

for key, value in CONFIG_DEFAULTS.items():
    CONFIG.setdefault(key, value)

RUN_PY_DEFAULTS = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "train_epochs": 10,
    "patience": 3,
    "num_workers": 10,
}

DATASET_SPECS = {
    "Weather": {
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "data": "custom",
        "features": "M",
        "freq": "t",
        "num_variates": 21,
        "seq_len": 96,
        "e_layers": 3,
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
        "d_model": 512,
        "d_ff": 512,
        "k_mode": "ratio",
    },
    "Electricity": {
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "data": "custom",
        "features": "M",
        "freq": "h",
        "num_variates": 321,
        "seq_len": 96,
        "e_layers": 3,
        "enc_in": 321,
        "dec_in": 321,
        "c_out": 321,
        "d_model": 512,
        "d_ff": 512,
        "batch_size": 16,
        "learning_rate": 0.0005,
        "k_mode": "ratio",
    },
    "Traffic": {
        "root_path": "./dataset/traffic/",
        "data_path": "traffic.csv",
        "data": "custom",
        "features": "M",
        "freq": "h",
        "num_variates": 862,
        "seq_len": 96,
        "e_layers": 4,
        "enc_in": 862,
        "dec_in": 862,
        "c_out": 862,
        "d_model": 512,
        "d_ff": 512,
        "batch_size": 16,
        "learning_rate": 0.001,
        "k_mode": "ratio",
    },
    "Solar": {
        "root_path": "./dataset/Solar/",
        "data_path": "solar_AL.txt",
        "data": "Solar",
        "features": "M",
        "freq": "t",
        "target": "none",
        "has_date": False,
        "num_variates": 137,
        "seq_len": 96,
        "e_layers": 2,
        "enc_in": 137,
        "dec_in": 137,
        "c_out": 137,
        "d_model": 512,
        "d_ff": 512,
        "learning_rate": 0.0005,
        "k_mode": "ratio",
    },
    "ETTh1": {
        "root_path": "./dataset/ETT/",
        "data_path": "ETTh1.csv",
        "data": "ETTh1",
        "features": "M",
        "freq": "h",
        "num_variates": 7,
        "seq_len": 96,
        "e_layers": 2,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "k_mode": "absolute",
        "d_model_by_pred_len": {
            96: 256,
            192: 256,
            336: 512,
            720: 512,
        },
        "d_ff_by_pred_len": {
            96: 256,
            192: 256,
            336: 512,
            720: 512,
        },
    },
}


CSV_COLUMNS = [
    'dataset', 'pred_len',
    'selection_id', 'selected_recipe', 'manual_override',
    'variate_reduction_type', 'variate_expansion_type', 'expansion_weight_source',
    'variate_decode_stage', 'loss_variant',
    'forecast_loss_type', 'huber_delta',
    'compute_reducer_aux_losses',
    'variate_token_split_factor',
    'num_variates', 'target_variate_tokens', 'source_variate_tokens',
    'reduced_variate_k', 'reduced_variate_tokens',
    'target_k_ratio', 'actual_k_ratio', 'source_k_ratio',
    'logical_k', 'executed_k', 'logical_k_ratio', 'executed_k_ratio',
    'k_budget_max', 'k_budget_exception',
    'lowrank_reducer_rank', 'decoder_residual_gate_init',
    'backbone_residual_gate_init', 'backbone_residual_gate_type',
    'backbone_residual_gate', 'backbone_residual_gate_abs_mean', 'backbone_residual_gate_max',
    'k_selection_mode', 'k_ratio_denominator', 'target_k_value',
    'original_encoder_tokens', 'reduced_encoder_tokens', 'num_extra_tokens',
    'token_ratio', 'token_reduction_percent',
    'attention_score_ratio', 'attention_score_reduction_percent',
    'eval_split', 'mse', 'mae', 'rmse', 'mape', 'mspe',
    'orthogonal_loss_weight', 'reconstruction_loss_weight',
    'coverage_loss_weight', 'assignment_entropy_loss_weight',
    'wcomp_entropy_loss_weight', 'group_attention_entropy_loss_weight',
    'group_attention_entropy_target',
    'linear_coverage_loss_weight', 'biorthogonal_loss_weight',
    'linear_weight_l2_loss_weight', 'decoder_init_l2_loss_weight',
    'linear_entropy_loss_weight', 'linear_cosine_loss_weight', 'linear_decoder_coverage_loss_weight',
    'support_overlap_loss_weight',
    'use_cycle_slot_loss', 'cycle_loss_weight', 'cycle_warmup_ratio',
    'cycle_topr', 'cycle_topr_multiplier', 'cycle_b_norm', 'cycle_eps',
    'export_slot_diagnostics',
    'cycle_loss', 'weighted_cycle_loss', 'cycle_current_weight',
    'mae_loss_weight',
    'train_time_sec', 'train_iter_count', 'avg_train_iter_time_sec', 'elapsed_sec',
    'peak_gpu_allocated_mb', 'peak_gpu_reserved_mb', 'gpu_memory_footprint_mb',
    'expansion_temperature', 'expansion_topk',
    'sparse_compress_topk', 'sparse_expand_topk',
    'cycle_topr_jaccard', 'cycle_a_effective_support', 'cycle_b_effective_support',
    'cycle_slot_overlap', 'cycle_topr_size',
    'residual_gate_mean', 'residual_gate_abs_mean', 'residual_gate_max',
    'residual_gate_min', 'residual_gate_value_max', 'residual_gate_range',
    'residual_gate_std', 'residual_gate_numel', 'residual_gate_is_variate',
    'hybrid_linear_gate',
    'local_temporal_branch', 'local_temporal_init', 'local_temporal_gate_init',
    'local_temporal_rank', 'local_temporal_gate',
    'zero_init_projector', 'skip_backbone',
    'output_calibration', 'output_calibration_scale_abs_mean', 'output_calibration_bias_abs_mean',
    'status', 'skip_reason', 'timestamp', 'git_commit', 'git_dirty', 'error', 'command', 'returncode',
    'gpu', 'cuda_visible_devices', 'data', 'root_path', 'data_path', 'target', 'freq',
    'seed', 'itr', 'model', 'model_id', 'setting',
    'seq_len', 'label_len', 'e_layers', 'd_model', 'd_ff',
    'batch_size', 'learning_rate', 'train_epochs', 'patience', 'num_workers',
    'use_norm', 'checkpoint_dir', 'result_dir', 'weight_matrix_dir', 'slot_diagnostic_dir',
]


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_SOURCE = Path("/home/yulim/MSH/patch-latent/dataset")


def quote_cmd(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def expansion_weight_source(reduction_type, expansion_type):
    if reduction_type == "none":
        return ""
    if expansion_type in {
        "id_query_decoder",
        "id_query_scalar_residual",
        "id_query_fixed_scalar_residual",
        "id_topk_decoder",
        "id_topk_scalar_residual",
        "id_topk_fixed_scalar_residual",
        "query_decoder",
        "query_decoder_scalar_residual",
        "query_decoder_variate_residual",
        "token_query_decoder",
    }:
        return "query_decoder"
    if expansion_type in {
        "fixed_assignment",
        "fixed_assignment_scalar_residual",
        "fixed_masked_linear",
        "fixed_masked_scalar_residual",
        "fixed_masked_variate_residual",
    }:
        return "fixed_expand_weight"
    if expansion_type in {"masked_linear", "masked_linear_scalar_residual"}:
        return "masked_learned_expand_linear"
    if expansion_type in {
        "masked_softmax",
        "masked_softmax_scalar_residual",
        "masked_softmax_variate_residual",
    }:
        return "masked_softmax_logits"
    if expansion_type == "slot_learned_linear":
        return "slot_learned_expand_linear"
    if reduction_type in {
        "mlp_static_combination",
        "mlp_slot_attention",
        "mlp_sparse_slot_attention",
        "mlp_sparse_slot_attention_id",
        "mlp_linear_sparse_slot_attention_id",
        "mlp_static_sparse_slot_attention_id",
    }:
        return "assignment_transpose"
    return "attention_or_anchor_expand"


def csv_fieldnames(csv_path=None):
    fieldnames = list(CSV_COLUMNS)
    if csv_path and csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            with csv_path.open(newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
            return existing_header + [column for column in fieldnames if column not in existing_header]
        except Exception:
            return fieldnames
    return fieldnames


def ensure_csv_schema(csv_path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames or []
        fieldnames = existing_header + [column for column in CSV_COLUMNS if column not in existing_header]
        if existing_header == fieldnames:
            return
        rows = list(reader)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in rows:
            writer.writerow({column: existing_row.get(column, "") for column in fieldnames})
    tmp_path.replace(csv_path)


def append_csv_row(path, row):
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_csv_schema(csv_path)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    fieldnames = csv_fieldnames(csv_path)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in fieldnames})


def ensure_dataset_symlink():
    link_path = REPO_ROOT / "dataset"
    desired = DATASET_SOURCE.resolve()
    if not DATASET_SOURCE.exists():
        raise RuntimeError("Dataset source does not exist: {}".format(DATASET_SOURCE))
    if not link_path.exists() and not link_path.is_symlink():
        link_path.symlink_to(DATASET_SOURCE)
        print("Created dataset symlink: {} -> {}".format(link_path, DATASET_SOURCE))
        return
    if link_path.is_symlink() and link_path.resolve() == desired:
        return
    if link_path.is_symlink():
        raise RuntimeError(
            "./dataset is a symlink, but it points to {} instead of {}".format(
                link_path.resolve(), desired
            )
        )
    raise RuntimeError(
        "./dataset exists and is not the desired symlink. Refusing to delete or overwrite: {}".format(link_path)
    )


def read_header(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def validate_datasets(dataset_names):
    resolved_targets = {}
    errors = []
    for dataset in dataset_names:
        if dataset not in DATASET_SPECS:
            errors.append("Unknown dataset in CONFIG: {}".format(dataset))
            continue
        spec = DATASET_SPECS[dataset]
        csv_path = REPO_ROOT / spec["root_path"] / spec["data_path"]
        if not csv_path.exists():
            errors.append("{} missing file: {}".format(dataset, csv_path))
            continue
        header = read_header(csv_path)
        if not spec.get("has_date", True):
            if len(header) != spec["num_variates"]:
                errors.append(
                    "{} expected {} columns, found {} in {}".format(
                        dataset, spec["num_variates"], len(header), csv_path
                    )
                )
                continue
            try:
                [float(value) for value in header]
            except ValueError:
                errors.append("{} first row is not numeric in {}".format(dataset, csv_path))
                continue
            resolved_targets[dataset] = spec.get("target", "none")
            continue
        date_column = spec.get("date_column", "date")
        date_matches = [
            column for column in header
            if column == date_column or str(column).lower() == str(date_column).lower()
            or str(column).lower() in ("date", "datetime")
        ]
        if not date_matches:
            errors.append("{} CSV has no date/Datetime column: {}".format(dataset, csv_path))
            continue
        date_columns = set(date_matches)
        non_date_columns = [column for column in header if column not in date_columns]
        if len(non_date_columns) != spec["num_variates"]:
            errors.append(
                "{} expected {} non-date columns, found {} in {}".format(
                    dataset, spec["num_variates"], len(non_date_columns), csv_path
                )
            )
            continue
        target = spec.get("target", "OT")
        if target not in header:
            if spec["data"] == "custom" and target == "OT":
                target = non_date_columns[-1]
                print("{}: target OT not found; using last non-date column '{}'.".format(dataset, target))
            else:
                errors.append("{} target '{}' is not in {}".format(dataset, target, csv_path))
                continue
        resolved_targets[dataset] = target
    if errors:
        raise RuntimeError("Dataset validation failed:\n" + "\n".join("- " + error for error in errors))
    return resolved_targets


def round_half_up(value):
    return int(math.floor(value + 0.5))


def strict_k_budget(num_variates):
    k_max = int(math.ceil(0.30 * int(num_variates))) - 1
    if k_max >= 1:
        return k_max, False
    return 1, True


def spec_for_pred_len(dataset, pred_len):
    spec = dict(DATASET_SPECS[dataset])
    if "d_model_by_pred_len" in spec:
        spec["d_model"] = spec["d_model_by_pred_len"][pred_len]
    if "d_ff_by_pred_len" in spec:
        spec["d_ff"] = spec["d_ff_by_pred_len"][pred_len]
    return spec


def k_ratio_denominators(dataset):
    denominator = str(CONFIG.get("K_RATIO_DENOMINATOR", "target") or "target")
    num_variates = DATASET_SPECS[dataset]["num_variates"]
    requested = ["target", "source"] if denominator == "both" else [denominator]
    result = []
    seen = set()
    for basis in requested:
        value = num_variates * variate_token_split_factor() if basis == "source" else num_variates
        if value in seen:
            continue
        seen.add(value)
        result.append((basis, value))
    return result


def make_k_spec(dataset, raw_k, selection_mode, target_ratio=None, target_value=None, ratio_denominator=""):
    num_variates = DATASET_SPECS[dataset]["num_variates"]
    source_tokens = num_variates * variate_token_split_factor()
    raw_k = int(raw_k)
    target_value = raw_k if target_value is None else target_value
    target_ratio = float(raw_k) / float(num_variates) if target_ratio is None else float(target_ratio)
    k_budget_max, k_budget_exception = strict_k_budget(num_variates)
    skip_reasons = []
    if raw_k > source_tokens:
        skip_reasons.append("K={} exceeds source tokens V_eff={}".format(raw_k, source_tokens))
    if selection_mode in {"fixed", "absolute"} and raw_k > num_variates:
        skip_reasons.append("K={} exceeds original variables V={}".format(raw_k, num_variates))
    if skip_reasons:
        return {
            "k_selection_mode": selection_mode,
            "k_ratio_denominator": ratio_denominator,
            "target_k_ratio": target_ratio,
            "target_k_value": target_value,
            "reduced_variate_k": raw_k,
            "actual_k_ratio": raw_k / num_variates,
            "k_budget_max": k_budget_max,
            "k_budget_exception": k_budget_exception,
            "label": "{}{}".format("ratio" if selection_mode == "ratio" else "k", raw_k),
            "skip": True,
            "skip_reason": "{} (V={}, V_eff={}, split={})".format(
                "; ".join(skip_reasons), num_variates, source_tokens, variate_token_split_factor()
            ),
        }
    reduced_k = raw_k
    if CONFIG.get("ENFORCE_K_BUDGET", True):
        reduced_k = max(1, min(raw_k, k_budget_max))
        if reduced_k != raw_k:
            print("{} {} K clamped from {} to strict budget {}".format(
                dataset,
                selection_mode,
                raw_k,
                reduced_k,
            ))
    if selection_mode == "ratio":
        basis = "src" if ratio_denominator == "source" else "v"
        label = "ratio{}_{}__k{}".format(basis, fmt_float(target_ratio), reduced_k)
    else:
        label = "k{}".format(reduced_k)
    return {
        "k_selection_mode": selection_mode,
        "k_ratio_denominator": ratio_denominator,
        "target_k_ratio": target_ratio,
        "target_k_value": target_value,
        "reduced_variate_k": reduced_k,
        "actual_k_ratio": reduced_k / num_variates,
        "k_budget_max": k_budget_max,
        "k_budget_exception": k_budget_exception,
        "label": label,
        "skip": False,
        "skip_reason": "",
    }


def resolved_k_specs(dataset):
    spec = DATASET_SPECS[dataset]
    num_variates = spec["num_variates"]
    if CONFIG.get("K_VALUES"):
        return dedupe_k_specs([
            make_k_spec(dataset, int(requested_k), "absolute", target_value=int(requested_k))
            for requested_k in CONFIG["K_VALUES"]
        ])
    result = []
    if spec["k_mode"] == "ratio":
        if CONFIG.get("INCLUDE_RATIO_K", True):
            for ratio_denominator, denominator in k_ratio_denominators(dataset):
                for ratio in CONFIG["K_RATIOS"]:
                    raw_k = round_half_up(denominator * ratio)
                    result.append(make_k_spec(
                        dataset,
                        raw_k,
                        "ratio",
                        target_ratio=ratio,
                        target_value=raw_k,
                        ratio_denominator=ratio_denominator,
                    ))
    else:
        for requested_k in CONFIG["ETTH1_K_VALUES"]:
            result.append(make_k_spec(dataset, int(requested_k), "absolute", target_value=int(requested_k)))

    for requested_k in CONFIG.get("FIXED_K_VALUES") or []:
        result.append(make_k_spec(dataset, int(requested_k), "fixed", target_value=int(requested_k)))
    return dedupe_k_specs(result)


def dedupe_k_specs(k_specs):
    deduped = []
    seen = set()
    for k_spec in k_specs:
        if k_spec.get("skip"):
            key = (
                "skip",
                k_spec.get("k_selection_mode", ""),
                k_spec.get("k_ratio_denominator", ""),
                int(k_spec.get("reduced_variate_k", -1)),
                k_spec.get("skip_reason", ""),
            )
        else:
            key = (
                "run",
                k_spec.get("k_selection_mode", ""),
                k_spec.get("k_ratio_denominator", ""),
                int(k_spec.get("reduced_variate_k", -1)),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(k_spec)
    return deduped


def baseline_k_spec(num_variates):
    return {
        "k_selection_mode": "baseline",
        "k_ratio_denominator": "",
        "target_k_ratio": 1.0,
        "target_k_value": num_variates,
        "reduced_variate_k": num_variates,
        "actual_k_ratio": 1.0,
        "k_budget_max": strict_k_budget(num_variates)[0],
        "k_budget_exception": strict_k_budget(num_variates)[1],
        "label": "baseline",
    }


def fmt_float(value, digits=2):
    return ("{:.%df}" % digits).format(float(value))


def fmt_weight(value):
    return "{:.3g}".format(float(value))


def safe_token(value, max_len=None):
    value = str(value)
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    token = "".join(safe).strip("_")
    if max_len is not None and len(token) > int(max_len):
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
        keep = max(1, int(max_len) - len(digest) - 1)
        return "{}_{}".format(token[:keep].rstrip("_"), digest)
    return token


def compact_model_id(model_id, max_len=120):
    if len(model_id) <= int(max_len):
        return model_id
    digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:10]
    keep = int(max_len) - len(digest) - 1
    return "{}_{}".format(model_id[:keep].rstrip("_"), digest)


def _loss_variant(name, orthogonal=0.01, reconstruction=0.1, linear_weight_l2=0.001,
                  linear_coverage=0.0, biorthogonal=0.0,
                  linear_entropy=0.0, linear_cosine=0.0, linear_decoder_coverage=0.0,
                  support_overlap=0.0, mae=0.0,
                  forecast_loss_type="mse", huber_delta=1.0):
    return {
        "NAME": name,
        "ORTHOGONAL_LOSS_WEIGHT": orthogonal,
        "RECONSTRUCTION_LOSS_WEIGHT": reconstruction,
        "COVERAGE_LOSS_WEIGHT": 0.0,
        "ASSIGNMENT_ENTROPY_LOSS_WEIGHT": 0.0,
        "LINEAR_COVERAGE_LOSS_WEIGHT": linear_coverage,
        "BIORTHOGONAL_LOSS_WEIGHT": biorthogonal,
        "LINEAR_WEIGHT_L2_LOSS_WEIGHT": linear_weight_l2,
        "LINEAR_ENTROPY_LOSS_WEIGHT": linear_entropy,
        "LINEAR_COSINE_LOSS_WEIGHT": linear_cosine,
        "LINEAR_DECODER_COVERAGE_LOSS_WEIGHT": linear_decoder_coverage,
        "SUPPORT_OVERLAP_LOSS_WEIGHT": support_overlap,
        "MAE_LOSS_WEIGHT": mae,
        "FORECAST_LOSS_TYPE": forecast_loss_type,
        "HUBER_DELTA": huber_delta,
    }


def _perf_loss_variants():
    return [
        _loss_variant("mse_only", orthogonal=0.0, reconstruction=0.0, linear_weight_l2=0.0),
        _loss_variant("mae0.01", orthogonal=0.0, reconstruction=0.0, linear_weight_l2=0.0, mae=0.01),
        _loss_variant("mae0.05", orthogonal=0.0, reconstruction=0.0, linear_weight_l2=0.0, mae=0.05),
        _loss_variant("mae0.1", orthogonal=0.0, reconstruction=0.0, linear_weight_l2=0.0, mae=0.1),
        _loss_variant("huber0.5", orthogonal=0.0, reconstruction=0.0, linear_weight_l2=0.0, forecast_loss_type="huber", huber_delta=0.5),
        _loss_variant("huber1.0", orthogonal=0.0, reconstruction=0.0, linear_weight_l2=0.0, forecast_loss_type="huber", huber_delta=1.0),
        _loss_variant("huber0.32", orthogonal=0.0, reconstruction=0.0, linear_weight_l2=0.0, forecast_loss_type="huber", huber_delta=0.32),
        _loss_variant("huber0.5_orth0.001", orthogonal=0.001, reconstruction=0.0, linear_weight_l2=0.0, forecast_loss_type="huber", huber_delta=0.5),
        _loss_variant("orth0.001", orthogonal=0.001, reconstruction=0.0, linear_weight_l2=0.0),
        _loss_variant("orth0.01", orthogonal=0.01, reconstruction=0.0, linear_weight_l2=0.0),
        _loss_variant("orth0.1", orthogonal=0.1, reconstruction=0.0, linear_weight_l2=0.0),
        _loss_variant("orth0.01_rec0.01", orthogonal=0.01, reconstruction=0.01, linear_weight_l2=0.0),
    ]

def apply_sweep_preset(name):
    if not name:
        return
    presets = {
        "traffic_perf_short": {
            "DATASETS": ["Traffic"],
            "PRED_LENS": [96, 192],
            "VARIATE_TOKEN_SPLIT_FACTOR": 1,
            "LOSS_VARIANTS": _perf_loss_variants(),
            "LOCAL_TEMPORAL_BRANCH": "persistence_gate",
            "LOCAL_TEMPORAL_GATE_INIT": 0.5,
            "WEIGHT_HEATMAP_DIR": "./results/weight_heatmaps_traffic_perf_short",
            "WANDB_TAGS": ["variate-reduction", "traffic-perf-short", name],
        },
        "target_perf_short": {
            "DATASETS": ["Weather", "Electricity", "Traffic", "Solar"],
            "PRED_LENS": [96, 192, 336, 720],
            "VARIATE_TOKEN_SPLIT_FACTOR": 1,
            "LOSS_VARIANTS": _perf_loss_variants(),
            "LOCAL_TEMPORAL_BRANCH": "persistence_gate",
            "LOCAL_TEMPORAL_GATE_INIT": 0.5,
            "WEIGHT_HEATMAP_DIR": "./results/weight_heatmaps_target_perf_short",
            "WANDB_TAGS": ["variate-reduction", "target-perf-short", name],
        },
    }
    if name not in presets:
        raise RuntimeError("Unknown sweep preset: {}".format(name))
    base_update = {
        "DATASETS": ["Weather", "Electricity", "Traffic", "Solar"],
        "PRED_LENS": [96, 192, 336, 720],
        "K_RATIOS": [0.1, 0.2, 0.3],
        "METHODS": ["mlp_slot_attention"],
        "RUN_BASELINE": False,
        "VARIATE_EXPANSION_TYPE": "transpose",
        "VARIATE_EXPANSION_TYPES": ["transpose"],
        "EXPANSION_TEMPERATURE": 1.0,
        "EXPANSION_TOPK": 0,
        "ORTHOGONAL_LOSS_WEIGHT": 0.01,
        "RECONSTRUCTION_LOSS_WEIGHT": 0.1,
        "COVERAGE_LOSS_WEIGHT": 0.0,
        "ASSIGNMENT_ENTROPY_LOSS_WEIGHT": 0.0,
        "SEEDS": [2021],
        "TRAIN_EPOCHS": None,
        "PATIENCE": None,
        "NUM_WORKERS": 0,
        "USE_WANDB": True,
        "WANDB_TAGS": ["variate-reduction", "loss-sweep", name],
    }
    base_update.update(presets[name])
    CONFIG.update(base_update)


def expansion_types():
    values = CONFIG.get("VARIATE_EXPANSION_TYPES")
    if values:
        return list(values)
    return [CONFIG["VARIATE_EXPANSION_TYPE"]]


def anchor_map_path_for(dataset, pred_len, spec, k_spec):
    template = CONFIG.get("VARIATE_ANCHOR_MAP_TEMPLATE", "") or ""
    if template:
        return template.format(
            dataset=dataset,
            pred_len=pred_len,
            num_variates=spec["num_variates"],
            k=k_spec["reduced_variate_k"],
            reduced_k=k_spec["reduced_variate_k"],
            target_k_ratio=k_spec["target_k_ratio"],
            actual_k_ratio=k_spec["actual_k_ratio"],
        )
    return CONFIG.get("VARIATE_ANCHOR_MAP_PATH", "") or ""


def variate_token_split_factor():
    split_factor = int(CONFIG.get("VARIATE_TOKEN_SPLIT_FACTOR", 1))
    if split_factor < 1:
        raise RuntimeError("VARIATE_TOKEN_SPLIT_FACTOR must be at least 1")
    return split_factor


def training_runtime_fields(spec):
    return {
        "batch_size": int(spec.get("batch_size", RUN_PY_DEFAULTS["batch_size"])),
        "learning_rate": float(spec.get("learning_rate", RUN_PY_DEFAULTS["learning_rate"])),
        "train_epochs": int(
            CONFIG["TRAIN_EPOCHS"]
            if CONFIG["TRAIN_EPOCHS"] is not None
            else RUN_PY_DEFAULTS["train_epochs"]
        ),
        "patience": int(
            CONFIG["PATIENCE"]
            if CONFIG["PATIENCE"] is not None
            else RUN_PY_DEFAULTS["patience"]
        ),
        "num_workers": int(
            CONFIG["NUM_WORKERS"]
            if CONFIG["NUM_WORKERS"] is not None
            else RUN_PY_DEFAULTS["num_workers"]
        ),
    }


def configured_loss_weights(overrides=None):
    overrides = overrides or {}
    weights = {
        "orthogonal_loss_weight": CONFIG.get("ORTHOGONAL_LOSS_WEIGHT", 0.0),
        "reconstruction_loss_weight": CONFIG.get("RECONSTRUCTION_LOSS_WEIGHT", 0.0),
        "coverage_loss_weight": CONFIG.get("COVERAGE_LOSS_WEIGHT", 0.0),
        "assignment_entropy_loss_weight": CONFIG.get("ASSIGNMENT_ENTROPY_LOSS_WEIGHT", 0.0),
        "wcomp_entropy_loss_weight": CONFIG.get("WCOMP_ENTROPY_LOSS_WEIGHT", 0.0),
        "group_attention_entropy_loss_weight": CONFIG.get("GROUP_ATTENTION_ENTROPY_LOSS_WEIGHT", 0.0),
        "group_attention_entropy_target": CONFIG.get("GROUP_ATTENTION_ENTROPY_TARGET", 0.35),
        "linear_coverage_loss_weight": CONFIG.get("LINEAR_COVERAGE_LOSS_WEIGHT", 0.0),
        "biorthogonal_loss_weight": CONFIG.get("BIORTHOGONAL_LOSS_WEIGHT", 0.0),
        "linear_weight_l2_loss_weight": CONFIG.get("LINEAR_WEIGHT_L2_LOSS_WEIGHT", 0.0),
        "decoder_init_l2_loss_weight": CONFIG.get("DECODER_INIT_L2_LOSS_WEIGHT", 0.0),
        "linear_entropy_loss_weight": CONFIG.get("LINEAR_ENTROPY_LOSS_WEIGHT", 0.0),
        "linear_cosine_loss_weight": CONFIG.get("LINEAR_COSINE_LOSS_WEIGHT", 0.0),
        "linear_decoder_coverage_loss_weight": CONFIG.get("LINEAR_DECODER_COVERAGE_LOSS_WEIGHT", 0.0),
        "support_overlap_loss_weight": CONFIG.get("SUPPORT_OVERLAP_LOSS_WEIGHT", 0.0),
        "mae_loss_weight": CONFIG.get("MAE_LOSS_WEIGHT", 0.0),
        "forecast_loss_type": CONFIG.get("FORECAST_LOSS_TYPE", "mse"),
        "huber_delta": CONFIG.get("HUBER_DELTA", 1.0),
    }
    mapping = {
        "ORTHOGONAL_LOSS_WEIGHT": "orthogonal_loss_weight",
        "RECONSTRUCTION_LOSS_WEIGHT": "reconstruction_loss_weight",
        "COVERAGE_LOSS_WEIGHT": "coverage_loss_weight",
        "ASSIGNMENT_ENTROPY_LOSS_WEIGHT": "assignment_entropy_loss_weight",
        "WCOMP_ENTROPY_LOSS_WEIGHT": "wcomp_entropy_loss_weight",
        "GROUP_ATTENTION_ENTROPY_LOSS_WEIGHT": "group_attention_entropy_loss_weight",
        "GROUP_ATTENTION_ENTROPY_TARGET": "group_attention_entropy_target",
        "LINEAR_COVERAGE_LOSS_WEIGHT": "linear_coverage_loss_weight",
        "BIORTHOGONAL_LOSS_WEIGHT": "biorthogonal_loss_weight",
        "LINEAR_WEIGHT_L2_LOSS_WEIGHT": "linear_weight_l2_loss_weight",
        "DECODER_INIT_L2_LOSS_WEIGHT": "decoder_init_l2_loss_weight",
        "LINEAR_ENTROPY_LOSS_WEIGHT": "linear_entropy_loss_weight",
        "LINEAR_COSINE_LOSS_WEIGHT": "linear_cosine_loss_weight",
        "LINEAR_DECODER_COVERAGE_LOSS_WEIGHT": "linear_decoder_coverage_loss_weight",
        "SUPPORT_OVERLAP_LOSS_WEIGHT": "support_overlap_loss_weight",
        "MAE_LOSS_WEIGHT": "mae_loss_weight",
        "FORECAST_LOSS_TYPE": "forecast_loss_type",
        "HUBER_DELTA": "huber_delta",
    }
    for config_key, weight_key in mapping.items():
        if config_key in overrides:
            weights[weight_key] = overrides[config_key]
    if variate_token_split_factor() > 1:
        weights["reconstruction_loss_weight"] = 0.0
        weights["coverage_loss_weight"] = 0.0
        weights["assignment_entropy_loss_weight"] = 0.0
    return weights


def configured_loss_variants():
    variants = CONFIG.get("LOSS_VARIANTS")
    if not variants:
        return [("", configured_loss_weights())]
    configured = [
        (variant.get("NAME", ""), configured_loss_weights(variant))
        for variant in variants
    ]
    deduped = []
    seen = set()
    for name, weights in configured:
        numeric_columns = [
            "orthogonal_loss_weight",
            "reconstruction_loss_weight",
            "coverage_loss_weight",
            "assignment_entropy_loss_weight",
            "wcomp_entropy_loss_weight",
            "group_attention_entropy_loss_weight",
            "group_attention_entropy_target",
            "linear_coverage_loss_weight",
            "biorthogonal_loss_weight",
            "linear_weight_l2_loss_weight",
            "decoder_init_l2_loss_weight",
            "linear_entropy_loss_weight",
            "linear_cosine_loss_weight",
            "linear_decoder_coverage_loss_weight",
            "support_overlap_loss_weight",
            "mae_loss_weight",
            "huber_delta",
        ]
        key = tuple(
            normalize_float(weights[column])
            for column in numeric_columns
        ) + (str(weights.get("forecast_loss_type", "mse")),)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, weights))
    return deduped


def aux_config_suffix(expansion_type, reconstruction_weight, coverage_weight, entropy_weight,
                      linear_coverage_weight, biorthogonal_weight, linear_weight_l2_weight,
                      decoder_init_l2_weight,
                      wcomp_entropy_weight=0.0,
                      group_attention_entropy_weight=0.0,
                      group_attention_entropy_target=0.35,
                      linear_entropy_weight=0.0,
                      linear_cosine_weight=0.0,
                      linear_decoder_coverage_weight=0.0,
                      support_overlap_weight=0.0,
                      mae_loss_weight=0.0,
                      forecast_loss_type="mse",
                      huber_delta=1.0,
                      expansion_temperature=1.0,
                      expansion_topk=0,
                      decoder_residual_gate_init=0.0,
                      backbone_residual_gate_init=1.0,
                      backbone_residual_gate_type="scalar",
                      variate_anchor_map_path=""):
    suffix = []
    if expansion_type != "transpose":
        suffix.append(expansion_type)
    if expansion_type in {
        "linear_scalar_residual",
        "linear_variate_residual",
        "fixed_masked_variate_residual",
        "masked_linear_scalar_residual",
        "masked_softmax_scalar_residual",
        "masked_softmax_variate_residual",
        "query_decoder_scalar_residual",
        "query_decoder_variate_residual",
    }:
        suffix.append("dres{}".format(fmt_weight(decoder_residual_gate_init)))
    if expansion_type in {"sharpened_column_normalized", "topk_column_normalized"} and float(expansion_temperature) != 1.0:
        suffix.append("temp{}".format(fmt_float(expansion_temperature)))
    if expansion_type == "topk_column_normalized" and int(expansion_topk) > 0:
        suffix.append("topk{}".format(int(expansion_topk)))
    if reconstruction_weight > 0.0:
        suffix.append("rec{}".format(fmt_weight(reconstruction_weight)))
    if coverage_weight > 0.0:
        suffix.append("cov{}".format(fmt_weight(coverage_weight)))
    if entropy_weight > 0.0:
        suffix.append("ent{}".format(fmt_weight(entropy_weight)))
    if wcomp_entropy_weight > 0.0:
        suffix.append("went{}".format(fmt_weight(wcomp_entropy_weight)))
    if group_attention_entropy_weight > 0.0:
        suffix.append("gent{}t{}".format(
            fmt_weight(group_attention_entropy_weight),
            fmt_weight(group_attention_entropy_target),
        ))
    if linear_coverage_weight > 0.0:
        suffix.append("lincov{}".format(fmt_weight(linear_coverage_weight)))
    if biorthogonal_weight > 0.0:
        suffix.append("bio{}".format(fmt_weight(biorthogonal_weight)))
    if linear_weight_l2_weight > 0.0:
        suffix.append("wl2{}".format(fmt_weight(linear_weight_l2_weight)))
    if decoder_init_l2_weight > 0.0:
        suffix.append("dinit{}".format(fmt_weight(decoder_init_l2_weight)))
    if linear_entropy_weight > 0.0:
        suffix.append("linent{}".format(fmt_weight(linear_entropy_weight)))
    if linear_cosine_weight > 0.0:
        suffix.append("cos{}".format(fmt_weight(linear_cosine_weight)))
    if linear_decoder_coverage_weight > 0.0:
        suffix.append("deccov{}".format(fmt_weight(linear_decoder_coverage_weight)))
    if support_overlap_weight > 0.0:
        suffix.append("supov{}".format(fmt_weight(support_overlap_weight)))
    if mae_loss_weight > 0.0:
        suffix.append("mae{}".format(fmt_weight(mae_loss_weight)))
    if forecast_loss_type in {"smooth_l1", "huber"}:
        suffix.append("{}{}".format(forecast_loss_type, fmt_weight(huber_delta)))
    if float(backbone_residual_gate_init) != 1.0 or backbone_residual_gate_type != "scalar":
        gate_prefix = "vbgate" if backbone_residual_gate_type == "variate" else "bgate"
        suffix.append("{}{}".format(gate_prefix, fmt_weight(backbone_residual_gate_init)))
    if variate_anchor_map_path:
        suffix.append("amap{}".format(safe_token(Path(variate_anchor_map_path).stem, max_len=32)))
    return "_" + "_".join(suffix) if suffix else ""


def split_suffix(split_factor):
    return "_split{}".format(split_factor) if int(split_factor) > 1 else ""


def local_temporal_suffix(branch, init, rank=4, zero_init_projector=False, gate_init=1.0, skip_backbone=False):
    if branch == "none":
        suffix = ""
        if zero_init_projector:
            suffix += "_zeroproj"
        if skip_backbone:
            suffix += "_skipbb"
        return suffix
    if branch == "lowrank_linear" or "lowrank" in branch or "group" in branch:
        suffix = "_local{}{}-{}".format(branch, int(rank), init)
    else:
        suffix = "_local{}-{}".format(branch, init)
    if float(gate_init) != 1.0:
        suffix += "_gate{}".format(fmt_weight(gate_init))
    if zero_init_projector:
        suffix += "_zeroproj"
    if skip_backbone:
        suffix += "_skipbb"
    return suffix


def output_calibration_suffix(output_calibration):
    if output_calibration in ("", None, "none"):
        return ""
    return "_ocal{}".format(output_calibration)


def decode_stage_suffix(decode_stage):
    if decode_stage in ("", None, "feature"):
        return ""
    return "_{}decode".format(decode_stage)


def cycle_suffix():
    if not bool(CONFIG.get("USE_CYCLE_SLOT_LOSS", False)):
        return ""
    suffix = "_cycle{}".format(fmt_weight(CONFIG.get("CYCLE_LOSS_WEIGHT", 0.0)))
    if float(CONFIG.get("CYCLE_WARMUP_RATIO", 0.0)) != 0.0:
        suffix += "_cw{}".format(fmt_weight(CONFIG.get("CYCLE_WARMUP_RATIO", 0.0)))
    if int(CONFIG.get("CYCLE_TOPR", 0)) > 0:
        suffix += "_ctr{}".format(int(CONFIG.get("CYCLE_TOPR", 0)))
    elif float(CONFIG.get("CYCLE_TOPR_MULTIPLIER", 1.0)) != 1.0:
        suffix += "_ctm{}".format(fmt_weight(CONFIG.get("CYCLE_TOPR_MULTIPLIER", 1.0)))
    return suffix


def model_id_for(
    dataset,
    pred_len,
    method,
    k_spec,
    loss_weights,
    seed,
    expansion_type="transpose",
    expansion_temperature=1.0,
    split_factor=1,
    local_temporal_branch="none",
    local_temporal_init="persistence",
    local_temporal_rank=4,
    local_temporal_gate_init=1.0,
    zero_init_projector=False,
    skip_backbone=False,
    decoder_residual_gate_init=0.0,
    backbone_residual_gate_init=1.0,
    backbone_residual_gate_type="scalar",
    variate_anchor_map_path="",
    output_calibration="none",
    variate_decode_stage="feature",
    expansion_topk=0,
    sparse_compress_topk=0,
    sparse_expand_topk=0,
):
    local_suffix = local_temporal_suffix(
        local_temporal_branch,
        local_temporal_init,
        local_temporal_rank,
        zero_init_projector,
        local_temporal_gate_init,
        skip_backbone,
    )
    output_suffix = output_calibration_suffix(output_calibration)
    decode_suffix = decode_stage_suffix(variate_decode_stage)
    cyc_suffix = cycle_suffix()
    if method == "none":
        return compact_model_id(
            "{}_pl{}_baseline_seed{}".format(dataset, pred_len, seed) + local_suffix + output_suffix
        )
    ortho_weight = loss_weights["orthogonal_loss_weight"]
    suffix = aux_config_suffix(
        expansion_type,
        loss_weights["reconstruction_loss_weight"],
        loss_weights["coverage_loss_weight"],
        loss_weights["assignment_entropy_loss_weight"],
        loss_weights["linear_coverage_loss_weight"],
        loss_weights["biorthogonal_loss_weight"],
        loss_weights["linear_weight_l2_loss_weight"],
        loss_weights["decoder_init_l2_loss_weight"],
        wcomp_entropy_weight=loss_weights.get("wcomp_entropy_loss_weight", 0.0),
        group_attention_entropy_weight=loss_weights.get("group_attention_entropy_loss_weight", 0.0),
        group_attention_entropy_target=loss_weights.get("group_attention_entropy_target", 0.35),
        linear_entropy_weight=loss_weights["linear_entropy_loss_weight"],
        linear_cosine_weight=loss_weights["linear_cosine_loss_weight"],
        linear_decoder_coverage_weight=loss_weights["linear_decoder_coverage_loss_weight"],
        support_overlap_weight=loss_weights["support_overlap_loss_weight"],
        mae_loss_weight=loss_weights["mae_loss_weight"],
        forecast_loss_type=loss_weights.get("forecast_loss_type", "mse"),
        huber_delta=loss_weights.get("huber_delta", 1.0),
        expansion_temperature=expansion_temperature,
        expansion_topk=expansion_topk,
        decoder_residual_gate_init=decoder_residual_gate_init,
        backbone_residual_gate_init=backbone_residual_gate_init,
        backbone_residual_gate_type=backbone_residual_gate_type,
        variate_anchor_map_path=variate_anchor_map_path,
    )
    topk_suffix = ""
    if method in {
        "mlp_sparse_slot_attention",
        "mlp_sparse_slot_attention_id",
        "mlp_linear_sparse_slot_attention_id",
        "mlp_static_sparse_slot_attention_id",
    } and int(expansion_topk) > 0:
        topk_suffix = "_slottopk{}".format(int(expansion_topk))
    if method == "mlp_sparse_representative":
        topk_suffix += "_s{}_r{}".format(int(sparse_compress_topk), int(sparse_expand_topk))
    if k_spec["k_selection_mode"] == "ratio":
        ratio_name = "ratioSrc" if k_spec.get("k_ratio_denominator") == "source" else "ratio"
        return compact_model_id("{}_pl{}_{}_{}{}_k{}_ortho{}_seed{}".format(
            dataset,
            pred_len,
            method,
            ratio_name,
            fmt_float(k_spec["target_k_ratio"]),
            k_spec["reduced_variate_k"],
            fmt_weight(ortho_weight),
            seed,
        ) + topk_suffix + split_suffix(split_factor) + local_suffix + output_suffix + decode_suffix + suffix + cyc_suffix)
    return compact_model_id("{}_pl{}_{}_k{}_ortho{}_seed{}".format(
        dataset,
        pred_len,
        method,
        k_spec["reduced_variate_k"],
        fmt_weight(ortho_weight),
        seed,
    ) + topk_suffix + split_suffix(split_factor) + local_suffix + output_suffix + decode_suffix + suffix + cyc_suffix)


def add_arg(command, name, value):
    command.extend([name, str(value)])


def build_command(run):
    spec = run["spec"]
    command = [
        sys.executable,
        "run.py",
        "--is_training", "1",
        "--model_id", run["model_id"],
        "--model", "iTransformer",
        "--data", spec["data"],
        "--root_path", spec["root_path"],
        "--data_path", spec["data_path"],
        "--features", spec["features"],
        "--target", run["target"],
        "--freq", spec.get("freq", "h"),
        "--seq_len", spec["seq_len"],
        "--pred_len", run["pred_len"],
        "--enc_in", spec["enc_in"],
        "--dec_in", spec["dec_in"],
        "--c_out", spec["c_out"],
        "--e_layers", spec["e_layers"],
        "--d_model", spec["d_model"],
        "--d_ff", spec["d_ff"],
        "--des", run["des"],
        "--use_gpu", "True",
        "--gpu", "0",
        "--use_multi_gpu", "False",
        "--variate_reduction_type", run["method"],
        "--reduced_variate_k", run["k_spec"]["reduced_variate_k"],
        "--lowrank_reducer_rank", run["lowrank_reducer_rank"],
        "--variate_expansion_type", run["variate_expansion_type"],
        "--variate_decode_stage", run["variate_decode_stage"],
        "--expansion_temperature", run["expansion_temperature"],
        "--expansion_topk", run["expansion_topk"],
        "--wcomp_normalization", run["wcomp_normalization"],
        "--entmax_alpha", run["entmax_alpha"],
        "--sparse_compress_topk", run["sparse_compress_topk"],
        "--sparse_expand_topk", run["sparse_expand_topk"],
        "--decoder_residual_gate_init", run["decoder_residual_gate_init"],
        "--backbone_residual_gate_init", run["backbone_residual_gate_init"],
        "--backbone_residual_gate_type", run["backbone_residual_gate_type"],
        "--orthogonal_loss_weight", run["orthogonal_loss_weight"],
        "--use_cycle_slot_loss", run.get("use_cycle_slot_loss", False),
        "--cycle_loss_weight", run.get("cycle_loss_weight", 0.0),
        "--cycle_warmup_ratio", run.get("cycle_warmup_ratio", 0.0),
        "--cycle_topr", run.get("cycle_topr", 0),
        "--cycle_topr_multiplier", run.get("cycle_topr_multiplier", 1.0),
        "--cycle_b_norm", run.get("cycle_b_norm", "row_l1"),
        "--cycle_eps", run.get("cycle_eps", 1e-8),
        "--export_slot_diagnostics", run.get("export_slot_diagnostics", False),
        "--mae_loss_weight", run["mae_loss_weight"],
        "--forecast_loss_type", run.get("forecast_loss_type", "mse"),
        "--huber_delta", run.get("huber_delta", 1.0),
        "--variate_token_split_factor", run["variate_token_split_factor"],
        "--enforce_k_budget", run["enforce_k_budget"],
        "--local_temporal_branch", run["local_temporal_branch"],
        "--local_temporal_init", run["local_temporal_init"],
        "--local_temporal_gate_init", run["local_temporal_gate_init"],
        "--local_temporal_rank", run["local_temporal_rank"],
        "--zero_init_projector", run["zero_init_projector"],
        "--skip_backbone", run["skip_backbone"],
        "--output_calibration", run["output_calibration"],
        "--loss_variant", run["loss_variant"],
        "--result_csv", run["result_csv"],
        "--seed", run["seed"],
        "--experiment_tag", run["dataset"],
        "--selection_id", run.get("selection_id", ""),
        "--selected_recipe", run.get("selected_recipe", ""),
        "--manual_override", run.get("manual_override", "0"),
        "--target_k_ratio", run["k_spec"]["target_k_ratio"],
        "--target_k_value", run["k_spec"]["target_k_value"],
        "--k_selection_mode", run["k_spec"]["k_selection_mode"],
        "--k_ratio_denominator", run["k_spec"].get("k_ratio_denominator", ""),
        "--variate_anchor_map_path", run["variate_anchor_map_path"],
        "--use_wandb", run["use_wandb"],
        "--wandb_project", run["wandb_project"],
        "--wandb_entity", run["wandb_entity"],
        "--wandb_mode", run["wandb_mode"],
        "--wandb_group", run["wandb_group"],
        "--wandb_tags", run["wandb_tags"],
        "--weight_heatmap_dir", run["weight_heatmap_dir"],
        "--skip_test_eval", run["skip_test_eval"],
        "--skip_epoch_test_eval", run.get("skip_epoch_test_eval", False),
    ]
    command.extend([
        "--reconstruction_loss_weight", run["reconstruction_loss_weight"],
        "--coverage_loss_weight", run["coverage_loss_weight"],
        "--assignment_entropy_loss_weight", run["assignment_entropy_loss_weight"],
        "--wcomp_entropy_loss_weight", run["wcomp_entropy_loss_weight"],
        "--group_attention_entropy_loss_weight", run["group_attention_entropy_loss_weight"],
        "--group_attention_entropy_target", run["group_attention_entropy_target"],
    ])
    add_arg(command, "--batch_size", run["batch_size"])
    add_arg(command, "--learning_rate", run["learning_rate"])
    add_arg(command, "--train_epochs", run["train_epochs"])
    add_arg(command, "--patience", run["patience"])
    add_arg(command, "--num_workers", run["num_workers"])
    return [str(part) for part in command]


def generate_runs(result_csv, seeds):
    runs = []
    split_factor = variate_token_split_factor()
    if split_factor > 1:
        raise RuntimeError("VARIATE_TOKEN_SPLIT_FACTOR > 1 was only used by removed generation reducers")
    removed_methods = {
        "mlp_generation",
        "mlp_generation_id",
        "mlp_lowrank_generation",
        "mlp_sparse_representative",
    }
    requested_removed = sorted(removed_methods.intersection(CONFIG["METHODS"]))
    if requested_removed:
        raise RuntimeError("Removed generation reducer(s) requested: {}".format(", ".join(requested_removed)))
    if CONFIG["RUN_BASELINE"]:
        for dataset in CONFIG["DATASETS"]:
            for pred_len in CONFIG["PRED_LENS"]:
                spec = spec_for_pred_len(dataset, pred_len)
                runtime_fields = training_runtime_fields(spec)
                for seed in seeds:
                    k_spec = baseline_k_spec(spec["num_variates"])
                    zero_losses = {
                        "orthogonal_loss_weight": 0.0,
                        "reconstruction_loss_weight": 0.0,
                        "coverage_loss_weight": 0.0,
                        "assignment_entropy_loss_weight": 0.0,
                        "wcomp_entropy_loss_weight": 0.0,
                        "group_attention_entropy_loss_weight": 0.0,
                        "group_attention_entropy_target": CONFIG["GROUP_ATTENTION_ENTROPY_TARGET"],
                        "linear_coverage_loss_weight": 0.0,
                        "biorthogonal_loss_weight": 0.0,
                        "linear_weight_l2_loss_weight": 0.0,
                        "decoder_init_l2_loss_weight": 0.0,
                        "linear_entropy_loss_weight": 0.0,
                        "linear_cosine_loss_weight": 0.0,
                        "linear_decoder_coverage_loss_weight": 0.0,
                        "support_overlap_loss_weight": 0.0,
                        "mae_loss_weight": 0.0,
                        "forecast_loss_type": "mse",
                        "huber_delta": 1.0,
                    }
                    model_id = model_id_for(dataset, pred_len, "none", k_spec, zero_losses, seed, split_factor=1)
                    run = {
                        "dataset": dataset,
                        "pred_len": pred_len,
                        "method": "none",
                        "orthogonal_loss_weight": zero_losses["orthogonal_loss_weight"],
                        "variate_expansion_type": "transpose",
                        "variate_decode_stage": "feature",
                        "expansion_temperature": 1.0,
                        "expansion_topk": 0,
                        "wcomp_normalization": CONFIG["WCOMP_NORMALIZATION"],
                        "entmax_alpha": CONFIG["ENTMAX_ALPHA"],
                        "reconstruction_loss_weight": zero_losses["reconstruction_loss_weight"],
                        "coverage_loss_weight": zero_losses["coverage_loss_weight"],
                        "assignment_entropy_loss_weight": zero_losses["assignment_entropy_loss_weight"],
                        "wcomp_entropy_loss_weight": zero_losses["wcomp_entropy_loss_weight"],
                        "group_attention_entropy_loss_weight": zero_losses["group_attention_entropy_loss_weight"],
                        "group_attention_entropy_target": zero_losses["group_attention_entropy_target"],
                        "linear_coverage_loss_weight": zero_losses["linear_coverage_loss_weight"],
                        "biorthogonal_loss_weight": zero_losses["biorthogonal_loss_weight"],
                        "linear_weight_l2_loss_weight": zero_losses["linear_weight_l2_loss_weight"],
                        "decoder_init_l2_loss_weight": zero_losses["decoder_init_l2_loss_weight"],
                        "linear_entropy_loss_weight": zero_losses["linear_entropy_loss_weight"],
                        "linear_cosine_loss_weight": zero_losses["linear_cosine_loss_weight"],
                        "linear_decoder_coverage_loss_weight": zero_losses["linear_decoder_coverage_loss_weight"],
                        "support_overlap_loss_weight": zero_losses["support_overlap_loss_weight"],
                        "use_cycle_slot_loss": False,
                        "cycle_loss_weight": 0.0,
                        "cycle_warmup_ratio": 0.0,
                        "cycle_topr": 0,
                        "cycle_topr_multiplier": 1.0,
                        "cycle_b_norm": "row_l1",
                        "cycle_eps": 1e-8,
                        "export_slot_diagnostics": False,
                        "mae_loss_weight": zero_losses["mae_loss_weight"],
                        "forecast_loss_type": zero_losses["forecast_loss_type"],
                        "huber_delta": zero_losses["huber_delta"],
                        "loss_variant": "",
                        "variate_token_split_factor": 1,
                        "seed": seed,
                        "k_spec": k_spec,
                        "model_id": model_id,
                        "des": "baseline",
                        "spec": spec,
                        "result_csv": result_csv,
                        "weight_heatmap_dir": CONFIG["WEIGHT_HEATMAP_DIR"],
                        "enforce_k_budget": CONFIG["ENFORCE_K_BUDGET"],
                        "lowrank_reducer_rank": CONFIG["LOWRANK_REDUCER_RANK"],
                        "decoder_residual_gate_init": 0.0,
                        "backbone_residual_gate_init": 1.0,
                        "backbone_residual_gate_type": "scalar",
                        "variate_anchor_map_path": "",
                        "local_temporal_branch": "none",
                        "local_temporal_init": CONFIG["LOCAL_TEMPORAL_INIT"],
                        "local_temporal_gate_init": CONFIG["LOCAL_TEMPORAL_GATE_INIT"],
                        "local_temporal_rank": CONFIG["LOCAL_TEMPORAL_RANK"],
                        "zero_init_projector": False,
                        "skip_backbone": False,
                        "output_calibration": "none",
                        "selection_id": CONFIG["SELECTION_ID"],
                        "selected_recipe": CONFIG["SELECTED_RECIPE"],
                        "manual_override": CONFIG["MANUAL_OVERRIDE"],
                        "skip_test_eval": CONFIG.get("SKIP_TEST_EVAL", False),
                        "skip_epoch_test_eval": CONFIG.get("SKIP_EPOCH_TEST_EVAL", False),
                        "sparse_compress_topk": 0,
                        "sparse_expand_topk": 0,
                        "skip_reason": "",
                    }
                    run.update(runtime_fields)
                    runs.append(run)

    for dataset in CONFIG["DATASETS"]:
        for pred_len in CONFIG["PRED_LENS"]:
            spec = spec_for_pred_len(dataset, pred_len)
            runtime_fields = training_runtime_fields(spec)
            if split_factor > 1 and spec["seq_len"] % split_factor != 0:
                raise RuntimeError("{} seq_len={} is not divisible by split factor {}".format(
                    dataset, spec["seq_len"], split_factor
                ))
            for method in CONFIG["METHODS"]:
                for k_spec in resolved_k_specs(dataset):
                    for expansion_type in expansion_types():
                        for loss_variant, loss_weights in configured_loss_variants():
                            for seed in seeds:
                                variate_anchor_map_path = anchor_map_path_for(dataset, pred_len, spec, k_spec)
                                model_id = model_id_for(
                                    dataset,
                                    pred_len,
                                    method,
                                    k_spec,
                                    loss_weights,
                                    seed,
                                    expansion_type,
                                    expansion_temperature=CONFIG["EXPANSION_TEMPERATURE"],
                                    split_factor=split_factor,
                                    local_temporal_branch=CONFIG["LOCAL_TEMPORAL_BRANCH"],
                                    local_temporal_init=CONFIG["LOCAL_TEMPORAL_INIT"],
                                    local_temporal_rank=CONFIG["LOCAL_TEMPORAL_RANK"],
                                    local_temporal_gate_init=CONFIG["LOCAL_TEMPORAL_GATE_INIT"],
                                    zero_init_projector=CONFIG["ZERO_INIT_PROJECTOR"],
                                    skip_backbone=CONFIG["SKIP_BACKBONE"],
                                    decoder_residual_gate_init=CONFIG["DECODER_RESIDUAL_GATE_INIT"],
                                    backbone_residual_gate_init=CONFIG["BACKBONE_RESIDUAL_GATE_INIT"],
                                    backbone_residual_gate_type=CONFIG["BACKBONE_RESIDUAL_GATE_TYPE"],
                                    variate_anchor_map_path=variate_anchor_map_path,
                                    output_calibration=CONFIG["OUTPUT_CALIBRATION"],
                                    variate_decode_stage=CONFIG["VARIATE_DECODE_STAGE"],
                                    expansion_topk=CONFIG["EXPANSION_TOPK"],
                                    sparse_compress_topk=CONFIG["SPARSE_COMPRESS_TOPK"],
                                    sparse_expand_topk=CONFIG["SPARSE_EXPAND_TOPK"],
                                )
                                run = {
                                    "dataset": dataset,
                                    "pred_len": pred_len,
                                    "method": method,
                                    "orthogonal_loss_weight": loss_weights["orthogonal_loss_weight"],
                                    "variate_expansion_type": expansion_type,
                                    "variate_decode_stage": CONFIG["VARIATE_DECODE_STAGE"],
                                    "expansion_temperature": CONFIG["EXPANSION_TEMPERATURE"],
                                    "expansion_topk": CONFIG["EXPANSION_TOPK"],
                                    "wcomp_normalization": CONFIG["WCOMP_NORMALIZATION"],
                                    "entmax_alpha": CONFIG["ENTMAX_ALPHA"],
                                    "reconstruction_loss_weight": loss_weights["reconstruction_loss_weight"],
                                    "coverage_loss_weight": loss_weights["coverage_loss_weight"],
                                    "assignment_entropy_loss_weight": loss_weights["assignment_entropy_loss_weight"],
                                    "wcomp_entropy_loss_weight": loss_weights["wcomp_entropy_loss_weight"],
                                    "group_attention_entropy_loss_weight": loss_weights[
                                        "group_attention_entropy_loss_weight"
                                    ],
                                    "group_attention_entropy_target": loss_weights[
                                        "group_attention_entropy_target"
                                    ],
                                    "linear_coverage_loss_weight": loss_weights["linear_coverage_loss_weight"],
                                    "biorthogonal_loss_weight": loss_weights["biorthogonal_loss_weight"],
                                    "linear_weight_l2_loss_weight": loss_weights["linear_weight_l2_loss_weight"],
                                    "decoder_init_l2_loss_weight": loss_weights["decoder_init_l2_loss_weight"],
                                    "linear_entropy_loss_weight": loss_weights["linear_entropy_loss_weight"],
                                    "linear_cosine_loss_weight": loss_weights["linear_cosine_loss_weight"],
                                    "linear_decoder_coverage_loss_weight": loss_weights["linear_decoder_coverage_loss_weight"],
                                    "support_overlap_loss_weight": loss_weights["support_overlap_loss_weight"],
                                    "use_cycle_slot_loss": CONFIG["USE_CYCLE_SLOT_LOSS"],
                                    "cycle_loss_weight": CONFIG["CYCLE_LOSS_WEIGHT"],
                                    "cycle_warmup_ratio": CONFIG["CYCLE_WARMUP_RATIO"],
                                    "cycle_topr": CONFIG["CYCLE_TOPR"],
                                    "cycle_topr_multiplier": CONFIG["CYCLE_TOPR_MULTIPLIER"],
                                    "cycle_b_norm": CONFIG["CYCLE_B_NORM"],
                                    "cycle_eps": CONFIG["CYCLE_EPS"],
                                    "export_slot_diagnostics": CONFIG["EXPORT_SLOT_DIAGNOSTICS"],
                                    "mae_loss_weight": loss_weights["mae_loss_weight"],
                                    "forecast_loss_type": loss_weights.get("forecast_loss_type", "mse"),
                                    "huber_delta": loss_weights.get("huber_delta", 1.0),
                                    "loss_variant": loss_variant,
                                    "variate_token_split_factor": split_factor,
                                    "seed": seed,
                                    "k_spec": k_spec,
                                    "k_ratio_denominator": k_spec.get("k_ratio_denominator", ""),
                                    "model_id": model_id,
                                    "des": "vr",
                                    "spec": spec,
                                    "result_csv": result_csv,
                                    "weight_heatmap_dir": CONFIG["WEIGHT_HEATMAP_DIR"],
                                    "enforce_k_budget": CONFIG["ENFORCE_K_BUDGET"],
                                    "lowrank_reducer_rank": CONFIG["LOWRANK_REDUCER_RANK"],
                                    "decoder_residual_gate_init": CONFIG["DECODER_RESIDUAL_GATE_INIT"],
                                    "backbone_residual_gate_init": CONFIG["BACKBONE_RESIDUAL_GATE_INIT"],
                                    "backbone_residual_gate_type": CONFIG["BACKBONE_RESIDUAL_GATE_TYPE"],
                                    "variate_anchor_map_path": variate_anchor_map_path,
                                    "local_temporal_branch": CONFIG["LOCAL_TEMPORAL_BRANCH"],
                                    "local_temporal_init": CONFIG["LOCAL_TEMPORAL_INIT"],
                                    "local_temporal_gate_init": CONFIG["LOCAL_TEMPORAL_GATE_INIT"],
                                    "local_temporal_rank": CONFIG["LOCAL_TEMPORAL_RANK"],
                                    "zero_init_projector": CONFIG["ZERO_INIT_PROJECTOR"],
                                    "skip_backbone": CONFIG["SKIP_BACKBONE"],
                                    "output_calibration": CONFIG["OUTPUT_CALIBRATION"],
                                    "selection_id": CONFIG["SELECTION_ID"],
                                    "selected_recipe": CONFIG["SELECTED_RECIPE"],
                                    "manual_override": CONFIG["MANUAL_OVERRIDE"],
                                    "skip_test_eval": CONFIG.get("SKIP_TEST_EVAL", False),
                                    "skip_epoch_test_eval": CONFIG.get("SKIP_EPOCH_TEST_EVAL", False),
                                    "sparse_compress_topk": CONFIG["SPARSE_COMPRESS_TOPK"],
                                    "sparse_expand_topk": CONFIG["SPARSE_EXPAND_TOPK"],
                                    "skip_reason": k_spec.get("skip_reason", ""),
                                }
                                run.update(runtime_fields)
                                runs.append(run)
    return runs


def normalize_float(value):
    try:
        value = float(value)
    except Exception:
        value = 0.0
    if math.isnan(value):
        value = 0.0
    return round(value, 12)


def row_float(row, name, default):
    value = row.get(name, "")
    if value in ("", None, "nan", "NaN", "None"):
        return default
    return float(value)


def run_key(run):
    k_spec = run["k_spec"]
    return (
        run["dataset"],
        int(run["pred_len"]),
        int(k_spec["reduced_variate_k"]),
        normalize_float(k_spec["target_k_ratio"]),
        int(k_spec["target_k_value"]),
        k_spec["k_selection_mode"],
        k_spec.get("k_ratio_denominator", "target" if k_spec["k_selection_mode"] == "ratio" else ""),
        run["method"],
        run["variate_expansion_type"],
        run.get("variate_decode_stage", "feature") or "feature",
        normalize_float(run["orthogonal_loss_weight"]),
        normalize_float(run["reconstruction_loss_weight"]),
        normalize_float(run["coverage_loss_weight"]),
        normalize_float(run["assignment_entropy_loss_weight"]),
        normalize_float(run.get("wcomp_entropy_loss_weight", 0.0)),
        normalize_float(run.get("group_attention_entropy_loss_weight", 0.0)),
        normalize_float(run.get("group_attention_entropy_target", 0.35)),
        normalize_float(run["linear_weight_l2_loss_weight"]),
        normalize_float(run["decoder_init_l2_loss_weight"]),
        normalize_float(run["linear_coverage_loss_weight"]),
        normalize_float(run["biorthogonal_loss_weight"]),
        normalize_float(run["mae_loss_weight"]),
        str(run.get("forecast_loss_type", "mse")),
        normalize_float(run.get("huber_delta", 1.0)),
        normalize_float(run.get("linear_entropy_loss_weight", 0.0)),
        normalize_float(run.get("linear_cosine_loss_weight", 0.0)),
        normalize_float(run.get("linear_decoder_coverage_loss_weight", 0.0)),
        normalize_float(run.get("support_overlap_loss_weight", 0.0)),
        bool(run.get("use_cycle_slot_loss", False)),
        normalize_float(run.get("cycle_loss_weight", 0.0)),
        normalize_float(run.get("cycle_warmup_ratio", 0.0)),
        int(run.get("cycle_topr", 0)),
        normalize_float(run.get("cycle_topr_multiplier", 1.0)),
        run.get("cycle_b_norm", "row_l1") or "row_l1",
        normalize_float(run.get("cycle_eps", 1e-8)),
        bool(run.get("export_slot_diagnostics", False)),
        int(run.get("variate_token_split_factor", 1)),
        run.get("local_temporal_branch", "none"),
        run.get("local_temporal_init", "persistence"),
        normalize_float(run.get("local_temporal_gate_init", 1.0)),
        int(run.get("local_temporal_rank", 4)),
        bool(run.get("zero_init_projector", False)),
        bool(run.get("skip_backbone", False)),
        run.get("output_calibration", "none") or "none",
        int(run.get("lowrank_reducer_rank", 8)),
        normalize_float(run.get("decoder_residual_gate_init", 0.0)),
        normalize_float(run.get("expansion_temperature", 1.0)),
        int(run.get("expansion_topk", 0)),
        run.get("wcomp_normalization", "softmax") or "softmax",
        normalize_float(run.get("entmax_alpha", 1.5)),
        int(run.get("sparse_compress_topk", 0)),
        int(run.get("sparse_expand_topk", 0)),
        normalize_float(run.get("backbone_residual_gate_init", 1.0)),
        run.get("backbone_residual_gate_type", "scalar"),
        run.get("variate_anchor_map_path", "") or "",
        run.get("selection_id", "") or "",
        run.get("selected_recipe", "") or "",
        str(run.get("manual_override", "0")),
        "val" if bool(run.get("skip_test_eval", False)) else "test",
        int(run.get("batch_size", RUN_PY_DEFAULTS["batch_size"])),
        normalize_float(run.get("learning_rate", RUN_PY_DEFAULTS["learning_rate"])),
        int(run.get("train_epochs", RUN_PY_DEFAULTS["train_epochs"])),
        int(run.get("patience", RUN_PY_DEFAULTS["patience"])),
        int(run.get("num_workers", RUN_PY_DEFAULTS["num_workers"])),
        int(run["seed"]),
    )


def completed_keys_from_csv(result_csv):
    path = Path(result_csv)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return set()
    keys = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                status = row.get("status", "success") or "success"
                if status != "success":
                    continue
                keys.add((
                    row["dataset"],
                    int(row["pred_len"]),
                    int(row["reduced_variate_k"]),
                    normalize_float(row["target_k_ratio"]),
                    int(row_float(row, "target_k_value", 0)),
                    row["k_selection_mode"],
                    row.get(
                        "k_ratio_denominator",
                        "target" if row.get("k_selection_mode") == "ratio" else "",
                    ) or ("target" if row.get("k_selection_mode") == "ratio" else ""),
                    row["variate_reduction_type"],
                    row.get("variate_expansion_type", "transpose") or "transpose",
                    row.get("variate_decode_stage", "feature") or "feature",
                    normalize_float(row_float(row, "orthogonal_loss_weight", 0.0)),
                    normalize_float(row_float(row, "reconstruction_loss_weight", 0.0)),
                    normalize_float(row_float(row, "coverage_loss_weight", 0.0)),
                    normalize_float(row_float(row, "assignment_entropy_loss_weight", 0.0)),
                    normalize_float(row_float(row, "wcomp_entropy_loss_weight", 0.0)),
                    normalize_float(row_float(row, "group_attention_entropy_loss_weight", 0.0)),
                    normalize_float(row_float(row, "group_attention_entropy_target", 0.35)),
                    normalize_float(row_float(row, "linear_weight_l2_loss_weight", 0.0)),
                    normalize_float(row_float(row, "decoder_init_l2_loss_weight", 0.0)),
                    normalize_float(row_float(row, "linear_coverage_loss_weight", 0.0)),
                    normalize_float(row_float(row, "biorthogonal_loss_weight", 0.0)),
                    normalize_float(row_float(row, "mae_loss_weight", 0.0)),
                    str(row.get("forecast_loss_type", "mse") or "mse"),
                    normalize_float(row_float(row, "huber_delta", 1.0)),
                    normalize_float(row_float(row, "linear_entropy_loss_weight", 0.0)),
                    normalize_float(row_float(row, "linear_cosine_loss_weight", 0.0)),
                    normalize_float(row_float(row, "linear_decoder_coverage_loss_weight", 0.0)),
                    normalize_float(row_float(row, "support_overlap_loss_weight", 0.0)),
                    str(row.get("use_cycle_slot_loss", "False")).lower() in {"true", "1", "yes"},
                    normalize_float(row_float(row, "cycle_loss_weight", 0.0)),
                    normalize_float(row_float(row, "cycle_warmup_ratio", 0.0)),
                    int(row_float(row, "cycle_topr", 0)),
                    normalize_float(row_float(row, "cycle_topr_multiplier", 1.0)),
                    row.get("cycle_b_norm", "row_l1") or "row_l1",
                    normalize_float(row_float(row, "cycle_eps", 1e-8)),
                    str(row.get("export_slot_diagnostics", "False")).lower() in {"true", "1", "yes"},
                    int(row_float(row, "variate_token_split_factor", 1)),
                    row.get("local_temporal_branch", "none") or "none",
                    row.get("local_temporal_init", "persistence") or "persistence",
                    normalize_float(row_float(row, "local_temporal_gate_init", 1.0)),
                    int(row_float(row, "local_temporal_rank", 4)),
                    str(row.get("zero_init_projector", "False")).lower() in {"true", "1", "yes"},
                    str(row.get("skip_backbone", "False")).lower() in {"true", "1", "yes"},
                    row.get("output_calibration", "none") or "none",
                    int(row_float(row, "lowrank_reducer_rank", 8)),
                    normalize_float(row_float(row, "decoder_residual_gate_init", 0.0)),
                    normalize_float(row_float(row, "expansion_temperature", 1.0)),
                    int(row_float(row, "expansion_topk", 0)),
                    row.get("normalization", row.get("wcomp_normalization", "softmax")) or "softmax",
                    normalize_float(row_float(row, "entmax_alpha", 1.5)),
                    int(row_float(row, "sparse_compress_topk", 0)),
                    int(row_float(row, "sparse_expand_topk", 0)),
                    normalize_float(row_float(row, "backbone_residual_gate_init", 1.0)),
                    row.get("backbone_residual_gate_type", "scalar") or "scalar",
                    row.get("variate_anchor_map_path", "") or "",
                    row.get("selection_id", "") or "",
                    row.get("selected_recipe", "") or "",
                    str(row.get("manual_override", "0")),
                    row.get("eval_split", "test") or "test",
                    int(row_float(row, "batch_size", -1)),
                    normalize_float(row_float(row, "learning_rate", -1.0)),
                    int(row_float(row, "train_epochs", -1)),
                    int(row_float(row, "patience", -1)),
                    int(row_float(row, "num_workers", -1)),
                    int(row_float(row, "seed", 2021)),
                ))
            except Exception:
                continue
    return keys


def completed_keys(result_csv, reference_csvs=None):
    keys = set()
    for csv_path in [result_csv] + list(reference_csvs or []):
        keys.update(completed_keys_from_csv(csv_path))
    return keys


def print_dry_run(runs):
    baselines = [run for run in runs if run["method"] == "none"]
    reducers = [run for run in runs if run["method"] != "none"]
    if reducers:
        variant_text = "; ".join(
            "{}: ortho={} rec={} cov={} ent={} mae={}".format(
                name or "default",
                fmt_weight(weights["orthogonal_loss_weight"]),
                fmt_weight(weights["reconstruction_loss_weight"]),
                fmt_weight(weights["coverage_loss_weight"]),
                fmt_weight(weights["assignment_entropy_loss_weight"]),
                fmt_weight(weights["mae_loss_weight"]),
            )
            for name, weights in configured_loss_variants()
        )
        print(
            "Reducer options: expansions={} decode_stage={} temp={} topk={} dres={} local_branch={} local_rank={} zero_proj={} skip_bb={} variants=[{}]".format(
                ",".join(expansion_types()),
                CONFIG["VARIATE_DECODE_STAGE"],
                CONFIG["EXPANSION_TEMPERATURE"],
                CONFIG["EXPANSION_TOPK"],
                fmt_weight(CONFIG["DECODER_RESIDUAL_GATE_INIT"]),
                CONFIG["LOCAL_TEMPORAL_BRANCH"],
                CONFIG["LOCAL_TEMPORAL_RANK"],
                CONFIG["ZERO_INIT_PROJECTOR"],
                CONFIG["SKIP_BACKBONE"],
                variant_text,
            )
        )
        print()

    if baselines:
        print("[BASELINE]")
        for run in baselines:
            print("{} pred_len={} baseline seed={}".format(run["dataset"], run["pred_len"], run["seed"]))
        print()

    grouped = defaultdict(list)
    for run in reducers:
        grouped[(run["dataset"], run["pred_len"])].append(run)

    for dataset in CONFIG["DATASETS"]:
        for pred_len in CONFIG["PRED_LENS"]:
            group = grouped.get((dataset, pred_len), [])
            if not group:
                continue
            print("[{} / pred_len={}]".format(dataset, pred_len))
            for run in group:
                k_spec = run["k_spec"]
                if k_spec["k_selection_mode"] == "ratio":
                    base = (
                        "method={} split={} ratio_basis={} ratio={:.2f} K={} actual_ratio={:.4f} "
                        "source_ratio={:.4f} variant={} decode={} local={} lrank={} zero_proj={} skip_bb={} dres={} ortho={} rec={} mae={} seed={}"
                    ).format(
                        run["method"],
                        run["variate_token_split_factor"],
                        k_spec.get("k_ratio_denominator", "target") or "target",
                        k_spec["target_k_ratio"],
                        k_spec["reduced_variate_k"],
                        k_spec["actual_k_ratio"],
                        k_spec["reduced_variate_k"] / (
                            DATASET_SPECS[run["dataset"]]["num_variates"] * run["variate_token_split_factor"]
                        ),
                        run.get("loss_variant", "") or "default",
                        run.get("variate_decode_stage", "feature"),
                        run.get("local_temporal_branch", "none"),
                        run.get("local_temporal_rank", 4),
                        run.get("zero_init_projector", False),
                        run.get("skip_backbone", False),
                        fmt_weight(run.get("decoder_residual_gate_init", 0.0)),
                        fmt_weight(run["orthogonal_loss_weight"]),
                        fmt_weight(run["reconstruction_loss_weight"]),
                        fmt_weight(run["mae_loss_weight"]),
                        run["seed"],
                    )
                else:
                    base = (
                        "method={} split={} K={} actual_ratio={:.4f} variant={} decode={} local={} lrank={} zero_proj={} skip_bb={} dres={} ortho={} rec={} mae={} seed={}".format(
                            run["method"],
                            run["variate_token_split_factor"],
                            k_spec["reduced_variate_k"],
                            k_spec["actual_k_ratio"],
                            run.get("loss_variant", "") or "default",
                            run.get("variate_decode_stage", "feature"),
                            run.get("local_temporal_branch", "none"),
                            run.get("local_temporal_rank", 4),
                            run.get("zero_init_projector", False),
                            run.get("skip_backbone", False),
                            fmt_weight(run.get("decoder_residual_gate_init", 0.0)),
                            fmt_weight(run["orthogonal_loss_weight"]),
                            fmt_weight(run["reconstruction_loss_weight"]),
                            fmt_weight(run["mae_loss_weight"]),
                            run["seed"],
                        )
                    )
                if run.get("variate_anchor_map_path"):
                    base += " amap={}".format(run["variate_anchor_map_path"])
                if run["k_spec"].get("skip"):
                    base += " SKIP={}".format(run["k_spec"].get("skip_reason", "skipped"))
                print(base)
        print()


def dedupe_runs(runs):
    deduped = []
    seen = set()
    for run in runs:
        key = run_key(run)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(run)
    return deduped


def append_failed_row(run, command, returncode, error, gpu):
    print("결과 CSV에는 실패 row를 기록하지 않습니다. returncode={}".format(returncode), flush=True)


def append_skipped_row(run, gpu):
    k_spec = run["k_spec"]
    spec = run["spec"]
    source_tokens = spec["num_variates"] * int(run.get("variate_token_split_factor", 1))
    row = {
        "dataset": run["dataset"],
        "pred_len": run["pred_len"],
        "selection_id": run.get("selection_id", ""),
        "selected_recipe": run.get("selected_recipe", ""),
        "manual_override": run.get("manual_override", "0"),
        "variate_reduction_type": run["method"],
        "variate_expansion_type": run["variate_expansion_type"],
        "expansion_weight_source": expansion_weight_source(run["method"], run["variate_expansion_type"]),
        "variate_decode_stage": run.get("variate_decode_stage", "feature"),
        "loss_variant": run.get("loss_variant", ""),
        "compute_reducer_aux_losses": "",
        "variate_token_split_factor": run.get("variate_token_split_factor", 1),
        "num_variates": spec["num_variates"],
        "target_variate_tokens": spec["num_variates"],
        "source_variate_tokens": source_tokens,
        "reduced_variate_k": k_spec["reduced_variate_k"],
        "reduced_variate_tokens": "",
        "target_k_ratio": k_spec["target_k_ratio"],
        "actual_k_ratio": k_spec["actual_k_ratio"],
        "source_k_ratio": float(k_spec["reduced_variate_k"]) / float(source_tokens) if source_tokens else "",
        "logical_k": k_spec["reduced_variate_k"],
        "executed_k": "",
        "logical_k_ratio": k_spec["actual_k_ratio"],
        "executed_k_ratio": "",
        "k_budget_max": k_spec["k_budget_max"],
        "k_budget_exception": k_spec["k_budget_exception"],
        "lowrank_reducer_rank": run.get("lowrank_reducer_rank", ""),
        "decoder_residual_gate_init": run.get("decoder_residual_gate_init", ""),
        "backbone_residual_gate_init": run.get("backbone_residual_gate_init", ""),
        "backbone_residual_gate_type": run.get("backbone_residual_gate_type", ""),
        "k_selection_mode": k_spec["k_selection_mode"],
        "k_ratio_denominator": k_spec.get("k_ratio_denominator", ""),
        "target_k_value": k_spec["target_k_value"],
        "eval_split": "val" if bool(run.get("skip_test_eval", False)) else "test",
        "orthogonal_loss_weight": run["orthogonal_loss_weight"],
        "reconstruction_loss_weight": run["reconstruction_loss_weight"],
        "coverage_loss_weight": run["coverage_loss_weight"],
        "assignment_entropy_loss_weight": run["assignment_entropy_loss_weight"],
        "wcomp_entropy_loss_weight": run.get("wcomp_entropy_loss_weight", 0.0),
        "group_attention_entropy_loss_weight": run.get("group_attention_entropy_loss_weight", 0.0),
        "group_attention_entropy_target": run.get("group_attention_entropy_target", 0.35),
        "linear_coverage_loss_weight": run["linear_coverage_loss_weight"],
        "biorthogonal_loss_weight": run["biorthogonal_loss_weight"],
        "linear_weight_l2_loss_weight": run["linear_weight_l2_loss_weight"],
        "decoder_init_l2_loss_weight": run["decoder_init_l2_loss_weight"],
        "linear_entropy_loss_weight": run.get("linear_entropy_loss_weight", 0.0),
        "linear_cosine_loss_weight": run.get("linear_cosine_loss_weight", 0.0),
        "linear_decoder_coverage_loss_weight": run.get("linear_decoder_coverage_loss_weight", 0.0),
        "support_overlap_loss_weight": run.get("support_overlap_loss_weight", 0.0),
        "use_cycle_slot_loss": run.get("use_cycle_slot_loss", False),
        "cycle_loss_weight": run.get("cycle_loss_weight", 0.0),
        "cycle_warmup_ratio": run.get("cycle_warmup_ratio", 0.0),
        "cycle_topr": run.get("cycle_topr", 0),
        "cycle_topr_multiplier": run.get("cycle_topr_multiplier", 1.0),
        "cycle_b_norm": run.get("cycle_b_norm", "row_l1"),
        "cycle_eps": run.get("cycle_eps", 1e-8),
        "export_slot_diagnostics": run.get("export_slot_diagnostics", False),
        "mae_loss_weight": run["mae_loss_weight"],
        "forecast_loss_type": run.get("forecast_loss_type", "mse"),
        "huber_delta": run.get("huber_delta", 1.0),
        "expansion_temperature": run["expansion_temperature"],
        "expansion_topk": run["expansion_topk"],
        "normalization": run.get("wcomp_normalization", "softmax"),
        "entmax_alpha": run.get("entmax_alpha", 1.5),
        "sparse_compress_topk": run.get("sparse_compress_topk", 0),
        "sparse_expand_topk": run.get("sparse_expand_topk", 0),
        "local_temporal_branch": run.get("local_temporal_branch", ""),
        "local_temporal_init": run.get("local_temporal_init", ""),
        "local_temporal_gate_init": run.get("local_temporal_gate_init", ""),
        "local_temporal_rank": run.get("local_temporal_rank", ""),
        "zero_init_projector": run.get("zero_init_projector", ""),
        "skip_backbone": run.get("skip_backbone", ""),
        "output_calibration": run.get("output_calibration", ""),
        "status": "skipped",
        "skip_reason": k_spec.get("skip_reason", "skipped"),
        "error": k_spec.get("skip_reason", "skipped"),
        "returncode": "",
        "gpu": gpu,
        "data": spec["data"],
        "root_path": spec["root_path"],
        "data_path": spec["data_path"],
        "target": run.get("target", ""),
        "freq": spec.get("freq", ""),
        "seed": run["seed"],
        "itr": 0,
        "model": "iTransformer",
        "model_id": run["model_id"],
        "setting": "",
        "seq_len": spec["seq_len"],
        "label_len": 48,
        "e_layers": spec["e_layers"],
        "d_model": spec["d_model"],
        "d_ff": spec["d_ff"],
        "batch_size": run.get("batch_size", ""),
        "learning_rate": run.get("learning_rate", ""),
        "train_epochs": run.get("train_epochs", ""),
        "patience": run.get("patience", ""),
        "num_workers": run.get("num_workers", ""),
        "command": "",
    }
    append_csv_row(run["result_csv"], row)


def run_command(command, env):
    print(quote_cmd(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail = deque(maxlen=40)
    for line in process.stdout:
        print(line, end="")
        tail.append(line.rstrip())
    returncode = process.wait()
    return returncode, "\n".join(tail)


def progress_description(run):
    k_spec = run["k_spec"]
    if k_spec["k_selection_mode"] == "ratio":
        k_text = "ratio_basis={} ratio={:.2f} K={}".format(
            k_spec.get("k_ratio_denominator", "target") or "target",
            k_spec["target_k_ratio"],
            k_spec["reduced_variate_k"],
        )
    else:
        k_text = "K={}".format(k_spec["reduced_variate_k"])
    return (
        "dataset={dataset} pred_len={pred_len} method={method} "
        "expansion={expansion} decode={decode} variant={variant} local={local} split={split} "
        "{k_text} rec={rec} seed={seed}"
    ).format(
        dataset=run["dataset"],
        pred_len=run["pred_len"],
        method=run["method"],
        expansion=run["variate_expansion_type"],
        decode=run.get("variate_decode_stage", "feature"),
        variant=run.get("loss_variant", "") or "default",
        local=run.get("local_temporal_branch", "none"),
        split=run.get("variate_token_split_factor", 1),
        k_text=k_text,
        rec=fmt_weight(run.get("reconstruction_loss_weight", 0.0)),
        seed=run["seed"],
    )


def config_wandb_tags():
    tags = CONFIG.get("WANDB_TAGS", [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    else:
        tags = [str(tag) for tag in tags]
    split_factor = variate_token_split_factor()
    if split_factor > 1:
        tags.append("split{}".format(split_factor))
    return ",".join(tags)


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--result_csv", type=str, default="./results/attention_variate_reduction_results.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument(
        "--skip_reference_csv",
        type=str,
        action="append",
        default=[],
        help="Additional CSV to use for skip_completed without writing to it. Can be passed multiple times.",
    )
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--weight_heatmap_dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--pred_lens", type=int, nargs="+", default=None)
    parser.add_argument("--k_ratios", type=float, nargs="+", default=None)
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=None,
        help="Explicit absolute K values for any selected dataset. Values are still clamped by strict K/V<0.30.",
    )
    parser.add_argument(
        "--fixed_k_values",
        type=int,
        nargs="+",
        default=None,
        help="Additional fixed K values to run alongside ratio K values. Values with K > V_eff are recorded as skipped.",
    )
    parser.add_argument(
        "--no_ratio_k",
        action="store_true",
        help="Use only fixed/absolute K values and do not generate ratio K settings.",
    )
    parser.add_argument(
        "--no_fixed_k",
        action="store_true",
        help="Do not generate preset fixed K settings. Useful for ratio-only queue shards.",
    )
    parser.add_argument(
        "--k_ratio_denominator",
        choices=["target", "source", "both"],
        default=None,
        help="Use target V, source V_eff, or both when converting K ratios to integer K.",
    )
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument(
        "--expansion_types",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "transpose",
            "column_normalized",
            "sharpened_column_normalized",
            "topk_column_normalized",
            "fixed_assignment",
            "fixed_assignment_scalar_residual",
            "fixed_masked_linear",
            "fixed_masked_scalar_residual",
            "fixed_masked_variate_residual",
            "masked_linear",
            "masked_softmax",
            "masked_softmax_scalar_residual",
            "masked_linear_scalar_residual",
            "masked_softmax_variate_residual",
            "slot_learned_linear",
            "id_query_decoder",
            "id_query_scalar_residual",
            "id_query_fixed_scalar_residual",
            "id_topk_decoder",
            "id_topk_scalar_residual",
            "id_topk_fixed_scalar_residual",
            "query_decoder",
            "query_decoder_scalar_residual",
            "query_decoder_variate_residual",
            "token_query_decoder",
        ],
    )
    parser.add_argument("--decoder_residual_gate_init", type=float, default=None)
    parser.add_argument("--backbone_residual_gate_init", type=float, default=None)
    parser.add_argument("--backbone_residual_gate_type", type=str, default=None, choices=["scalar", "variate"])
    parser.add_argument("--variate_decode_stage", type=str, default=None, choices=["feature", "forecast"])
    parser.add_argument("--variate_anchor_map_path", type=str, default=None)
    parser.add_argument(
        "--variate_anchor_map_template",
        type=str,
        default=None,
        help="Per-run anchor map template. Supports {dataset}, {pred_len}, {k}, {reduced_k}, {num_variates}, {target_k_ratio}, {actual_k_ratio}.",
    )
    parser.add_argument("--train_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--split_factor", type=int, default=None)
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--no_baseline", action="store_true")
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="Generate only iTransformer baseline runs for the selected datasets/horizons/seeds.",
    )
    parser.add_argument("--local_temporal_branch", type=str, default=None, choices=["none", "linear", "nlinear", "nlinear_affine", "nlinear_lowrank", "nlinear_lowrank_affine", "nlinear_decomp", "nlinear_decomp_affine", "anchor_residual_nlinear", "anchor_residual_nlinear_affine", "anchor_delta_nlinear", "anchor_delta_nlinear_affine", "anchor_mask_delta_nlinear", "anchor_mask_delta_nlinear_affine", "anchor_group_nlinear", "anchor_group_nlinear_affine", "nlinear_group", "nlinear_group_affine", "lowrank_linear", "persistence", "persistence_gate"])
    parser.add_argument("--local_temporal_init", type=str, default=None, choices=["persistence", "zero"])
    parser.add_argument("--local_temporal_gate_init", type=float, default=None)
    parser.add_argument("--local_temporal_rank", type=int, default=None)
    parser.add_argument("--zero_init_projector", action="store_true")
    parser.add_argument("--skip_backbone", action="store_true")
    parser.add_argument("--output_calibration", type=str, default=None, choices=["none", "variate_affine"])
    parser.add_argument("--selection_id", type=str, default=None)
    parser.add_argument("--selected_recipe", type=str, default=None)
    parser.add_argument("--manual_override", type=str, default=None)
    parser.add_argument("--lowrank_reducer_rank", type=int, default=None)
    parser.add_argument("--expansion_temperature", type=float, default=None)
    parser.add_argument("--expansion_topk", type=int, default=None)
    parser.add_argument("--wcomp_normalization", type=str, default=None, choices=["softmax", "entmax15"])
    parser.add_argument("--entmax_alpha", type=float, default=None)
    parser.add_argument(
        "--sparse_compress_topk",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sparse_expand_topk",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--skip_test_eval", action="store_true")
    parser.add_argument(
        "--skip_epoch_test_eval",
        action="store_true",
        help="Skip per-epoch test-set validation but still run the final test evaluation and heatmap export.",
    )
    parser.add_argument("--no_enforce_k_budget", action="store_true")
    parser.add_argument("--orthogonal_loss_weight", type=float, default=None)
    parser.add_argument("--reconstruction_loss_weight", type=float, default=None)
    parser.add_argument("--coverage_loss_weight", type=float, default=None)
    parser.add_argument("--assignment_entropy_loss_weight", type=float, default=None)
    parser.add_argument("--wcomp_entropy_loss_weight", type=float, default=None)
    parser.add_argument("--group_attention_entropy_loss_weight", type=float, default=None)
    parser.add_argument("--group_attention_entropy_target", type=float, default=None)
    parser.add_argument("--linear_coverage_loss_weight", type=float, default=None)
    parser.add_argument("--biorthogonal_loss_weight", type=float, default=None)
    parser.add_argument("--linear_weight_l2_loss_weight", type=float, default=None)
    parser.add_argument("--decoder_init_l2_loss_weight", type=float, default=None)
    parser.add_argument("--linear_entropy_loss_weight", type=float, default=None)
    parser.add_argument("--linear_cosine_loss_weight", type=float, default=None)
    parser.add_argument("--linear_decoder_coverage_loss_weight", type=float, default=None)
    parser.add_argument("--support_overlap_loss_weight", type=float, default=None)
    parser.add_argument("--use_cycle_slot_loss", action="store_true")
    parser.add_argument("--cycle_loss_weight", type=float, default=None)
    parser.add_argument("--cycle_warmup_ratio", type=float, default=None)
    parser.add_argument("--cycle_topr", type=int, default=None)
    parser.add_argument("--cycle_topr_multiplier", type=float, default=None)
    parser.add_argument("--cycle_b_norm", type=str, default=None, choices=["row_l1"])
    parser.add_argument("--cycle_eps", type=float, default=None)
    parser.add_argument("--export_slot_diagnostics", action="store_true")
    parser.add_argument("--mae_loss_weight", type=float, default=None)
    parser.add_argument("--forecast_loss_type", type=str, default=None, choices=["mse", "smooth_l1", "huber"])
    parser.add_argument("--huber_delta", type=float, default=None)
    parser.add_argument(
        "--loss_variant_names",
        type=str,
        nargs="+",
        default=None,
        help="Run only the named loss variants from the selected sweep preset, preserving loss_variant names in CSVs.",
    )
    parser.add_argument(
        "--sweep_preset",
        type=str,
        default="",
        choices=[
            "",
            "traffic_perf_short",
            "target_perf_short",
        ],
        help="Apply a predefined loss sweep CONFIG override.",
    )
    return parser.parse_args()


def wandb_settings(args):
    use_wandb = bool(CONFIG.get("USE_WANDB", True))
    if args.use_wandb:
        use_wandb = True
    if args.no_wandb:
        use_wandb = False
    return {
        "use_wandb": use_wandb,
        "wandb_project": args.wandb_project if args.wandb_project is not None else CONFIG.get("WANDB_PROJECT", "iTransformer-variate-reduction"),
        "wandb_entity": args.wandb_entity if args.wandb_entity is not None else CONFIG.get("WANDB_ENTITY", ""),
        "wandb_mode": args.wandb_mode if args.wandb_mode is not None else CONFIG.get("WANDB_MODE", "online"),
        "wandb_group": args.wandb_group if args.wandb_group is not None else CONFIG.get("WANDB_GROUP", ""),
        "wandb_tags": args.wandb_tags if args.wandb_tags is not None else config_wandb_tags(),
    }


def main():
    args = parse_args()
    apply_sweep_preset(args.sweep_preset)
    if args.datasets is not None:
        CONFIG["DATASETS"] = args.datasets
    if args.pred_lens is not None:
        CONFIG["PRED_LENS"] = args.pred_lens
    if args.k_ratios is not None:
        CONFIG["K_RATIOS"] = args.k_ratios
    if args.k_values is not None:
        CONFIG["K_VALUES"] = args.k_values
    if args.fixed_k_values is not None:
        CONFIG["FIXED_K_VALUES"] = args.fixed_k_values
    if args.no_ratio_k:
        CONFIG["INCLUDE_RATIO_K"] = False
    if args.no_fixed_k:
        CONFIG["FIXED_K_VALUES"] = []
    if args.k_ratio_denominator is not None:
        CONFIG["K_RATIO_DENOMINATOR"] = args.k_ratio_denominator
    if args.methods is not None:
        CONFIG["METHODS"] = args.methods
    if args.expansion_types is not None:
        CONFIG["VARIATE_EXPANSION_TYPES"] = args.expansion_types
    if args.decoder_residual_gate_init is not None:
        CONFIG["DECODER_RESIDUAL_GATE_INIT"] = args.decoder_residual_gate_init
    if args.backbone_residual_gate_init is not None:
        CONFIG["BACKBONE_RESIDUAL_GATE_INIT"] = args.backbone_residual_gate_init
    if args.backbone_residual_gate_type is not None:
        CONFIG["BACKBONE_RESIDUAL_GATE_TYPE"] = args.backbone_residual_gate_type
    if args.variate_decode_stage is not None:
        CONFIG["VARIATE_DECODE_STAGE"] = args.variate_decode_stage
    if args.expansion_temperature is not None:
        CONFIG["EXPANSION_TEMPERATURE"] = args.expansion_temperature
    if args.expansion_topk is not None:
        CONFIG["EXPANSION_TOPK"] = args.expansion_topk
    if args.wcomp_normalization is not None:
        CONFIG["WCOMP_NORMALIZATION"] = args.wcomp_normalization
    if args.entmax_alpha is not None:
        CONFIG["ENTMAX_ALPHA"] = args.entmax_alpha
    if args.sparse_compress_topk is not None:
        CONFIG["SPARSE_COMPRESS_TOPK"] = args.sparse_compress_topk
    if args.sparse_expand_topk is not None:
        CONFIG["SPARSE_EXPAND_TOPK"] = args.sparse_expand_topk
    if args.variate_anchor_map_path is not None:
        CONFIG["VARIATE_ANCHOR_MAP_PATH"] = args.variate_anchor_map_path
    if args.variate_anchor_map_template is not None:
        CONFIG["VARIATE_ANCHOR_MAP_TEMPLATE"] = args.variate_anchor_map_template
    if args.train_epochs is not None:
        CONFIG["TRAIN_EPOCHS"] = args.train_epochs
    if args.patience is not None:
        CONFIG["PATIENCE"] = args.patience
    if args.num_workers is not None:
        CONFIG["NUM_WORKERS"] = args.num_workers
    if args.split_factor is not None:
        CONFIG["VARIATE_TOKEN_SPLIT_FACTOR"] = args.split_factor
    if args.run_baseline:
        CONFIG["RUN_BASELINE"] = True
    if args.no_baseline:
        CONFIG["RUN_BASELINE"] = False
    if args.baseline_only:
        CONFIG["RUN_BASELINE"] = True
        CONFIG["METHODS"] = []
    if args.local_temporal_branch is not None:
        CONFIG["LOCAL_TEMPORAL_BRANCH"] = args.local_temporal_branch
    if args.local_temporal_init is not None:
        CONFIG["LOCAL_TEMPORAL_INIT"] = args.local_temporal_init
    if args.local_temporal_gate_init is not None:
        CONFIG["LOCAL_TEMPORAL_GATE_INIT"] = args.local_temporal_gate_init
    if args.local_temporal_rank is not None:
        CONFIG["LOCAL_TEMPORAL_RANK"] = args.local_temporal_rank
    if args.zero_init_projector:
        CONFIG["ZERO_INIT_PROJECTOR"] = True
    if args.skip_backbone:
        CONFIG["SKIP_BACKBONE"] = True
    if args.output_calibration is not None:
        CONFIG["OUTPUT_CALIBRATION"] = args.output_calibration
    if args.selection_id is not None:
        CONFIG["SELECTION_ID"] = args.selection_id
    if args.selected_recipe is not None:
        CONFIG["SELECTED_RECIPE"] = args.selected_recipe
    if args.manual_override is not None:
        CONFIG["MANUAL_OVERRIDE"] = args.manual_override
    if args.lowrank_reducer_rank is not None:
        CONFIG["LOWRANK_REDUCER_RANK"] = args.lowrank_reducer_rank
    if args.skip_test_eval:
        CONFIG["SKIP_TEST_EVAL"] = True
    if args.skip_epoch_test_eval:
        CONFIG["SKIP_EPOCH_TEST_EVAL"] = True
    if args.use_cycle_slot_loss:
        CONFIG["USE_CYCLE_SLOT_LOSS"] = True
    if args.cycle_loss_weight is not None:
        CONFIG["CYCLE_LOSS_WEIGHT"] = args.cycle_loss_weight
    if args.cycle_warmup_ratio is not None:
        CONFIG["CYCLE_WARMUP_RATIO"] = args.cycle_warmup_ratio
    if args.cycle_topr is not None:
        CONFIG["CYCLE_TOPR"] = args.cycle_topr
    if args.cycle_topr_multiplier is not None:
        CONFIG["CYCLE_TOPR_MULTIPLIER"] = args.cycle_topr_multiplier
    if args.cycle_b_norm is not None:
        CONFIG["CYCLE_B_NORM"] = args.cycle_b_norm
    if args.cycle_eps is not None:
        CONFIG["CYCLE_EPS"] = args.cycle_eps
    if args.export_slot_diagnostics:
        CONFIG["EXPORT_SLOT_DIAGNOSTICS"] = True
    if args.no_enforce_k_budget:
        CONFIG["ENFORCE_K_BUDGET"] = False
    loss_overrides = {
        "ORTHOGONAL_LOSS_WEIGHT": args.orthogonal_loss_weight,
        "RECONSTRUCTION_LOSS_WEIGHT": args.reconstruction_loss_weight,
        "COVERAGE_LOSS_WEIGHT": args.coverage_loss_weight,
        "ASSIGNMENT_ENTROPY_LOSS_WEIGHT": args.assignment_entropy_loss_weight,
        "WCOMP_ENTROPY_LOSS_WEIGHT": args.wcomp_entropy_loss_weight,
        "GROUP_ATTENTION_ENTROPY_LOSS_WEIGHT": args.group_attention_entropy_loss_weight,
        "GROUP_ATTENTION_ENTROPY_TARGET": args.group_attention_entropy_target,
        "LINEAR_COVERAGE_LOSS_WEIGHT": args.linear_coverage_loss_weight,
        "BIORTHOGONAL_LOSS_WEIGHT": args.biorthogonal_loss_weight,
        "LINEAR_WEIGHT_L2_LOSS_WEIGHT": args.linear_weight_l2_loss_weight,
        "DECODER_INIT_L2_LOSS_WEIGHT": args.decoder_init_l2_loss_weight,
        "LINEAR_ENTROPY_LOSS_WEIGHT": args.linear_entropy_loss_weight,
        "LINEAR_COSINE_LOSS_WEIGHT": args.linear_cosine_loss_weight,
        "LINEAR_DECODER_COVERAGE_LOSS_WEIGHT": args.linear_decoder_coverage_loss_weight,
        "SUPPORT_OVERLAP_LOSS_WEIGHT": args.support_overlap_loss_weight,
        "MAE_LOSS_WEIGHT": args.mae_loss_weight,
        "FORECAST_LOSS_TYPE": args.forecast_loss_type,
        "HUBER_DELTA": args.huber_delta,
    }
    provided_loss_overrides = {
        key: value for key, value in loss_overrides.items()
        if value is not None
    }
    removed_loss_overrides = {
        key: value for key, value in provided_loss_overrides.items()
        if key in {
            "LINEAR_COVERAGE_LOSS_WEIGHT",
            "BIORTHOGONAL_LOSS_WEIGHT",
            "LINEAR_WEIGHT_L2_LOSS_WEIGHT",
            "DECODER_INIT_L2_LOSS_WEIGHT",
            "LINEAR_ENTROPY_LOSS_WEIGHT",
            "LINEAR_COSINE_LOSS_WEIGHT",
            "LINEAR_DECODER_COVERAGE_LOSS_WEIGHT",
            "SUPPORT_OVERLAP_LOSS_WEIGHT",
        }
    }
    nonzero_removed_losses = [
        key for key, value in removed_loss_overrides.items()
        if value is not None and float(value) != 0.0
    ]
    if nonzero_removed_losses:
        raise RuntimeError(
            "generation-only loss overrides were removed; set these to 0 or drop them: {}".format(
                ", ".join(sorted(nonzero_removed_losses))
            )
        )
    for key in removed_loss_overrides:
        provided_loss_overrides.pop(key, None)
    if args.loss_variant_names is not None:
        if provided_loss_overrides:
            raise RuntimeError("--loss_variant_names cannot be combined with explicit loss weight overrides")
        variants = CONFIG.get("LOSS_VARIANTS") or []
        by_name = {str(variant.get("NAME", "")): variant for variant in variants}
        missing = [name for name in args.loss_variant_names if name not in by_name]
        if missing:
            raise RuntimeError(
                "Unknown loss variant(s): {}. Available: {}".format(
                    ", ".join(missing),
                    ", ".join(sorted(by_name)),
                )
            )
        CONFIG["LOSS_VARIANTS"] = [by_name[name] for name in args.loss_variant_names]
    if provided_loss_overrides:
        CONFIG.update(provided_loss_overrides)
        CONFIG["LOSS_VARIANTS"] = None
    removed_methods = {
        "mlp_generation",
        "mlp_generation_id",
        "mlp_lowrank_generation",
        "mlp_sparse_representative",
    }
    requested_removed = sorted(removed_methods.intersection(CONFIG["METHODS"]))
    if requested_removed:
        raise RuntimeError("Removed generation reducer(s) requested: {}".format(", ".join(requested_removed)))
    ensure_dataset_symlink()
    seeds = args.seeds if args.seeds is not None else CONFIG["SEEDS"]
    selected_datasets = CONFIG["DATASETS"]
    resolved_targets = validate_datasets(selected_datasets)
    wandb_config = wandb_settings(args)
    runs = dedupe_runs(generate_runs(args.result_csv, seeds))
    for run in runs:
        run["target"] = resolved_targets[run["dataset"]]
        if args.weight_heatmap_dir is not None:
            run["weight_heatmap_dir"] = args.weight_heatmap_dir
        run.update(wandb_config)

    if args.skip_completed:
        done = completed_keys(args.result_csv, args.skip_reference_csv)
        before = len(runs)
        runs = [run for run in runs if run_key(run) not in done]
        print("skip_completed: skipped {} completed runs".format(before - len(runs)))

    if args.max_runs is not None:
        runs = runs[:args.max_runs]

    if args.dry_run:
        print_dry_run(runs)
        print("Total runs: {}".format(len(runs)))
        return

    total_runs = len(runs)
    print("Planned runs after skip/max filtering: {}".format(total_runs), flush=True)
    if total_runs == 0:
        print("No runs to execute.", flush=True)
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-itformer")
    for run_index, run in enumerate(runs, start=1):
        print(
            "\n[RUN {}/{}] START {}".format(
                run_index,
                total_runs,
                progress_description(run),
            ),
            flush=True,
        )
        if run["k_spec"].get("skip"):
            append_skipped_row(run, args.gpu)
            print(
                "[RUN {}/{}] SKIPPED {}".format(
                    run_index,
                    total_runs,
                    run["k_spec"].get("skip_reason", "skipped"),
                ),
                flush=True,
            )
            continue
        command = build_command(run)
        returncode, output_tail = run_command(command, env)
        status = "SUCCESS" if returncode == 0 else "FAILED"
        print(
            "[RUN {}/{}] {} returncode={}".format(
                run_index,
                total_runs,
                status,
                returncode,
            ),
            flush=True,
        )
        if returncode != 0:
            error = output_tail[-2000:] if output_tail else "Process failed with return code {}".format(returncode)
            append_failed_row(run, command, returncode, error, args.gpu)


if __name__ == "__main__":
    main()
