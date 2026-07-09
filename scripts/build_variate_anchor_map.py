import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def strict_k_budget(num_variates):
    k_max = int(math.ceil(0.30 * int(num_variates))) - 1
    if k_max >= 1:
        return k_max, False
    return 1, True


def read_multivariate_csv(root_path, data_path, features, target):
    path = Path(root_path) / data_path
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip().split(",")
    try:
        [float(item) for item in first]
        has_header = False
    except ValueError:
        has_header = True

    if not has_header:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        columns = [str(index) for index in range(data.shape[1])]
        return data, columns

    df_raw = pd.read_csv(path)
    date_cols = [col for col in df_raw.columns if str(col).lower() in {"date", "datetime"}]
    if date_cols and target in df_raw.columns and features in {"M", "MS"}:
        cols = list(df_raw.columns)
        cols.remove(target)
        for date_col in date_cols:
            cols.remove(date_col)
        ordered = date_cols[:1] + cols + [target]
        df_raw = df_raw[ordered]
        df_data = df_raw[df_raw.columns[1:]]
    else:
        if date_cols:
            df_raw = df_raw.drop(columns=date_cols)
        if features == "S" and target in df_raw.columns:
            df_data = df_raw[[target]]
        else:
            df_data = df_raw
    df_data = df_data.apply(pd.to_numeric, errors="coerce")
    df_data = df_data.ffill(limit=len(df_data)).bfill(limit=len(df_data))
    return df_data.values.astype(np.float32), [str(col) for col in df_data.columns]


def standardize_raw_train(data, train_ratio, stride):
    train_len = int(len(data) * train_ratio)
    train = data[:train_len]
    if stride > 1:
        train = train[::stride]
    mean = np.nanmean(train, axis=0, keepdims=True)
    std = np.nanstd(train, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train = (train - mean) / std
    train = np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0)
    return train.astype(np.float32), train_len


def standardize_train(data, train_ratio, stride, similarity_transform="raw"):
    train, train_len = standardize_raw_train(data, train_ratio, stride)
    if similarity_transform == "diff":
        if train.shape[0] < 2:
            raise ValueError("diff similarity requires at least two train rows")
        train = np.diff(train, axis=0)
    elif similarity_transform == "raw_diff":
        if train.shape[0] < 2:
            raise ValueError("raw_diff similarity requires at least two train rows")
        train = np.concatenate([train[1:], np.diff(train, axis=0)], axis=0)
    return train.astype(np.float32), train_len


def abs_correlation(train):
    denom = np.linalg.norm(train, axis=0, keepdims=True)
    normalized = train / (denom + 1e-8)
    corr = np.abs(normalized.T @ normalized).astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    return corr


def signed_correlation(train):
    denom = np.linalg.norm(train, axis=0, keepdims=True)
    normalized = train / (denom + 1e-8)
    corr = (normalized.T @ normalized).astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    return corr


def rank_columns(values):
    order = np.argsort(values, axis=0, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    col_ids = np.arange(values.shape[1])
    ranks[order, col_ids] = np.arange(values.shape[0], dtype=np.float32)[:, None]
    ranks = ranks - ranks.mean(axis=0, keepdims=True)
    std = ranks.std(axis=0, keepdims=True)
    return ranks / np.where(std < 1e-6, 1.0, std)


def similarity_matrix(train, raw_train, metric, profile_raw_weight):
    metric = str(metric or "abs_pearson")
    if metric == "abs_pearson":
        return abs_correlation(train)
    if metric == "positive_pearson":
        corr = signed_correlation(train)
        corr = np.clip(corr, 0.0, 1.0).astype(np.float32)
        np.fill_diagonal(corr, 1.0)
        return corr
    if metric == "abs_spearman":
        return abs_correlation(rank_columns(train))
    if metric == "positive_spearman":
        corr = signed_correlation(rank_columns(train))
        corr = np.clip(corr, 0.0, 1.0).astype(np.float32)
        np.fill_diagonal(corr, 1.0)
        return corr
    if metric == "forecast_profile":
        return forecast_profile_similarity(raw_train)
    if metric == "raw_profile_mix":
        return raw_profile_similarity(raw_train, profile_raw_weight)
    raise ValueError("unknown similarity metric: {}".format(metric))


def lag_correlation_features(train, lags):
    features = []
    for lag in lags:
        if lag <= 0 or train.shape[0] <= lag:
            continue
        left = train[:-lag]
        right = train[lag:]
        numerator = np.mean(left * right, axis=0)
        denominator = np.sqrt(np.mean(left ** 2, axis=0) * np.mean(right ** 2, axis=0))
        features.append(numerator / np.maximum(denominator, 1e-8))
    if not features:
        return np.zeros((train.shape[1], 0), dtype=np.float32)
    return np.stack(features, axis=1).astype(np.float32)


def forecast_profile_similarity(raw_train):
    lag_set = [1, 2, 4, 8, 12, 24, 48, 96]
    feature_blocks = [lag_correlation_features(raw_train, lag_set)]
    if raw_train.shape[0] > 2:
        diff_train = np.diff(raw_train, axis=0)
        feature_blocks.append(lag_correlation_features(diff_train, [1, 2, 4, 8, 12, 24, 48]))
        diff_energy = np.sqrt(np.mean(diff_train ** 2, axis=0, keepdims=True)).T
        feature_blocks.append(diff_energy.astype(np.float32))
    features = np.concatenate([block for block in feature_blocks if block.shape[1] > 0], axis=1)
    if features.shape[1] == 0:
        return np.ones((raw_train.shape[1], raw_train.shape[1]), dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = (features - mean) / np.where(std < 1e-6, 1.0, std)
    dist2 = np.mean((features[:, None, :] - features[None, :, :]) ** 2, axis=-1)
    upper = dist2[np.triu_indices(dist2.shape[0], k=1)]
    scale = float(np.median(upper[upper > 1e-8])) if np.any(upper > 1e-8) else 1.0
    sim = np.exp(-dist2 / max(scale, 1e-8)).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


def raw_profile_similarity(raw_train, raw_weight):
    raw_weight = float(raw_weight)
    if raw_weight < 0.0 or raw_weight > 1.0:
        raise ValueError("profile_raw_weight must be in [0, 1]")
    raw_corr = abs_correlation(raw_train)
    profile_sim = forecast_profile_similarity(raw_train)
    sim = raw_weight * raw_corr + (1.0 - raw_weight) * profile_sim
    sim = np.clip(sim, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


def temporal_importance(raw_train):
    if raw_train.shape[0] < 2:
        return np.zeros(raw_train.shape[1], dtype=np.float32)
    diff = np.diff(raw_train, axis=0)
    energy = np.sqrt(np.mean(diff ** 2, axis=0))
    lo = float(np.quantile(energy, 0.10))
    hi = float(np.quantile(energy, 0.90))
    if hi <= lo + 1e-8:
        return np.zeros_like(energy, dtype=np.float32)
    importance = (energy - lo) / (hi - lo)
    return np.clip(importance, 0.0, 1.0).astype(np.float32)


def farthest_anchors(corr, k, priority=None, priority_weight=0.0):
    priority_weight = float(priority_weight)
    if priority is None or priority_weight <= 0:
        priority = np.zeros(corr.shape[0], dtype=np.float32)
        priority_weight = 0.0
    else:
        priority = np.asarray(priority, dtype=np.float32)
        if priority.shape[0] != corr.shape[0]:
            raise ValueError("anchor priority length does not match corr")
    centrality = corr.mean(axis=1)
    start_score = centrality * (1.0 + priority_weight * priority)
    anchors = [int(np.argmax(start_score))]
    best_sim = corr[:, anchors[0]].copy()
    for _ in range(1, k):
        novelty = 1.0 - best_sim
        score = novelty * (1.0 + priority_weight * priority)
        score[np.asarray(anchors, dtype=np.int64)] = -np.inf
        index = int(np.argmax(score))
        anchors.append(index)
        best_sim = np.maximum(best_sim, corr[:, index])
    return np.array(anchors, dtype=np.int64)


def assign_groups(corr, anchors):
    return np.argmax(corr[:, anchors], axis=1).astype(np.int64)


def assign_groups_balanced(corr, anchors, max_group_size):
    num_variates = corr.shape[0]
    k = len(anchors)
    group_ids = np.full(num_variates, -1, dtype=np.int64)
    counts = np.zeros(k, dtype=np.int64)
    for group_idx, anchor in enumerate(anchors):
        group_ids[int(anchor)] = group_idx
        counts[group_idx] += 1

    preferences = np.argsort(-corr[:, anchors], axis=1)
    confidence = corr[np.arange(num_variates), anchors[preferences[:, 0]]]
    for var_idx in np.argsort(-confidence):
        if group_ids[var_idx] >= 0:
            continue
        for group_idx in preferences[var_idx]:
            if counts[group_idx] < max_group_size:
                group_ids[var_idx] = int(group_idx)
                counts[group_idx] += 1
                break
        if group_ids[var_idx] < 0:
            group_idx = int(np.argmin(counts))
            group_ids[var_idx] = group_idx
            counts[group_idx] += 1
    return group_ids


def refine_medoids(
    corr,
    anchors,
    group_ids,
    rounds,
    balanced=False,
    max_group_size=None,
    priority=None,
    priority_weight=0.0,
):
    anchors = anchors.copy()
    priority_weight = float(priority_weight)
    if priority is None or priority_weight <= 0:
        priority = np.zeros(corr.shape[0], dtype=np.float32)
        priority_weight = 0.0
    else:
        priority = np.asarray(priority, dtype=np.float32)
    for _ in range(rounds):
        new_anchors = anchors.copy()
        for group_idx in range(len(anchors)):
            members = np.flatnonzero(group_ids == group_idx)
            if members.size == 0:
                continue
            local_corr = corr[np.ix_(members, members)]
            medoid_score = local_corr.mean(axis=1) + priority_weight * priority[members]
            new_anchors[group_idx] = int(members[int(np.argmax(medoid_score))])
        anchors = new_anchors
        if balanced:
            group_ids = assign_groups_balanced(corr, anchors, max_group_size)
        else:
            group_ids = assign_groups(corr, anchors)
    return anchors.astype(np.int64), group_ids.astype(np.int64)


def _coverage_score(corr, anchors, quantile=0.05, min_weight=0.10, mean_weight=1.0,
                    group_size_weight=0.03):
    anchors = np.asarray(anchors, dtype=np.int64)
    sim = corr[:, anchors]
    group_ids = np.argmax(sim, axis=1).astype(np.int64)
    assigned = sim[np.arange(corr.shape[0]), group_ids]
    group_sizes = np.bincount(group_ids, minlength=len(anchors)).astype(np.float32)
    score = (
        mean_weight * float(np.mean(assigned)) +
        0.35 * float(np.quantile(assigned, quantile)) +
        min_weight * float(np.min(assigned)) -
        group_size_weight * float(np.max(group_sizes) / float(corr.shape[0]))
    )
    return score, group_ids, assigned


def exact_tail_coverage_anchors(corr, k, max_exact_combinations=200000):
    num_variates = corr.shape[0]
    total = math.comb(num_variates, k)
    if total > int(max_exact_combinations):
        return None
    best_score = None
    best_anchors = None
    best_tuple = None
    for anchors_tuple in itertools.combinations(range(num_variates), k):
        anchors = np.asarray(anchors_tuple, dtype=np.int64)
        score, group_ids, assigned = _coverage_score(corr, anchors)
        group_sizes = np.bincount(group_ids, minlength=k)
        # Tie-break toward the actual tail coverage, then mean coverage, then smaller largest group.
        compare = (
            score,
            float(np.quantile(assigned, 0.05)),
            float(np.min(assigned)),
            float(np.mean(assigned)),
            -int(np.max(group_sizes)),
        )
        if best_tuple is None or compare > best_tuple:
            best_tuple = compare
            best_score = score
            best_anchors = anchors
    return best_anchors.astype(np.int64), float(best_score)


def tail_coverage_swap_refine(corr, anchors, rounds=3):
    anchors = np.asarray(anchors, dtype=np.int64).copy()
    num_variates = corr.shape[0]
    k = len(anchors)
    if k <= 0:
        return anchors
    current_score, _, _ = _coverage_score(corr, anchors)
    for _ in range(max(0, int(rounds))):
        sim = corr[:, anchors]
        order = np.argsort(-sim, axis=1)
        best_pos = order[:, 0]
        best = sim[np.arange(num_variates), best_pos]
        if k > 1:
            second_pos = order[:, 1]
            second = sim[np.arange(num_variates), second_pos]
        else:
            second_pos = np.zeros(num_variates, dtype=np.int64)
            second = np.zeros(num_variates, dtype=np.float32)
        anchor_set = set(int(item) for item in anchors)
        best_update = None
        best_compare = None
        for replace_pos in range(k):
            without_replaced = np.where(best_pos == replace_pos, second, best)
            fallback_pos = np.where(best_pos == replace_pos, second_pos, best_pos)
            for candidate in range(num_variates):
                if candidate in anchor_set:
                    continue
                candidate_sim = corr[:, candidate]
                new_best = np.maximum(without_replaced, candidate_sim)
                new_group_ids = np.where(
                    candidate_sim >= without_replaced,
                    replace_pos,
                    fallback_pos,
                ).astype(np.int64)
                group_sizes = np.bincount(new_group_ids, minlength=k)
                score = (
                    float(np.mean(new_best)) +
                    0.35 * float(np.quantile(new_best, 0.05)) +
                    0.10 * float(np.min(new_best)) -
                    0.03 * float(np.max(group_sizes) / float(num_variates))
                )
                compare = (
                    score,
                    float(np.quantile(new_best, 0.05)),
                    float(np.min(new_best)),
                    float(np.mean(new_best)),
                    -int(np.max(group_sizes)),
                )
                if score <= current_score + 1e-8:
                    continue
                if best_compare is None or compare > best_compare:
                    best_compare = compare
                    best_update = (replace_pos, candidate, score)
        if best_update is None:
            break
        replace_pos, candidate, current_score = best_update
        anchors[replace_pos] = int(candidate)
    return anchors.astype(np.int64)


def select_tail_coverage_anchors(corr, k, max_exact_combinations=200000, swap_rounds=3,
                                 priority=None, priority_weight=0.0):
    exact = exact_tail_coverage_anchors(corr, k, max_exact_combinations=max_exact_combinations)
    if exact is not None:
        anchors, _ = exact
        return anchors.astype(np.int64)
    anchors = farthest_anchors(corr, k, priority=priority, priority_weight=priority_weight)
    group_ids = assign_groups(corr, anchors)
    anchors, _ = refine_medoids(
        corr,
        anchors,
        group_ids,
        rounds=2,
        priority=priority,
        priority_weight=priority_weight,
    )
    return tail_coverage_swap_refine(corr, anchors, rounds=swap_rounds)


def build_expand_mask(corr, anchors, group_ids, top_m):
    if top_m <= 0:
        return None
    top_m = min(int(top_m), len(anchors))
    top_groups = np.argsort(-corr[:, anchors], axis=1)[:, :top_m]
    mask = np.zeros((corr.shape[0], len(anchors)), dtype=np.float32)
    row_ids = np.arange(corr.shape[0])[:, None]
    mask[row_ids, top_groups] = 1.0
    mask[np.arange(corr.shape[0]), group_ids] = 1.0
    return mask


def build_expand_init_weight(corr, anchors, group_ids, expand_mask, mode, power=1.0):
    if mode == "onehot":
        return None
    if mode != "corr":
        raise ValueError("unknown expand init mode: {}".format(mode))
    power = float(power)
    if power <= 0:
        raise ValueError("expand_init_power must be positive")
    if expand_mask is None:
        expand_mask = np.zeros((corr.shape[0], len(anchors)), dtype=np.float32)
        expand_mask[np.arange(corr.shape[0]), group_ids] = 1.0
    weights = np.power(corr[:, anchors].astype(np.float64), power) * expand_mask.astype(np.float64)
    weights[np.arange(corr.shape[0]), group_ids] = np.maximum(
        weights[np.arange(corr.shape[0]), group_ids],
        np.power(corr[np.arange(corr.shape[0]), anchors[group_ids]].astype(np.float64), power),
    )
    row_sum = weights.sum(axis=1, keepdims=True)
    fallback = row_sum[:, 0] <= 1e-8
    if np.any(fallback):
        weights[fallback] = 0.0
        weights[np.flatnonzero(fallback), group_ids[fallback]] = 1.0
        row_sum = weights.sum(axis=1, keepdims=True)
    return (weights / np.maximum(row_sum, 1e-8)).astype(np.float32)


def write_summary(out_path, data_path, columns, anchors, group_ids, corr, train_len, stride,
                  balanced=False, max_group_size=0, expand_top_m=0, similarity_transform="raw",
                  similarity_metric="abs_pearson",
                  profile_raw_weight=0.8, expand_init="onehot", expand_init_power=1.0,
                  anchor_priority="coverage", anchor_priority_weight=0.0, priority_values=None,
                  anchor_objective="medoid", tail_exact_combinations=200000, tail_swap_rounds=0):
    group_sizes = np.bincount(group_ids, minlength=len(anchors))
    assigned_corr = corr[np.arange(corr.shape[0]), anchors[group_ids]]
    summary = {
        "data_path": str(data_path),
        "num_variates": int(corr.shape[0]),
        "reduced_k": int(len(anchors)),
        "k_ratio": float(len(anchors) / corr.shape[0]),
        "train_rows": int(train_len),
        "stride": int(stride),
        "similarity_transform": str(similarity_transform),
        "similarity_metric": str(similarity_metric),
        "profile_raw_weight": float(profile_raw_weight),
        "balanced": bool(balanced),
        "max_group_size": int(max_group_size),
        "expand_top_m": int(expand_top_m),
        "expand_init": str(expand_init),
        "expand_init_power": float(expand_init_power),
        "anchor_objective": str(anchor_objective),
        "tail_exact_combinations": int(tail_exact_combinations),
        "tail_swap_rounds": int(tail_swap_rounds),
        "anchor_priority": str(anchor_priority),
        "anchor_priority_weight": float(anchor_priority_weight),
        "mean_abs_corr_to_anchor": float(np.mean(assigned_corr)),
        "p05_abs_corr_to_anchor": float(np.quantile(assigned_corr, 0.05)),
        "min_abs_corr_to_anchor": float(np.min(assigned_corr)),
        "group_size_min": int(np.min(group_sizes)),
        "group_size_p50": float(np.quantile(group_sizes, 0.50)),
        "group_size_max": int(np.max(group_sizes)),
        "anchors": [int(item) for item in anchors],
        "anchor_names": [columns[int(item)] for item in anchors],
    }
    if priority_values is not None:
        selected_priority = np.asarray(priority_values, dtype=np.float32)[anchors]
        summary.update({
            "anchor_priority_mean": float(np.mean(selected_priority)),
            "anchor_priority_p50": float(np.quantile(selected_priority, 0.50)),
            "anchor_priority_max": float(np.max(selected_priority)),
        })
    md = [
        "# Variate Anchor Map 요약",
        "",
        "- source: `{}`".format(summary["data_path"]),
        "- V/K/KV: `{}` / `{}` / `{:.6f}`".format(
            summary["num_variates"], summary["reduced_k"], summary["k_ratio"]
        ),
        "- train row: `{}`, stride: `{}`".format(summary["train_rows"], summary["stride"]),
        "- similarity transform: `{}`".format(summary["similarity_transform"]),
        "- similarity metric: `{}`".format(summary["similarity_metric"]),
        "- profile raw weight: `{:.4g}`".format(summary["profile_raw_weight"]),
        "- balanced: `{}`, 최대 group size: `{}`".format(summary["balanced"], summary["max_group_size"]),
        "- expand top-m mask: `{}`".format(summary["expand_top_m"]),
        "- expand init: `{}`".format(summary["expand_init"]),
        "- expand init power: `{:.4g}`".format(summary["expand_init_power"]),
        "- anchor objective: `{}`".format(summary["anchor_objective"]),
        "- tail exact combination: `{}`, swap round `{}`".format(
            summary["tail_exact_combinations"], summary["tail_swap_rounds"]
        ),
        "- anchor priority: `{}`, weight `{:.4g}`".format(
            summary["anchor_priority"], summary["anchor_priority_weight"]
        ),
        "- assigned anchor abs corr: mean `{:.6f}`, p05 `{:.6f}`, min `{:.6f}`".format(
            summary["mean_abs_corr_to_anchor"],
            summary["p05_abs_corr_to_anchor"],
            summary["min_abs_corr_to_anchor"],
        ),
        "- group size: min `{}`, p50 `{:.1f}`, max `{}`".format(
            summary["group_size_min"], summary["group_size_p50"], summary["group_size_max"]
        ),
        "",
        "## 주요 Anchor",
        "",
    ]
    for idx, anchor in enumerate(anchors[:30]):
        md.append("- group {}: var {} `{}` size {}".format(
            idx, int(anchor), columns[int(anchor)], int(group_sizes[idx])
        ))
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_path.with_suffix(".summary.md").write_text("\n".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--features", default="M")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--reduced_k", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--similarity_transform", choices=["raw", "diff", "raw_diff", "raw_profile"], default="raw")
    parser.add_argument(
        "--similarity_metric",
        choices=[
            "abs_pearson",
            "positive_pearson",
            "abs_spearman",
            "positive_spearman",
            "forecast_profile",
            "raw_profile_mix",
        ],
        default="abs_pearson",
    )
    parser.add_argument("--profile_raw_weight", type=float, default=0.8)
    parser.add_argument("--anchor_objective", choices=["medoid", "tail_coverage"], default="medoid")
    parser.add_argument("--tail_exact_combinations", type=int, default=200000)
    parser.add_argument("--tail_swap_rounds", type=int, default=3)
    parser.add_argument("--anchor_priority", choices=["coverage", "delta_energy"], default="coverage")
    parser.add_argument("--anchor_priority_weight", type=float, default=0.0)
    parser.add_argument("--refine_rounds", type=int, default=2)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--max_group_size", type=int, default=0)
    parser.add_argument("--expand_top_m", type=int, default=0)
    parser.add_argument("--expand_init", choices=["onehot", "corr"], default="onehot")
    parser.add_argument("--expand_init_power", type=float, default=1.0)
    parser.add_argument("--enforce_k_budget", action="store_true")
    args = parser.parse_args()

    data, columns = read_multivariate_csv(args.root_path, args.data_path, args.features, args.target)
    num_variates = data.shape[1]
    k = int(args.reduced_k)
    if args.enforce_k_budget:
        k_max, exception = strict_k_budget(num_variates)
        if k > k_max:
            raise ValueError("reduced_k={} exceeds K_max={} for V={}".format(k, k_max, num_variates))
        if exception and k != 1:
            raise ValueError("V={} only allows K=1 exception".format(num_variates))
    raw_train, train_len = standardize_raw_train(data, args.train_ratio, max(1, args.stride))
    if args.similarity_transform == "raw_profile" and args.similarity_metric == "abs_pearson":
        train = raw_train
        corr = raw_profile_similarity(raw_train, args.profile_raw_weight)
        similarity_metric = "raw_profile_mix"
    else:
        train, train_len = standardize_train(
            data,
            args.train_ratio,
            max(1, args.stride),
            similarity_transform=args.similarity_transform,
        )
        corr = similarity_matrix(train, raw_train, args.similarity_metric, args.profile_raw_weight)
        similarity_metric = args.similarity_metric
    priority_values = None
    priority_weight = float(args.anchor_priority_weight)
    if args.anchor_priority == "delta_energy":
        priority_values = temporal_importance(raw_train)
    elif priority_weight > 0:
        raise ValueError("anchor_priority_weight requires a non-coverage anchor_priority")
    if args.anchor_objective == "tail_coverage":
        anchors = select_tail_coverage_anchors(
            corr,
            k,
            max_exact_combinations=int(args.tail_exact_combinations),
            swap_rounds=int(args.tail_swap_rounds),
            priority=priority_values,
            priority_weight=priority_weight,
        )
    else:
        anchors = farthest_anchors(corr, k, priority_values, priority_weight)
    max_group_size = int(args.max_group_size)
    if args.balanced and max_group_size <= 0:
        max_group_size = int(math.ceil(num_variates / float(k)))
    if args.balanced:
        group_ids = assign_groups_balanced(corr, anchors, max_group_size)
    else:
        group_ids = assign_groups(corr, anchors)
    if args.anchor_objective == "medoid":
        anchors, group_ids = refine_medoids(
            corr,
            anchors,
            group_ids,
            max(0, args.refine_rounds),
            balanced=args.balanced,
            max_group_size=max_group_size if args.balanced else None,
            priority=priority_values,
            priority_weight=priority_weight,
        )
    else:
        group_ids = assign_groups_balanced(corr, anchors, max_group_size) if args.balanced else assign_groups(corr, anchors)
    group_sizes = np.bincount(group_ids, minlength=k)
    if np.any(group_sizes <= 0):
        raise RuntimeError("empty anchor group after refinement")
    expand_mask = build_expand_mask(corr, anchors, group_ids, int(args.expand_top_m))
    expand_init_weight = build_expand_init_weight(
        corr,
        anchors,
        group_ids,
        expand_mask,
        args.expand_init,
        power=args.expand_init_power,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        anchor_indices=anchors.astype(np.int64),
        group_ids=group_ids.astype(np.int64),
        group_sizes=group_sizes.astype(np.int64),
        assigned_abs_corr=corr[np.arange(num_variates), anchors[group_ids]].astype(np.float32),
        columns=np.array(columns),
        similarity_transform=np.array([args.similarity_transform]),
        similarity_metric=np.array([similarity_metric]),
        profile_raw_weight=np.array([float(args.profile_raw_weight)]),
        balanced=np.array([bool(args.balanced)]),
        max_group_size=np.array([int(max_group_size)]),
        anchor_priority=np.array([args.anchor_priority]),
        anchor_priority_weight=np.array([float(priority_weight)]),
        anchor_objective=np.array([args.anchor_objective]),
        tail_exact_combinations=np.array([int(args.tail_exact_combinations)]),
        tail_swap_rounds=np.array([int(args.tail_swap_rounds)]),
    )
    if priority_values is not None:
        payload["anchor_priority_values"] = priority_values.astype(np.float32)
    if expand_mask is not None:
        payload["expand_mask"] = expand_mask
        payload["expand_top_m"] = np.array([int(args.expand_top_m)])
    if expand_init_weight is not None:
        payload["expand_init_weight"] = expand_init_weight
        payload["expand_init"] = np.array([args.expand_init])
        payload["expand_init_power"] = np.array([float(args.expand_init_power)])
    np.savez(out_path, **payload)
    write_summary(
        out_path,
        Path(args.root_path) / args.data_path,
        columns,
        anchors,
        group_ids,
        corr,
        train_len,
        args.stride,
        balanced=args.balanced,
        max_group_size=max_group_size,
        expand_top_m=int(args.expand_top_m),
        similarity_transform=args.similarity_transform,
        similarity_metric=similarity_metric,
        profile_raw_weight=float(args.profile_raw_weight),
        expand_init=args.expand_init,
        expand_init_power=args.expand_init_power,
        anchor_priority=args.anchor_priority,
        anchor_priority_weight=priority_weight,
        priority_values=priority_values,
        anchor_objective=args.anchor_objective,
        tail_exact_combinations=int(args.tail_exact_combinations),
        tail_swap_rounds=int(args.tail_swap_rounds),
    )
    print("wrote {}".format(out_path))
    print("V={} K={} K/V={:.6f}".format(num_variates, k, k / float(num_variates)))
    print("group size min/median/max: {} / {:.1f} / {}".format(
        int(np.min(group_sizes)), float(np.median(group_sizes)), int(np.max(group_sizes))
    ))


if __name__ == "__main__":
    main()
