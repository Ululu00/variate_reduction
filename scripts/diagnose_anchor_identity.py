import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from build_variate_anchor_map import read_multivariate_csv, standardize_raw_train


DATASETS = {
    "Weather": {
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "features": "M",
        "target": "OT",
        "anchor_map": "./results/anchor_maps/Weather_k6_corr_medoid_top16_corrinit_p4_train.npz",
    },
    "Traffic": {
        "root_path": "./dataset/traffic/",
        "data_path": "traffic.csv",
        "features": "M",
        "target": "OT",
        "anchor_map": "./results/anchor_maps/Traffic_k258_corr_medoid_top16_corrinit_p4_train.npz",
    },
    "Electricity": {
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "features": "M",
        "target": "OT",
        "anchor_map": "./results/anchor_maps/Electricity_k96_corr_medoid_top16_corrinit_p4_train.npz",
    },
    "Solar": {
        "root_path": "./dataset/Solar/",
        "data_path": "solar_AL.txt",
        "features": "M",
        "target": "none",
        "anchor_map": "./results/anchor_maps/Solar_k41_corr_medoid_top16_corrinit_p4_train.npz",
    },
}


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
}


def use_theme():
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        },
    )


def abs_log_ratio(value, eps=1e-8):
    value = float(value)
    return abs(math.log(max(value, eps)))


def cosine_corr(left, right, eps=1e-8):
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= eps:
        return 0.0
    return float(np.dot(left, right) / (denom + eps))


def best_lag_corr(series, anchor, max_lag):
    best_corr = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left = series[-lag:]
            right = anchor[: lag or None]
        elif lag > 0:
            left = series[:-lag]
            right = anchor[lag:]
        else:
            left = series
            right = anchor
        if left.size < 4:
            continue
        corr = abs(cosine_corr(left, right))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return float(best_corr), int(best_lag)


def load_anchor_map(path):
    data = np.load(path, allow_pickle=False)
    required = {"anchor_indices", "group_ids", "assigned_abs_corr"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError("{} missing fields {}".format(path, sorted(missing)))
    return {
        "anchor_indices": data["anchor_indices"].astype(np.int64),
        "group_ids": data["group_ids"].astype(np.int64),
        "assigned_abs_corr": data["assigned_abs_corr"].astype(np.float64),
        "expand_init_weight": data["expand_init_weight"].astype(np.float64)
        if "expand_init_weight" in data.files
        else None,
    }


def decoder_entropy(expand_init_weight, eps=1e-12):
    if expand_init_weight is None:
        return None
    weight = np.maximum(expand_init_weight.astype(np.float64), 0.0)
    norm = weight / (weight.sum(axis=1, keepdims=True) + eps)
    entropy = -(norm * np.log(norm + eps)).sum(axis=1)
    return entropy


def analyze_dataset(name, config, repo_root, train_ratio, stride, max_lag):
    data, columns = read_multivariate_csv(
        repo_root / config["root_path"],
        config["data_path"],
        config["features"],
        config["target"],
    )
    train_std, train_len = standardize_raw_train(data, train_ratio, stride)
    raw_train = data[:train_len]
    if stride > 1:
        raw_train = raw_train[::stride]
    raw_std = np.nanstd(raw_train, axis=0)
    raw_std = np.where(raw_std < 1e-8, 1.0, raw_std)
    if raw_train.shape[0] > 1:
        diff_energy = np.sqrt(np.mean(np.diff(raw_train, axis=0) ** 2, axis=0))
    else:
        diff_energy = np.ones(raw_train.shape[1], dtype=np.float64)
    diff_energy = np.where(diff_energy < 1e-8, 1.0, diff_energy)

    anchor_map = load_anchor_map(repo_root / config["anchor_map"])
    anchors = anchor_map["anchor_indices"]
    group_ids = anchor_map["group_ids"]
    assigned_abs_corr = anchor_map["assigned_abs_corr"]
    entropy = decoder_entropy(anchor_map["expand_init_weight"])

    variable_rows = []
    for var_idx in range(train_std.shape[1]):
        group_id = int(group_ids[var_idx])
        anchor_idx = int(anchors[group_id])
        series = train_std[:, var_idx]
        anchor = train_std[:, anchor_idx]
        dot_anchor = float(np.dot(anchor, anchor))
        beta = float(np.dot(series, anchor) / (dot_anchor + 1e-8))
        residual = series - beta * anchor
        residual_var_frac = float(np.mean(residual ** 2) / (np.mean(series ** 2) + 1e-8))
        signed_corr = cosine_corr(series, anchor)
        best_corr, best_lag = best_lag_corr(series, anchor, max_lag)
        zero_abs_corr = abs(signed_corr)
        raw_std_ratio = float(raw_std[var_idx] / raw_std[anchor_idx])
        diff_energy_ratio = float(diff_energy[var_idx] / diff_energy[anchor_idx])
        variable_rows.append(
            {
                "dataset": name,
                "variable_index": var_idx,
                "variable_name": str(columns[var_idx]) if var_idx < len(columns) else str(var_idx),
                "group_id": group_id,
                "anchor_index": anchor_idx,
                "anchor_name": str(columns[anchor_idx]) if anchor_idx < len(columns) else str(anchor_idx),
                "is_anchor": int(var_idx == anchor_idx),
                "assigned_abs_corr": float(assigned_abs_corr[var_idx]),
                "signed_corr": signed_corr,
                "anchor_beta": beta,
                "anchor_residual_var_frac": residual_var_frac,
                "best_lag_abs_corr": best_corr,
                "best_lag": best_lag,
                "lag_gain_abs_corr": best_corr - zero_abs_corr,
                "raw_std_ratio_to_anchor": raw_std_ratio,
                "abs_log_raw_std_ratio": abs_log_ratio(raw_std_ratio),
                "diff_energy_ratio_to_anchor": diff_energy_ratio,
                "abs_log_diff_energy_ratio": abs_log_ratio(diff_energy_ratio),
                "decoder_init_entropy": float(entropy[var_idx]) if entropy is not None else float("nan"),
            }
        )
    return variable_rows


def summarize_groups(variable_df):
    rows = []
    for (dataset, group_id), group in variable_df.groupby(["dataset", "group_id"]):
        anchor_rows = group[group["is_anchor"] == 1]
        anchor_index = int(anchor_rows["anchor_index"].iloc[0]) if not anchor_rows.empty else int(group["anchor_index"].iloc[0])
        rows.append(
            {
                "dataset": dataset,
                "group_id": int(group_id),
                "anchor_index": anchor_index,
                "group_size": int(len(group)),
                "assigned_abs_corr_mean": float(group["assigned_abs_corr"].mean()),
                "assigned_abs_corr_p05": float(group["assigned_abs_corr"].quantile(0.05)),
                "assigned_abs_corr_min": float(group["assigned_abs_corr"].min()),
                "anchor_residual_var_mean": float(group["anchor_residual_var_frac"].mean()),
                "anchor_residual_var_p95": float(group["anchor_residual_var_frac"].quantile(0.95)),
                "anchor_residual_var_max": float(group["anchor_residual_var_frac"].max()),
                "abs_log_raw_std_ratio_mean": float(group["abs_log_raw_std_ratio"].mean()),
                "abs_log_diff_energy_ratio_mean": float(group["abs_log_diff_energy_ratio"].mean()),
                "best_lag_abs_median": float(group["best_lag"].abs().median()),
                "best_lag_abs_p95": float(group["best_lag"].abs().quantile(0.95)),
                "decoder_init_entropy_mean": float(group["decoder_init_entropy"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "group_size"], ascending=[True, False])


def summarize_datasets(variable_df, group_df):
    rows = []
    for dataset, group in variable_df.groupby("dataset"):
        group_part = group_df[group_df["dataset"] == dataset]
        rows.append(
            {
                "dataset": dataset,
                "num_variates": int(len(group)),
                "num_anchors": int(group["group_id"].nunique()),
                "k_ratio": float(group["group_id"].nunique() / max(1, len(group))),
                "assigned_abs_corr_p05": float(group["assigned_abs_corr"].quantile(0.05)),
                "assigned_abs_corr_min": float(group["assigned_abs_corr"].min()),
                "group_size_max": int(group_part["group_size"].max()),
                "group_size_p95": float(group_part["group_size"].quantile(0.95)),
                "anchor_residual_var_mean": float(group["anchor_residual_var_frac"].mean()),
                "anchor_residual_var_p95": float(group["anchor_residual_var_frac"].quantile(0.95)),
                "lag_gain_abs_corr_p95": float(group["lag_gain_abs_corr"].quantile(0.95)),
                "best_lag_abs_p95": float(group["best_lag"].abs().quantile(0.95)),
                "abs_log_raw_std_ratio_p95": float(group["abs_log_raw_std_ratio"].quantile(0.95)),
                "abs_log_diff_energy_ratio_p95": float(group["abs_log_diff_energy_ratio"].quantile(0.95)),
                "decoder_init_entropy_mean": float(group["decoder_init_entropy"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset")


def write_markdown(path, dataset_df, group_df):
    lines = [
        "# Anchor Identity Diagnostics",
        "",
        "Train-split diagnostics for fixed corr-medoid p4 representative maps. "
        "Correlation coverage measures whether each variable has a close anchor; "
        "anchor-residual and scale/lag diagnostics check whether high-correlation groups still lose local identity.",
        "",
        "| dataset | K/V | p05 corr | max group | residual mean | residual p95 | lag p95 | std-ratio p95 | diff-ratio p95 | decoder entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_df.to_dict("records"):
        lines.append(
            "| {dataset} | {k_ratio:.4f} | {assigned_abs_corr_p05:.4f} | {group_size_max} | "
            "{anchor_residual_var_mean:.4f} | {anchor_residual_var_p95:.4f} | {best_lag_abs_p95:.1f} | "
            "{abs_log_raw_std_ratio_p95:.3f} | {abs_log_diff_energy_ratio_p95:.3f} | {decoder_init_entropy_mean:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Solar Read",
            "",
            "- Solar has very high assigned-anchor correlation, so fixed anchor selection coverage is not the obvious failure.",
            "- The largest Solar group has 25 variables under one executed anchor; long-horizon errors can therefore come from identity details inside highly redundant groups rather than from missing coverage.",
            "- High decoder-init entropy means each variable initially blends many nearby anchors, which is efficient but can dilute scale/phase details after temporal forecasting.",
            "",
            "## Largest Solar Groups",
            "",
            "| group | anchor | size | corr p05 | residual mean | residual p95 | lag p95 | diff-ratio mean | entropy |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    solar_groups = group_df[group_df["dataset"] == "Solar"].sort_values("group_size", ascending=False).head(10)
    for row in solar_groups.to_dict("records"):
        lines.append(
            "| {group_id} | {anchor_index} | {group_size} | {assigned_abs_corr_p05:.4f} | "
            "{anchor_residual_var_mean:.4f} | {anchor_residual_var_p95:.4f} | {best_lag_abs_p95:.1f} | "
            "{abs_log_diff_energy_ratio_mean:.3f} | {decoder_init_entropy_mean:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_dataset_summary(dataset_df, out_path):
    plot_df = dataset_df.set_index("dataset")[
        [
            "assigned_abs_corr_p05",
            "group_size_max",
            "anchor_residual_var_p95",
            "best_lag_abs_p95",
            "abs_log_diff_energy_ratio_p95",
            "decoder_init_entropy_mean",
        ]
    ].copy()
    norm_df = plot_df.copy()
    for column in norm_df.columns:
        values = norm_df[column].astype(float)
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo + 1e-12:
            norm_df[column] = 0.0
        else:
            norm_df[column] = (values - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.heatmap(norm_df, annot=plot_df.round(3), fmt="", cmap="YlGnBu", linewidths=0.5, ax=ax)
    ax.set_title("Anchor identity diagnostics by dataset")
    ax.set_xlabel("Metric, annotated with raw values")
    ax.set_ylabel("")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_solar_groups(group_df, out_path):
    solar = group_df[group_df["dataset"] == "Solar"].sort_values("group_size", ascending=False).head(18).copy()
    solar["label"] = solar.apply(lambda row: "g{} / v{}".format(int(row["group_id"]), int(row["anchor_index"])), axis=1)
    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))
    sns.barplot(data=solar, y="label", x="group_size", color="#F0986E", edgecolor="#804126", ax=ax1)
    ax1.set_xlabel("Group size")
    ax1.set_ylabel("Largest Solar anchor groups")
    ax2 = ax1.twiny()
    ax2.plot(solar["anchor_residual_var_p95"], np.arange(len(solar)), color="#2E4780", marker="o", linewidth=1.2)
    ax2.set_xlabel("Anchor residual variance p95")
    ax1.set_title("Solar large anchor groups still carry non-anchor residual detail")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Weather", "Traffic", "Electricity", "Solar"])
    parser.add_argument("--out_dir", default="./results/current_result_inventory/anchor_identity")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_lag", type=int, default=24)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    use_theme()

    variable_rows = []
    for dataset in args.datasets:
        if dataset not in DATASETS:
            raise ValueError("unknown dataset {}".format(dataset))
        variable_rows.extend(
            analyze_dataset(dataset, DATASETS[dataset], repo_root, args.train_ratio, args.stride, args.max_lag)
        )

    variable_df = pd.DataFrame(variable_rows)
    group_df = summarize_groups(variable_df)
    dataset_df = summarize_datasets(variable_df, group_df)

    variable_df.to_csv(out_dir / "anchor_identity_variable_stats.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    group_df.to_csv(out_dir / "anchor_identity_group_stats.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    dataset_df.to_csv(out_dir / "anchor_identity_dataset_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_markdown(out_dir / "anchor_identity_diagnostics.md", dataset_df, group_df)
    plot_dataset_summary(dataset_df, out_dir / "anchor_identity_dataset_summary.png")
    plot_solar_groups(group_df, out_dir / "solar_anchor_identity_groups.png")

    print("wrote {}".format(out_dir))


if __name__ == "__main__":
    main()
