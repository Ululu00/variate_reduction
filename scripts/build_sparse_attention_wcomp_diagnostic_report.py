import argparse
import base64
import html
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ["Weather", "Electricity", "Traffic", "Solar"]
PRED_ORDER = [96, 720]


def as_float(series):
    return pd.to_numeric(series, errors="coerce")


def geo_mean(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    values = values[values > 0]
    if len(values) == 0:
        return float("nan")
    return float(math.exp(np.mean(np.log(values))))


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def encode_image(path):
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def img_tag(path, alt="", cls="chart"):
    return (
        f'<img class="{cls}" src="data:image/png;base64,{encode_image(path)}" '
        f'alt="{html.escape(alt)}" loading="lazy">'
    )


def row_norm_abs(matrix, eps=1e-12):
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.abs(matrix)
    denom = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(denom, eps)


def entropy_support(matrix, eps=1e-12):
    probs = row_norm_abs(matrix, eps=eps)
    entropy = -(probs * np.log(np.maximum(probs, eps))).sum(axis=1)
    return np.exp(entropy)


def coverage_stats(matrix, top=5):
    probs = row_norm_abs(matrix)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] == 0:
        return {}
    rows, cols = probs.shape
    top = max(1, min(int(top), cols))
    top_idx = np.argsort(-probs, axis=1)[:, :top]
    top1 = top_idx[:, 0]
    unique_top = np.unique(top_idx)
    unique_top1 = np.unique(top1)
    counts_top1 = np.bincount(top1, minlength=cols)
    counts_top = np.bincount(top_idx.reshape(-1), minlength=cols)
    return {
        f"top{top}_coverage_count": int(len(unique_top)),
        f"top{top}_coverage_frac": float(len(unique_top) / cols),
        "top1_coverage_count": int(len(unique_top1)),
        "top1_coverage_frac": float(len(unique_top1) / cols),
        "top1_unique_per_latent": float(len(unique_top1) / rows),
        "top1_repeat_rate": float(1.0 - len(unique_top1) / rows),
        "top1_max_latent_share": float(counts_top1.max() / rows),
        f"top{top}_max_latent_share": float(counts_top.max() / rows),
        "mean_top1_mass": float(np.take_along_axis(probs, top1[:, None], axis=1).mean()),
        f"mean_top{top}_mass": float(np.take_along_axis(probs, top_idx, axis=1).sum(axis=1).mean()),
        "mean_effective_support": float(entropy_support(probs).mean()),
    }


def load_weight_stats(plot_dir):
    stats = {}
    if not isinstance(plot_dir, str) or not plot_dir:
        return stats
    path = Path(plot_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    compress = path / "compress_weight.npy"
    expand = path / "expand_weight.npy"
    effective = path / "effective_weight.npy"
    if compress.exists():
        w = np.load(compress)
        stats.update({f"comp_{k}": v for k, v in coverage_stats(w, top=5).items()})
        stats.update({f"comp10_{k}": v for k, v in coverage_stats(w, top=10).items()})
    if expand.exists():
        w = np.load(expand)
        stats.update({f"exp_{k}": v for k, v in coverage_stats(w, top=1).items()})
        stats["exp_mean_effective_support"] = float(entropy_support(w).mean())
    if effective.exists():
        w = np.load(effective)
        stats.update({f"eff_{k}": v for k, v in coverage_stats(w, top=5).items()})
        stats["eff_mean_effective_support"] = float(entropy_support(w).mean())
    return stats


def resolve_plot_dir(plot_dir):
    path = Path(str(plot_dir))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def render_matrix(ax, matrix, title, xlabel, ylabel):
    matrix = row_norm_abs(matrix)
    vmax = np.nanpercentile(matrix, 99.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = None
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    rows, cols = matrix.shape
    if cols <= 30:
        ax.set_xticks(np.arange(cols))
        ax.tick_params(axis="x", labelsize=6, rotation=0)
    else:
        ax.set_xticks(np.linspace(0, cols - 1, 6).astype(int))
        ax.tick_params(axis="x", labelsize=7)
    if rows <= 30:
        ax.set_yticks(np.arange(rows))
        ax.tick_params(axis="y", labelsize=6)
    else:
        ax.set_yticks(np.linspace(0, rows - 1, 6).astype(int))
        ax.tick_params(axis="y", labelsize=7)
    return im


def render_triptych(row, out_path):
    plot_dir = resolve_plot_dir(row["plot_dir"])
    matrices = []
    for name, filename, xlabel, ylabel in [
        ("W_comp", "compress_weight.npy", "source variable", "latent token"),
        ("W_exp", "expand_weight.npy", "latent token", "target variable"),
        ("W_eff = W_exp W_comp", "effective_weight.npy", "source variable", "target variable"),
    ]:
        path = plot_dir / filename
        if path.exists():
            matrices.append((name, np.load(path), xlabel, ylabel))
    if not matrices:
        return None
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.2), constrained_layout=True)
    if len(matrices) == 1:
        axes = [axes]
    for ax, (name, matrix, xlabel, ylabel) in zip(axes, matrices):
        im = render_matrix(ax, matrix, name, xlabel, ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    title = (
        f"{row['dataset']} pred={int(row['pred_len'])} | {row['method_name']} | "
        f"K={int(row['num_latents_K'])}, ratio={row['k_ratio']}"
    )
    fig.suptitle(title, fontsize=12, y=1.04)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_best_ratio_heatmap(best, out_path):
    data = np.full((len(DATASET_ORDER), len(PRED_ORDER)), np.nan)
    labels = [["" for _ in PRED_ORDER] for _ in DATASET_ORDER]
    for _, row in best.iterrows():
        i = DATASET_ORDER.index(row["dataset"])
        j = PRED_ORDER.index(int(row["pred_len"]))
        data[i, j] = row["mse_ratio"]
        labels[i][j] = f"{row['mse_ratio']:.2f} / {row['mae_ratio']:.2f}\n{row['short_method']}\nK={int(row['num_latents_K'])}"
    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0.95, vmax=max(3.2, np.nanmax(data)))
    ax.set_xticks(np.arange(len(PRED_ORDER)))
    ax.set_xticklabels([str(v) for v in PRED_ORDER])
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels(DATASET_ORDER)
    ax.set_xlabel("prediction length")
    ax.set_title("Best candidate per dataset × horizon: MSE/MAE ratio vs iTransformer")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MSE ratio, lower is better")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_ratio_distribution(candidates, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    positions = np.arange(len(DATASET_ORDER))
    data = [
        candidates.loc[candidates["dataset"] == dataset, "mse_ratio"].dropna().values
        for dataset in DATASET_ORDER
    ]
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor("#1d4ed8")
    for median in bp["medians"]:
        median.set_color("#111827")
    ax.axhline(1.00, color="#166534", linewidth=1.2, linestyle="-", label="exact win")
    ax.axhline(1.01, color="#a16207", linewidth=1.2, linestyle="--", label="pass threshold")
    ax.set_xticks(positions)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("MSE ratio vs baseline")
    ax.set_title("Candidate MSE ratio distribution by dataset")
    ax.set_ylim(0.9, max(3.8, candidates["mse_ratio"].max() * 1.05))
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_sparsity_scatter(candidates, out_path):
    colors = {
        "Weather": "#2563eb",
        "Electricity": "#ca8a04",
        "Traffic": "#dc2626",
        "Solar": "#16a34a",
    }
    markers = {"softmax": "o", "entmax15": "^"}
    fig, ax = plt.subplots(figsize=(9.3, 5.4), constrained_layout=True)
    for dataset in DATASET_ORDER:
        for norm, marker in markers.items():
            sub = candidates[(candidates["dataset"] == dataset) & (candidates["normalization"] == norm)]
            ax.scatter(
                sub["support_frac"],
                sub["mse_ratio"],
                s=np.where(sub["k_ratio"] == 0.3, 58, 34),
                alpha=0.70,
                marker=marker,
                color=colors[dataset],
                edgecolor="#111827",
                linewidth=0.35,
                label=f"{dataset} {norm}",
            )
    ax.axhline(1.00, color="#166534", linewidth=1.1)
    ax.axhline(1.01, color="#a16207", linewidth=1.1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("avg effective support / C (log scale)")
    ax.set_ylabel("MSE ratio vs baseline (log scale)")
    ax.set_title("Sparser W_comp did not imply better forecasting")
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered = []
    for h, label in zip(handles, labels):
        if label not in seen:
            filtered.append((h, label))
            seen.add(label)
    ax.legend([h for h, _ in filtered], [l for _, l in filtered], ncol=2, fontsize=7, frameon=False)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_sparsity_bars(grouped, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    labels = [f"{d}\n{n}" for d, n in zip(grouped["dataset"], grouped["normalization"])]
    x = np.arange(len(grouped))
    axes[0].bar(x, grouped["med_support_frac"], color="#60a5fa", edgecolor="#1e3a8a", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("median effective support / C")
    axes[0].set_title("Lower support often means hard selection, not better coverage")
    axes[1].bar(x, grouped["med_top5_coverage_frac"], color="#fbbf24", edgecolor="#92400e", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("median unique top-5 source coverage / C")
    axes[1].set_title("Coverage remains limited after sparsification")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def short_method(name):
    name = str(name)
    family = "LQ" if name.startswith("latent_query_attention") else "MLP"
    norm = "E15" if "_entmax" in name else "SM"
    entropy = "+H" if name.endswith("_entropy") else ""
    return f"{family}-{norm}{entropy}"


def df_to_html_table(df, classes="table"):
    return df.to_html(index=False, escape=False, classes=classes, border=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_csv", default="./results/sparse_attention_wcomp/results.csv")
    parser.add_argument("--plot_root", default="./results/sparse_attention_wcomp/plots")
    parser.add_argument("--out_html", default="./results/sparse_attention_wcomp/sparse_attention_wcomp_diagnostic_report.html")
    parser.add_argument("--asset_dir", default="./results/sparse_attention_wcomp/diagnostic_report_assets")
    args = parser.parse_args()

    result_csv = Path(args.result_csv)
    out_html = Path(args.out_html)
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(result_csv)
    for col in [
        "pred_len",
        "mse",
        "mae",
        "train_time_sec",
        "avg_train_iter_time_sec",
        "parameter_count",
        "peak_gpu_allocated_mb",
        "k_ratio",
        "num_latents_K",
        "num_variates",
        "avg_comp_entropy",
        "avg_effective_support",
        "avg_top1_mass",
        "avg_top5_mass",
        "avg_top10_mass",
        "avg_pairwise_overlap_top5",
        "avg_pairwise_overlap_top10",
        "avg_pairwise_cosine_between_latents",
        "w_comp_density",
        "w_exp_density",
        "w_eff_density",
        "avg_w_eff_entropy",
        "avg_w_eff_top5_mass",
    ]:
        if col in df:
            df[col] = as_float(df[col])

    baseline = df[df["method_family"] == "iTransformer"].copy()
    candidates = df[df["method_family"] != "iTransformer"].copy()
    baseline_metrics = baseline[
        ["dataset", "pred_len", "mse", "mae", "train_time_sec", "avg_train_iter_time_sec", "parameter_count", "peak_gpu_allocated_mb"]
    ].rename(
        columns={
            "mse": "baseline_mse",
            "mae": "baseline_mae",
            "train_time_sec": "baseline_train_time_sec",
            "avg_train_iter_time_sec": "baseline_avg_train_iter_time_sec",
            "parameter_count": "baseline_parameter_count",
            "peak_gpu_allocated_mb": "baseline_peak_gpu_allocated_mb",
        }
    )
    candidates = candidates.merge(baseline_metrics, on=["dataset", "pred_len"], how="left")
    candidates["mse_ratio"] = candidates["mse"] / candidates["baseline_mse"]
    candidates["mae_ratio"] = candidates["mae"] / candidates["baseline_mae"]
    candidates["pass_mse"] = candidates["mse_ratio"] < 1.01
    candidates["pass_mae"] = candidates["mae_ratio"] < 1.01
    candidates["pass_both"] = candidates["pass_mse"] & candidates["pass_mae"]
    candidates["exact_mse_win"] = candidates["mse_ratio"] < 1.0
    candidates["exact_mae_win"] = candidates["mae_ratio"] < 1.0
    candidates["support_frac"] = candidates["avg_effective_support"] / candidates["num_variates"]
    candidates["uniform_top5_mass"] = np.minimum(5, candidates["num_variates"]) / candidates["num_variates"]
    candidates["top5_lift_vs_uniform"] = candidates["avg_top5_mass"] / candidates["uniform_top5_mass"]
    candidates["short_method"] = candidates["method_name"].map(short_method)

    weight_stats = []
    for _, row in candidates.iterrows():
        stats = load_weight_stats(row.get("plot_dir", ""))
        stats["__index"] = row.name
        weight_stats.append(stats)
    weight_stats = pd.DataFrame(weight_stats).set_index("__index")
    candidates = pd.concat([candidates, weight_stats], axis=1)

    best = (
        candidates.sort_values(["dataset", "pred_len", "mse_ratio", "mae_ratio"])
        .groupby(["dataset", "pred_len"], as_index=False)
        .head(1)
        .sort_values(["dataset", "pred_len"])
    )

    pass_rows = candidates[candidates["pass_both"]].copy().sort_values(["dataset", "pred_len", "mse_ratio"])
    worst = candidates.sort_values("mse_ratio", ascending=False).head(8)
    summary_by_dataset = (
        candidates.groupby("dataset")
        .agg(
            n=("mse_ratio", "size"),
            pass_rate=("pass_both", "mean"),
            gm_mse=("mse_ratio", geo_mean),
            gm_mae=("mae_ratio", geo_mean),
            best_mse=("mse_ratio", "min"),
            median_mse=("mse_ratio", "median"),
            median_support_frac=("support_frac", "median"),
            median_top5=("avg_top5_mass", "median"),
            median_top5_coverage=("comp_top5_coverage_frac", "median"),
            median_top1_repeat=("comp_top1_repeat_rate", "median"),
        )
        .reset_index()
    )
    summary_by_norm = (
        candidates.groupby(["dataset", "normalization"])
        .agg(
            n=("mse_ratio", "size"),
            gm_mse=("mse_ratio", geo_mean),
            med_support_frac=("support_frac", "median"),
            med_top5_lift=("top5_lift_vs_uniform", "median"),
            med_top5_coverage_frac=("comp_top5_coverage_frac", "median"),
            med_top1_repeat=("comp_top1_repeat_rate", "median"),
            med_density=("w_comp_density", "median"),
            med_overlap5=("avg_pairwise_overlap_top5", "median"),
        )
        .reset_index()
    )
    summary_by_method = (
        candidates.groupby(["method_family", "method_name", "k_ratio"])
        .agg(
            n=("mse_ratio", "size"),
            pass_rate=("pass_both", "mean"),
            gm_mse=("mse_ratio", geo_mean),
            gm_mae=("mae_ratio", geo_mean),
            med_support_frac=("support_frac", "median"),
            med_top5=("avg_top5_mass", "median"),
            med_top5_coverage=("comp_top5_coverage_frac", "median"),
            med_overlap5=("avg_pairwise_overlap_top5", "median"),
            med_cosine=("avg_pairwise_cosine_between_latents", "median"),
        )
        .reset_index()
        .sort_values("gm_mse")
    )

    save_best_ratio_heatmap(best, asset_dir / "best_ratio_heatmap.png")
    save_ratio_distribution(candidates, asset_dir / "mse_ratio_distribution.png")
    save_sparsity_scatter(candidates, asset_dir / "sparsity_vs_mse.png")
    save_sparsity_bars(summary_by_norm, asset_dir / "sparsity_bars.png")

    triptych_rows = []
    for _, row in best.iterrows():
        triptych_rows.append(row)
    extra_specs = [
        ("Traffic", 96, "latent_query_attention_softmax_entropy", 0.3),
        ("Traffic", 96, "MLP_attention_entmax_entropy", 0.3),
        ("Weather", 720, "MLP_attention_softmax_entropy", 0.3),
    ]
    for dataset, pred_len, method_name, k_ratio in extra_specs:
        sub = candidates[
            (candidates["dataset"] == dataset)
            & (candidates["pred_len"] == pred_len)
            & (candidates["method_name"] == method_name)
            & (candidates["k_ratio"] == k_ratio)
        ]
        if not sub.empty:
            triptych_rows.append(sub.iloc[0])

    seen = set()
    triptych_cards = []
    for row in triptych_rows:
        key = (row["dataset"], int(row["pred_len"]), row["method_name"], float(row["k_ratio"]))
        if key in seen:
            continue
        seen.add(key)
        filename = f"triptych_{row['dataset']}_p{int(row['pred_len'])}_{row['method_name']}_kr{str(row['k_ratio']).replace('.', 'p')}.png"
        out = render_triptych(row, asset_dir / filename)
        if out is None:
            continue
        interp = []
        if row["support_frac"] < 0.05 and row.get("comp_top1_repeat_rate", 0) > 0.5:
            interp.append("W_comp는 매우 sparse하지만 여러 latent가 같은 source에 몰리는 collapse 패턴이다.")
        elif row["support_frac"] > 0.3:
            interp.append("W_comp support가 커서 대표 변수 선택이라기보다 전역 혼합에 가깝다.")
        else:
            interp.append("W_comp는 일부 sparsity를 만들었지만 coverage와 expansion에서 정보가 다시 섞인다.")
        if row.get("w_eff_density", np.nan) > 0.8:
            interp.append("W_eff가 거의 dense라 최종 target 관점에서는 sparse 구조가 보존되지 않는다.")
        if row["mse_ratio"] > 1.05 or row["mae_ratio"] > 1.05:
            interp.append("성능 ratio가 1.05를 넘어 압축 정보 손실이 예측 성능을 압도한다.")
        card = f"""
        <article class="heat-card">
          <h4>{html.escape(row['dataset'])} pred={int(row['pred_len'])} · {html.escape(row['method_name'])} · K={int(row['num_latents_K'])}</h4>
          <p class="metric-line">MSE ratio <strong>{fmt(row['mse_ratio'])}</strong>, MAE ratio <strong>{fmt(row['mae_ratio'])}</strong>,
          support/C <strong>{fmt(row['support_frac'])}</strong>, top-5 coverage/C <strong>{fmt(row.get('comp_top5_coverage_frac', np.nan))}</strong>,
          top1 repeat <strong>{fmt(row.get('comp_top1_repeat_rate', np.nan))}</strong></p>
          {img_tag(out, alt=filename, cls="heatmap")}
          <ul>{''.join('<li>' + html.escape(x) + '</li>' for x in interp)}</ul>
        </article>
        """
        triptych_cards.append(card)

    best_table = best[
        [
            "dataset",
            "pred_len",
            "method_name",
            "k_ratio",
            "num_latents_K",
            "baseline_mse",
            "mse",
            "mse_ratio",
            "baseline_mae",
            "mae",
            "mae_ratio",
            "support_frac",
            "comp_top5_coverage_frac",
            "comp_top1_repeat_rate",
            "avg_pairwise_overlap_top5",
            "w_eff_density",
        ]
    ].copy()
    for col in best_table.columns:
        if col not in ["dataset", "method_name"]:
            best_table[col] = best_table[col].map(fmt)

    pass_table = pass_rows[
        ["dataset", "pred_len", "method_name", "k_ratio", "num_latents_K", "mse_ratio", "mae_ratio", "support_frac", "comp_top5_coverage_frac"]
    ].copy()
    for col in pass_table.columns:
        if col not in ["dataset", "method_name"]:
            pass_table[col] = pass_table[col].map(fmt)

    dataset_table = summary_by_dataset.copy()
    for col in dataset_table.columns:
        if col != "dataset":
            dataset_table[col] = dataset_table[col].map(fmt)

    method_table = summary_by_method.head(12).copy()
    for col in method_table.columns:
        if col not in ["method_family", "method_name"]:
            method_table[col] = method_table[col].map(fmt)

    worst_table = worst[
        [
            "dataset",
            "pred_len",
            "method_name",
            "k_ratio",
            "num_latents_K",
            "mse_ratio",
            "mae_ratio",
            "support_frac",
            "avg_top5_mass",
            "comp_top5_coverage_frac",
            "comp_top1_repeat_rate",
            "w_comp_density",
            "w_eff_density",
        ]
    ].copy()
    for col in worst_table.columns:
        if col not in ["dataset", "method_name"]:
            worst_table[col] = worst_table[col].map(fmt)

    total_candidates = len(candidates)
    pass_count = int(candidates["pass_both"].sum())
    exact_mse_wins = int(candidates["exact_mse_win"].sum())
    exact_mae_wins = int(candidates["exact_mae_win"].sum())
    best_cell_pass_count = int(((best["mse_ratio"] < 1.01) & (best["mae_ratio"] < 1.01)).sum())
    gm_mse = geo_mean(candidates["mse_ratio"])
    gm_mae = geo_mean(candidates["mae_ratio"])

    css = """
    <style>
    :root { color-scheme: light; --ink:#111827; --muted:#4b5563; --line:#d1d5db; --soft:#f8fafc; --blue:#1d4ed8; --red:#b91c1c; --green:#166534; --gold:#a16207; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#ffffff; line-height:1.55; }
    main { max-width:1180px; margin:0 auto; padding:34px 28px 70px; }
    h1 { font-size:32px; margin:0 0 16px; letter-spacing:0; }
    h2 { font-size:23px; margin:38px 0 12px; border-top:1px solid var(--line); padding-top:24px; }
    h3 { font-size:18px; margin:26px 0 10px; }
    h4 { font-size:16px; margin:0 0 8px; }
    p { margin:8px 0 14px; }
    .summary { background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px; padding:18px 20px; }
    .summary ul { margin:8px 0 0 20px; padding:0; }
    .callout { border-left:4px solid var(--red); background:#fff7ed; padding:12px 16px; margin:16px 0; }
    .ok { border-left-color:var(--green); background:#f0fdf4; }
    .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }
    .metric { border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; }
    .metric strong { display:block; font-size:24px; line-height:1.15; color:var(--blue); }
    .metric span { color:var(--muted); font-size:13px; }
    .chart { display:block; width:100%; max-width:980px; margin:12px auto 20px; border:1px solid var(--line); border-radius:8px; background:white; }
    .heatmap { display:block; width:100%; max-width:100%; border:1px solid var(--line); border-radius:8px; background:white; }
    .heat-card { border:1px solid var(--line); border-radius:8px; padding:14px; margin:18px 0 24px; background:#fff; }
    .metric-line { color:var(--muted); font-size:14px; }
    table { border-collapse:collapse; width:100%; font-size:13px; margin:12px 0 20px; }
    th, td { border-bottom:1px solid #e5e7eb; padding:7px 8px; vertical-align:top; text-align:right; }
    th { background:#f8fafc; color:#374151; font-weight:650; }
    td:first-child, th:first-child, td:nth-child(3), th:nth-child(3) { text-align:left; }
    code { background:#f3f4f6; padding:1px 4px; border-radius:4px; }
    .small { color:var(--muted); font-size:13px; }
    @media (max-width:860px) { .metric-grid { grid-template-columns:1fr 1fr; } main { padding:24px 14px 50px; } }
    </style>
    """

    html_doc = f"""<!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Sparse Attention W_comp 진단 리포트</title>
      {css}
    </head>
    <body>
    <main>
      <h1>Sparse Attention W_comp 진단 리포트</h1>
      <section class="summary">
        <h2 style="border-top:0;padding-top:0;margin-top:0">기술 요약</h2>
        <ul>
          <li><strong>결론: 현재 attention-style W_comp는 iTransformer baseline 대체 후보로 실패했다.</strong> candidate 128개 중 MSE와 MAE를 동시에 pass한 row는 {pass_count}개뿐이고, dataset × horizon별 best candidate 8개는 모두 strict pass를 만족하지 못했다.</li>
          <li><strong>Sparse 자체는 일부 조합에서 만들어졌지만, representative가 되지는 않았다.</strong> entmax/MLP 계열은 특히 Traffic에서 support가 극단적으로 작아지며 같은 source에 몰리는 collapse가 생겼고, softmax/latent-query 계열은 여전히 전역 혼합에 가까웠다.</li>
          <li><strong>성능 실패의 핵심 원인은 변수 identity와 coverage 손실이다.</strong> W_comp가 변수를 고르게 대표하지 못한 상태에서 encoder에 K개 token만 들어가고, W_exp는 빠진 target-specific 정보를 복원할 수 없다.</li>
          <li><strong>다음 실험은 단순 sparsity가 아니라 coverage-constrained representative selection이어야 한다.</strong> latent별 sparse 선택뿐 아니라 source coverage, latent 간 중복 억제, target별 제한적 expansion을 같이 설계해야 한다.</li>
        </ul>
      </section>

      <section>
        <h2>정량 결과는 거의 전 구간에서 baseline보다 나쁘다</h2>
        <div class="metric-grid">
          <div class="metric"><strong>{pass_count}/{total_candidates}</strong><span>candidate row pass<br>MSE &lt; 1.01, MAE &lt; 1.01</span></div>
          <div class="metric"><strong>{best_cell_pass_count}/8</strong><span>best-per-cell pass</span></div>
          <div class="metric"><strong>{fmt(gm_mse)}</strong><span>geometric mean MSE ratio</span></div>
          <div class="metric"><strong>{fmt(gm_mae)}</strong><span>geometric mean MAE ratio</span></div>
        </div>
        <p>아래 heatmap은 각 dataset × horizon에서 MSE 기준 가장 좋은 candidate만 남긴 것이다. 숫자는 MSE ratio / MAE ratio이며 1보다 작으면 baseline보다 좋고, 1.01 미만이면 pass다. Weather 720에서만 일부 row가 pass에 가까웠고, Traffic은 모든 후보가 크게 무너졌다.</p>
        {img_tag(asset_dir / "best_ratio_heatmap.png", "best ratio heatmap")}
        <p class="small">Exact win은 MSE 기준 {exact_mse_wins}개, MAE 기준 {exact_mae_wins}개다. 전체 candidate의 GM ratio는 MSE {fmt(gm_mse)}, MAE {fmt(gm_mae)}로 baseline보다 크게 높다.</p>
        <h3>Best candidate per cell</h3>
        {df_to_html_table(best_table)}
        <h3>Pass row는 Weather 720의 MLP softmax 계열 2개뿐이다</h3>
        {df_to_html_table(pass_table) if len(pass_table) else "<p>Pass row 없음.</p>"}
      </section>

      <section>
        <h2>Dataset별 실패 양상이 다르지만, 공통 원인은 대표성 부족이다</h2>
        <p>Weather는 작은 C=21이라 혼합이 어느 정도 버티지만 pass가 안정적이지 않다. Electricity와 Solar는 5~18% 악화 수준에서 머물고, Traffic은 변수 수가 크고 이질적이라 잘못된 압축이 곧바로 1.7~3.6배 MSE 악화로 이어진다.</p>
        {img_tag(asset_dir / "mse_ratio_distribution.png", "MSE ratio distribution")}
        <h3>Dataset summary</h3>
        {df_to_html_table(dataset_table)}
      </section>

      <section>
        <h2>W_comp는 두 가지 실패 모드로 나뉜다: dense mixing 또는 sparse collapse</h2>
        <p><strong>의도한 sparse representative</strong>라면 latent별로 적은 source를 보되, latent들이 서로 다른 source 영역을 커버해야 한다. 하지만 softmax는 밀도가 거의 1에 가까워 dense mixing이고, entmax는 sparse해지는 대신 coverage가 낮거나 top source가 반복되는 경우가 많았다.</p>
        {img_tag(asset_dir / "sparsity_vs_mse.png", "sparsity versus mse")}
        <p>Scatter에서 왼쪽으로 갈수록 W_comp가 sparse하지만, 성능이 좋아지지 않는다. 특히 Traffic은 sparse한 점들도 위쪽에 몰려 있다. 이는 sparsity가 정보 보존이 아니라 변수 제거/collapse로 작동했음을 의미한다.</p>
        {img_tag(asset_dir / "sparsity_bars.png", "sparsity bars")}
        <h3>Method summary, GM MSE 기준 상위 12개</h3>
        {df_to_html_table(method_table)}
      </section>

      <section>
        <h2>Heatmap 진단: sparse하게 된 경우도 대표 변수 생성은 아니다</h2>
        <p>아래 그림은 각 조합의 <code>W_comp</code>, learned <code>W_exp</code>, 그리고 target 관점의 effective mapping <code>W_eff = W_exp W_comp</code>를 같은 방식으로 row-normalized absolute value로 본 것이다. 밝은 영역은 해당 row가 상대적으로 강하게 의존하는 source/token을 의미한다.</p>
        {''.join(triptych_cards)}
      </section>

      <section>
        <h2>가장 나쁜 실패 사례는 Traffic 96에서 반복된다</h2>
        <p>Traffic 96은 대부분 후보가 baseline 대비 3배 이상 나빠졌다. 흥미로운 점은 dense한 후보와 극단적으로 sparse한 후보가 모두 실패했다는 것이다. 즉 문제는 “sparse 여부” 하나가 아니라, 어떤 변수를 얼마나 균형 있게 보존하고 target으로 되돌리는지가 핵심이다.</p>
        {df_to_html_table(worst_table)}
      </section>

      <section>
        <h2>왜 성능이 나쁜가</h2>
        <div class="callout">
          <p><strong>1. W_comp의 sparsity가 coverage를 보장하지 않는다.</strong> entmax는 확률을 0으로 만들 수 있지만, 서로 다른 latent가 같은 고신호 변수만 반복해서 선택하는 것을 막지 않는다. Traffic 96 best 후보도 support는 작지만 top-1 반복과 top-5 overlap이 커서 대표 토큰 집합으로 보기 어렵다.</p>
          <p><strong>2. W_exp는 빠진 변수 identity를 복원하지 못한다.</strong> encoder에는 K개 latent만 들어가므로, W_comp에서 약하게 들어간 변수의 국소 패턴은 이미 손실된다. learned linear expansion은 남은 latent를 target C개로 재분배할 뿐, 사라진 variable-specific temporal state를 다시 만들 수 없다.</p>
          <p><strong>3. latent_query_attention은 query가 전역적으로 비슷한 key를 보며 global mixture가 되기 쉽다.</strong> support와 cosine이 높고 W_eff density도 높은 조합은 계산 graph는 줄였지만 의미상 변수 압축이라기보다 전체 평균적 mixture에 가깝다.</p>
          <p><strong>4. orthogonal loss는 이 실패를 직접 막지 못한다.</strong> hidden representation 간 각도는 벌릴 수 있어도, source coverage나 target별 identity preservation을 보장하지 않는다.</p>
        </div>
      </section>

      <section>
        <h2>다음 실험에서 유의할 점</h2>
        <ul>
          <li><strong>TopS sparse만으로는 부족하다.</strong> source coverage 또는 load-balancing constraint가 없으면 sparse collapse가 생긴다.</li>
          <li><strong>W_exp도 제한해야 한다.</strong> target이 모든 latent를 볼 수 있으면 W_eff가 다시 dense해지고, 압축 구조의 해석 가능성이 사라진다. target별 top-r latent expansion 또는 residual identity path를 비교해야 한다.</li>
          <li><strong>Traffic은 대표성 검증의 stress test로 써야 한다.</strong> Weather만 보면 global mixture가 통하는 것처럼 보일 수 있지만, Traffic에서는 identity 손실이 즉시 드러난다.</li>
          <li><strong>새 loss는 sparsity가 아니라 coverage/diversity/information preservation에 직접 연결되어야 한다.</strong> 예: source coverage entropy, latent-source load balance, target reconstruction residual, grouped representative constraints.</li>
        </ul>
      </section>

      <section>
        <h2>Scope와 metric 정의</h2>
        <p>분석 범위는 <code>results/sparse_attention_wcomp/results.csv</code>의 full 136 rows다. Baseline은 iTransformer 8 rows, candidates는 4 datasets × 2 horizons × 2 k_ratio × 2 method families × 2 normalizations × entropy on/off = 128 rows다. Horizon은 이번 v3 runner 설정에 맞춰 96과 720만 포함한다.</p>
        <p><code>MSE ratio</code>와 <code>MAE ratio</code>는 candidate metric을 같은 dataset × horizon의 iTransformer baseline metric으로 나눈 값이다. <code>support/C</code>는 W_comp row의 effective support를 변수 수 C로 나눈 값이고, <code>top-5 coverage/C</code>는 latent별 top-5 source를 모두 모았을 때 unique source가 전체 변수 중 차지하는 비율이다.</p>
      </section>
    </main>
    </body>
    </html>
    """

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_doc, encoding="utf-8")
    candidates.to_csv(asset_dir / "diagnostic_candidates_enriched.csv", index=False)
    best.to_csv(asset_dir / "diagnostic_best_per_cell.csv", index=False)
    print(f"wrote {out_html}")
    print(f"assets {asset_dir}")


if __name__ == "__main__":
    main()
