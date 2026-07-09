import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLOR_FAMILIES = {
    "blue": {
        "open": TOKENS["panel"],
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "gold": {
        "open": TOKENS["panel"],
        "xlight": "#FFF4C2",
        "light": "#FFEA8F",
        "base": "#FFE15B",
        "mid": "#B8A037",
        "dark": "#736422",
    },
    "orange": {
        "open": TOKENS["panel"],
        "xlight": "#FFEDDE",
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
}


def use_chart_theme():
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.monospace": ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"],
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(fig, ax, title, subtitle):
    if not title or not subtitle:
        raise ValueError("title and subtitle are required")
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.97, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.925, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])
    sns.despine(ax=ax)


def load_anchor_map(path):
    payload = np.load(path, allow_pickle=False)
    required = {"anchor_indices", "group_ids", "group_sizes", "assigned_abs_corr"}
    missing = required.difference(payload.files)
    if missing:
        raise ValueError("missing required fields in {}: {}".format(path, sorted(missing)))
    data = {
        "anchor_indices": payload["anchor_indices"].astype(np.int64),
        "group_ids": payload["group_ids"].astype(np.int64),
        "group_sizes": payload["group_sizes"].astype(np.int64),
        "assigned_abs_corr": payload["assigned_abs_corr"].astype(np.float64),
        "columns": payload["columns"].astype(str) if "columns" in payload.files else None,
        "expand_top_m": int(payload["expand_top_m"][0]) if "expand_top_m" in payload.files else 0,
        "expand_mask": payload["expand_mask"].astype(np.float32) if "expand_mask" in payload.files else None,
        "expand_init_weight": payload["expand_init_weight"].astype(np.float32) if "expand_init_weight" in payload.files else None,
        "expand_init_power": float(payload["expand_init_power"][0]) if "expand_init_power" in payload.files else 1.0,
    }
    return data


def make_group_frame(data):
    group_ids = data["group_ids"]
    assigned_corr = data["assigned_abs_corr"]
    anchor_indices = data["anchor_indices"]
    group_sizes = data["group_sizes"]
    rows = []
    for group_id, anchor_idx in enumerate(anchor_indices):
        members = np.flatnonzero(group_ids == group_id)
        corr_values = assigned_corr[members]
        rows.append(
            {
                "group_id": int(group_id),
                "anchor_index": int(anchor_idx),
                "anchor_name": str(data["columns"][anchor_idx]) if data["columns"] is not None else str(anchor_idx),
                "group_size": int(group_sizes[group_id]),
                "assigned_corr_mean": float(np.mean(corr_values)),
                "assigned_corr_p05": float(np.quantile(corr_values, 0.05)),
                "assigned_corr_min": float(np.min(corr_values)),
            }
        )
    df = pd.DataFrame(rows)
    df["group_size_rank"] = df["group_size"].rank(method="first", ascending=False).astype(int)
    return df.sort_values("group_size", ascending=False)


def plot_group_size_rank(group_df, out_path, dataset):
    top = group_df.head(24).copy()
    top["label"] = top.apply(
        lambda row: "g{} / v{}".format(int(row["group_id"]), int(row["anchor_index"])),
        axis=1,
    )
    plot_df = top.sort_values("group_size", ascending=True)
    family = COLOR_FAMILIES["orange"]
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    sns.barplot(
        data=plot_df,
        x="group_size",
        y="label",
        color=family["base"],
        edgecolor=family["dark"],
        linewidth=1.0,
        ax=ax,
    )
    median_size = float(group_df["group_size"].median())
    ax.axvline(median_size, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.text(median_size, -0.8, "median {:.0f}".format(median_size), color=TOKENS["muted"], fontsize=8)
    for patch, value in zip(ax.patches, plot_df["group_size"]):
        ax.text(
            patch.get_width() + 1.0,
            patch.get_y() + patch.get_height() / 2,
            "{:.0f}".format(value),
            va="center",
            ha="left",
            fontsize=8,
            color=TOKENS["ink"],
        )
    ax.set_xlabel("Variables assigned to anchor")
    ax.set_ylabel("Largest anchor groups")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    add_chart_header(
        fig,
        ax,
        "{} anchor map has a long group-size tail".format(dataset),
        "Top 24 groups from the train-split correlation-medoid map; large groups are where non-anchor identity loss is most likely.",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_corr_distribution(data, out_path, dataset):
    assigned_corr = data["assigned_abs_corr"]
    family = COLOR_FAMILIES["blue"]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.histplot(
        assigned_corr,
        bins=32,
        color=family["base"],
        edgecolor=family["dark"],
        linewidth=1.0,
        ax=ax,
    )
    for value, label in [
        (float(np.quantile(assigned_corr, 0.05)), "p05"),
        (float(np.mean(assigned_corr)), "mean"),
        (float(np.min(assigned_corr)), "min"),
    ]:
        ax.axvline(value, color=TOKENS["ink"], linestyle=":" if label != "mean" else "-", linewidth=1.0)
        ax.text(value, ax.get_ylim()[1] * 0.9, "{} {:.3f}".format(label, value), rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("Absolute correlation to assigned anchor")
    ax.set_ylabel("Variable count")
    ax.set_xlim(max(0.0, float(assigned_corr.min()) - 0.03), 1.005)
    add_chart_header(
        fig,
        ax,
        "{} assigned-anchor correlation distribution".format(dataset),
        "Distribution across all variables; weak left tails indicate under-represented variables, while strong tails shift suspicion to forecast identity/detail loss.",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_decoder_mask_summary(data, group_df, out_path, dataset):
    expand_mask = data["expand_mask"]
    if expand_mask is None:
        return False
    top_groups = group_df.head(30)["group_id"].to_numpy(dtype=np.int64)
    matrix = expand_mask[:, top_groups]
    row_activity = matrix.sum(axis=1)
    order = np.argsort(-row_activity)[: min(160, matrix.shape[0])]
    matrix = matrix[order, :]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    family = COLOR_FAMILIES["gold"]
    cmap = sns.blend_palette([TOKENS["panel"], family["xlight"], family["light"], family["base"]], as_cmap=True)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        cbar=False,
        linewidths=0.0,
        xticklabels=["g{}".format(item) for item in top_groups],
        yticklabels=False,
    )
    ax.set_xlabel("Largest anchor groups included as decoder columns")
    ax.set_ylabel("Variables, sorted by active decoder anchors")
    add_chart_header(
        fig,
        ax,
        "{} top-m decoder mask stays sparse but targets large groups often".format(dataset),
        "Rows are variables and columns are the 30 largest anchor groups; yellow cells mark anchors used by masked expansion.",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_decoder_init_weight_summary(data, group_df, out_path, dataset):
    init_weight = data["expand_init_weight"]
    if init_weight is None:
        return False
    top_groups = group_df.head(30)["group_id"].to_numpy(dtype=np.int64)
    matrix = init_weight[:, top_groups]
    row_sum = init_weight.sum(axis=1, keepdims=True)
    normalized = init_weight / np.maximum(row_sum, 1e-8)
    entropy = -(normalized * np.log(np.maximum(normalized, 1e-8))).sum(axis=1)
    order = np.argsort(-entropy)[: min(160, matrix.shape[0])]
    matrix = matrix[order, :]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    family = COLOR_FAMILIES["blue"]
    cmap = sns.blend_palette(
        [TOKENS["panel"], family["xlight"], family["light"], family["base"], family["mid"]],
        as_cmap=True,
    )
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        cbar=True,
        cbar_kws={"label": "Initial decoder weight"},
        linewidths=0.0,
        xticklabels=["g{}".format(item) for item in top_groups],
        yticklabels=False,
        vmin=0.0,
        vmax=float(max(0.2, np.quantile(init_weight, 0.99))),
    )
    ax.set_xlabel("Largest anchor groups")
    ax.set_ylabel("Variables with most distributed corr-init weights")
    add_chart_header(
        fig,
        ax,
        "{} corr-init decoder weights remain local, not dense".format(dataset),
        "Rows with highest init entropy; blue intensity is normalized train-correlation weight inside the sparse top-m anchor neighborhood.",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_notes(out_path, dataset, anchor_path, group_df, data, figures):
    assigned_corr = data["assigned_abs_corr"]
    init_weight = data["expand_init_weight"]
    init_summary = {}
    if init_weight is not None:
        row_sum = init_weight.sum(axis=1, keepdims=True)
        normalized = init_weight / np.maximum(row_sum, 1e-8)
        entropy = -(normalized * np.log(np.maximum(normalized, 1e-8))).sum(axis=1)
        init_summary = {
            "decoder_init_weight_max_mean": float(np.max(init_weight, axis=1).mean()),
            "decoder_init_weight_entropy_mean": float(entropy.mean()),
            "decoder_init_weight_active_mean": float((init_weight > 0).sum(axis=1).mean()),
        }
    summary = {
        "dataset": dataset,
        "anchor_path": str(anchor_path),
        "num_variates": int(len(data["group_ids"])),
        "k": int(len(data["anchor_indices"])),
        "k_ratio": float(len(data["anchor_indices"]) / len(data["group_ids"])),
        "expand_top_m": int(data["expand_top_m"]),
        "expand_init_power": float(data["expand_init_power"]),
        "group_size_min": int(group_df["group_size"].min()),
        "group_size_median": float(group_df["group_size"].median()),
        "group_size_max": int(group_df["group_size"].max()),
        "assigned_abs_corr_mean": float(np.mean(assigned_corr)),
        "assigned_abs_corr_p05": float(np.quantile(assigned_corr, 0.05)),
        "assigned_abs_corr_min": float(np.min(assigned_corr)),
        "figures": [str(item) for item in figures],
    }
    summary.update(init_summary)
    corr_interpretation = (
        "assigned-anchor correlation이 충분히 높아, 실패 원인이 약한 anchor matching만이라고 보기는 어렵다."
        if summary["assigned_abs_corr_p05"] >= 0.8
        else "assigned-anchor correlation의 lower tail이 약해 strict K budget에서 이 dataset이 under-represent될 수 있다."
    )
    group_interpretation = (
        "group-size tail이 길어 소수 anchor가 많은 변수를 대표한다. non-anchor identity가 희석될 때 이 group들이 MAE 손실의 원천일 가능성이 높다."
        if summary["group_size_max"] >= max(16, summary["group_size_median"] * 4)
        else "group size가 비교적 완만하다. 이 dataset이 실패한다면 하나의 극단적 group보다 decoder 또는 horizon dynamics가 더 유력하다."
    )
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Anchor Map 진단",
        "",
        "- dataset: `{}`".format(dataset),
        "- source: `{}`".format(anchor_path),
        "- V/K/KV: `{}` / `{}` / `{:.6f}`".format(
            summary["num_variates"], summary["k"], summary["k_ratio"]
        ),
        "- expand top-m: `{}`".format(summary["expand_top_m"]),
        "- expand init power: `{:.4g}`".format(summary["expand_init_power"]),
        "- group size min/median/max: `{}` / `{:.1f}` / `{}`".format(
            summary["group_size_min"], summary["group_size_median"], summary["group_size_max"]
        ),
        "- assigned abs corr mean/p05/min: `{:.6f}` / `{:.6f}` / `{:.6f}`".format(
            summary["assigned_abs_corr_mean"],
            summary["assigned_abs_corr_p05"],
            summary["assigned_abs_corr_min"],
        ),
    ]
    if init_summary:
        lines.extend([
            "- decoder init max-weight mean: `{:.6f}`".format(summary["decoder_init_weight_max_mean"]),
            "- decoder init entropy mean: `{:.6f}`".format(summary["decoder_init_weight_entropy_mean"]),
            "- decoder init active anchors mean: `{:.2f}`".format(summary["decoder_init_weight_active_mean"]),
        ])
    lines.extend([
        "",
        "## 해석",
        "",
        "- {}".format(corr_interpretation),
        "- {}".format(group_interpretation),
        "- anchor-relative residual forecasting이 MAE를 개선한다면, 문제는 token collapse보다 local non-anchor detail 손실이었다는 가설을 지지한다.",
        "",
        "## 그림",
        "",
    ])
    for figure in figures:
        lines.append("- `{}`".format(figure))
    out_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_map", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    use_chart_theme()
    anchor_path = Path(args.anchor_map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or "{}_anchor_map".format(args.dataset.lower())

    data = load_anchor_map(anchor_path)
    group_df = make_group_frame(data)
    group_csv = out_dir / "{}_group_summary.csv".format(prefix)
    group_df.to_csv(group_csv, index=False)

    figures = []
    group_size_png = out_dir / "{}_group_size_rank.png".format(prefix)
    plot_group_size_rank(group_df, group_size_png, args.dataset)
    figures.append(group_size_png)

    corr_png = out_dir / "{}_assigned_corr_hist.png".format(prefix)
    plot_corr_distribution(data, corr_png, args.dataset)
    figures.append(corr_png)

    mask_png = out_dir / "{}_decoder_mask_heatmap.png".format(prefix)
    if plot_decoder_mask_summary(data, group_df, mask_png, args.dataset):
        figures.append(mask_png)

    init_png = out_dir / "{}_decoder_init_weight_heatmap.png".format(prefix)
    if plot_decoder_init_weight_summary(data, group_df, init_png, args.dataset):
        figures.append(init_png)

    notes_path = out_dir / "{}_diagnostics".format(prefix)
    write_notes(notes_path, args.dataset, anchor_path, group_df, data, figures)
    print("wrote {}".format(group_csv))
    for figure in figures:
        print("wrote {}".format(figure))
    print("wrote {}".format(notes_path.with_suffix(".md")))
    print("wrote {}".format(notes_path.with_suffix(".json")))


if __name__ == "__main__":
    main()
