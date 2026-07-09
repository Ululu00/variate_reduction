import argparse
import base64
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SPARSITY_COLUMNS = [
    "avg_comp_entropy",
    "avg_effective_support",
    "avg_top1_mass",
    "avg_top5_mass",
    "avg_top10_mass",
    "avg_pairwise_overlap_top5",
    "avg_pairwise_overlap_top10",
    "avg_pairwise_cosine_between_latents",
    "w_comp_density",
    "w_eff_density",
]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fmt(value, digits=4):
    value = as_float(value)
    if math.isnan(value):
        return ""
    return ("{:.%df}" % digits).format(value)


def image_data_uri(path):
    path = Path(path)
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:image/png;base64,{}".format(data)


def save_heatmap(matrix, path, title, cmap="viridis", symmetric=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    fig_w = max(7.0, min(14.0, matrix.shape[1] / 60.0))
    fig_h = max(4.0, min(12.0, matrix.shape[0] / 60.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    kwargs = {}
    if symmetric:
        vmax = np.nanmax(np.abs(matrix)) if matrix.size else 1.0
        kwargs.update(vmin=-vmax, vmax=vmax)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("input/source index")
    ax.set_ylabel("output/target index")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ensure_plot_set(run_dir):
    run_dir = Path(run_dir)
    compress_path = run_dir / "compress_weight.npy"
    expand_path = run_dir / "expand_weight.npy"
    effective_path = run_dir / "effective_weight.npy"
    if not compress_path.exists():
        return []
    A = np.load(compress_path)
    generated = []
    if not (run_dir / "w_comp_heatmap.png").exists():
        save_heatmap(A, run_dir / "w_comp_heatmap.png", "W_comp attention response")
        generated.append(run_dir / "w_comp_heatmap.png")
    eps = 1e-8
    entropy = -(A * np.log(A + eps)).sum(axis=-1)
    support = np.exp(entropy)
    sorted_mass = np.sort(A, axis=-1)[:, ::-1]
    top1 = sorted_mass[:, :1].sum(axis=-1)
    top5 = sorted_mass[:, : min(5, sorted_mass.shape[-1])].sum(axis=-1)
    top10 = sorted_mass[:, : min(10, sorted_mass.shape[-1])].sum(axis=-1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entropy_plot = run_dir / "entropy_plot.png"
    if not entropy_plot.exists():
        fig, ax = plt.subplots(figsize=(7, 3), dpi=140)
        ax.plot(np.arange(entropy.shape[0]), entropy, marker="o", linewidth=1)
        ax.set_title("Per-latent W_comp entropy")
        ax.set_xlabel("latent token index")
        ax.set_ylabel("entropy")
        fig.tight_layout()
        fig.savefig(entropy_plot)
        plt.close(fig)
        generated.append(entropy_plot)

    support_plot = run_dir / "effective_support_hist.png"
    if not support_plot.exists():
        fig, ax = plt.subplots(figsize=(6, 3), dpi=140)
        ax.hist(support, bins=min(20, max(3, support.size)))
        ax.set_title("Effective support size")
        ax.set_xlabel("exp(entropy)")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(support_plot)
        plt.close(fig)
        generated.append(support_plot)

    topk_plot = run_dir / "topk_mass_hist.png"
    if not topk_plot.exists():
        fig, ax = plt.subplots(figsize=(6, 3), dpi=140)
        ax.hist(top1, alpha=0.55, label="top1")
        ax.hist(top5, alpha=0.55, label="top5")
        ax.hist(top10, alpha=0.55, label="top10")
        ax.set_title("Top-k mass")
        ax.set_xlabel("mass")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(topk_plot)
        plt.close(fig)
        generated.append(topk_plot)

    overlap_plot = run_dir / "overlap_matrix.png"
    if not overlap_plot.exists():
        topk = min(10, A.shape[-1])
        indices = np.argsort(A, axis=-1)[:, -topk:]
        mask = np.zeros_like(A, dtype=np.float32)
        for row_idx, cols in enumerate(indices):
            mask[row_idx, cols] = 1.0
        overlap = mask @ mask.T / float(topk)
        save_heatmap(overlap, overlap_plot, "Selected variable overlap top10")
        generated.append(overlap_plot)

    if expand_path.exists() and not (run_dir / "w_exp_heatmap.png").exists():
        save_heatmap(np.load(expand_path), run_dir / "w_exp_heatmap.png", "W_exp learned K->C", symmetric=True)
        generated.append(run_dir / "w_exp_heatmap.png")
    if effective_path.exists() and not (run_dir / "w_eff_heatmap.png").exists():
        save_heatmap(np.load(effective_path), run_dir / "w_eff_heatmap.png", "W_eff = W_exp @ W_comp", symmetric=True)
        generated.append(run_dir / "w_eff_heatmap.png")
    return generated


def collect_plot_dirs(plot_root):
    root = Path(plot_root)
    if not root.exists():
        return {}
    result = defaultdict(list)
    for path in root.rglob("compress_weight.npy"):
        run_dir = path.parent
        method = run_dir.parent.name
        result[method].append(run_dir)
    return result


def table_html(rows, columns):
    out = ["<table>", "<thead><tr>"]
    for column in columns:
        out.append("<th>{}</th>".format(html.escape(column)))
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for column in columns:
            value = row.get(column, "")
            if column in {"mse", "mae"} or column in SPARSITY_COLUMNS:
                value = fmt(value)
            out.append("<td>{}</td>".format(html.escape(str(value))))
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def display_family(name):
    return "latent-query attention" if name == "latent_query_attention" else name


def build_html(rows, plot_dirs):
    candidates = [row for row in rows if row.get("method_family") != "iTransformer"]
    baselines = [row for row in rows if row.get("method_family") == "iTransformer"]
    for dirs in plot_dirs.values():
        for run_dir in dirs:
            ensure_plot_set(run_dir)
    style = """
    body { font-family: Arial, sans-serif; margin: 24px; color: #172033; }
    h1, h2, h3 { color: #0b2e6f; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d7deea; padding: 6px 8px; text-align: right; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) { text-align: left; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }
    .figure { border: 1px solid #d7deea; padding: 10px; border-radius: 6px; }
    .figure img { max-width: 100%; height: auto; display: block; }
    .muted { color: #5b667a; }
    """
    out = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Sparse Attention W_comp Report</title>",
        "<style>{}</style></head><body>".format(style),
        "<h1>Sparse Attention W_comp Quick Report</h1>",
        "<p>Dense linear baseline is excluded. Baseline is the original iTransformer without variate reduction.</p>",
        "<h2>Experiment Summary</h2>",
        "<ul>",
        "<li>Methods: MLP_attention, latent-query attention</li>",
        "<li>Normalization: softmax, entmax-1.5</li>",
        "<li>Entropy loss: 0 or 1e-3 on W_comp attention response</li>",
        "<li>Datasets/horizons are read from the result CSV.</li>",
        "</ul>",
        "<h2>iTransformer Baseline</h2>",
        table_html(baselines, ["dataset", "pred_len", "seed", "method_family", "method_name", "mse", "mae", "train_time_sec", "peak_gpu_allocated_mb"]),
        "<h2>Candidate Quantitative Results</h2>",
        table_html(candidates, ["dataset", "pred_len", "seed", "k_ratio", "method_family", "method_name", "normalization", "lambda_entropy", "mse", "mae", "train_time_sec", "peak_gpu_allocated_mb"]),
        "<h2>Sparsity Behavior Summary</h2>",
        table_html(candidates, ["dataset", "pred_len", "k_ratio", "method_name"] + SPARSITY_COLUMNS),
        "<h2>Dataset-wise Analysis</h2>",
    ]
    by_dataset = defaultdict(list)
    for row in candidates:
        by_dataset[row.get("dataset", "")].append(row)
    for dataset, dataset_rows in sorted(by_dataset.items()):
        entmax_rows = [row for row in dataset_rows if row.get("normalization") == "entmax15"]
        softmax_rows = [row for row in dataset_rows if row.get("normalization") == "softmax"]
        ent_support = np.nanmean([as_float(row.get("avg_effective_support")) for row in entmax_rows])
        soft_support = np.nanmean([as_float(row.get("avg_effective_support")) for row in softmax_rows])
        out.append("<h3>{}</h3>".format(html.escape(dataset)))
        out.append(
            "<p>Average effective support: softmax {:.4f}, entmax {:.4f}. Lower support means a smaller selected variable subset.</p>".format(
                soft_support,
                ent_support,
            )
        )
    out.append("<h2>Method-wise Figures</h2>")
    for method, dirs in sorted(plot_dirs.items()):
        out.append("<h3>{}</h3>".format(html.escape(display_family(method))))
        for run_dir in dirs[:4]:
            out.append("<div class='figure'><p class='muted'>{}</p><div class='grid'>".format(html.escape(str(run_dir))))
            for filename in [
                "w_comp_heatmap.png",
                "entropy_plot.png",
                "effective_support_hist.png",
                "topk_mass_hist.png",
                "overlap_matrix.png",
                "w_exp_heatmap.png",
                "w_eff_heatmap.png",
            ]:
                uri = image_data_uri(run_dir / filename)
                if uri:
                    out.append("<div><strong>{}</strong><img src='{}'></div>".format(html.escape(filename), uri))
            out.append("</div></div>")
    out.append("<h2>Decision Guide</h2>")
    out.append("<p>PROMOTE if entmax or entropy variants reduce effective support and increase top-k mass without large MSE/MAE degradation or latent subset collapse.</p>")
    out.append("</body></html>")
    return "\n".join(out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_csv", required=True)
    parser.add_argument("--plot_root", required=True)
    parser.add_argument("--out_html", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.result_csv)
    plot_dirs = collect_plot_dirs(args.plot_root)
    html_text = build_html(rows, plot_dirs)
    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    print("wrote {}".format(out_path))


if __name__ == "__main__":
    main()
