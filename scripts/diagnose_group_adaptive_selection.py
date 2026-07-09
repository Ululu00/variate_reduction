#!/usr/bin/env python3
"""Diagnose group-wise adaptive representative choices on Solar windows."""

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import Dataset_Solar


def entropy(probs):
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def selection_counts(dataset, group_ids, anchor_indices, stride):
    data = np.asarray(dataset.data_x, dtype=np.float32)
    seq_len = int(dataset.seq_len)
    pred_len = int(dataset.pred_len)
    num_windows = len(dataset)
    starts = list(range(0, num_windows, max(1, int(stride))))
    num_groups = int(group_ids.max()) + 1
    num_variates = data.shape[1]
    counts = np.zeros((num_groups, num_variates), dtype=np.int64)

    members_by_group = [np.nonzero(group_ids == group_idx)[0] for group_idx in range(num_groups)]
    for start in starts:
        window = data[start:start + seq_len]
        centered = window - window.mean(axis=0, keepdims=True)
        scale = np.sqrt(centered.var(axis=0, keepdims=True) + 1e-5)
        shape = centered / scale
        group_mean = np.zeros((seq_len, num_groups), dtype=np.float32)
        for group_idx, members in enumerate(members_by_group):
            group_mean[:, group_idx] = shape[:, members].mean(axis=1)
        scores = np.zeros(num_variates, dtype=np.float32)
        for group_idx, members in enumerate(members_by_group):
            diff = shape[:, members] - group_mean[:, group_idx:group_idx + 1]
            scores[members] = -(diff * diff).mean(axis=0)
        scores[anchor_indices] += 1e-4
        for group_idx, members in enumerate(members_by_group):
            selected = members[int(np.argmax(scores[members]))]
            counts[group_idx, selected] += 1
    return counts, len(starts)


def summarize_counts(split, counts, total_windows, group_ids, anchor_indices):
    rows = []
    for group_idx in range(counts.shape[0]):
        group_counts = counts[group_idx]
        members = np.nonzero(group_ids == group_idx)[0]
        member_counts = group_counts[members]
        denom = max(1, int(member_counts.sum()))
        probs = member_counts.astype(np.float64) / denom
        top_order = np.argsort(member_counts)[::-1]
        top_member = int(members[top_order[0]])
        top_count = int(member_counts[top_order[0]])
        anchor_idx = int(anchor_indices[group_idx])
        anchor_count = int(group_counts[anchor_idx])
        rows.append(
            {
                "split": split,
                "group_id": group_idx,
                "group_size": int(members.size),
                "anchor_idx": anchor_idx,
                "anchor_select_rate": anchor_count / denom,
                "unique_selected": int((member_counts > 0).sum()),
                "selection_entropy": entropy(probs),
                "selection_entropy_norm": entropy(probs) / math.log(max(2, members.size)),
                "top_selected_idx": top_member,
                "top_selected_rate": top_count / denom,
                "num_windows": int(total_windows),
            }
        )
    return rows


def write_heatmap(path, counts, group_ids):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    denom = counts.sum(axis=1, keepdims=True).clip(min=1)
    rates = counts / denom
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(rates, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Variable index")
    ax.set_ylabel("Group id")
    ax.set_title("Group-adaptive selected-variable rate")
    fig.colorbar(im, ax=ax, label="selection rate")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="./dataset/Solar/")
    parser.add_argument("--data_path", default="solar_AL.txt")
    parser.add_argument("--anchor_map", required=True)
    parser.add_argument("--pred_len", type=int, default=192)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--out_dir", default="./results/current_result_inventory/group_adaptive")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = np.load(args.anchor_map)
    anchor_indices = anchor["anchor_indices"].astype(np.int64)
    group_ids = anchor["group_ids"].astype(np.int64)

    all_rows = []
    md_lines = [
        "# Group-Adaptive 대표변수 진단",
        "",
        "`variate_group_adaptive_selection`과 같은 scoring rule을 쓴다. 각 변수의 input window를 normalize하고 "
        "group centroid shape을 계산한 뒤, 그 centroid와 가장 가까운 group member를 선택한다.",
        "",
        f"Anchor map: `{args.anchor_map}`",
        f"Window stride: `{args.stride}`",
        "",
    ]
    for split in args.splits:
        dataset = Dataset_Solar(
            root_path=args.root_path,
            flag=split,
            size=[args.seq_len, args.label_len, args.pred_len],
            features="M",
            data_path=args.data_path,
            target="none",
            scale=True,
            timeenc=0,
            freq="t",
        )
        counts, total = selection_counts(dataset, group_ids, anchor_indices, args.stride)
        rows = summarize_counts(split, counts, total, group_ids, anchor_indices)
        all_rows.extend(rows)
        df = pd.DataFrame(rows)
        mean_anchor = df["anchor_select_rate"].mean()
        mean_unique = df["unique_selected"].mean()
        mean_entropy = df["selection_entropy_norm"].mean()
        md_lines.extend(
            [
                f"## {split}",
                "",
                f"- sampled window: `{total}`",
                f"- 평균 anchor 선택률: `{mean_anchor:.4f}`",
                f"- group당 평균 unique 선택 변수 수: `{mean_unique:.2f}`",
                f"- 평균 normalized selection entropy: `{mean_entropy:.4f}`",
                "",
            ]
        )
        np.save(out_dir / f"{split}_selection_counts.npy", counts)
        write_heatmap(out_dir / f"{split}_selection_heatmap.png", counts, group_ids)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "group_adaptive_selection_stats.csv", index=False)
    md_lines.append("상세 group row: `group_adaptive_selection_stats.csv`.")
    (out_dir / "group_adaptive_selection_diagnostics.md").write_text("\n".join(md_lines) + "\n")
    print(f"{out_dir}에 진단 결과 작성 완료")


if __name__ == "__main__":
    main()
