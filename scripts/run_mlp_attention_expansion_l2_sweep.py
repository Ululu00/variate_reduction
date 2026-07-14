#!/usr/bin/env python3
"""Run the new MLP-attention softmax loss grid.

The existing sparse-attention CSV supplies iTransformer baselines for the
report; this script launches only configurations not present in this grid.
"""

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

from run_sparse_attention_wcomp_grid import DATASET_SPECS, k_for_ratio


ORTHOGONAL_WEIGHT = 0.01
RECONSTRUCTION_WEIGHTS = (0.01, 0.1)
EXPANSION_L2_WEIGHTS = (1e-4, 1e-3)
WCOMP_ENTROPY_WEIGHTS = (1e-3, 1e-2)


def float_key(value):
    return "{:.12g}".format(float(value))


def token(value):
    return ("{:.0e}".format(float(value))).replace("-", "m")


def method_name(wcomp_normalization, reconstruction, expansion_l2, entropy):
    return "MLP_attention_{}_orth{}_rec{}_wexp_l2{}_ent{}".format(
        wcomp_normalization,
        token(ORTHOGONAL_WEIGHT), token(reconstruction), token(expansion_l2), token(entropy)
    )


def build_runs(args):
    runs = []
    for dataset in args.datasets:
        spec = DATASET_SPECS[dataset]
        for pred_len in args.pred_lens:
            for k_ratio in args.k_ratios:
                k_value = k_for_ratio(spec["num_variates"], k_ratio)
                for reconstruction in RECONSTRUCTION_WEIGHTS:
                    for expansion_l2 in EXPANSION_L2_WEIGHTS:
                        for entropy in WCOMP_ENTROPY_WEIGHTS:
                            runs.append({
                                "dataset": dataset,
                                "pred_len": int(pred_len),
                                "k_ratio": float(k_ratio),
                                "k_value": int(k_value),
                                "seed": int(args.seed),
                                "reconstruction": float(reconstruction),
                                "expansion_l2": float(expansion_l2),
                                "entropy": float(entropy),
                                "wcomp_normalization": args.wcomp_normalization,
                                "method_name": method_name(
                                    args.wcomp_normalization, reconstruction, expansion_l2, entropy
                                ),
                                "spec": spec,
                            })
    return runs


def run_key(run):
    return (
        run["dataset"], str(run["pred_len"]), float_key(run["k_ratio"]),
        float_key(ORTHOGONAL_WEIGHT), float_key(run["reconstruction"]),
        str(run["wcomp_normalization"]), float_key(run["expansion_l2"]),
        float_key(run["entropy"]), str(run["seed"]),
    )


def normalization_from_result(row):
    value = str(row.get("wcomp_normalization") or row.get("normalization") or "").strip().lower()
    if value:
        return value
    method_name = str(row.get("method_name", "")).lower()
    if "entmax" in method_name:
        return "entmax15"
    return "softmax"


def completed_keys(result_csv):
    path = Path(result_csv)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (
                row.get("dataset", ""), str(row.get("pred_len", "")),
                float_key(row.get("k_ratio", row.get("target_k_ratio", 0.0))),
                float_key(row.get("orthogonal_loss_weight", 0.0)),
                float_key(row.get("reconstruction_loss_weight", 0.0)),
                normalization_from_result(row),
                float_key(row.get("expansion_weight_l2_loss_weight", 0.0)),
                float_key(row.get("wcomp_entropy_loss_weight", 0.0)), str(row.get("seed", "")),
            )
            for row in rows
            if row.get("status", "success") in {"", "success"}
        }


def command_for(run, args):
    spec = run["spec"]
    setting = "{}_pl{}_ratio{}_k{}_seed{}".format(
        run["dataset"], run["pred_len"], str(run["k_ratio"]).replace(".", "p"),
        run["k_value"], run["seed"],
    )
    plot_dir = Path(args.plot_root) / setting / run["method_name"]
    train_epochs = 1 if args.smoke else args.train_epochs
    patience = 1 if args.smoke else args.patience
    command = [
        sys.executable, "run.py", "--is_training", "1",
        "--model_id", "{}_{}".format(setting, run["method_name"]),
        "--model", "iTransformer", "--data", spec["data"],
        "--root_path", spec["root_path"], "--data_path", spec["data_path"],
        "--features", spec["features"], "--target", spec.get("target", "OT"),
        "--freq", spec.get("freq", "h"), "--seq_len", spec["seq_len"],
        "--pred_len", run["pred_len"], "--enc_in", spec["enc_in"],
        "--dec_in", spec["dec_in"], "--c_out", spec["c_out"],
        "--e_layers", spec["e_layers"], "--d_model", spec["d_model"],
        "--d_ff", spec["d_ff"], "--des", "mlp_attention_expansion_l2_sweep",
        "--use_gpu", "True", "--gpu", args.gpu, "--use_multi_gpu", "False",
        "--variate_reduction_type", "MLP_attention",
        "--reduced_variate_k", run["k_value"],
        "--variate_expansion_type", "slot_learned_linear",
        "--variate_decode_stage", "feature", "--method_family", "MLP_attention",
        "--method_name", run["method_name"], "--wcomp_normalization", run["wcomp_normalization"],
        "--entmax_alpha", args.entmax_alpha, "--orthogonal_loss_weight", ORTHOGONAL_WEIGHT,
        "--reconstruction_loss_weight", run["reconstruction"],
        "--expansion_weight_l2_loss_weight", run["expansion_l2"],
        "--wcomp_entropy_loss_weight", run["entropy"],
        "--coverage_loss_weight", "0.0", "--assignment_entropy_loss_weight", "0.0",
        "--group_attention_entropy_loss_weight", "0.0", "--use_cycle_slot_loss", "False",
        "--cycle_loss_weight", "0.0", "--mae_loss_weight", "0.0",
        "--variate_token_split_factor", "1", "--enforce_k_budget", "True",
        "--result_csv", args.result_csv, "--seed", run["seed"],
        "--experiment_tag", run["dataset"], "--target_k_ratio", run["k_ratio"],
        "--target_k_value", run["k_value"], "--k_selection_mode", "ratio",
        "--k_ratio_denominator", "", "--use_wandb", "False", "--wandb_mode", "disabled",
        "--weight_heatmap_dir", plot_dir, "--skip_epoch_test_eval", "True",
        "--batch_size", spec["batch_size"], "--learning_rate", spec["learning_rate"],
        "--train_epochs", train_epochs, "--patience", patience, "--num_workers", args.num_workers,
    ]
    if args.smoke:
        command.extend(["--max_train_batches", args.max_train_batches, "--max_eval_batches", args.max_eval_batches])
    return [str(value) for value in command]


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["Weather", "Electricity", "Traffic", "Solar"])
    parser.add_argument("--pred_lens", type=int, nargs="+", default=[96, 720])
    parser.add_argument("--k_ratios", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--wcomp_normalization", choices=("softmax", "entmax15"), default="softmax")
    parser.add_argument("--entmax_alpha", type=float, default=1.5)
    parser.add_argument("--result_csv", default="./results/mlp_attention_expansion_l2/260714_mlp_attention_expansion_l2_grid.csv")
    parser.add_argument("--plot_root", default="./results/mlp_attention_expansion_l2/260714_heatmaps")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=2)
    parser.add_argument("--max_eval_batches", type=int, default=2)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = set(args.datasets) - set(DATASET_SPECS)
    if unknown:
        raise ValueError("unknown datasets: {}".format(sorted(unknown)))
    Path(args.result_csv).parent.mkdir(parents=True, exist_ok=True)
    runs = build_runs(args)
    if args.smoke:
        runs = runs[:1]
    if args.dry_run:
        print("planned candidates: {} (MLP-attention {} runs)".format(
            len(runs), args.wcomp_normalization
        ))
        for run in runs[:12]:
            print(run)
        return
    if args.skip_completed:
        completed = completed_keys(args.result_csv)
        before = len(runs)
        runs = [run for run in runs if run_key(run) not in completed]
        print("skip_completed: {} of {} already complete".format(before - len(runs), before), flush=True)
    failures = []
    for index, run in enumerate(runs, start=1):
        command = command_for(run, args)
        print("[RUN {}/{}] {} pl{} K/C={} norm={} rec={} wexp_l2={} entropy={}".format(
            index, len(runs), run["dataset"], run["pred_len"], run["k_ratio"],
            run["wcomp_normalization"], run["reconstruction"], run["expansion_l2"], run["entropy"]), flush=True)
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
        if completed.returncode:
            failures.append((run, completed.returncode))
            if args.stop_on_error:
                break
    if failures:
        raise RuntimeError("{} candidate(s) failed".format(len(failures)))


if __name__ == "__main__":
    main()
