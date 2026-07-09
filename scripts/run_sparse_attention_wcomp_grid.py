import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


DATASET_SPECS = {
    "Weather": {
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "data": "custom",
        "features": "M",
        "freq": "t",
        "target": "OT",
        "num_variates": 21,
        "seq_len": 96,
        "e_layers": 3,
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
        "d_model": 512,
        "d_ff": 512,
        "batch_size": 32,
        "learning_rate": 0.0001,
    },
    "Electricity": {
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "data": "custom",
        "features": "M",
        "freq": "h",
        "target": "OT",
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
    },
    "Traffic": {
        "root_path": "./dataset/traffic/",
        "data_path": "traffic.csv",
        "data": "custom",
        "features": "M",
        "freq": "h",
        "target": "OT",
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
    },
    "Solar": {
        "root_path": "./dataset/Solar/",
        "data_path": "solar_AL.txt",
        "data": "Solar",
        "features": "M",
        "freq": "t",
        "target": "none",
        "num_variates": 137,
        "seq_len": 96,
        "e_layers": 2,
        "enc_in": 137,
        "dec_in": 137,
        "c_out": 137,
        "d_model": 512,
        "d_ff": 512,
        "batch_size": 32,
        "learning_rate": 0.0005,
    },
}


METHODS = ("MLP_attention", "latent_query_attention")
NORMALIZATIONS = ("softmax", "entmax15")
ENTROPY_WEIGHTS = (0.0, 1e-3)


def strict_k_budget(num_variates):
    k_max = int(math.ceil(0.30 * int(num_variates))) - 1
    return (k_max, False) if k_max >= 1 else (1, True)


def round_half_up(value):
    return int(math.floor(float(value) + 0.5))


def k_for_ratio(num_variates, ratio):
    k_budget_max, _ = strict_k_budget(num_variates)
    return max(1, min(round_half_up(float(num_variates) * float(ratio)), k_budget_max))


def fmt_float_token(value):
    return ("{:.3g}".format(float(value))).replace(".", "p").replace("-", "m")


def method_name(method, normalization, entropy_weight):
    norm_name = "entmax" if normalization == "entmax15" else "softmax"
    suffix = "_entropy" if float(entropy_weight) > 0.0 else ""
    return "{}_{}{}".format(method, norm_name, suffix)


def build_runs(args):
    if args.stage == "dense":
        return [], []
    include_baseline = args.stage in {"baseline", "all"}
    include_candidates = args.stage in {"candidates", "all"}
    baseline_runs = []
    candidate_runs = []
    for dataset in args.datasets:
        spec = DATASET_SPECS[dataset]
        for pred_len in args.pred_lens:
            for seed in args.seeds:
                if include_baseline:
                    baseline_runs.append({
                        "dataset": dataset,
                        "pred_len": pred_len,
                        "seed": seed,
                        "method_family": "iTransformer",
                        "method_name": "iTransformer",
                        "method": "none",
                        "normalization": "",
                        "entropy_weight": 0.0,
                        "k_ratio": "",
                        "k_value": spec["num_variates"],
                        "spec": spec,
                    })
                if include_candidates:
                    for k_ratio in args.k_ratios:
                        k_value = k_for_ratio(spec["num_variates"], k_ratio)
                        for method in METHODS:
                            for normalization in NORMALIZATIONS:
                                for entropy_weight in ENTROPY_WEIGHTS:
                                    if args.smoke:
                                        keep = (
                                            method == "MLP_attention"
                                            and normalization == "softmax"
                                            and float(entropy_weight) == 0.0
                                        ) or (
                                            method == "latent_query_attention"
                                            and normalization == "entmax15"
                                            and float(entropy_weight) == 0.0
                                        )
                                        if not keep:
                                            continue
                                    candidate_runs.append({
                                        "dataset": dataset,
                                        "pred_len": pred_len,
                                        "seed": seed,
                                        "method_family": method,
                                        "method_name": method_name(method, normalization, entropy_weight),
                                        "method": method,
                                        "normalization": normalization,
                                        "entropy_weight": float(entropy_weight),
                                        "k_ratio": float(k_ratio),
                                        "k_value": k_value,
                                        "spec": spec,
                                    })
    return baseline_runs, candidate_runs


def run_key(run):
    return (
        run["dataset"],
        str(run["pred_len"]),
        str(run["seed"]),
        run["method_family"],
        run["method_name"],
        str(run["k_ratio"]),
    )


def completed_keys(path):
    result = set()
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return result
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "success") not in {"", "success"}:
                continue
            result.add((
                row.get("dataset", ""),
                row.get("pred_len", ""),
                row.get("seed", ""),
                row.get("method_family", ""),
                row.get("method_name", ""),
                row.get("k_ratio", ""),
            ))
    return result


def base_command(run, args):
    spec = run["spec"]
    smoke_suffix = "_smoke" if args.smoke else ""
    if run["method"] == "none":
        model_id = "{}_pl{}_iTransformer_seed{}{}".format(
            run["dataset"],
            run["pred_len"],
            run["seed"],
            smoke_suffix,
        )
        des = "baseline"
        weight_dir = "./results/sparse_attention_wcomp/plots/iTransformer"
    else:
        model_id = "{}_pl{}_{}_ratio{}_k{}_seed{}{}".format(
            run["dataset"],
            run["pred_len"],
            run["method_name"],
            fmt_float_token(run["k_ratio"]),
            run["k_value"],
            run["seed"],
            smoke_suffix,
        )
        des = "sparse_wcomp"
        weight_dir = "./results/sparse_attention_wcomp/plots/{}".format(run["method_name"])
        if args.smoke:
            weight_dir = "./results/sparse_attention_wcomp/smoke/plots/{}".format(run["method_name"])
    train_epochs = 1 if args.smoke else args.train_epochs
    patience = 1 if args.smoke else args.patience
    batch_size = int(spec.get("batch_size", 32))
    command = [
        sys.executable,
        "run.py",
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "iTransformer",
        "--data", spec["data"],
        "--root_path", spec["root_path"],
        "--data_path", spec["data_path"],
        "--features", spec["features"],
        "--target", spec.get("target", "OT"),
        "--freq", spec.get("freq", "h"),
        "--seq_len", spec["seq_len"],
        "--pred_len", run["pred_len"],
        "--enc_in", spec["enc_in"],
        "--dec_in", spec["dec_in"],
        "--c_out", spec["c_out"],
        "--e_layers", spec["e_layers"],
        "--d_model", spec["d_model"],
        "--d_ff", spec["d_ff"],
        "--des", des,
        "--use_gpu", "True",
        "--gpu", args.gpu,
        "--use_multi_gpu", "False",
        "--variate_reduction_type", run["method"],
        "--reduced_variate_k", run["k_value"],
        "--variate_expansion_type", "slot_learned_linear",
        "--variate_decode_stage", "feature",
        "--method_family", run["method_family"],
        "--method_name", run["method_name"],
        "--wcomp_normalization", run["normalization"] or "softmax",
        "--entmax_alpha", "1.5",
        "--orthogonal_loss_weight", "0.01" if run["method"] != "none" else "0.0",
        "--wcomp_entropy_loss_weight", run["entropy_weight"],
        "--reconstruction_loss_weight", "0.0",
        "--coverage_loss_weight", "0.0",
        "--assignment_entropy_loss_weight", "0.0",
        "--linear_coverage_loss_weight", "0.0",
        "--biorthogonal_loss_weight", "0.0",
        "--linear_weight_l2_loss_weight", "0.0",
        "--decoder_init_l2_loss_weight", "0.0",
        "--mae_loss_weight", "0.0",
        "--variate_token_split_factor", "1",
        "--enforce_k_budget", "True",
        "--result_csv", args.result_csv,
        "--seed", run["seed"],
        "--experiment_tag", run["dataset"],
        "--target_k_ratio", run["k_ratio"] if run["k_ratio"] != "" else "1.0",
        "--target_k_value", run["k_value"],
        "--k_selection_mode", "baseline" if run["method"] == "none" else "ratio",
        "--k_ratio_denominator", "",
        "--use_wandb", "False",
        "--wandb_mode", "disabled",
        "--weight_heatmap_dir", weight_dir,
        "--skip_epoch_test_eval", "True",
        "--batch_size", batch_size,
        "--learning_rate", spec.get("learning_rate", 0.0001),
        "--train_epochs", train_epochs,
        "--patience", patience,
        "--num_workers", args.num_workers,
    ]
    if args.smoke:
        command.extend([
            "--max_train_batches", "2",
            "--max_eval_batches", "2",
        ])
    return [str(part) for part in command]


def print_dry_run(baseline_runs, candidate_runs):
    print("baseline: {}".format(len(baseline_runs)))
    print("candidates: {}".format(len(candidate_runs)))
    print("total: {}".format(len(baseline_runs) + len(candidate_runs)))
    for run in baseline_runs[:8]:
        print("BASELINE {} pred_len={} seed={}".format(run["dataset"], run["pred_len"], run["seed"]))
    for run in candidate_runs[:12]:
        print(
            "CANDIDATE {} pred_len={} k_ratio={} K={} {}".format(
                run["dataset"],
                run["pred_len"],
                run["k_ratio"],
                run["k_value"],
                run["method_name"],
            )
        )


def entmax15(scores, dim=-1, n_iter=50, eps=1e-8):
    import torch

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


def print_entmax_smoke():
    import torch

    torch.manual_seed(2021)
    scores = torch.randn(3, 4, 21)
    A = entmax15(scores, dim=-1)
    sums = A.sum(dim=-1)
    has_bad = bool(torch.isnan(A).any().item() or torch.isinf(A).any().item())
    print("ENTMAX_SMOKE min(A): {:.8f}".format(float(A.min().item())))
    print("ENTMAX_SMOKE max(A): {:.8f}".format(float(A.max().item())))
    print("ENTMAX_SMOKE mean(sum over C): {:.8f}".format(float(sums.mean().item())))
    print("ENTMAX_SMOKE max absolute deviation from 1 over C: {:.8e}".format(float((sums - 1.0).abs().max().item())))
    print("ENTMAX_SMOKE zero ratio: {:.8f}".format(float((A == 0).float().mean().item())))
    print("ENTMAX_SMOKE has NaN/Inf: {}".format(has_bad))


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry_run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--stage", choices=["baseline", "candidates", "all", "dense"], required=True)
    parser.add_argument("--datasets", nargs="+", default=["Weather", "Electricity", "Traffic", "Solar"])
    parser.add_argument("--pred_lens", type=int, nargs="+", default=[96, 720])
    parser.add_argument("--k_ratios", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2021])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--result_csv", default="./results/sparse_attention_wcomp/results.csv")
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "dense":
        print("dense stage is disabled for this experiment")
        return
    for dataset in args.datasets:
        if dataset not in DATASET_SPECS:
            raise RuntimeError("Unknown dataset: {}".format(dataset))
    baseline_runs, candidate_runs = build_runs(args)
    if args.dry_run:
        print_dry_run(baseline_runs, candidate_runs)
        return
    if args.smoke:
        print_entmax_smoke()
    runs = baseline_runs + candidate_runs
    if args.skip_completed:
        done = completed_keys(args.result_csv)
        before = len(runs)
        runs = [run for run in runs if run_key(run) not in done]
        print("skip_completed: skipped {} completed runs".format(before - len(runs)), flush=True)
    print("planned runs: {}".format(len(runs)), flush=True)
    for index, run in enumerate(runs, start=1):
        command = base_command(run, args)
        print(
            "[RUN {}/{}] {} {} pred_len={} method={} K={}".format(
                index,
                len(runs),
                run["dataset"],
                run["seed"],
                run["pred_len"],
                run["method_name"],
                run["k_value"] if run["method"] != "none" else "",
            ),
            flush=True,
        )
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
        if completed.returncode != 0:
            raise RuntimeError("run failed with return code {}".format(completed.returncode))


if __name__ == "__main__":
    main()
