#!/usr/bin/env python3
"""Build a baseline comparison report from current local result artifacts."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_MD = RESULTS / "vardrop_comparison_report.md"
OUT_HTML = RESULTS / "vardrop_comparison_report.html"
OUT_CSV = RESULTS / "vardrop_comparison_table.csv"


VARDROP_PERF = {
    ("Electricity", 96): (0.153, 0.245, 0.150, 0.242),
    ("Electricity", 192): (0.167, 0.257, 0.166, 0.256),
    ("Electricity", 336): (0.183, 0.275, 0.184, 0.276),
    ("Electricity", 720): (0.220, 0.305, 0.214, 0.302),
    ("Traffic", 96): (0.396, 0.274, 0.398, 0.272),
    ("Traffic", 192): (0.417, 0.281, 0.418, 0.279),
    ("Traffic", 336): (0.435, 0.289, 0.431, 0.286),
    ("Traffic", 720): (0.472, 0.308, 0.465, 0.304),
    ("Weather", 96): (0.178, 0.218, 0.176, 0.216),
    ("Weather", 192): (0.227, 0.258, 0.225, 0.257),
    ("Weather", 336): (0.281, 0.297, 0.281, 0.299),
    ("Weather", 720): (0.357, 0.347, 0.358, 0.350),
    ("Solar", 96): (0.202, 0.238, 0.205, 0.236),
    ("Solar", 192): (0.237, 0.262, 0.239, 0.263),
    ("Solar", 336): (0.254, 0.275, 0.249, 0.273),
    ("Solar", 720): (0.252, 0.274, 0.250, 0.275),
}

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f3(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def f1(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


def f2(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def load_current_summary() -> dict[tuple[str, int], dict[str, str]]:
    rows = read_rows(RESULTS / "current_result_inventory" / "final_coverage_matrix_summary.csv")
    return {(r["dataset"], int(r["pred_len"])): r for r in rows}


def load_runtime_rows() -> dict[tuple[str, int], dict[str, str]]:
    runtime_sources = [
        RESULTS / "traffic_anchor_corrinit_p4_top16_valscreen.csv",
        RESULTS / "traffic_anchor_corrinit_p4_top16_horizons_valscreen.csv",
        RESULTS / "electricity_anchor_corrinit_p4_top16_horizons_valscreen.csv",
        RESULTS / "solar_anchor_corrinit_p4_top16_horizons_valscreen.csv",
        RESULTS / "learned_anchor_tokenquery_tailcov_weather_traffic_valscreen.csv",
        RESULTS / "weather_learned_anchor_tailcov_horizons_valscreen.csv",
    ]
    out: dict[tuple[str, int], dict[str, str]] = {}
    for path in runtime_sources:
        if not path.exists():
            continue
        for row in read_rows(path):
            if row.get("status", "success") != "success":
                continue
            key = (row["dataset"], int(row["pred_len"]))
            if key not in out:
                out[key] = row
    return out


def load_local_baseline_rows() -> dict[tuple[str, int], dict[str, str]]:
    rows = read_rows(RESULTS / "validation_baseline_screens.csv")
    return {(r["dataset"], int(r["pred_len"])): r for r in rows}


def load_local_test_baseline_rows() -> dict[tuple[str, int], dict[str, str]]:
    path = RESULTS / "current_result_inventory" / "baseline_by_cell.csv"
    rows = read_rows(path)
    out = {}
    for row in rows:
        if row.get("eval_split") != "test":
            continue
        out[(row["dataset"], int(row["pred_len"]))] = row
    return out


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_candidate_test_rows() -> dict[tuple[str, int], dict[str, str]]:
    path = RESULTS / "coverage_fallback_test_for_vardrop.csv"
    if not path.exists():
        return {}
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in read_rows(path):
        if row.get("status", "success") != "success":
            continue
        if row.get("eval_split") != "test":
            continue
        key = (row["dataset"], int(row["pred_len"]))
        grouped.setdefault(key, []).append(row)

    out = {}
    for key, rows in grouped.items():
        seeds = sorted({r.get("seed", "") for r in rows if r.get("seed", "")})
        out[key] = {
            "mse": f"{mean([float(r['mse']) for r in rows]):.12g}",
            "mae": f"{mean([float(r['mae']) for r in rows]):.12g}",
            "seed_count": str(len(seeds) or len(rows)),
            "seeds": ",".join(seeds),
        }
    return out


def build_performance_rows(
    summary: dict[tuple[str, int], dict[str, str]],
    local_test_baseline: dict[tuple[str, int], dict[str, str]],
    candidate_test: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in ["Electricity", "Traffic", "Weather", "Solar"]:
        dataset_rows = []
        for pred_len in [96, 192, 336, 720]:
            vd_mse, vd_mae, it_mse, it_mae = VARDROP_PERF[(dataset, pred_len)]
            ours = summary.get((dataset, pred_len), {})
            test_base = local_test_baseline.get((dataset, pred_len), {})
            test_candidate = candidate_test.get((dataset, pred_len), {})
            base_mse = float(test_base["mse"]) if test_base else None
            base_mae = float(test_base["mae"]) if test_base else None
            cand_mse = float(test_candidate["mse"]) if test_candidate else None
            cand_mae = float(test_candidate["mae"]) if test_candidate else None
            row = {
                "dataset": dataset,
                "pred_len": str(pred_len),
                "method_note": ours.get("route", "pending"),
                "vardrop_mse": f3(vd_mse),
                "vardrop_mae": f3(vd_mae),
                "paper_itransformer_mse": f3(it_mse),
                "paper_itransformer_mae": f3(it_mae),
                "local_itransformer_test_mse": f3(base_mse) if base_mse is not None else "pending",
                "local_itransformer_test_mae": f3(base_mae) if base_mae is not None else "pending",
                "ours_test_mse": f3(cand_mse) if cand_mse is not None else "pending",
                "ours_test_mae": f3(cand_mae) if cand_mae is not None else "pending",
                "ours_test_mse_ratio": f3(cand_mse / base_mse) if cand_mse is not None and base_mse else "pending",
                "ours_test_mae_ratio": f3(cand_mae / base_mae) if cand_mae is not None and base_mae else "pending",
                "ours_test_seed_count": test_candidate.get("seed_count", "pending"),
                "ours_val_mse": f3(float(ours["candidate_mse_mean"])) if ours else "pending",
                "ours_val_mae": f3(float(ours["candidate_mae_mean"])) if ours else "pending",
                "ours_val_mse_ratio": f3(float(ours["mse_ratio_of_means"])) if ours else "pending",
                "ours_val_mae_ratio": f3(float(ours["mae_ratio_of_means"])) if ours else "pending",
            }
            rows.append(row)
            dataset_rows.append(row)

        def avg_field(field: str) -> str:
            values = [float(r[field]) for r in dataset_rows if r[field] != "pending"]
            return f3(mean(values)) if len(values) == 4 else "pending"

        avg = {
            "dataset": dataset,
            "pred_len": "Avg",
            "method_note": "coverage-routed",
            "vardrop_mse": f3(sum(float(r["vardrop_mse"]) for r in dataset_rows) / 4),
            "vardrop_mae": f3(sum(float(r["vardrop_mae"]) for r in dataset_rows) / 4),
            "paper_itransformer_mse": f3(sum(float(r["paper_itransformer_mse"]) for r in dataset_rows) / 4),
            "paper_itransformer_mae": f3(sum(float(r["paper_itransformer_mae"]) for r in dataset_rows) / 4),
            "local_itransformer_test_mse": avg_field("local_itransformer_test_mse"),
            "local_itransformer_test_mae": avg_field("local_itransformer_test_mae"),
            "ours_test_mse": avg_field("ours_test_mse"),
            "ours_test_mae": avg_field("ours_test_mae"),
            "ours_test_mse_ratio": avg_field("ours_test_mse_ratio"),
            "ours_test_mae_ratio": avg_field("ours_test_mae_ratio"),
            "ours_test_seed_count": min(
                (r["ours_test_seed_count"] for r in dataset_rows if r["ours_test_seed_count"] != "pending"),
                default="pending",
            ),
            "ours_val_mse": avg_field("ours_val_mse"),
            "ours_val_mae": avg_field("ours_val_mae"),
            "ours_val_mse_ratio": avg_field("ours_val_mse_ratio"),
            "ours_val_mae_ratio": avg_field("ours_val_mae_ratio"),
        }
        rows.append(avg)
    return rows


def build_efficiency_rows(
    runtime: dict[tuple[str, int], dict[str, str]],
    baseline: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in ["Electricity", "Traffic", "Weather", "Solar"]:
        for pred_len in [96, 192, 336, 720]:
            r = runtime.get((dataset, pred_len))
            b = baseline.get((dataset, pred_len))
            base_ms = float(b["avg_train_iter_time_sec"]) * 1000.0 if b else None
            ours_ms = float(r["avg_train_iter_time_sec"]) * 1000.0 if r else None
            base_mem = float(b["peak_gpu_allocated_mb"]) / 1024.0 if b else None
            ours_mem = float(r["peak_gpu_allocated_mb"]) / 1024.0 if r else None
            rows.append({
                "dataset": dataset,
                "pred_len": str(pred_len),
                "baseline_ms": f1(base_ms) if base_ms is not None else "pending",
                "ours_ms": f1(ours_ms) if ours_ms is not None else "pending",
                "time_ratio": f"{ours_ms / base_ms:.3f}" if base_ms and ours_ms else "pending",
                "time_delta": f"{(ours_ms / base_ms - 1.0) * 100.0:+.1f}%" if base_ms and ours_ms else "pending",
                "baseline_mem_gb": f2(base_mem) if base_mem is not None else "pending",
                "ours_mem_gb": f2(ours_mem) if ours_mem is not None else "pending",
                "mem_ratio": f"{ours_mem / base_mem:.3f}" if base_mem and ours_mem else "pending",
                "mem_delta": f"{(ours_mem / base_mem - 1.0) * 100.0:+.1f}%" if base_mem and ours_mem else "pending",
            })
    return rows


def summarize_efficiency(rows: list[dict[str, str]]) -> dict[str, str]:
    time_ratios = [float(r["time_ratio"]) for r in rows if r["time_ratio"] != "pending"]
    mem_ratios = [float(r["mem_ratio"]) for r in rows if r["mem_ratio"] != "pending"]
    time_better_5 = sum(r <= 0.95 for r in time_ratios)
    time_worse_1 = sum(r > 1.01 for r in time_ratios)
    mem_better_5 = sum(r <= 0.95 for r in mem_ratios)
    mem_worse_1 = sum(r > 1.01 for r in mem_ratios)
    return {
        "time_avg_ratio": f"{sum(time_ratios) / len(time_ratios):.3f}" if time_ratios else "pending",
        "mem_avg_ratio": f"{sum(mem_ratios) / len(mem_ratios):.3f}" if mem_ratios else "pending",
        "time_better_5": f"{time_better_5}/{len(time_ratios)}",
        "time_worse_1": f"{time_worse_1}/{len(time_ratios)}",
        "mem_better_5": f"{mem_better_5}/{len(mem_ratios)}",
        "mem_worse_1": f"{mem_worse_1}/{len(mem_ratios)}",
    }


def md_table(rows: list[dict[str, str]], cols: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label in cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(key, "") for key, _ in cols) + " |")
    return "\n".join(lines)


def html_table(rows: list[dict[str, str]], cols: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in cols)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(row.get(key, ''))}</td>" for key, _ in cols) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_csv(perf_rows, efficiency_rows) -> None:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "dataset_or_method", "pred_len_or_basis", "metric_a", "metric_b", "metric_c", "metric_d"])
        for row in perf_rows:
            writer.writerow([
                "performance",
                row["dataset"],
                row["pred_len"],
                row["method_note"],
                row["ours_test_mse"],
                row["ours_test_mae"],
                row["ours_test_mse_ratio"] + "/" + row["ours_test_mae_ratio"],
            ])
        for row in efficiency_rows:
            writer.writerow([
                "efficiency_vs_local_itransformer",
                row["dataset"],
                row["pred_len"],
                row["baseline_ms"] + "/" + row["ours_ms"],
                row["time_ratio"],
                row["baseline_mem_gb"] + "/" + row["ours_mem_gb"],
                row["mem_ratio"],
            ])


def main() -> None:
    summary = load_current_summary()
    runtime = load_runtime_rows()
    baseline = load_local_baseline_rows()
    local_test_baseline = load_local_test_baseline_rows()
    candidate_test = load_candidate_test_rows()
    perf_rows = build_performance_rows(summary, local_test_baseline, candidate_test)
    efficiency_rows = build_efficiency_rows(runtime, baseline)
    efficiency_summary = summarize_efficiency(efficiency_rows)
    write_csv(perf_rows, efficiency_rows)

    perf_cols = [
        ("dataset", "Dataset"),
        ("pred_len", "Pred"),
        ("method_note", "Ours method"),
        ("vardrop_mse", "VarDrop MSE"),
        ("vardrop_mae", "VarDrop MAE"),
        ("paper_itransformer_mse", "Paper iTransformer MSE"),
        ("paper_itransformer_mae", "Paper iTransformer MAE"),
        ("local_itransformer_test_mse", "Local iTransformer-test MSE"),
        ("local_itransformer_test_mae", "Local iTransformer-test MAE"),
        ("ours_test_mse", "Ours-test MSE"),
        ("ours_test_mae", "Ours-test MAE"),
        ("ours_test_mse_ratio", "Ours/local test MSE ratio"),
        ("ours_test_mae_ratio", "Ours/local test MAE ratio"),
        ("ours_test_seed_count", "Ours test seeds"),
        ("ours_val_mse_ratio", "Ours/local val MSE ratio"),
        ("ours_val_mae_ratio", "Ours/local val MAE ratio"),
    ]
    efficiency_cols = [
        ("dataset", "Dataset"),
        ("pred_len", "Pred"),
        ("baseline_ms", "iTransformer ms/iter"),
        ("ours_ms", "Ours ms/iter"),
        ("time_ratio", "Time ratio"),
        ("time_delta", "Time delta"),
        ("baseline_mem_gb", "iTransformer peak GB"),
        ("ours_mem_gb", "Ours peak GB"),
        ("mem_ratio", "Memory ratio"),
        ("mem_delta", "Memory delta"),
    ]

    caveat = (
        "주의: VarDrop 논문 표는 test 성능/논문 하드웨어 기준이고, 현재 Ours 성능은 "
        "selected route test 실행 전에는 pending으로 표시된다. 효율 표는 VarDrop 논문 하드웨어 숫자를 쓰지 않고 "
        "현재 저장소의 local iTransformer validation 로그만 기준으로 비교했다. 이 Codex 세션에서는 CUDA가 "
        "보이지 않고 escalated GPU 실행도 정책상 거절되어 test 재실행과 CUDA median profile을 완료하지 못했다."
    )
    md = f"""# VarDrop 성능표와 iTransformer 기준 Ours 비교

## 기술 요약

- 현재 Ours selected route는 `coverage-routed representative`: Weather는 `learned_anchor_selection + token_query_decoder`, 나머지는 `variate_anchor_selection + masked_linear`다.
- 성능 표에는 VarDrop 논문 test 숫자, local iTransformer test baseline, 그리고 `coverage_fallback_test_for_vardrop.csv`의 Ours test를 함께 둔다.
- Ours test가 `pending`이면 selected route test 실행이 아직 끝나지 않은 상태다. 마지막 두 val-ratio 열은 현재 validation selection 참고값이다.
- 효율 표는 논문 하드웨어 값을 제외하고, 같은 저장소/같은 로그의 `Local iTransformer` 대비 Ours ratio만 계산했다.
- 현재 이 세션에서 escalated GPU 실행 승인이 정책상 거절되어, missing test와 CUDA median profile은 pending이다.

## 성능 비교 표

{md_table(perf_rows, perf_cols)}

## iTransformer baseline 대비 효율

{md_table(efficiency_rows, efficiency_cols)}

## 효율 요약

- 평균 train ms/iter ratio: {efficiency_summary["time_avg_ratio"]}
- 평균 peak allocated memory ratio: {efficiency_summary["mem_avg_ratio"]}
- train time 5% 이상 개선 cell: {efficiency_summary["time_better_5"]}
- train time 1% 이상 악화 cell: {efficiency_summary["time_worse_1"]}
- memory 5% 이상 개선 cell: {efficiency_summary["mem_better_5"]}
- memory 1% 이상 악화 cell: {efficiency_summary["mem_worse_1"]}

## 해석

- Ours-val은 Electricity/Traffic/Weather에서 local validation baseline 대비 대부분 개선되지만, Solar는 192/336/720에서 약하다.
- 효율은 Traffic에서 memory가 크게 줄어드나, Solar는 현재 fixed sparse route 로그상 train time과 memory가 모두 나빠서 구조/구현 병목이 남아 있다.
- Weather는 learned route가 성능을 개선하지만 train time은 느려져서, fixed route 전환 또는 K-selection overhead 제거 ablation이 필요하다.
- VarDrop 성능표에 공정하게 들어가려면 selected route의 final test와 같은 GPU에서 median latency/train-step/memory profile이 필요하다.

## 필요한 추가 실행

```bash
cd /home/yulim/MSH/patch-latent
bash iTransformer/scripts/run_vardrop_performance_table.sh

# 최종 3-seed 표까지 만들 때
SEEDS="2021 2022 2023" bash iTransformer/scripts/run_vardrop_performance_table.sh

# 별도 CUDA median efficiency profile
cd /home/yulim/MSH/patch-latent/iTransformer
/home/yulim/anaconda3/envs/patchtst/bin/python -u scripts/profile_coverage_fallback_efficiency.py --device cuda --datasets Electricity Traffic Weather Solar --pred_lens 96 --output_csv ./results/coverage_fallback_efficiency_profile_vardrop96.csv --warmup 5 --repeat 15 --train_repeat 7
```

{caveat}
"""
    OUT_MD.write_text(md, encoding="utf-8")

    css = """
body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#122033;line-height:1.45}
h1{font-size:28px;margin:0 0 18px}
h2{font-size:20px;margin:28px 0 10px;border-bottom:1px solid #d5dde8;padding-bottom:6px}
table{border-collapse:collapse;width:max-content;min-width:100%;font-size:13px}
th,td{border:1px solid #d8e0eb;padding:6px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}
thead th{background:#eef3f8}
.table-wrap{overflow-x:auto;margin:10px 0 18px}
.note{background:#fff7df;border:1px solid #f0d480;padding:10px 12px;border-radius:6px}
code,pre{background:#f5f7fb;border:1px solid #d8e0eb;border-radius:6px}
pre{padding:12px;overflow:auto}
"""
    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>VarDrop 성능표와 iTransformer 기준 Ours 비교</title>
<style>{css}</style>
</head>
<body>
<h1>VarDrop 성능표와 iTransformer 기준 Ours 비교</h1>
<section>
<h2>기술 요약</h2>
<ul>
<li>현재 Ours selected route는 <code>coverage-routed representative</code>다.</li>
<li>성능 표는 paper test 숫자, local iTransformer test baseline, Ours test를 함께 보여준다.</li>
<li>Ours test가 pending이면 <code>coverage_fallback_test_for_vardrop.csv</code> 생성이 아직 끝나지 않은 상태다.</li>
<li>효율 표는 논문 하드웨어 값을 제외하고 같은 저장소의 local iTransformer 대비 ratio만 보여준다.</li>
<li>현재 세션에서는 escalated GPU 실행이 정책상 거절되어 missing test/CUDA median profile이 pending이다.</li>
</ul>
</section>
<section>
<h2>성능 비교 표</h2>
<div class="table-wrap">{html_table(perf_rows, perf_cols)}</div>
</section>
<section>
<h2>iTransformer baseline 대비 효율</h2>
<div class="table-wrap">{html_table(efficiency_rows, efficiency_cols)}</div>
</section>
<section>
<h2>효율 요약</h2>
<ul>
<li>평균 train ms/iter ratio: {efficiency_summary["time_avg_ratio"]}</li>
<li>평균 peak allocated memory ratio: {efficiency_summary["mem_avg_ratio"]}</li>
<li>train time 5% 이상 개선 cell: {efficiency_summary["time_better_5"]}</li>
<li>train time 1% 이상 악화 cell: {efficiency_summary["time_worse_1"]}</li>
<li>memory 5% 이상 개선 cell: {efficiency_summary["mem_better_5"]}</li>
<li>memory 1% 이상 악화 cell: {efficiency_summary["mem_worse_1"]}</li>
</ul>
</section>
<section>
<h2>해석</h2>
<p>Ours-val은 Electricity/Traffic/Weather에서 local validation baseline 대비 대부분 개선되지만 Solar 장기 horizon이 약하다. 효율은 Traffic에서 memory가 크게 줄어드는 반면, Weather learned route와 Solar fixed route에서는 runtime 또는 memory 병목이 남아 있다.</p>
<p class="note">{html.escape(caveat)}</p>
</section>
<section>
<h2>필요한 추가 실행</h2>
<pre>cd /home/yulim/MSH/patch-latent
bash iTransformer/scripts/run_vardrop_performance_table.sh

# 최종 3-seed 표까지 만들 때
SEEDS="2021 2022 2023" bash iTransformer/scripts/run_vardrop_performance_table.sh

# 별도 CUDA median efficiency profile
cd /home/yulim/MSH/patch-latent/iTransformer
/home/yulim/anaconda3/envs/patchtst/bin/python -u scripts/profile_coverage_fallback_efficiency.py --device cuda --datasets Electricity Traffic Weather Solar --pred_lens 96 --output_csv ./results/coverage_fallback_efficiency_profile_vardrop96.csv --warmup 5 --repeat 15 --train_repeat 7</pre>
</section>
</body>
</html>
"""
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_HTML}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
