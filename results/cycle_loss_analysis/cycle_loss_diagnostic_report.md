# Cycle Loss 실험 진단

- 후보 48개 중 pass 0, exact win 0
- GM MSE ratio 1.365, GM MAE ratio 1.293
- 실제 결과 CSV: `iTransformer/results/attention_variate_reduction_results.csv`
- 분석 CSV: `iTransformer/results/cycle_loss_analysis/cycle_candidate_ratios.csv`, `iTransformer/results/cycle_loss_analysis/cycle_matrix_diagnostics.csv`
- HTML: `iTransformer/results/cycle_loss_analysis/cycle_loss_diagnostic_report.html`

핵심 해석: cycle loss는 A/B top-r overlap을 맞추는 방향으로 계산됐지만, expansion B가 dense하게 퍼지는 문제를 막지 못했다. 특히 Traffic에서 B effective support가 평균 711까지 커져 target 변수 identity가 거의 복원되지 않았다.
