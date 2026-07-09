# VarDrop 성능표와 iTransformer 기준 Ours 비교

## 기술 요약

- 현재 Ours selected route는 `coverage-routed representative`: Weather는 `learned_anchor_selection + token_query_decoder`, 나머지는 `variate_anchor_selection + masked_linear`다.
- 성능 표에는 VarDrop 논문 test 숫자, local iTransformer test baseline, 그리고 `coverage_fallback_test_for_vardrop.csv`의 Ours test를 함께 둔다.
- Ours test가 `pending`이면 selected route test 실행이 아직 끝나지 않은 상태다. 마지막 두 val-ratio 열은 현재 validation selection 참고값이다.
- 효율 표는 논문 하드웨어 값을 제외하고, 같은 저장소/같은 로그의 `Local iTransformer` 대비 Ours ratio만 계산했다.
- 현재 이 세션에서 escalated GPU 실행 승인이 정책상 거절되어, missing test와 CUDA median profile은 pending이다.

## 성능 비교 표

| Dataset | Pred | Ours method | VarDrop MSE | VarDrop MAE | Paper iTransformer MSE | Paper iTransformer MAE | Local iTransformer-test MSE | Local iTransformer-test MAE | Ours-test MSE | Ours-test MAE | Ours/local test MSE ratio | Ours/local test MAE ratio | Ours test seeds | Ours/local val MSE ratio | Ours/local val MAE ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Electricity | 96 | fixed_sparse_representative | 0.153 | 0.245 | 0.150 | 0.242 | 0.149 | 0.240 | 0.145 | 0.242 | 0.977 | 1.006 | 1 | 0.929 | 0.984 |
| Electricity | 192 | fixed_sparse_representative | 0.167 | 0.257 | 0.166 | 0.256 | 0.166 | 0.258 | 0.166 | 0.261 | 0.995 | 1.013 | 1 | 0.941 | 0.984 |
| Electricity | 336 | fixed_sparse_representative | 0.183 | 0.275 | 0.184 | 0.276 | 0.179 | 0.272 | 0.181 | 0.276 | 1.010 | 1.014 | 1 | 0.961 | 0.996 |
| Electricity | 720 | fixed_sparse_representative | 0.220 | 0.305 | 0.214 | 0.302 | 0.217 | 0.303 | 0.222 | 0.310 | 1.023 | 1.022 | 1 | 0.955 | 0.994 |
| Electricity | Avg | coverage-routed | 0.181 | 0.271 | 0.178 | 0.269 | 0.178 | 0.268 | 0.178 | 0.272 | 1.001 | 1.014 | 1 | 0.947 | 0.990 |
| Traffic | 96 | fixed_sparse_representative | 0.396 | 0.274 | 0.398 | 0.272 | 0.392 | 0.268 | 0.457 | 0.318 | 1.165 | 1.189 | 1 | 0.964 | 0.959 |
| Traffic | 192 | fixed_sparse_representative | 0.417 | 0.281 | 0.418 | 0.279 | 0.411 | 0.276 | 0.468 | 0.320 | 1.137 | 1.159 | 1 | 0.950 | 0.935 |
| Traffic | 336 | fixed_sparse_representative | 0.435 | 0.289 | 0.431 | 0.286 | 0.422 | 0.282 | 0.484 | 0.331 | 1.146 | 1.172 | 1 | 0.949 | 0.947 |
| Traffic | 720 | fixed_sparse_representative | 0.472 | 0.308 | 0.465 | 0.304 | 0.458 | 0.300 | 0.509 | 0.343 | 1.111 | 1.141 | 1 | 0.934 | 0.920 |
| Traffic | Avg | coverage-routed | 0.430 | 0.288 | 0.428 | 0.285 | 0.421 | 0.282 | 0.480 | 0.328 | 1.140 | 1.165 | 1 | 0.949 | 0.940 |
| Weather | 96 | learned_representative | 0.178 | 0.218 | 0.176 | 0.216 | 0.175 | 0.213 | 0.178 | 0.220 | 1.018 | 1.030 | 1 | 0.931 | 0.958 |
| Weather | 192 | learned_representative | 0.227 | 0.258 | 0.225 | 0.257 | 0.224 | 0.257 | 0.227 | 0.260 | 1.012 | 1.013 | 1 | 0.982 | 0.997 |
| Weather | 336 | learned_representative | 0.281 | 0.297 | 0.281 | 0.299 | 0.283 | 0.300 | 0.282 | 0.301 | 0.996 | 1.003 | 1 | 0.966 | 0.975 |
| Weather | 720 | learned_representative | 0.357 | 0.347 | 0.358 | 0.350 | 0.359 | 0.352 | 0.358 | 0.350 | 0.995 | 0.994 | 1 | 0.983 | 0.993 |
| Weather | Avg | coverage-routed | 0.261 | 0.280 | 0.260 | 0.280 | 0.260 | 0.280 | 0.261 | 0.283 | 1.005 | 1.010 | 1 | 0.966 | 0.981 |
| Solar | 96 | fixed_sparse_representative | 0.202 | 0.238 | 0.205 | 0.236 | 0.202 | 0.237 | 0.207 | 0.249 | 1.024 | 1.051 | 1 | 1.017 | 0.990 |
| Solar | 192 | fixed_sparse_representative | 0.237 | 0.262 | 0.239 | 0.263 | 0.237 | 0.263 | 0.239 | 0.273 | 1.007 | 1.039 | 1 | 1.097 | 1.050 |
| Solar | 336 | fixed_sparse_representative | 0.254 | 0.275 | 0.249 | 0.273 | 0.249 | 0.273 | 0.255 | 0.286 | 1.025 | 1.045 | 1 | 1.106 | 1.080 |
| Solar | 720 | fixed_sparse_representative | 0.252 | 0.274 | 0.250 | 0.275 | 0.249 | 0.275 | 0.256 | 0.288 | 1.026 | 1.044 | 1 | 1.034 | 1.038 |
| Solar | Avg | coverage-routed | 0.236 | 0.262 | 0.236 | 0.262 | 0.234 | 0.262 | 0.239 | 0.274 | 1.020 | 1.045 | 1 | 1.063 | 1.040 |

## iTransformer baseline 대비 효율

| Dataset | Pred | iTransformer ms/iter | Ours ms/iter | Time ratio | Time delta | iTransformer peak GB | Ours peak GB | Memory ratio | Memory delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Electricity | 96 | 998.7 | 935.9 | 0.937 | -6.3% | 0.79 | 0.86 | 1.081 | +8.1% |
| Electricity | 192 | 926.5 | 989.2 | 1.068 | +6.8% | 0.80 | 0.87 | 1.085 | +8.5% |
| Electricity | 336 | 983.9 | 1013.5 | 1.030 | +3.0% | 0.81 | 0.88 | 1.082 | +8.2% |
| Electricity | 720 | 994.0 | 1064.4 | 1.071 | +7.1% | 0.88 | 0.93 | 1.052 | +5.2% |
| Traffic | 96 | 1794.6 | 1753.8 | 0.977 | -2.3% | 4.96 | 2.49 | 0.501 | -49.9% |
| Traffic | 192 | 1674.0 | 1947.4 | 1.163 | +16.3% | 4.98 | 2.51 | 0.504 | -49.6% |
| Traffic | 336 | 1706.7 | 1905.3 | 1.116 | +11.6% | 5.00 | 2.55 | 0.510 | -49.0% |
| Traffic | 720 | 1850.4 | 1956.9 | 1.058 | +5.8% | 5.06 | 2.66 | 0.525 | -47.5% |
| Weather | 96 | 142.6 | 164.4 | 1.153 | +15.3% | 0.17 | 0.15 | 0.893 | -10.7% |
| Weather | 192 | 192.0 | 218.0 | 1.135 | +13.5% | 0.17 | 0.15 | 0.898 | -10.2% |
| Weather | 336 | 196.1 | 220.2 | 1.123 | +12.3% | 0.17 | 0.15 | 0.907 | -9.3% |
| Weather | 720 | 199.4 | 212.0 | 1.063 | +6.3% | 0.18 | 0.17 | 0.944 | -5.6% |
| Solar | 96 | 144.1 | 184.0 | 1.277 | +27.7% | 0.38 | 0.68 | 1.762 | +76.2% |
| Solar | 192 | 1831.6 | 1933.6 | 1.056 | +5.6% | 0.39 | 0.68 | 1.754 | +75.4% |
| Solar | 336 | 1981.6 | 1829.2 | 0.923 | -7.7% | 0.40 | 0.70 | 1.752 | +75.2% |
| Solar | 720 | 1910.4 | 1737.9 | 0.910 | -9.0% | 0.44 | 0.73 | 1.662 | +66.2% |

## 효율 요약

- 평균 train ms/iter ratio: 1.066
- 평균 peak allocated memory ratio: 1.057
- train time 5% 이상 개선 cell: 3/16
- train time 1% 이상 악화 cell: 12/16
- memory 5% 이상 개선 cell: 8/16
- memory 1% 이상 악화 cell: 8/16

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

주의: VarDrop 논문 표는 test 성능/논문 하드웨어 기준이고, 현재 Ours 성능은 selected route test 실행 전에는 pending으로 표시된다. 효율 표는 VarDrop 논문 하드웨어 숫자를 쓰지 않고 현재 저장소의 local iTransformer validation 로그만 기준으로 비교했다. 이 Codex 세션에서는 CUDA가 보이지 않고 escalated GPU 실행도 정책상 거절되어 test 재실행과 CUDA median profile을 완료하지 못했다.
