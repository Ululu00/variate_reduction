# Variate Reduction Handoff

이 저장소는 iTransformer에 variate token reduction plug-in을 붙여 실험하기 위한 작업본이다.
현재 원격 서버에서 이어서 돌릴 때 필요한 핵심 구현과 실행 스크립트를 포함한다.

## 핵심 구현

- `layers/VariateReduction.py`: MLP/attention/anchor/grouped soft representative 계열 reducer 구현
- `model/iTransformer.py`: variate reduction module을 encoder 앞뒤에 연결
- `experiments/exp_long_term_forecasting.py`: reduction loss, diagnostics, result row 기록
- `run.py`: reduction 관련 CLI 인자 추가
- `scripts/run_mlp_variate_reduction_grid.py`: validation/test grid runner
- `scripts/build_variate_anchor_map.py`: similarity 기반 anchor/group map 생성
- `scripts/queue_grouped_soft_representative_screen.sh`: grouped soft representative screen queue
- `scripts/run_sparse_attention_wcomp_grid.py`: sparse attention W_comp grid runner

## 현재 중요한 방향

W 기반 dense generation은 성능이 좋지 않아 주력 방향에서 제외했다.
현재 우선순위는 변수 유사도로 group/anchor를 만들고, 각 latent token이 자기 group 안에서 soft representative를 학습하도록 하는 방식이다.

대표 실행 명령:

```bash
cd /home/yulim/MSH/patch-latent/iTransformer
mkdir -p ./results/grouped_soft_representative/logs

nohup env PY=/home/yulim/anaconda3/envs/patchtst/bin/python GPU=0 SEEDS=2021 \
  SIMILARITIES="abs_pearson positive_pearson" \
  NORMALIZATIONS="softmax entmax15" \
  GROUP_ENTROPY_WEIGHTS="0 0.01" \
  GROUP_ENTROPY_TARGET=0.35 \
  bash scripts/queue_grouped_soft_representative_screen.sh \
  > ./results/grouped_soft_representative/logs/grouped_soft_rep_screen.log 2>&1 &
```

결과 위치:

```bash
./results/grouped_soft_representative/results_valscreen.csv
./results/grouped_soft_representative/logs/grouped_soft_rep_screen.log
```

주의: 현재 로컬에는 grouped soft representative full result가 없고 smoke 결과만 있다.

```bash
./results/grouped_soft_representative/smoke/results_smoke.csv
```

## 데이터 경로

`dataset`은 로컬 symlink라 Git에는 포함하지 않는다. 새 서버에서는 데이터 폴더를 기존 코드 인자(`--root_path`, `--data_path`)에 맞게 다시 지정하거나 symlink를 새로 만들어야 한다.

## Git 제외 대상

다음 산출물은 Git에 올리지 않는다.

- `checkpoints/`
- `wandb/`
- `test_results/`
- 대용량 `results/**/metrics.npy`, plot 이미지, checkpoint/tensor 파일
- 로컬 데이터 symlink `dataset`

CSV/HTML/MD 요약 결과만 필요할 때 선별적으로 commit한다.
