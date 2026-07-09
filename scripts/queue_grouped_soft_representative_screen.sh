#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/yulim/anaconda3/envs/patchtst/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-2021}
SIMILARITIES=${SIMILARITIES:-"abs_pearson positive_pearson"}
NORMALIZATIONS=${NORMALIZATIONS:-"softmax entmax15"}
GROUP_ENTROPY_WEIGHTS=${GROUP_ENTROPY_WEIGHTS:-"0 0.01"}
GROUP_ENTROPY_TARGET=${GROUP_ENTROPY_TARGET:-0.35}
EXPAND_TOP_M=${EXPAND_TOP_M:-16}
EXPAND_INIT_POWER=${EXPAND_INIT_POWER:-4}
RESULT_CSV=${RESULT_CSV:-./results/grouped_soft_representative/results_valscreen.csv}
MAP_ROOT=${MAP_ROOT:-./results/anchor_maps/grouped_soft_rep}
HEATMAP_ROOT=${HEATMAP_ROOT:-./results/grouped_soft_representative/heatmaps}
LOG_ROOT=${LOG_ROOT:-./results/grouped_soft_representative/logs}
MAX_RUNS=${MAX_RUNS:-0}

mkdir -p "${MAP_ROOT}" "${HEATMAP_ROOT}" "${LOG_ROOT}"

dataset_root() {
  case "$1" in
    Weather) echo "./dataset/weather/" ;;
    Electricity) echo "./dataset/electricity/" ;;
    Traffic) echo "./dataset/traffic/" ;;
    Solar) echo "./dataset/Solar/" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

dataset_path() {
  case "$1" in
    Weather) echo "weather.csv" ;;
    Electricity) echo "electricity.csv" ;;
    Traffic) echo "traffic.csv" ;;
    Solar) echo "solar_AL.txt" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

dataset_data() {
  case "$1" in
    Solar) echo "Solar" ;;
    Weather|Electricity|Traffic) echo "custom" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

dataset_target() {
  case "$1" in
    Solar) echo "none" ;;
    Weather|Electricity|Traffic) echo "OT" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

dataset_k_values() {
  case "$1" in
    Weather) echo "2 4 6" ;;
    Electricity) echo "32 64 96" ;;
    Traffic) echo "86 172 258" ;;
    Solar) echo "14 27 41" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

metric_token() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_' '_'
}

build_maps_for_similarity() {
  local sim="$1"
  local sim_token
  sim_token=$(metric_token "${sim}")
  for dataset in Weather Electricity Traffic Solar; do
    for k in $(dataset_k_values "${dataset}"); do
      local out="${MAP_ROOT}/${dataset}_k${k}_${sim_token}_medoid_top${EXPAND_TOP_M}_corrinit_p${EXPAND_INIT_POWER}_train.npz"
      if [[ -f "${out}" ]]; then
        echo "[MAP] exists ${out}"
        continue
      fi
      echo "[MAP] build dataset=${dataset} K=${k} similarity=${sim} -> ${out}"
      "${PY}" scripts/build_variate_anchor_map.py \
        --root_path "$(dataset_root "${dataset}")" \
        --data_path "$(dataset_path "${dataset}")" \
        --features M \
        --target "$(dataset_target "${dataset}")" \
        --reduced_k "${k}" \
        --similarity_metric "${sim}" \
        --anchor_objective medoid \
        --refine_rounds 2 \
        --expand_top_m "${EXPAND_TOP_M}" \
        --expand_init corr \
        --expand_init_power "${EXPAND_INIT_POWER}" \
        --out "${out}" \
        --enforce_k_budget
    done
  done
}

run_grid_for_combo() {
  local sim="$1"
  local norm="$2"
  local gent="$3"
  local sim_token
  sim_token=$(metric_token "${sim}")
  local extra=()
  if [[ "${MAX_RUNS}" != "0" ]]; then
    extra+=(--max_runs "${MAX_RUNS}")
  fi
  echo "[GRID] similarity=${sim} normalization=${norm} group_entropy=${gent}"
  "${PY}" -u scripts/run_mlp_variate_reduction_grid.py --run \
    --gpu "${GPU}" \
    --datasets Weather Electricity Traffic Solar \
    --pred_lens 96 192 336 720 \
    --k_ratios 0.1 0.2 0.3 \
    --methods grouped_soft_representative \
    --expansion_types masked_linear \
    --variate_anchor_map_template "${MAP_ROOT}/{dataset}_k{reduced_k}_${sim_token}_medoid_top${EXPAND_TOP_M}_corrinit_p${EXPAND_INIT_POWER}_train.npz" \
    --no_baseline \
    --train_epochs 10 \
    --patience 3 \
    --num_workers 0 \
    --skip_test_eval \
    --skip_epoch_test_eval \
    --local_temporal_branch nlinear_affine \
    --local_temporal_init persistence \
    --orthogonal_loss_weight 0.0 \
    --reconstruction_loss_weight 0.0 \
    --coverage_loss_weight 0.0 \
    --assignment_entropy_loss_weight 0.0 \
    --wcomp_entropy_loss_weight 0.0 \
    --group_attention_entropy_loss_weight "${gent}" \
    --group_attention_entropy_target "${GROUP_ENTROPY_TARGET}" \
    --wcomp_normalization "${norm}" \
    --entmax_alpha 1.5 \
    --result_csv "${RESULT_CSV}" \
    --weight_heatmap_dir "${HEATMAP_ROOT}" \
    --skip_completed \
    --no_wandb \
    --wandb_mode disabled \
    "${extra[@]}"
}

echo "[INFO] grouped soft representative screen"
echo "[INFO] result_csv=${RESULT_CSV}"
echo "[INFO] similarities=${SIMILARITIES}"
echo "[INFO] normalizations=${NORMALIZATIONS}"
echo "[INFO] group_entropy_weights=${GROUP_ENTROPY_WEIGHTS}"

for sim in ${SIMILARITIES}; do
  build_maps_for_similarity "${sim}"
done

for sim in ${SIMILARITIES}; do
  for norm in ${NORMALIZATIONS}; do
    for gent in ${GROUP_ENTROPY_WEIGHTS}; do
      run_grid_for_combo "${sim}" "${norm}" "${gent}"
    done
  done
done

echo "[DONE] grouped soft representative screen"
