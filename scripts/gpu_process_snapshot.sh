#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "host: $(hostname)"
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

echo "[nvidia-smi]"
nvidia-smi
echo

echo "[compute processes]"
mapfile -t rows < <(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)

if [[ "${#rows[@]}" -eq 0 ]]; then
  echo "No compute process reported by nvidia-smi."
else
  for row in "${rows[@]}"; do
    IFS=',' read -r raw_pid raw_name raw_mem <<<"$row"
    pid="$(echo "$raw_pid" | xargs)"
    proc_name="$(echo "$raw_name" | xargs)"
    mem_mib="$(echo "$raw_mem" | xargs)"
    echo "===== PID $pid | ${mem_mib}MiB | $proc_name ====="
    ps -o pid,ppid,etime,stat,pcpu,pmem,cmd -p "$pid" || true
    echo "cwd: $(readlink "/proc/$pid/cwd" 2>/dev/null || echo '?')"
    cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    parent_pid="$(ps -o ppid= -p "$pid" 2>/dev/null | xargs || true)"
    parent_cmd=""
    if [[ -n "$parent_pid" && -r "/proc/$parent_pid/cmdline" ]]; then
      parent_cmd="$(tr '\0' ' ' <"/proc/$parent_pid/cmdline" 2>/dev/null || true)"
    fi
    echo "cmdline: ${cmdline:-?}"
    echo "parent: ${parent_pid:-?} ${parent_cmd}"
    if [[ "$cmdline" =~ --model_id[[:space:]]+([^[:space:]]+) ]]; then
      echo "model_id: ${BASH_REMATCH[1]}"
    fi
    if [[ "$cmdline" =~ --result_csv[[:space:]]+([^[:space:]]+) ]]; then
      echo "result_csv: ${BASH_REMATCH[1]}"
    fi
    if [[ "$cmdline" =~ --experiment_tag[[:space:]]+([^[:space:]]+) ]]; then
      echo "experiment_tag: ${BASH_REMATCH[1]}"
    fi
    echo
  done
fi

echo "[recent logs updated within 2 hours]"
if [[ -d logs ]]; then
  find logs -maxdepth 1 -type f -mmin -120 -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort || true
else
  echo "logs directory not found"
fi

echo
echo "[recent csv/results updated within 2 hours]"
if [[ -d results ]]; then
  find results -maxdepth 2 \( -name '*.csv' -o -name '*.md' -o -name '*.log' \) -mmin -120 -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort || true
else
  echo "results directory not found"
fi
