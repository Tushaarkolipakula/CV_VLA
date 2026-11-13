#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODE=${1:-clean}
RUN_NAME=${2:-$(date -u +%Y%m%dT%H%M%SZ)}
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
case "$MODE" in clean|fixed|curriculum) ;; *) echo 'Mode must be clean, fixed, or curriculum' >&2; exit 2;; esac
DATA_ROOT=${DATA_ROOT:-/home/tushaar/data/LIBERO-datasets}
RECOVERY_ROOT=${RECOVERY_ROOT:-$ROOT/data/contact_v1}
OUT="$ROOT/checkpoints/smolvla_${MODE}_${RUN_NAME}"
if [[ -e "$OUT" || -e "${OUT}.log" ]]; then echo "Run already exists: $OUT" >&2; exit 2; fi
DATA=()
read -r -a SUITE_LIST <<< "${SUITES:-libero_spatial libero_object libero_goal libero_10}"
for suite in "${SUITE_LIST[@]}"; do
  if [[ ! -d "$DATA_ROOT/$suite" ]]; then echo "Missing dataset: $DATA_ROOT/$suite" >&2; exit 2; fi
  DATA+=("$DATA_ROOT/$suite")
  if [[ "$MODE" != clean ]]; then
    if [[ ! -d "$RECOVERY_ROOT/$suite" ]]; then echo "Generate verified contact recoveries first: $RECOVERY_ROOT/$suite" >&2; exit 2; fi
    DATA+=("$RECOVERY_ROOT/$suite")
  fi
done
VLM_PATH=$(cat "$ROOT/results/smolvla_setup/vlm_path.txt")
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
"$ROOT/.venv-smolvla/bin/python" -u -m smolvla_pipeline.train \
  --data "${DATA[@]}" --mode "$MODE" --out "$OUT" \
  --steps "${STEPS:-20000}" --batch-size "${BATCH_SIZE:-4}" --accumulation "${ACCUMULATION:-4}" \
  --lr "${LR:-0.0001}" --seed "${SEED:-0}" --clean-action-offset 1 \
  --save-every 1000 --validate-every 500 --keep-last 2 --vlm-path "$VLM_PATH" "$@" \
  2>&1 | tee "${OUT}.log"
