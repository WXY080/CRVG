#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${CRVG_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPUS:-${GPU:-0}}"
read -r -a DATASET_ARGS <<< "${DATASETS:-refcoco_val}"
ARGS=(--backbone vlmr1
  --model-path "${CRVG_MODEL:?Set CRVG_MODEL}"
  --data-root "${CRVG_DATA:?Set CRVG_DATA}"
  --image-root "${CRVG_IMAGES:?Set CRVG_IMAGES}"
  --qwen-model "${CRVG_QWEN3:?Set CRVG_QWEN3}"
  --dino-model "${CRVG_DINO:?Set CRVG_DINO}"
  --log-dir "${CRVG_LOGS:-logs/vlmr1}"
  --config "${CRVG_CONFIG:-configs/default.yaml}"
  --datasets "${DATASET_ARGS[@]}"
  --batch-size "${BATCH_SIZE:-4}"
  --backbone-python "${CRVG_BACKBONE_PYTHON:-$PYTHON}"
  --verifier-python "${CRVG_VERIFIER_PYTHON:-$PYTHON}"
  --stop-after "${STOP_AFTER:-final}")
if [[ -n "${CRVG_CONTROLLER:-}" ]]; then ARGS+=(--controller "$CRVG_CONTROLLER"); fi
if [[ -n "${GAMMA0:-}" ]]; then ARGS+=(--gamma0 "$GAMMA0"); fi
if [[ -n "${GATE:-}" ]]; then ARGS+=(--gate "$GATE"); fi
if [[ -n "${GAMMA1:-}" ]]; then ARGS+=(--gamma1 "$GAMMA1"); fi
"$PYTHON" -m crvg.pipeline "${ARGS[@]}" "$@"
