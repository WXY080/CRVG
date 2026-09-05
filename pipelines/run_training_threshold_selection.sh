#!/usr/bin/env bash
# Reproduce the paper's TRAIN-only threshold-selection figure from cached evidence.
set -euo pipefail

: "${TRAIN_CACHE_DIR:?Set TRAIN_CACHE_DIR to the difficult-TRAIN cache directory}"

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTROLLER="${CRVG_CONTROLLER:-$ROOT/checkpoints/dino_three_domain_risk_controller}"
CALIBRATION="${DINO_CALIBRATION:-$ROOT/artifacts/controller_data/dino_three_domain_risk_calibration.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/training_threshold_selection}"

BASE_POOLS=(
  "$TRAIN_CACHE_DIR/rec_results_refcoco_train.json"
  "$TRAIN_CACHE_DIR/rec_results_refcoco+_train.json"
  "$TRAIN_CACHE_DIR/rec_results_refcocog_train.json"
)
EXPANDED_POOLS=(
  "$TRAIN_CACHE_DIR/rec_results_refcoco_train_expanded.json"
  "$TRAIN_CACHE_DIR/rec_results_refcoco+_train_expanded.json"
  "$TRAIN_CACHE_DIR/rec_results_refcocog_train_expanded.json"
)
QWEN_PICKS=(
  "$TRAIN_CACHE_DIR/crop_picks_refcoco_train.json"
  "$TRAIN_CACHE_DIR/crop_picks_refcoco+_train.json"
  "$TRAIN_CACHE_DIR/crop_picks_refcocog_train.json"
)

for path in "${BASE_POOLS[@]}" "${EXPANDED_POOLS[@]}" "${QWEN_PICKS[@]}" \
            "$CALIBRATION" "$CONTROLLER/risk_controller.pt" \
            "$CONTROLLER/risk_controller_config.json"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing TRAIN-only threshold input: $path" >&2
    exit 2
  fi
done

cd "$ROOT"
python -m analysis.training_threshold_selection \
  --base-pools "${BASE_POOLS[@]}" \
  --expanded-pools "${EXPANDED_POOLS[@]}" \
  --qwen-picks "${QWEN_PICKS[@]}" \
  --dino-calibration "$CALIBRATION" \
  --controller "$CONTROLLER" \
  --output-dir "$OUTPUT_DIR" \
  --selected-gamma0 "${GAMMA0:-0.50}" \
  --selected-gate "${GATE:-0.30}" \
  --selected-gamma1 "${GAMMA1:-0.35}" \
  --gate-calibration-gamma0 "${GATE_CALIBRATION_GAMMA0:-0.75}" \
  --max-qwen-intervention-pct "${MAX_QWEN_INTERVENTION_PCT:-10}"

echo "TRAIN-only threshold selection complete: $OUTPUT_DIR/training_threshold_selection.pdf"
