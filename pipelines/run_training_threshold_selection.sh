#!/usr/bin/env bash
# Reproduce the paper's TRAIN-only threshold-selection figure from cached evidence.
set -euo pipefail

: "${TRAIN_CACHE_DIR:?Set TRAIN_CACHE_DIR to the difficult-TRAIN cache directory}"

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTROLLER="${CRVG_CONTROLLER:-$ROOT/checkpoints/dino_three_domain_risk_controller}"
CALIBRATION="${DINO_CALIBRATION:-$ROOT/artifacts/controller_data/dino_three_domain_risk_calibration.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/training_threshold_selection}"
DINO_REPORT="${DINO_REPORT:-$OUTPUT_DIR/dino_route_training_sweep.json}"

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
mkdir -p "$OUTPUT_DIR"
python -m analysis.dino_route_calibration \
  --calibration "$CALIBRATION" \
  --controller "$CONTROLLER" \
  --thresholds "${GAMMA1_VALUES:-0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75}" \
  --min-domain-net "${MIN_DOMAIN_NET:-0}" \
  --min-total-net "${MIN_TOTAL_NET:-0}" \
  --out "$DINO_REPORT"

python -m analysis.training_threshold_selection \
  --base-pools "${BASE_POOLS[@]}" \
  --expanded-pools "${EXPANDED_POOLS[@]}" \
  --qwen-picks "${QWEN_PICKS[@]}" \
  --dino-calibration-report "$DINO_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-gamma0 "${GAMMA0:-0.50}" \
  --expected-gate "${GATE:-0.30}" \
  --expected-gamma1 "${GAMMA1:-0.35}" \
  --gate-calibration-gamma0 "${GATE_CALIBRATION_GAMMA0:-0.75}" \
  --max-qwen-intervention-pct "${MAX_QWEN_INTERVENTION_PCT:-10}" \
  --min-domain-net "${MIN_DOMAIN_NET:-0}" \
  --min-total-net "${MIN_TOTAL_NET:-0}" \
  --min-average-dmiou-pp "${MIN_AVERAGE_DIOU_PP:-0}" \
  --require-expected

echo "TRAIN-only threshold selection complete: $OUTPUT_DIR/training_threshold_selection.pdf"
