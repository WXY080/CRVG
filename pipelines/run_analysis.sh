#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${CRVG_PYTHON:-python}"
read -r -a DATASET_ARGS <<< "${DATASETS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"
ARGS=(--log-dir "${CRVG_LOGS:?Set CRVG_LOGS}" --datasets "${DATASET_ARGS[@]}")
"$PYTHON" -m tools.show_results "${ARGS[@]}"
"$PYTHON" -m analysis.rescue_damage "${ARGS[@]}"
"$PYTHON" -m analysis.routing_efficiency "${ARGS[@]}"
