# CRVG: Consensus-Routed Visual Grounding

Official implementation of **Knowing When to Intervene: Consensus-Routed Evidence Acquisition for Risk-Aware Visual Grounding**.

CRVG is a selective post-hoc intervention for referring expression comprehension. It diagnoses candidate agreement, acquires additional evidence only for ambiguous inputs, and preserves the current prediction unless a challenger passes the corresponding confidence and risk checks. The grounding backbone, Qwen3-VL-8B-Instruct, and Grounding-DINO-base remain frozen.

## Method Overview

1. **Routing.** One greedy prediction and eight stochastic decodes form the candidate pool B0. The cascade preserves the current prediction unless B0 holds at least two candidates with minimum pairwise IoU below **0.50**; otherwise equivariant views (horizontal flip plus three padded canvases) may challenge and update the current box.
2. **Evidence acquisition.** A frozen Qwen3-VL scores red-box crops and replaces the current box when the probability advantage exceeds **0.30**; if post-expansion consensus stays below **0.35**, Grounding-DINO appends up to eight external proposals, and every challenger is compared with the current box in both orders.
3. **Risk-aware termination.** Pairwise evidence, candidate geometry, detector confidence, and consensus state determine whether to keep the current box, switch to a challenger, or abstain. A switch is accepted only when it is order-consistent and clears the frozen utility threshold.

Default thresholds **gamma0=0.50 / gate=0.30 / gamma1=0.35** live in [configs/default.yaml](configs/default.yaml). Score definitions, the equation-to-code map, clustering and medoid details, and implementation settings are in the [method specification](docs/paper_alignment.md).

## Released Artifacts

| Path | Contents |
| --- | --- |
| `checkpoints/dino_three_domain_risk_controller/` | Frozen final-stage decision policy used in the reported experiments. |
| `artifacts/controller_data/dino_three_domain_risk_train.jsonl` | TRAIN-only current-challenger examples for the released policy. |
| `artifacts/controller_data/dino_three_domain_risk_calibration.jsonl` | Held-out TRAIN examples used to select the released operating policy. |
| `artifacts/controller_data/dino_three_domain_risk_audit.json` | Data composition and integrity statistics. |

## Installation

Use Linux for GPU execution. Install the PyTorch build appropriate for the server's CUDA version, then:

```bash
git clone https://github.com/WXY080/CRVG.git
cd CRVG
python -m pip install -e ".[inference,analysis]"
python -m pip check
```

This environment runs the frozen Qwen3 verifier, Grounding-DINO, the CRVG decision stage, and analysis using Transformers **4.57.6**. PaDT candidate generation is run with its native environment and imported through the cache interface below.

## Data

Each `DATA_ROOT/<dataset>.json` contains one referring expression per row:

```json
[
  {
    "dataset_index": 0,
    "image": "COCO_train2014_000000000001.jpg",
    "text": "the object on the right",
    "bbox": [30, 20, 80, 60],
    "width": 640,
    "height": 480
  }
]
```

Boxes use pixel **[x, y, width, height]**. An alternative `solution` field is interpreted as pixel xyxy. Image paths may be absolute or relative to the image root.

```bash
python -m tools.prepare_data --input /data/raw/refcoco_val.json \
  --image-root /data/coco/train2014 --dataset refcoco_val \
  --output /data/rec/refcoco_val.json
```

This command validates images and expands multi-sentence annotations. Ground truth supplies controller labels and evaluation metrics; it is not an inference feature or model-prompt input.

TRAIN domains: `refcoco_train`, `refcoco+_train`, `refcocog_train`.

Evaluation splits: `refcoco_val`, `refcoco_testA`, `refcoco_testB`, `refcoco+_val`, `refcoco+_testA`, `refcoco+_testB`, `refcocog_val`, `refcocog_test`.

## Run CRVG

First generate PaDT's greedy, BoN, and equivariant candidate caches with the original PaDT code. Convert the two output files for each split into CRVG's canonical cache format:

```bash
export CRVG_QWEN3=/models/Qwen3-VL-8B-Instruct
export CRVG_DINO=/models/grounding-dino-base
export CRVG_IMAGES=/data/coco/train2014
export CRVG_LOGS=logs/padt
export CRVG_CONTROLLER=checkpoints/dino_three_domain_risk_controller

python -m tools.import_candidates --format padt --dataset refcoco_val \
  --base /data/padt/rec_results_refcoco_val.json \
  --ece /data/padt/equivariant_candidates_refcoco_val.json \
  --out-base /data/crvg-cache/rec_results_refcoco_val.json \
  --out-ece /data/crvg-cache/ece_refcoco_val.json

python -m crvg.pipeline --backbone padt --candidate-cache-dir /data/crvg-cache \
  --image-root /data/coco/train2014 --qwen-model /models/Qwen3-VL-8B-Instruct \
  --dino-model /models/grounding-dino-base \
  --controller checkpoints/dino_three_domain_risk_controller \
  --log-dir logs/padt --datasets refcoco_val
```

Supply the greedy-initialized B0 and completed four-view evidence for every required input. The importer checks sample identity, source hashes, and matching current predictions.

`BATCH_SIZE` controls Qwen and Grounding-DINO inference. Run manifests record commands and input, output, and source-code hashes; use `--rebuild` after replacing a checkpoint at the same path.

### Optional Backbone Interfaces

The repository also includes generation adapters for VLM-R1 and the original InternVL3 checkpoint interface. They implement the same candidate-cache contract but are not part of the PaDT experiment path described above. Their launchers are `pipelines/run_vlmr1.sh` and `pipelines/run_internvl.sh`; InternVL's native interface uses the `.[internvl]` dependency set.

## Released Decision Policy

The reported pipeline loads the fixed parameters in `checkpoints/dino_three_domain_risk_controller/` and applies them unchanged on every evaluation split. Ground-truth boxes are used only to compute reported metrics, never as decision features or prompt inputs. The feature order and acceptance rule are documented in the [method specification](docs/paper_alignment.md).

## Evaluation

```bash
CRVG_LOGS=logs/padt bash pipelines/run_analysis.sh
```

Reports include Acc@0.50/0.75/0.90, mIoU, rescue/damage, and routing counts. **Average is the unweighted mean across the requested splits.**

Two offline analyses operate on cached run logs:

- `analysis.replay_thresholds` re-routes a finished run under alternative thresholds. It raises an error when the cached evidence is missing or the current box changed in a way that invalidates the pairwise scores.
- `analysis.threshold_sweep` scans a threshold grid over the cached evidence and reports covered combinations; combinations the cache does not cover are skipped and listed.

```bash
python -m analysis.replay_thresholds --log-dir logs/padt \
  --datasets refcoco_val --gamma0 0.50 --gate 0.30 --gamma1 0.35 \
  --output-dir outputs/replay
python -m analysis.threshold_sweep --log-dir logs/padt \
  --datasets refcoco_val --output-dir outputs/sensitivity
```

The paper's threshold-selection figure is a separate TRAIN-only analysis. Point `TRAIN_CACHE_DIR` to the aligned difficult-TRAIN BoN, broad-ECE, and frozen-Qwen caches for `refcoco_train`, `refcoco+_train`, and `refcocog_train`:

```bash
TRAIN_CACHE_DIR=/data/crvg-threshold-train \
  bash pipelines/run_training_threshold_selection.sh
```

This produces JSON/CSV source data, a Markdown summary, a caption, and PDF/SVG/PNG figures in `outputs/training_threshold_selection`. The ECE caches must cover the largest scanned `gamma0`; the default figure scans through 0.95. The script independently regenerates the TRAIN-only DINO route report and fails if the selected values do not reproduce `0.50/0.30/0.35`.

## Testing

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
python -m compileall -q crvg analysis tools tests
bash -n pipelines/run_internvl.sh
bash -n pipelines/run_vlmr1.sh
bash -n pipelines/run_analysis.sh
bash -n pipelines/run_training_threshold_selection.sh
```

The CPU suite covers method contracts and cached workflows with synthetic fixtures; it does not load real model checkpoints. GPU steps and coverage details are in the [testing guide](docs/testing.md).

Model weights and benchmark images/annotations follow their own licenses and are obtained from their official sources; this repository distributes code, the frozen controller, and the controller-development data. Code is MIT-licensed; see [third-party components](THIRD_PARTY.md).
