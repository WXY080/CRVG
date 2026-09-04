# CRVG: Consensus-Routed Visual Grounding

Official implementation of **Knowing When to Intervene: Consensus-Routed Evidence Acquisition for Risk-Aware Visual Grounding**.

CRVG is a selective post-hoc intervention for referring expression comprehension. The grounding backbone, the Qwen3-VL-8B-Instruct verifier, and Grounding-DINO-base all stay frozen; CRVG routes only low-consensus samples through extra evidence acquisition and a risk-aware switch decision.

## Released Artifacts

| Path | Contents |
| --- | --- |
| `checkpoints/dino_three_domain_risk_controller/` | Frozen risk controller used in the reported experiments: `risk_controller.pt` (weights) and `risk_controller_config.json` (36-D feature schema, normalization, utility costs, acceptance threshold, policy). |
| `artifacts/controller_data/dino_three_domain_risk_train.jsonl` | 3,502 difficult examples used for controller fitting. |
| `artifacts/controller_data/dino_three_domain_risk_calibration.jsonl` | 799 training-only examples used for policy selection (images disjoint from the fitting set). |
| `artifacts/controller_data/dino_three_domain_risk_audit.json` | Statistics of the two controller-development sets. |

## Method Overview

1. **Routing.** One greedy prediction and eight stochastic decodes form the candidate pool B0. The cascade preserves the current prediction unless B0 holds at least two candidates with minimum pairwise IoU below **0.50**; otherwise equivariant views (horizontal flip plus three padded canvases) may challenge and update the current box.
2. **Evidence acquisition.** A frozen Qwen3-VL scores red-box crops and replaces the current box when the probability advantage exceeds **0.30**; if post-expansion consensus stays below **0.35**, Grounding-DINO appends up to eight external proposals, and every challenger is compared with the current box in both orders.
3. **Risk-aware termination.** A frozen 36-feature controller predicts KEEP/SWITCH/ABSTAIN. The prediction is replaced only by an order-consistent challenger whose utility clears the saved threshold; KEEP and ABSTAIN both preserve it.

Default thresholds **gamma0=0.50 / gate=0.30 / gamma1=0.35** live in [configs/default.yaml](configs/default.yaml). Score definitions, the equation-to-code map, clustering and medoid details, and implementation settings are in the [method specification](docs/paper_alignment.md).

## Installation

Use Linux for GPU execution. Install the PyTorch build appropriate for the server's CUDA version, then:

```bash
git clone https://github.com/WXY080/CRVG.git
cd CRVG
python -m pip install -e ".[inference,analysis]"
python -m pip check
```

This installs every dependency used by the code: PyTorch, Transformers (pinned to **4.57.6**), accelerate, Pillow, tqdm, numpy, PyYAML, and matplotlib. Backbone generation and verification use the same Transformers stack, so a single environment is sufficient; the pipeline scripts also accept separate interpreters for the backbone and verifier stages (`CRVG_BACKBONE_PYTHON`, `CRVG_VERIFIER_PYTHON`).

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

```bash
export CRVG_MODEL=/models/InternVL3-9B
export CRVG_QWEN3=/models/Qwen3-VL-8B-Instruct
export CRVG_DINO=/models/grounding-dino-base
export CRVG_DATA=/data/rec
export CRVG_IMAGES=/data/coco/train2014
export CRVG_LOGS=logs/internvl
export CRVG_CONTROLLER=checkpoints/dino_three_domain_risk_controller

GPU=0 DATASETS="refcoco_val" bash pipelines/run_internvl.sh --dry-run
GPU=0 DATASETS="refcoco_val" bash pipelines/run_internvl.sh
```

For VLM-R1, set its checkpoint and use `pipelines/run_vlmr1.sh`. Run manifests record configuration, source-code hashes, and input/output hashes. Use `--rebuild` after changing checkpoint contents at the same path.

### PaDT Candidate Input

Generate the native PaDT BoN and equivariant caches, then convert them:

```bash
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

## Risk Controller

The released controller was fitted once on difficult cases mined from the three TRAIN domains using PaDT-REC-3B, then frozen; the same directory is reused unchanged on every backbone. To refit from the released controller-development data:

```bash
python -m crvg.controller.train \
  --train artifacts/controller_data/dino_three_domain_risk_train.jsonl \
  --val artifacts/controller_data/dino_three_domain_risk_calibration.jsonl \
  --output-dir checkpoints/risk --device cpu
```

Training exits with a nonzero status if calibration fails. Controller-development data can also be regenerated end to end: acquire evidence from the three TRAIN domains using PaDT-REC-3B with `--stop-after pairwise`, then partition and build with `crvg.controller.build_data`; labels follow Section 3.4 of the manuscript.

## Evaluation

```bash
CRVG_LOGS=logs/internvl bash pipelines/run_analysis.sh
```

Reports include Acc@0.50/0.75/0.90, mIoU, rescue/damage, and routing counts. **Average is the unweighted mean across the requested splits.**

Two offline analyses operate on cached run logs:

- `analysis.replay_thresholds` re-routes a finished run under alternative thresholds. It raises an error when the cached evidence is missing or the current box changed in a way that invalidates the pairwise scores.
- `analysis.threshold_sweep` scans a threshold grid over the cached evidence and reports covered combinations; combinations the cache does not cover are skipped and listed.

```bash
python -m analysis.replay_thresholds --log-dir logs/internvl \
  --datasets refcoco_val --gamma0 0.50 --gate 0.30 --gamma1 0.35 \
  --output-dir outputs/replay
python -m analysis.threshold_sweep --log-dir logs/internvl \
  --datasets refcoco_val --output-dir outputs/sensitivity
```

## Testing

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
python -m compileall -q crvg analysis tools tests
bash -n pipelines/run_internvl.sh
bash -n pipelines/run_vlmr1.sh
bash -n pipelines/run_analysis.sh
```

The CPU suite covers method contracts and cached workflows with synthetic fixtures; it does not load real model checkpoints. GPU steps and coverage details are in the [testing guide](docs/testing.md).

Model weights and benchmark images/annotations follow their own licenses and are obtained from their official sources; this repository distributes code, the frozen controller, and the controller-development data. Code is MIT-licensed; see [third-party components](THIRD_PARTY.md).
