# Testing

Run the CPU test suite and syntax checks:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
python -m compileall -q crvg analysis tools tests
bash -n pipelines/run_internvl.sh
bash -n pipelines/run_vlmr1.sh
bash -n pipelines/run_analysis.sh
bash -n pipelines/run_training_threshold_selection.sh
```

The same commands run in CI on every push and pull request.

## Coverage

The suite checks coordinate conversion, inverse views, provenance-preserving clustering, threshold equality, post-ECE candidate preservation, weak-Qwen continuation into DINO, order alignment, three-action termination, ground-truth-independent features, data identity, controller serialization, full-split metrics, and cached replay. Fixtures are synthetic, so the suite runs on CPU without model weights, images, or benchmark data. It verifies method contracts, not benchmark accuracy.

The PaDT cache tests cover import identity, threshold coverage, candidate alignment, and pipeline argument forwarding. Optional backbone-interface tests cover left-padded variable-length batches, sampled-output grouping, native InternVL loading and tile counts, and generation limits for ECE views.

## GPU Runs

Real checkpoints, images, and benchmark data are needed for GPU evaluation:

1. Record PaDT, Qwen3-VL, Grounding-DINO, library, and CUDA versions.
2. Convert a small PaDT cache and verify the imported current and candidate boxes.
3. Run the converted inputs through Qwen, DINO, pairwise comparison, and final selection.
4. Confirm sample coverage and retain the generated manifests with the results.

Use a small PaDT cache containing at least two image sizes, one early-exit input, and one routed input:

```bash
python -m tools.import_candidates --format padt --dataset refcoco_val \
  --base /data/padt/rec_results_refcoco_val.json \
  --ece /data/padt/equivariant_candidates_refcoco_val.json \
  --out-base /data/crvg-smoke/rec_results_refcoco_val.json \
  --out-ece /data/crvg-smoke/ece_refcoco_val.json

python -m crvg.pipeline --backbone padt \
  --candidate-cache-dir /data/crvg-smoke --image-root /data/coco/train2014 \
  --qwen-model /models/Qwen3-VL-8B-Instruct \
  --dino-model /models/grounding-dino-base \
  --controller checkpoints/dino_three_domain_risk_controller \
  --log-dir logs/padt-smoke --datasets refcoco_val
```

Check for one current box, the complete native PaDT candidate pool, four attempted equivariant views for routed inputs, and a final decision for every sample. Compare the converted cache and full-split metrics with the native PaDT files before reporting results.

Inference raises errors on missing required evidence, mismatched score sources, incompatible feature schemas, and invalid controller policies.
