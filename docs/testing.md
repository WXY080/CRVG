# Testing

Run the CPU test suite and syntax checks:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
python -m compileall -q crvg analysis tools tests
bash -n pipelines/run_internvl.sh
bash -n pipelines/run_vlmr1.sh
bash -n pipelines/run_analysis.sh
```

The same commands run in CI on every push and pull request.

## Coverage

The suite checks coordinate conversion, inverse views, provenance-preserving clustering, threshold equality, post-ECE candidate preservation, weak-Qwen continuation into DINO, order alignment, three-action termination, ground-truth-independent features, data identity, controller serialization, full-split metrics, and cached replay. Fixtures are synthetic, so the suite runs on CPU without model weights, images, or benchmark data. It verifies method contracts, not benchmark accuracy.

Backbone interface tests cover left-padded variable-length batches, sampled-output grouping, native InternVL loading and tile counts, generation limits for ECE views, and pipeline argument forwarding.

## GPU Runs

Real checkpoints, images, and benchmark data are needed for GPU evaluation:

1. Record checkpoint, processor, library, and CUDA versions.
2. Inspect generated boxes for each backbone's coordinate convention.
3. Run a complete small batch through all stages and confirm sample coverage.
4. Supply the same controller weights and policy to every backbone.
5. Keep configuration, manifests, and outputs with each reported run.

Use a small annotation file containing at least two different image sizes and expression lengths. In the environments described in the README, test each backbone with `BACKBONE_BATCH_SIZE=1`, then `BACKBONE_BATCH_SIZE=2`, and run through ECE:

```bash
GPU=0 BACKBONE_BATCH_SIZE=2 DATASETS="refcoco_val" STOP_AFTER=ece \
  CRVG_DATA=/data/rec-smoke CRVG_LOGS=logs/internvl-smoke \
  bash pipelines/run_internvl.sh
```

For VLM-R1, use `pipelines/run_vlmr1.sh` and the Transformers 4.57.6 interpreter. Check for one greedy and eight sampled outputs per input, valid box coordinates, and four attempted views for routed ECE inputs. Include a low-consensus input so ECE is exercised. Compare generated boxes and full-split metrics with the reference caches before reporting reproduction results; a fixed seed alone does not establish equivalence across inference implementations or batch sizes.

Inference raises errors on missing required evidence, mismatched score sources, incompatible feature schemas, and invalid controller policies.
