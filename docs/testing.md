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

## GPU Runs

Real checkpoints, images, and benchmark data are needed for GPU evaluation:

1. Record checkpoint, processor, library, and CUDA versions.
2. Inspect generated boxes for each backbone's coordinate convention.
3. Run a complete small batch through all stages and confirm sample coverage.
4. Supply the same controller weights and policy to every backbone.
5. Keep configuration, manifests, and outputs with each reported run.

Inference raises errors on missing required evidence, mismatched score sources, incompatible feature schemas, and invalid controller policies.
