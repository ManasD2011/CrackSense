# Model packaging and precision experiments

**Execution:** training/ground-station computer. Run from `03-segformer/` so default `Quantised/` paths remain consistent.

```bash
python model-packaging/quantize_segformer.py --input checkpoints/latest.pt --output-dir Quantised
python model-packaging/Verify_quantised.py
```

The exporter writes `<source-stem>_fp32.pt`, `<source-stem>_fp16.pt` and an experimental dynamic-INT8 package. It accepts a saved module or reconstructs a specific B2 configuration from a supported state dictionary. The B2 reconstruction path is not a generic B0/B1 exporter. Strict loading must succeed; do not suppress shape errors.

`Verify_quantised.py` inspects packages in `Quantised/`. `compare_videos.py` is an earlier precision comparison; inspect its hard-coded inputs/configuration first. Use the [Phase 1 benchmark](../../04-live-implementation/benchmarks/README.md) for the documented timing scope.

Live deployment uses FP16 on CUDA. Exported dynamic INT8 is not supported by the final live model loader. Metadata must retain the matching validated threshold, or the live command must receive an explicitly validated value.
