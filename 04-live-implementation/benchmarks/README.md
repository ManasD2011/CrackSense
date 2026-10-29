# FP32 and FP16 benchmark

**Runs on:** CUDA laptop, using `benchmark_phase1.py`. From the repository root:

```bash
python 04-live-implementation/benchmarks/benchmark_phase1.py --help
python 04-live-implementation/benchmarks/benchmark_phase1.py --video /path/to/video.mp4 --model-dir 03-segformer/Quantised --output phase1_results --frames 100 --stride 5 --warmup 20 --repeats 3
```

Inspect the script's expected package filenames and the source video's availability first. `requirements-working.txt` preserves the historical recorded environment; it is not a portable lockfile, and CUDA packages may require a separate index. The selected recorded output is in `../examples/phase1-summary.json`.

The timed scope excludes pre-sampled RGB conversion/resizing, video decoding, capture, transport, display and disk output. Agreement against FP32 is not ground-truth accuracy. [Result interpretation](../../docs/results-and-limitations.md).
