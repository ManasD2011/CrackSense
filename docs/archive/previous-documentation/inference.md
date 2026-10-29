> Historical document: superseded by the [current repository guide](../../../README.md). Paths, status and claims below may be outdated.

# Inference and Model Packaging

## Video inference

`Crack_Detector_main.py` is the main video script in this repository. It samples frames from a video, runs SegFormer inference, applies per-frame Otsu thresholding, resizes the binary mask back to the original frame dimensions, and saves both a red overlay and a binary mask.

Example:

```bash
python Crack_Detector_main.py --video path/to/video.mp4 --checkpoint path/to/model.pt --fps 2
```

It also supports `--num-frames`, `--start`, `--end`, `--max-frames`, `--output`, and `--tta`. Use `python Crack_Detector_main.py --help` for current defaults. TTA performs five predictions per frame and increases processing cost.

`crack_video_inference.py` is an earlier path with a `--threshold-method` option for calibrated or Otsu thresholding. Its default behavior and probability convention should be checked before comparing its output with the main script.

## Preprocessing and mask output

The main path converts OpenCV BGR frames to RGB, resizes the model input to 256 × 256, and applies ImageNet normalization. The two-class logits are converted to a crack probability map. After thresholding, nearest-neighbor resizing restores a binary mask to the source frame size. The overlay paints predicted crack pixels red and blends them with the original frame.

The model output is a visual segmentation aid. It does not estimate physical crack width or severity without a separately validated scale/calibration procedure.

## Checkpoint requirements

No trained checkpoint is shipped in this repository. The main script searches `Quantised/latest_fp32.pt`, `Quantised/latest_fp16.pt`, `Quantised/latest_int8.pt`, `checkpoints/latest.pt`, and `checkpoints/best.pt` in that order unless `--checkpoint` is supplied.

Some checkpoints contain serialized PyTorch model objects. Loading such a file requires a trusted source and a compatible Python/PyTorch/Transformers environment. Do not load untrusted checkpoint files.

## Quantization and benchmark scripts

- `quantize_segformer.py` exports FP32, FP16, and dynamic INT8 (linear layers) packages from a compatible source checkpoint.
- `Verify_quantised.py` inspects model packages and attempts a forward pass.
- `compare_videos.py` compares FP32/FP16 GPU and INT8 CPU paths, but its current benchmark uses a 224 × 224 input while the inference path uses 256 × 256. Record that difference in any performance report.

The final Black Book says quantized package performance and accuracy were not fully evaluated. Treat exported packages and benchmark scripts as experimental until rerun against the intended environment and validation data.

## Raspberry Pi daemon status

`drone_pipeline_PI.py` is experimental and is not runnable from a clean clone as currently packaged: it references `extract_frames_with_crack_detection_2.py`, which is absent, and expects `Quantised/latest_int8.pt`, which is also absent. The Black Book treats onboard inference as future work; do not present this helper as a validated Pi deployment.
