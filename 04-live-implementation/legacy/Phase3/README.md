# Historical Phase3

Laptop OpenCV live-inference baseline and its earlier revision. Predates the final timestamp-aware receiver.

These original notes contain old working directories, local addresses and commands. See [current startup](../../docs/startup-sequence.md).

<details>
<summary>Original project notes</summary>

# Phase 3 — Live FP16 bench application

Keep MediaMTX running on the Pi with the Phase 2 configuration. Close VLC and FFplay. This application runs on the Windows laptop, in the same Python environment that passed Phase 1. Preserve your existing scripts and model files.

Extract this folder and place `laptop_live_inference_old.py` in your project folder, beside the `Quantised` folder. It expects `Quantised/latest_fp16.pt`. Alternatively pass the full checkpoint path.

## 1. Receiver-only test (Windows CMD)

```cmd
python laptop_live_inference.py --stream rtsp://10.207.215.185:8554/cam0 --preview-only --duration 60
```

Check preview delay with the stopwatch again: this receiver differs from FFplay. The application uses the FFmpeg backend built into OpenCV; it does not launch the downloaded ffmpeg.exe or ffplay.exe. Do not install opencv-python-headless, since GUI windows are required.

## 2. Live segmentation

```cmd
python laptop_live_inference.py --stream rtsp://10.207.215.185:8554/cam0 --checkpoint Quantised/latest_fp16.pt --duration 900
```

Threshold is read from `checkpoint_metadata.best_threshold` or the package's `best_threshold`. If absent, supply your validated Phase 1 value with `--threshold VALUE`; do not literally enter VALUE or invent a threshold. The application prints an actionable error rather than silently choosing one.

Two windows appear: smooth raw preview and matched segmentation. Focus a window and press **Q** to quit or **S** to save the current matched original, binary mask and overlay.

Outputs are timestamped session folders under `phase3_results`, containing configuration, per-result CSV and summary statistics. Saved frame IDs are assigned at decoder delivery, not at Pi capture. Preview-only runs have no inference metrics.

## Behavior and timing

- Model stays loaded on CUDA in FP16, batch size one, 256 x 256, RGB/ImageNet normalization, TTA disabled.
- Probability matches Crack_Detector_main.py: sigmoid(crack logit minus background logit), after bilinear logit resizing. The attached older compare_videos.py instead applies sigmoid only to the crack logit; its predictions are not equivalent. Verify your successful Phase 1 benchmark's probability convention and threshold before comparing quality.
- Receiver continuously drains the stream. One waiting frame replaces older waiting frames. This bounds the application queue; it does not prove all upstream decoder/network queues have bounded latency. Measure with the stopwatch.
- Read/open timeouts trigger reconnect attempts. A connection generation prevents old results from being displayed after a receiver reconnect. Results expire after one second by default. Timely delivery of buffered old frames cannot be detected without capture timestamps.
- `processing_ms` includes preprocessing, GPU inference, CPU mask transfer and mask resizing. CPU transfer completes GPU work before timing ends.
- `decode_to_view_ms` ends before overlay annotation/display submission and excludes capture, encoder, network and decoder delay. It is NOT full camera-to-result delay or physical screen latency. Re-measure full delay using a visible stopwatch in the matched window.
- Reported inference FPS counts unique results submitted for display. Skipped-frame counts cover waiting frames displaced between successive processed frames, not network losses or results replaced before display.



## Troubleshooting

- `CUDA unavailable`: activate the working Phase 1 environment; do not replace its PyTorch installation.
- Missing imports: use that same environment. Required packages are torch, transformers, numpy and GUI-capable OpenCV. Packaged model loading may require its original Transformers version.
- No preview: check MediaMTX output and Pi IP. DHCP may change the address. Some OpenCV builds lack FFmpeg or timeout support; the receiver logs errors instead of falling back to unbounded reads. Inspect with `python -c "import cv2; print(cv2.getBuildInformation())"`.
- Model loading uses `weights_only=False` for your serialized packaged model; load only your trusted checkpoint.
- Red regions are model predictions; threshold and streaming compression must be validated on representative crack/no-crack scenes.



## Remaining validation

Phase 2's six screenshots gave an approximate 420–480 ms range and 443 ms mean. Endurance, recovery, thermal behavior and fine-crack detail remain unvalidated. Phase 3 is a bench integration, not flight approval. Test a 15-minute combined run, deliberate link loss, correct mask/frame pairing, and full camera-to-result delay before field trials.

Validation in this delivery: Python syntax, latest-frame replacement and generation invalidation passed. Live RTSP, GUI, CUDA inference and checkpoint loading require your laptop/Pi and were not executed here.
</details>
