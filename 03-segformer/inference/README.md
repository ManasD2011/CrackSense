# Offline video inference

**Execution:** ground-station laptop/training computer. These scripts process a local video, not the final live RTSP/telemetry application.

From the repository root:

```bash
cd 03-segformer
python inference/Crack_Detector_main.py --help
python inference/Crack_Detector_main.py --video /path/to/inspection.mp4 --checkpoint /path/to/model.pt --fps 2
```

`Crack_Detector_main.py` samples video frames and saves binary masks and overlays. Its main path uses per-frame Otsu thresholding. `crack_video_inference.py` is an earlier path with calibrated/Otsu options; inspect its help before use. Both run on the computer, not the Pi.

The images are converted to RGB, resized and normalized before inference. Mask reconstruction, optional TTA and threshold choices differ between experiments. Do not compare outputs as though these choices were identical. For live frame/range processing use [Phase 4](../../04-live-implementation/README.md).
