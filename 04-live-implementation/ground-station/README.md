# Ground-station processing

The laptop receives camera frames and telemetry, runs SegFormer, and displays the original image alongside the predicted crack overlay and distance readings.

| Module | Responsibility |
|---|---|
| `laptop_dashboard.py` | Reception checks, preview, GPU inference, display and recording |
| `rtsp_video.py` | Video reception and decoding |
| `model.py` | SegFormer preprocessing and segmentation |

Preserve the adjacent `shared/` folder. Follow the [startup sequence](../docs/startup-sequence.md), beginning with reception checks before loading a model.
