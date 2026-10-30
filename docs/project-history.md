# Project history

## 1 Initial CNN and U-Net work

The initial system used CNN classification and U-Net segmentation. Hardware comprised an F450 frame, 1000 KV motors, 40 A ESCs, FlySky FS-i6 transmitter/receiver, Pixhawk 2.4.8 and a LiPo battery. Mission Planner was used for tuning. Insufficient stability in that build and unsatisfactory model accuracy prevented real drone-feed model testing.

## 2 Transformer model development

The team moved to SegFormer, trained B0, B1 and B2, and tested them on test data. CNN and U-Net are also deep-learning architectures: this was a change from convolutional models to transformer-based segmentation. CrackSeg9k and CrackVision12K were used. The available training notebooks preserve development configurations, while the explained-model notebooks load saved checkpoints. They do not constitute complete independent training records for every final variant.

## 3 Hardware upgrade and integration

920 KV DJI motors replaced the initial motors; the team reports more than 500 g thrust per motor in its setup and a stable assembled drone. Raspberry Pi 5, IMX219 and two TFmini-S lidars were mounted. Work included camera tests and calibration, MAVLink heartbeat tests, receiving both lidar streams through Pixhawk, and SSH access during flight. The confirmed upgraded wiring diagram labels 30 A ESCs; the initial build used 40 A ESCs.

## 4 Live implementation

MediaMTX on the Pi streams IMX219 video. A Pi sender reads MAVLink measurements from Pixhawk and sends timestamped UDP telemetry. The laptop decodes frames, associates eligible ranges, runs SegFormer and displays paired raw/processed images. Saved sessions demonstrate operation and observed timing mappings. Physical acquisition-time synchronization, full visual latency and a quantitative airborne accuracy campaign are not established by those logs.

[Hardware](hardware-and-wiring.md) · [Data flow](system-data-flow.md) · [Results](results-and-limitations.md)

## Photographic development record

<table>
<tr><td width="50%" align="center"><img src="images/project/drone-v1.jpeg" alt="Initial F450 assembly used during offline model development." height="280"></td><td width="50%" align="center"><img src="images/project/drone-v2-front.jpeg" alt="Upgraded aircraft used for camera and live inspection integration." height="280"></td></tr>
<tr><td>Initial F450 assembly used during offline model development.</td><td>Upgraded aircraft used for camera and live inspection integration.</td></tr>
</table>

See the [project gallery](project-gallery.md) for additional photographs.
