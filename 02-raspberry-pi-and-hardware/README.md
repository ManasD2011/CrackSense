# Phase 2 — Raspberry Pi and hardware integration

[Project overview](../README.md) · [Getting started](../docs/getting-started.md)

This phase verifies the onboard camera, flight-controller connection and distance sensors before live segmentation. Diagnostic scripts run on the Raspberry Pi 5; Mission Planner runs on the laptop for flight-controller configuration.

Use the [confirmed wiring](../docs/hardware-and-wiring.md). Keep the aircraft disarmed and propellers removed for bench work. The following sensor tests do not require running the guided-flight experiments.

## Upgraded onboard platform

<img src="../docs/images/project/drone-v2-top.jpeg" alt="Top view of the upgraded drone showing the Pi enclosure, camera ribbon and onboard electronics" width="760">

*The upgraded 920 KV build with onboard computing and camera integration. Use the confirmed wiring guide for electrical connections; this photo shows physical placement.*

## Recommended test sequence

| Step | Run or read | Expected checkpoint |
|---|---|---|
| 1 | [Setup and SSH](docs/setup-and-ssh.md) | Remote shell and correct Python environment |
| 2 | [Camera tests](docs/camera-testing-and-calibration.md) | Detected IMX219, captured frames and measured capture FPS |
| 3 | Camera calibration notes in the same guide | Understand saved intrinsics and their scope |
| 4 | `tests/telemetry/heartbeat_test.py` | Pixhawk heartbeat, system and component ID |
| 5 | `tests/telemetry/Attitude_test.py` | Attitude messages arriving |
| 6 | `tests/telemetry/lidar2_mavlink_test.py` | Both ground and wall readings, identity, validity and age |
| 7 | [Live startup](../04-live-implementation/docs/startup-sequence.md) | Video and telemetry received on laptop |

Steps 4–6 are detailed in [MAVLink and lidar testing](docs/mavlink-and-lidar-testing.md). Run only one serial reader at a time and stop the test before starting the live sender.

## Folder roles

- `tests/camera/`: Picamera2 FPS test plus older V4L2/DirectShow camera probes.
- `tests/telemetry/`: recommended USB MAVLink communication tests.
- `camera/`: simple Flask camera streaming experiment.
- `dashboard/`: earlier Pi-hosted multi-camera/telemetry dashboard; not the final laptop segmentation UI.
- `experiments/direct-uart/`: earlier direct-UART and loopback experiments; not the final USB Pixhawk route.
- `experiments/VIO/`: preserved ArUco, optical flow, telemetry and flight-control exploration, including placeholders. Read its README first.

These hardware-facing scripts are manual diagnostics, not an automated test suite. Run the documented scripts individually rather than using automated test discovery. [Full file index](../docs/file-index.md).
