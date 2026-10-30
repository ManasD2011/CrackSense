# Camera diagnostics

Run these checks on the Raspberry Pi before starting the live stream. Begin with `camera_fps_test.py`, which uses Picamera2 to report capture performance. The OpenCV probes provide additional device diagnostics; check their selected camera index before use.

Follow the [camera testing and calibration guide](../../docs/camera-testing-and-calibration.md) for commands and expected results. Stop MediaMTX and other camera applications before running a test.
