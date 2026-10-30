# Automated software tests

Run these tests on a development computer from the repository root:

```bash
python -m unittest discover -s 04-live-implementation/tests -v
```

The integration tests require NumPy, OpenCV and PyAV with libx264 support. Core and sender tests use simulated inputs without flight hardware.

The suite checks software behaviour, including timing utilities, telemetry handling and local video processing. Hardware timing, flight performance and CUDA checkpoint compatibility require separate verification.
