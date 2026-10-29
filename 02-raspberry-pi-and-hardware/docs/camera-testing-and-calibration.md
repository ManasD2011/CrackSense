# Camera testing and calibration

Run on the Raspberry Pi, from the repository root, with the Pi environment active. Stop other camera owners first.

```bash
rpicam-hello --list-cameras
python 02-raspberry-pi-and-hardware/tests/camera/camera_fps_test.py
```

The FPS test captures 640×480 frames with Picamera2 and prints measured FPS, frame time and resolution. Stop with Ctrl+C. This measures capture-loop performance, not SegFormer speed or network latency.

For the earlier browser camera experiment:

```bash
python 02-raspberry-pi-and-hardware/camera/camera_stream.py
```

Open the address/port reported by Flask from the laptop. Stop this process before MediaMTX takes ownership of camera 0. `camera_debug.py` and `test_camera.py` use OpenCV video-device probes rather than the final Picamera2 path. The copied `camera.py` helper retains its original default index 2; inspect device selection before running it.

## Calibration evidence

The team calibrated the IMX219. Stored intrinsics and distortion coefficients are present in `dashboard/app.py` and ArUco scripts under `experiments/VIO/aruco/`. The source collection does not contain a standalone calibration capture/solve script, calibration image set or verified reprojection-error report. These should not be invented or described as included.

Inspect the camera matrix, distortion vector, image resolution, crop and marker dimensions before reusing them. Values from one camera setup are not a calibration for another. The VIO camera helper optionally applies undistortion when coefficients are passed. The final live SegFormer `model.py` does not use those coefficients; calibration completion does not imply that inference frames are undistorted.

Next: [MAVLink and lidar](mavlink-and-lidar-testing.md).
