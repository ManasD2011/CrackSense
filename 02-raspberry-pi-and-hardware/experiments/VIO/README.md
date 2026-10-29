# Historical VIO and ArUco experiments

**Execution:** Raspberry Pi, with some OpenCV camera helpers also supporting Windows. **Status:** experiments and scaffolding, not prerequisites for live crack segmentation.

The original relative layout is preserved so sibling imports remain usable. `camera2_web.py` and `new_camera_web.py` are camera/recording experiments; `optical_flow.py` and its test explore image motion. `aruco/` explores marker pose and external-navigation messages. `guided/` and `tests/modes/` can issue flight/mode commands and are not passive sensor checks. Read their source and perform separate controlled hardware validation before use.

`config.py`, `ekf.py`, `main.py`, `motion_estimator.py`, `pixhawk_interface.py` and `position_controller.py` are empty scaffolding. Multiple files under `tests/` are also empty or comment-only. The [file index](../../../docs/file-index.md) identifies each. Nonempty read-only telemetry examples include `tests/heartbeat_test.py`, `tests/attitude_test.py`, `tests/imu_test.py` and `tests/altitude.py`, using `tests/common.py`.

The ArUco external-navigation bench scripts support a dry run and require `--send` to publish ODOMETRY. They do not themselves arm or command movement. This is separate from the command-capable guided/mode scripts and is not a complete autonomous flight system. Camera calibration and marker-size constants must match the actual setup.
