# Camera calibration record

The team calibrated the IMX219. Saved camera matrices and distortion coefficients are embedded in [the Pi dashboard](../dashboard/app.py) and [ArUco scripts](../experiments/VIO/aruco/aruco_externalnav_test.py). The ArUco bench code labels its values for a 640×480 detector; check resolution, crop and mounting before using them.

The supplied files do not include the original calibration images, a standalone solver or a reprojection-error report. A new camera setup requires its own calibration and error assessment. Follow [camera testing and calibration](../docs/camera-testing-and-calibration.md) for how these values relate to the rest of the project.
