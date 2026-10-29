> Historical document: superseded by the [current repository guide](../../../README.md). Paths, status and claims below may be outdated.

# Future Scope

The following items are described as future work or incomplete experiments. They are not demonstrated capabilities of the current project.

## Vision-model improvements

- Hash-check the merged dataset for duplicate CrackSeg9K pairs and rerun the test if needed.
- Annotate representative drone-captured images and evaluate them with ground-truth masks.
- Save and consistently use the validation-calibrated threshold.
- Use overlapping tiled inference to retain thin cracks in high-resolution images.
- Fix and evaluate the test-time augmentation scale transforms.
- Compare B0, B1, and B2 under controlled training and test conditions.
- Benchmark FP32, FP16, and INT8 packages on the intended hardware.
- Validate any crack-width/severity estimate with camera calibration, range measurements, and physical measurements.

## ArUco marker-guided hover

The first planned autonomous demonstration is a low-risk hover over an ArUco marker in Guided mode. Earlier bench work used marker ID 7 with a 200 mm marker, but the report explicitly says this did not demonstrate Guided hover. A valid position estimate, EKF origin, accepted ExternalNav data, correct camera-to-flight coordinate transforms, and tested failsafes must be established first. The pilot must retain immediate manual takeover.

## Optical flow and visual-inertial estimation

Optical-flow/VIO and EKF-based position estimation are later steps, to be considered only after camera calibration and ArUco-guided hover are reliable. Files under `RaspberryPi5_codebase/VIO/` are exploratory work, not evidence that autonomous navigation is complete.

## Onboard inference

The Black Book proposes running segmentation on the Raspberry Pi or streaming to a ground PC for live analysis. Raspberry Pi performance, model accuracy under deployment conditions, and full in-flight latency have not been established. The `drone_pipeline_PI.py` helper in this repository currently depends on files/checkpoints that are not included.
