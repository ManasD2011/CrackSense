> Historical document: superseded by the [current repository guide](../../../README.md). Paths, status and claims below may be outdated.

# Project History and Development Phases

## Project goal

CrackSense explores a safer and more repeatable way to collect close-range images of structural surfaces and identify visible cracks at pixel level. The project deliberately separates image acquisition from image analysis so an unvalidated model cannot control the aircraft.

The final project report describes two workstreams developed in parallel: drone hardware and subsystem verification, and crack-dataset/model development. They meet in the inference pipeline.

## Phases

### 1. Define the inspection workflow

The team selected a manually piloted drone for image acquisition and PC-based analysis after capture. Manual piloting remained the stable baseline while the vision system was developed.

### 2. Build and verify the acquisition platform

The documented platform is an F450 quadcopter with a Pixhawk flight controller, Raspberry Pi 5 companion computer, and IMX219 camera. Work included camera bring-up, image capture and storage, local network preview/streaming, camera calibration, and MAVLink communication with Pixhawk. The rangefinder was verified on the Pixhawk side; reliable Pi-side reading and use in crack analysis remained open tasks.

### 3. Prepare the segmentation dataset

The project combined image/mask pairs from CrackSeg9K and CrackVision12K, discovered image-mask pairs across the source folder layouts, binarized masks, and created an 80:10:10 split with a fixed random seed for the B2 run. Training data received augmentation; validation and test data were not augmented.

The report flags a possible duplicate source: 11,298 CrackSeg9K pairs may appear both in a top-level folder and in train/test subfolders. A hash check is needed to determine whether the final test set was affected.

### 4. Explore and select the model

Earlier project work included CNN classification and U-Net segmentation. The report then compares SegFormer MiT-B0, MiT-B1, and MiT-B2 variants. B2 was selected based on the reported validation and held-out test results. This was not a controlled comparison: input sizes, training setup, and data splits differed across runs, and complete B1 metrics were unavailable in the report.

### 5. Build inference and review outputs

The model produces crack probability maps, which are thresholded into binary masks. Masks are resized to the original image dimensions and can be saved with red overlays. The repository includes video inference, explanation notebooks, model packaging, and verification/benchmark scripts.

### 6. Demonstrate acquisition and image analysis

The report's baseline workflow is manual flight, image capture/storage on the Pi, image transfer to a PC, and segmentation on the PC. A later live laptop-processing prototype received RTSP video and MAVLink telemetry and visually ran segmentation on camera images. The report does not establish synchronized sensor/frame alignment, recording validation, or end-to-end in-flight performance.

## Future scope

ArUco marker-guided hover is the first planned autonomous demonstration. Optical-flow/visual-inertial position estimation, closed-loop flight control, and onboard crack inference are later goals. They are not completed features of the current project.

## Project chronology and contributions

The source report describes the development stages and parallel workstreams but does not provide a complete date-by-date contributor log. Add exact dates, owners, and review milestones from Git history and team records rather than inferring them from filenames.
