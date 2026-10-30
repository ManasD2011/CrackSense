# Getting started with CrackSense

CrackSense has two connected workflows: developing a crack-segmentation model and operating a drone-to-laptop inspection system. You can study the model without assembling a drone, or test video and telemetry reception before adding inference.

## Choose a starting point

| Your goal | Start here | What you need |
|---|---|---|
| Understand the project | [System architecture](system-data-flow.md), then [development history](project-history.md) | No hardware or model required |
| Study the initial approach | [CNN and U-Net](../01-cnn-unet/README.md) | Separate TensorFlow environment and suitable data |
| Train or evaluate SegFormer | [Model development](../03-segformer/README.md) | Datasets, a training environment and appropriate compute |
| Assemble and test the onboard system | [Hardware integration](../02-raspberry-pi-and-hardware/README.md) | Pi, camera, Pixhawk and sensors |
| Run the live demonstration | [Live startup](../04-live-implementation/docs/startup-sequence.md) | Working hardware, shared network, CUDA laptop and compatible checkpoint |

## Recommended sequence for a complete build

1. **Understand the connections.** Review the [wiring guide](hardware-and-wiring.md). The ground-facing lidar connects to TELEM2 and the wall-facing lidar to SERIAL4/5.
2. **Prepare the Pi.** Configure its environment and establish [SSH access](../02-raspberry-pi-and-hardware/docs/setup-and-ssh.md) from the laptop.
3. **Test the camera.** Confirm detection and image capture, then review [camera calibration](../02-raspberry-pi-and-hardware/docs/camera-testing-and-calibration.md).
4. **Test communication and sensors.** Check the Pixhawk heartbeat, attitude messages and both lidar readings using the [MAVLink guide](../02-raspberry-pi-and-hardware/docs/mavlink-and-lidar-testing.md).
5. **Prepare a model.** Follow [training](../03-segformer/training/README.md) and [evaluation](../03-segformer/evaluation/README.md), or supply a compatible trained model through the [checkpoint guide](../03-segformer/checkpoints/README.md).
6. **Verify reception before inference.** Start Pi video and telemetry, run the laptop reception probe, and inspect the preview as described in [live startup](../04-live-implementation/docs/startup-sequence.md).
7. **Run laptop segmentation.** Load the model, inspect the predicted overlay and range readings, then enable recording if required.

Keep the aircraft disarmed with propellers removed for bench tests. Complete camera and sensor checks before moving to a controlled flight demonstration.

## Terms used in the guides

| Term | Meaning in this project |
|---|---|
| Segmentation | Predicting whether each image pixel belongs to a crack or the background |
| Mask / overlay | A binary prediction image / that prediction drawn over the camera image |
| Checkpoint | Saved model weights and configuration required to reproduce inference |
| Ground station | The laptop that receives the drone data and runs SegFormer |
| MAVLink | The protocol used to receive flight-controller and sensor messages |
| RTSP | The protocol used to deliver the camera video stream |
| Telemetry | Sensor readings and related timing or status information |
| FP16 | Half-precision floating-point execution used for GPU inference |
| Validation / test data | Data used for model selection / separate data reserved for final evaluation |

## Find the right code

The four phase guides identify training, Raspberry Pi and ground-station responsibilities. Use the [code reference](file-index.md) when you need a specific script. Folders named `experiments`, `legacy` and `archive` preserve development work; skip them during the first setup. Training notebooks and model walkthroughs serve different purposes: walkthroughs load existing weights and explain predictions.

[Return to the project overview](../README.md)
