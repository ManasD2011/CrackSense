> Historical document: superseded by the [current repository guide](../../../README.md). Paths, status and claims below may be outdated.

# Hardware Setup and Troubleshooting

## Reported hardware baseline

The final project report identifies an F450 quadcopter, Pixhawk 2.4.8 flight controller, Raspberry Pi 5 companion computer, and IMX219 camera. Flight is manually controlled. The Pi captures/stores images, while model inference runs on a PC or laptop. The report also notes hardware/source conflicts for the camera and rangefinder; verify the actual bill of materials before publishing a definitive parts list.

## Bring-up sequence from the project report

1. Install and configure the Raspberry Pi OS, SSH access, and a project-specific Python environment.
2. Connect the camera and verify detection/capture.
3. Capture and store images locally on the Pi.
4. Establish a local network preview or stream to check framing.
5. Connect Pixhawk over USB and identify its serial device; the report records `/dev/ttyACM0`.
6. Verify rangefinder telemetry on Pixhawk using Mission Planner.
7. Calibrate the camera with a chessboard. The final intrinsics and reprojection error still need to be recorded.
8. Fly manually, capture images, transfer them to the PC, and process them offline or use the separately documented live laptop prototype.

Do not reuse historical IP addresses as fixed configuration. Replace them with the current network addresses and avoid publishing private network details.

## Troubleshooting records

| Symptom | Cause or evidence | Response recorded in the project |
|---|---|---|
| Camera is not visible through the expected interface | Camera device/configuration mismatch | List video devices and check the Pi camera configuration/tools |
| A new process cannot open `/dev/video0` | Another capture/streaming process still owns the device | Identify the owner with `fuser`, then stop the old process before retrying |
| PC cannot view the camera stream | Stream server or address configuration is wrong | Confirm the Pi stream service is running and use its current local network address |
| Pixhawk is not reachable from the Pi | Serial port not identified or link not established | Identify the correct serial device (reported as `/dev/ttyACM0`) and verify MAVLink heartbeat |
| Rangefinder appears in Mission Planner but not on the Pi | Pixhawk-side sensor works, but Pi-side telemetry reception was unresolved in the report | Keep rangefinder data out of image analysis until the Pi data path is validated |

For each new incident, record exact error output, date, hardware/software versions, steps to reproduce, root cause, fix, and verification evidence. Redact credentials, serial identifiers where sensitive, and local network addresses before publication.

## Safety boundary

The documented system uses manual flight. ArUco-guided hover and other automated modes are future scope. Any future flight-control implementation needs staged bench and flight testing, velocity limits, loss-of-marker behavior, failsafes, and immediate manual takeover.
