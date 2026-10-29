# Local validation — 2026-10-19

- Python compilation: passed.
- 14 automated tests: passed.
- Deterministic test of the real Pi sender loop: TIMESYNC mapping and both sensor orientations passed; serial hardware was replaced by a test double.
- Local TCP RTSP / UDP telemetry integration with real H264 encode/decode: 63 decoded frames, 53 both-sensor matches, one deliberate reconnect recovered.
- Paired AVI recording: readable, 1280×620 output.
- UI layout: rendered and visually inspected with synthetic images/values.

Not executed here: Raspberry Pi camera capture; installed MediaMTX configuration startup; Pixhawk firmware TIMESYNC/measurement timestamp semantics; real Wi-Fi transport; Windows GUI; CUDA FP16/FP32 checkpoint loading/inference; physical timing or full visual latency; 15-minute endurance.

Run README steps in order. Probe success indicates observed clock mapping, not exact sensor-acquisition alignment.
