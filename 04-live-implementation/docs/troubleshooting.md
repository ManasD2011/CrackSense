# Live troubleshooting

| Symptom | Check |
|---|---|
| Missing `timing` or `telemetry_common` | Preserve sibling `shared/`, `raspberry-pi/` and `ground-station/` layout |
| Camera busy | Stop Flask/Picamera2 tests and older MediaMTX instances |
| No heartbeat | USB data cable, power, device path, permissions, sole serial ownership |
| No telemetry | Explicit laptop address, UDP 5005, firewall, matching sender/receiver version |
| Bind failure | Stop the old laptop telemetry receiver or another dashboard |
| `VIDEO_TIME_UNAVAILABLE` | Two RTCP reports, MediaMTX absolute timestamps, mapping reset/expiry |
| `SENSOR_CLOCK_UNAVAILABLE` | Pixhawk TIMESYNC response and clock-estimator diagnostics |
| Wrong/missing direction | Actual sensor ID/orientation; use `--ground-id` and `--wall-id` only after independent target checks |
| `NO_PRIOR_SAMPLE` | Clock domains, startup history, sample age and delayed telemetry |
| CUDA unavailable | Correct GPU PyTorch installation; preview/probe can be used without a model |
| Checkpoint mismatch | Matching architecture, full package schema, producer environment and validated threshold |
| Authenticated/non-H264 RTSP | Current narrow receiver supports an unauthenticated H264 stream over TCP; it is not a general RTSP client |

Retain session `config.json`, `summary.json` and frame metadata when investigating. Do not change flight-controller parameters merely to make the dashboard display numbers. Historical prototypes are under `legacy/`; use their documented version/layout together rather than mixing their UDP schema with the current sender.
