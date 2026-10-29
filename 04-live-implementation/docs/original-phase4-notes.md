> Historical instructions: original filenames and paths below predate repository organization. Use the current phase README and startup guide for live execution.

# CrackSense Phase 4 — ordered implementation

This separate package preserves your Phase 3 script. It implements:

1. Source video timestamp extraction and Pi/Pixhawk clock-mapping diagnostics.
2. Bounded per-sensor histories and causal frame/sample matching.
3. One OpenCV window with the same raw/processed frame and processed-side metadata.
4. Reconnect/stale-data handling, per-result logs, snapshots and optional paired recording.

**Run the timestamp probe first. Do not interpret a working display as proof of physical synchronization.** The code always labels matching **APPROXIMATE message-time association**. No GUI setting can silently promote it to verified capture/measurement alignment.

## What changed

- MediaMTX still captures the Pi camera directly. No separate FFmpeg publisher is introduced.
- `phase4.yml` preserves 1280×720, 25 FPS, software H264, 4 Mbps, IDR period 25 and baseline profile. It adds `useAbsoluteTimestamp: true` and correctly spells `rtspTransports: [tcp]`.
- `rtsp_video.py` is a deliberately narrow RTSP-over-TCP receiver: it retains RTP timestamps, reads RTCP sender reports, assembles baseline H264, and uses PyAV/FFmpeg to decode. It supports H264 single-NAL, STAP-A and FU-A packetization, a single video track, and an unauthenticated `rtsp://` URL. It is not a general replacement for VLC.
- The Pi sender is the sole serial reader. It requests DISTANCE_SENSOR at 10 Hz, exchanges TIMESYNC, and forwards v2 UDP packets. It never arms or commands motion.
- The laptop no longer uses OpenCV VideoCapture for reception. OpenCV handles layout/display; PyAV handles decoding. This receiver change is necessary to retain the timing metadata used by this implementation.
- The Phase 3 model pipeline is retained in `model.py`: trusted packaged model, RGB/ImageNet normalization, 256×256, two-class sigmoid(crack-background), validated threshold, nearest-neighbor final mask. FP16 is default. FP32 is supported through `--precision fp32`. INT8 is intentionally rejected until a compatible inference path/checkpoint is supplied; it is not merely a label.
- Video recording is off by default to reduce disk work. Metrics are always logged. Optional recording/snapshots use a bounded output worker, not the inference or capture worker.

## Files and dependencies

Keep the entire folder together on both machines. In particular, do not copy only the sender: `timing.py` and `telemetry_common.py` must be beside it.

| File | Purpose |
|---|---|
| `phase4.yml` | Separate MediaMTX configuration |
| `pi_telemetry_sender.py` | Pixhawk serial reader, TIMESYNC, UDP v2 sender |
| `telemetry_common.py` | Shared range validation and packet utilities |
| `timing.py` | Clock estimator, v2 validation, histories, frame matching |
| `rtsp_video.py` | Timestamp-preserving H264 receiver |
| `model.py` | Phase 3 model pipeline |
| `laptop_dashboard.py` | Probe, paired preview, inference UI and output |
| `tests/` | Unit tests, deterministic sender test, local video/UDP integration |

Pi: use the existing `(drone)` environment with pymavlink. No new Pi dependency is required.

Laptop: use the existing Phase 1/3 environment with torch, transformers, numpy and GUI-capable OpenCV. Add only PyAV:

```powershell
python -m pip install av
```

Do not install `opencv-python-headless` on the laptop; that cannot open the dashboard. Do not replace your working PyTorch installation. Local verification used PyAV and headless OpenCV only for offscreen tests; it did not validate your Windows GUI/GPU environment.

## Step 1 — place the files and start the Pi services

Extract the `CrackSense_Phase4` folder:

- Pi: `/home/pi5/VIO/CrackSense_Phase4/`
- Laptop: `D:\Web development\Crack_Detection_Project\B2_Exp\CrackSense_Phase4\`

The commands below assume Pi `10.207.215.185` and laptop Wi-Fi `10.207.215.7`, from your earlier setup. Substitute actual addresses if DHCP changed them.

Stop the old MediaMTX instance with Ctrl+C in its terminal. Keep `phase2.yml` unchanged for rollback. Start the separate configuration from the directory containing your MediaMTX binary:

```bash
cd /phase2
./mediamtx /home/pi5/VIO/CrackSense_Phase4/phase4.yml
```

If your binary is elsewhere, use that directory. If MediaMTX reports an unknown configuration field, retain the error and `./mediamtx --version` output; do not remove timestamp preservation just to make the probe appear successful.

In a second Pi terminal, activate your existing `(drone)` environment. Stop the previous telemetry sender and any heartbeat/lidar test that opens `/dev/ttyACM0`. Run:

```bash
cd /home/pi5/VIO
python CrackSense_Phase4/pi_telemetry_sender.py --host 10.207.215.7 --udp-port 5005 --device /dev/ttyACM0 --baud 115200 --rate 10
```

Initial output can say `clock=UNMAPPED`. After at least three accepted TIMESYNC replies, it should report `MAPPED_MESSAGE_TIME` with estimated clock uncertainty. Support and timestamp semantics depend on the installed firmware. If mapping never establishes, the laptop keeps distances unavailable instead of using arrival time as capture time.

The sender retries serial connection failures. A Pixhawk boot-time regression/wrap or a Pi clock step starts a new telemetry session and clears mapping. One serial owner only: use a MAVLink router/dispatcher if other applications also need this link.

## Step 2 — run the laptop timestamp probe (FIRST laptop command)

Close the old `laptop_telemetry_receiver.py` so port 5005 is free. Close Phase 3, VLC and FFplay for an isolated test. Keep both Pi terminals running.

Use PowerShell:

```powershell
cd 'D:\Web development\Crack_Detection_Project'
.\.venv\Scripts\Activate.ps1
cd B2_Exp
python .\CrackSense_Phase4\laptop_dashboard.py --stream rtsp://10.207.215.185:8554/cam0 --source-ip 10.207.215.185 --probe --duration 30
```

This opens **no window and loads no model**. It still uses the real video decoder and real telemetry receiver. It creates a timestamped `phase4_results` session with `config.json`, `frames.jsonl` and `summary.json`.

Expected progression:

- `received` increases: H264 decoding works.
- `telemetry` increases: v2 packets from the Pi are accepted.
- `mapped` increases after two consistent RTCP sender reports: a source-time mapping exists.
- `both_matched` increases: both configured sensors have eligible samples for those frame times.

Default sensor mapping, based on your observed MAVLink output:

| Direction | Sensor ID | Orientation |
|---|---:|---:|
| Forward/wall | 0 | 0 |
| Down/ground | 1 | 25 |

Change `--ground-id` / `--wall-id` only if the actual MAVLink IDs differ. Orientation alone is not enough when multiple systems/sensors match; ambiguity produces unavailable data.

A probe with fewer than 10 both-sensor matches exits with code 2 and `gate: NOT_READY`. Otherwise the summary says `MAPPING_OBSERVED_REQUIRES_HARDWARE_VALIDATION`. This is evidence that the mapping path works, **not a synchronization accuracy certificate**. Review mapped/matched counts and dropouts, not only the gate string.

**If this step fails, stop here and share `summary.json` plus both terminal outputs.** Do not change flight-controller parameters merely to make the UI show numbers.

## Step 3 — paired preview without inference

Once the probe is receiving and matching both sensors:

```powershell
python .\CrackSense_Phase4\laptop_dashboard.py --stream rtsp://10.207.215.185:8554/cam0 --source-ip 10.207.215.185 --preview-only --duration 60
```

A single window displays paired frames. Since no model is loaded, both images are raw and processing latency is `--`; metadata says `preview / NONE`. Move a target in front of each lidar independently and confirm that only the expected range changes. An invalid range should become unavailable, not 0 m.

## Step 4 — FP16 dashboard

From `B2_Exp`, run:

```powershell
python .\CrackSense_Phase4\laptop_dashboard.py --stream rtsp://10.207.215.185:8554/cam0 --source-ip 10.207.215.185 --checkpoint Quantised/latest_fp16.pt --duration 900
```

If the checkpoint does not contain a valid `best_threshold`, add `--threshold` followed by your validated Phase 1 value. Do not invent a threshold. Model deserialization uses `weights_only=False`; load your own trusted checkpoint.

- Left: clean raw frame, with its ID outside the image.
- Right: the **same frame** with predicted crack mask.
- Below processed view: checkpoint name, actual selected runtime precision, frame ID above, processing latency, received/displayed FPS, ground/wall ranges, matched sample ages, timing status.
- `Q` or closing the window: stop.
- `S`: save full-resolution raw frame, binary mask (if inference enabled), displayed dashboard and metadata JSON through the output worker.
- Add `--record` to save `paired.avi`. AVI uses fixed playback FPS (`--record-fps`, default 25); `recorded_frame_index` in JSONL identifies each recorded result. Actual sampling intervals remain in the metadata.

FP32 comparison uses `--precision fp32`. Converting FP16 weights to FP32 changes execution dtype; it does not restore precision already lost in the weights. Use your original FP32 package to compare original FP32 weights. This version does not implement INT8 inference.

## Step 5 — bench validation in this order

1. **Sensor identity:** move the forward target while the downward target stays fixed, then reverse. Check ID, orientation and displayed direction.
2. **Telemetry interruption:** Ctrl+C the Pi sender. Within about one second the UI must stop showing ranges (`TELEMETRY_STALE`). Restart it; a new sender session clears old history.
3. **Video interruption:** stop MediaMTX; the window must show waiting/disconnected rather than retain a live-looking result. Restart it; the video generation and timestamp mapping restart.
4. **Physical timing:** record a visible moving target/event in the camera view while changing the corresponding lidar distance. Compare `source_unix_ns`, matched `sample_unix_ns`, residual/uncertainty fields and observed event timing. This is a sanity check, not a precise acquisition-time calibration. Exact validation needs an independently timed reference/event or verified sensor/driver acquisition timestamps.
5. **Full visual latency:** use the same visible stopwatch method used for Phase 3. Compare baseline and new receiver on the same network/camera settings, with recording off first. Neither processing latency nor probe timestamps measure physical screen refresh.
6. **15-minute run:** check reconnects, invalid/stale intervals, output drops, thermals, mask detail and latency. Then repeat with optional recording if you need it.

Hardware-dependent tests above remain pending. Do not claim a latency improvement until measured against Phase 3.

## Timing definitions and limitations

- Video time: RTP timestamp mapped to the source absolute timeline via RTCP sender reports. Two consistent reports are required; mappings expire after 15 seconds and reset on a >20 ms discontinuity. Source timestamps still need verification against camera exposure timing.
- Sensor time: `DISTANCE_SENSOR.time_boot_ms` mapped by a rolling TIMESYNC estimate to Pi monotonic time, then to Pi wall time. This can be the firmware's message/update time, not the lidar's physical acquisition time. Driver/sensor delay is not included in the clock uncertainty.
- Clock estimator: requires at least three replies within 10 seconds; chooses the lowest RTT estimate, includes observed offset spread, RTT/2, a drift allowance and millisecond quantization. Rejects estimates over 25 ms. These are diagnostic estimates, **not guaranteed error bounds**.
- Matching: independently choose the latest sample at or before the estimated frame time. A newer invalid eligible sample suppresses the range rather than exposing an older valid value. Unknown clocks never fall back to latest-arrival matching. A maximum age of 250 ms including estimated clock uncertainty is provisional and configurable (`--max-sample-age-ms`).
- No intentional waiting for future telemetry. Delayed telemetry may therefore yield no match for an already selected frame. The association is frozen with the frame before inference and is not retroactively changed.
- Histories: at most 300 packets per sensor and 32 sensor identities. Sender sessions, sequence numbers and source-IP filtering reject duplicates, reordering and retired sessions. This is validation/filtering, not cryptographic authentication; use your controlled network.
- A fresh telemetry packet indicating video source time more than one second behind produces `VIDEO_SOURCE_STALE`; a reversed discrepancy produces `CLOCK_DOMAIN_MISMATCH`. This is a relative clock/lag diagnostic, not an end-to-end latency measurement.
- Latest waiting decoded frame replaces older work; it does not eliminate upstream TCP/encoder/network buffering. The receiver rejects malformed/lost H264 access units and reconnects. Compare real-world performance before relying on it.
- Ground range is along the downward beam, not corrected vertical altitude. Wall range is along the forward beam, not necessarily perpendicular wall distance. Neither measurement is per-pixel depth or crack size.
- Processing latency includes preprocessing, model execution, CPU mask transfer and mask resize. It excludes video capture, transport, decode, annotation and display. Preview has no inference metric.
- `decode_to_display_preparation_ms` ends before rendering and display submission. No field claims true camera-to-screen latency. Pi and laptop clocks are not directly subtracted for that purpose.

## Outputs and troubleshooting

`frames.jsonl` stores one accepted display/probe result per line: frame generation/ID, source/RTP time, source SSRC, decode/inference times, sample association and age for each lidar, estimated clock uncertainty, FPS, skipped waiting frames and precision. `summary.json` includes counts, errors, dropped output items and processing mean/p95 for the most recent 10,000 results.

The output queue has 16 slots. If disk cannot keep up, it drops new log/record items and increments `output_dropped_items`; it does not silently block inference. Snapshot queuing failures are printed. Output write failures stop the run. Frame images are not copied into telemetry packets.

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: timing` or `telemetry_common` | Keep all package Python files in one folder |
| `No module named av` | Install `av` with the active laptop environment's `python -m pip` |
| Cannot bind port | Stop the old laptop telemetry receiver/dashboard |
| Unknown schema | Stop old Pi sender; run this package's v2 sender |
| `SENSOR_CLOCK_UNAVAILABLE` | Inspect Pi TIMESYNC output; verify firmware/link support |
| `VIDEO_TIME_UNAVAILABLE` | Allow two RTCP reports; check `phase4.yml` and probe output |
| `NO_PRIOR_SAMPLE` | Check clocks, sensor IDs, network delay and startup history |
| `VIDEO_SOURCE_STALE` | Investigate buffering and source/telemetry clock agreement |
| `CLOCK_DOMAIN_MISMATCH` | Check Pi clock changes and source timestamp configuration |
| Telemetry count remains zero | Check Wi-Fi addresses and existing UDP 5005 firewall rule |
| CUDA/model error | Use the working Phase 3 environment and trusted checkpoint |
| RTSP 401/auth or non-H264 | Unsupported by this receiver; retain logs rather than silently changing transport |

Rollback: stop the Phase 4 processes, restart `./mediamtx phase2.yml`, run the old Pi telemetry sender and unchanged Phase 3 laptop application. Their independent console receiver can be restarted after Phase 4 releases UDP 5005.

## Local verification and sources

Run tests from the extracted folder:

```powershell
python -m unittest discover -s tests -v
```

Unit tests cover clock estimation/expiry/reset, wraparound, malformed packets, causal sample choice, missing/invalid/stale data, session replacement and ambiguity. The sender test exercises the real loop with a deterministic MAVLink transport double. The integration test uses real local TCP/UDP sockets, generated H264 with fragmentation, real PyAV decoding, paired AVI recording and forced reconnect. It requires `av`, `numpy` and OpenCV; no CUDA is used. These are software tests, not Pi/laptop hardware measurements.

Primary references used for implementation:

- https://mediamtx.org/docs/features/absolute-timestamps
- https://mediamtx.org/docs/references/configuration-file
- https://mavlink.io/en/services/timesync.html
- https://mavlink.io/en/messages/common.html#DISTANCE_SENSOR
- https://www.rfc-editor.org/rfc/rfc6184
- https://pyav.org/docs/stable/api/codec.html

Current documentation can differ from installed MediaMTX/firmware versions; the probe is deliberately the first on-device gate.
