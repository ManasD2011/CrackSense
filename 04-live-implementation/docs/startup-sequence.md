# Live startup sequence

Finish [Pi bring-up](../../02-raspberry-pi-and-hardware/README.md) first. Commands below use the root of `04-live-implementation/` as the working directory on both devices. Replace `PI_IP` and `LAPTOP_IP`; these tokens are placeholders, not literal addresses. Stop older camera owners, Pi serial readers and laptop UDP receivers.

## 1 Pi environment and video

Activate the Pi environment, install `raspberry-pi/requirements.txt`, and install a MediaMTX Linux ARM64 build using [the official instructions](https://mediamtx.org/docs/kickoff/install). Ensure it supports the saved configuration. Run the executable from wherever you installed it:

```bash
/path/to/mediamtx raspberry-pi/phase4.yml
```

Expected: camera 0 publishes `cam0` at 1280×720/25 FPS, software H.264, target 4 Mbps, RTSP/TCP 8554. `useAbsoluteTimestamp: true` preserves source timing for the receiver. Configuration support must be checked on the installed version.

## 2 Pi telemetry in a second SSH terminal

```bash
python raspberry-pi/pi_telemetry_sender.py --host LAPTOP_IP --udp-port 5005 --device /dev/ttyACM0 --baud 115200 --rate 10
```

Expected: heartbeat, clock-mapping diagnostics and both sensor streams. The sender now requires the laptop address explicitly to avoid transmitting to an old hard-coded address. Keep both SSH sessions/processes running.

## 3 Laptop dependencies and reception probe

Install `ground-station/requirements.txt` in the laptop environment. For subsequent inference, install a compatible CUDA-enabled PyTorch build using [the PyTorch selector](https://pytorch.org/get-started/locally/). Allow the chosen UDP port through the laptop's firewall for this network.

```bash
python ground-station/laptop_dashboard.py --stream rtsp://PI_IP:8554/cam0 --source-ip PI_IP --probe --duration 30
```

No model or window is loaded. Received, mapped and matched counts should increase. Fewer than ten both-range matches yields exit code 2 and `NOT_READY`. A successful mapping gate is `MAPPING_OBSERVED_REQUIRES_HARDWARE_VALIDATION`, not a synchronization certificate. Inspect `summary.json` in the generated session folder.

## 4 Paired preview without model

```bash
python ground-station/laptop_dashboard.py --stream rtsp://PI_IP:8554/cam0 --source-ip PI_IP --preview-only --duration 60
```

Both panels show the same raw frame. Independently move ground/wall targets and check the correct range changes. Stop the sender briefly to check stale data handling, then restart. This is a bench check before inference.

## 5 SegFormer on the laptop GPU

Obtain/package your trained model following [checkpoint requirements](../../03-segformer/checkpoints/README.md). For a package placed at `03-segformer/Quantised/latest_fp16.pt`:

```bash
python ground-station/laptop_dashboard.py --stream rtsp://PI_IP:8554/cam0 --source-ip PI_IP --checkpoint ../03-segformer/Quantised/latest_fp16.pt --duration 900
```

Expected: raw frame on the left, mask overlay for the same frame on the right, plus ranges, frame IDs and diagnostics. If checkpoint metadata lacks a threshold, supply `--threshold` with the value validated for that checkpoint; do not invent one. The live model accepts two-class floating-point packages, defaults to FP16, and rejects INT8.

## 6 Recording and shutdown

Add `--record` to the preceding command for raw/processed/paired videos. Press `S` for snapshots, `Q` to stop the UI, then Ctrl+C the Pi sender and MediaMTX. Recording uses fixed playback FPS; retain JSONL metadata for actual frame timing. Check summaries for dropped outputs and stale intervals.

Hardware checks still needed include physical timing, a visible-event camera-to-screen latency measurement, sustained thermal/network behaviour and quantitative masks from airborne images. [Troubleshooting](troubleshooting.md).
