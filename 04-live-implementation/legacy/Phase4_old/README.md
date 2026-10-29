# Historical Phase4_old

Earlier Pi telemetry sender and separate laptop console receiver with their matching telemetry_common.py. Do not mix with the current v2 pipeline.

These original notes contain old working directories, local addresses and commands. See [current startup](../../docs/startup-sequence.md).

<details>
<summary>Original project notes</summary>

# CrackSense Phase 4D — Pi to laptop lidar telemetry

Prepared 16 October 2026 for laptop Wi-Fi IPv4 **10.207.215.7**.

Both Benewake lidars connect to Pixhawk. Pi receives them on `/dev/ttyACM0` using the working 115200-baud connection. Observed sensor mapping: ID 0/orientation 0 = forward; ID 1/orientation 25 = downward. Labels follow the configured orientation; confirm them with separate target movements.

## Files and dependencies

- Pi: `pi_telemetry_sender.py` and `telemetry_common.py` in the same folder. Use the Python environment where your heartbeat/lidar test already works; it needs pymavlink.
- Laptop: `laptop_telemetry_receiver.py` and `telemetry_common.py` in the same folder. Python 3 and its standard library are sufficient.
- These programs do not change flight modes, arm, move the vehicle or write parameters. The Pi requests the DISTANCE_SENSOR telemetry interval.

## Start on Windows first

Extract the ZIP. Open a terminal in the extracted `CrackSense_Phase4_UDP` folder:

```powershell
python .\laptop_telemetry_receiver.py --port 5005
```

Optional detailed log, using a filename that does not already exist:

```powershell
python .\laptop_telemetry_receiver.py --port 5005 --log lidar_network_test.jsonl
```

## Then start on Raspberry Pi

Copy the sender and common module to the Pi. Stop `heartbeat_test.py` / `lidar_mavlink_test.py` first: only one program should directly read this serial port. If another MAVLink service owns it, use a routed endpoint instead of opening the port concurrently. Keep the existing camera publisher running.

From the folder containing the files:

```bash
python3 pi_telemetry_sender.py --host 10.207.215.7 --udp-port 5005 --device /dev/ttyACM0 --baud 115200 --rate 10
```

Ctrl+C stops either program. Restart the sender after unplugging/rebooting the Pixhawk. The receiver clears previous readings on a new sender session. If the Pi's IP changes, restart the receiver.

## Verify

1. Confirm forward and downward readings appear separately on the laptop; initially WAITING is normal.
2. Move a flat target only in front of the forward lidar, then only below the downward lidar. Compare against known distances above the minimum valid range.
3. Run for 60 seconds and inspect actual received Hz. Requested 10 Hz does not guarantee exactly 10 Hz, and packet rate is not proof of fresh physical samples at that rate.
4. Stop the sender. Within about one second plus one display refresh, both values should become `-- / STALE`.
5. Restart the sender and verify both recover. Any changed orientation or ID must be rechecked physically.

IN_RANGE means within the limits reported by Pixhawk, not independently calibrated accuracy. Zero, invalid sentinels and out-of-range values are transmitted as invalid measurements rather than valid zero metres.

## If laptop stays WAITING

- Check sender says Connected, its Sent count increases, and send_errors remains zero. UDP send success alone is not delivery confirmation.
- Both devices must be on a network that permits device-to-device traffic. Some hotspots/guest networks isolate clients.
- Confirm `ipconfig` still reports laptop Wi-Fi IPv4 10.207.215.7. The virtual Ethernet/WSL addresses are not the intended destination.
- Check Windows Firewall allows inbound UDP 5005 from the Pi. Do not disable the firewall. An optional narrow rule from Administrator PowerShell is shown below; use the actual Pi Wi-Fi address from `hostname -I`:

```powershell
$piAddress = "REPLACE_WITH_PI_WIFI_IPV4"
New-NetFirewallRule -DisplayName "CrackSense lidar UDP 5005" -Direction Inbound -Action Allow -Protocol UDP -LocalPort 5005 -RemoteAddress $piAddress -LocalAddress 10.207.215.7 -Profile Any
```

Use the real numeric IP, not the placeholder. This enables only the telemetry port from that address. Remove the rule when no longer needed:

```powershell
Remove-NetFirewallRule -DisplayName "CrackSense lidar UDP 5005"
```

If Windows reports the port is already in use, stop the previous receiver or choose another port on BOTH programs. The receiver prints errors and exits if binding fails.

## Timing and packet contract

Each compact UDP JSON packet contains schema version, sender session ID, per-sensor sequence, MAVLink source IDs, sensor ID, orientation, raw centimetres, valid metres or null, limits, quality, Pixhawk `time_boot_ms` and Pi reception monotonic timestamp. The receiver validates fields and units, ignores older/duplicate sequence numbers and optionally logs its own reception monotonic timestamp.

`age` in the laptop display is time since laptop receipt. Pi and laptop monotonic clocks are different; never subtract them for network latency. Pixhawk message timestamps are retained without claiming exact physical sampling time. This release does NOT provide frame synchronization or true camera-to-screen latency. The FFmpeg publishing command/source timestamp audit is needed next.

`gaps` counts skipped sequence positions when newer packets are accepted; late arrivals are counted as `old`. It is a diagnostic discontinuity count, not a calibrated network-loss measurement. Receiver memory and per-sensor rate windows are bounded. Logs grow while enabled and are flushed once per second; use this as a standalone test, not the final dashboard recording implementation.

UDP is unauthenticated local-network telemetry. The receiver locks to the first valid sender IP by default; `--source-ip ACTUAL_PI_IP` restricts it from startup. This is not cryptographic sender authentication. No commands are sent from the laptop to the aircraft.

## Verification performed during preparation

Python syntax checks, packet validation checks, and a localhost sender-to-receiver test with simulated MAVLink messages. Tested malformed packets, duplicate/older packets, invalid range, stale timeout and new session recovery. The actual Pi, Pixhawk, Wi-Fi and Windows firewall must still be tested on your equipment.

Reference: [Microsoft New-NetFirewallRule documentation](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule) for the optional port/address-scoped rule.

</details>
