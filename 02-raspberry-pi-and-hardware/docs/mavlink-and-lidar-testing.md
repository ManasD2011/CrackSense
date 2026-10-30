# MAVLink and lidar tests

Run on the Pi after camera checks. Connect Pixhawk micro-USB to Pi USB-A. The project uses `/dev/ttyACM0` with a configured baud value of 115200; verify the actual device. For USB CDC, that baud setting is not necessarily a physical UART transfer rate.

From the repository root:

```bash
python 02-raspberry-pi-and-hardware/tests/telemetry/heartbeat_test.py
```

Expected: heartbeat received, target system ID and component ID. The original simple test waits indefinitely; use Ctrl+C if the connection is absent. Check power, USB data cable, device name and port ownership.

Then run attitude reception, stopping it before the next test:

```bash
python 02-raspberry-pi-and-hardware/tests/telemetry/Attitude_test.py
python 02-raspberry-pi-and-hardware/tests/telemetry/lidar2_mavlink_test.py --port /dev/ttyACM0 --baud 115200 --rate 10
```

The second command requests DISTANCE_SENSOR messages and reports sensor ID, orientation, range, validity, reception age and packet rate. Optional `--csv readings.csv` creates a new log and refuses to overwrite an existing file. Receiving at 10 Hz does not prove 10 distinct physical samples per second.

| Physical sensor | Confirmed Pixhawk port | Live default identity |
|---|---|---|
| Ground/downward | TELEM2 | ID 1, orientation 25 |
| Wall/forward | SERIAL4/5 | ID 0, orientation 0 |

Move a target in front of each lidar separately to verify direction/identity. Distances arrive in centimetres and are converted to metres. Invalid readings must not be treated as zero distance. If heartbeat works but ranges do not arrive, inspect the installed firmware's rangefinder setup, requested stream response and sensor wiring. No universal firmware parameter file is included.

`Lidartest.py` under `experiments/direct-uart/` parses a lidar directly on Pi UART. It is not the recommended route for the final build, where both lidars connect through Pixhawk. Several VIO tests such as `battery_test.py` and `distance_sensor_test.py` are placeholders, not completed diagnostics.

Stop every test and earlier dashboard before the [live sender](../../04-live-implementation/README.md) owns the serial connection. Continue with [live startup](../../04-live-implementation/docs/startup-sequence.md).
