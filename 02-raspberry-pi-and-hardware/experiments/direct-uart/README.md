# Historical direct UART tests

These Raspberry Pi experiments explore direct serial communication. `Lidartest.py` reads native TFmini packets from Pi UART, while `test_connection.py` checks `/dev/serial0`.

The current inspection system connects both lidar sensors to Pixhawk and receives their data over USB MAVLink. Follow the [current sensor-testing guide](../../docs/mavlink-and-lidar-testing.md) for that configuration. Inspect wiring and device selection before using these older tests.
