# direct-uart

Historical Pi UART and loopback tests. Lidartest.py reads native TFmini packets directly on Pi UART. test_connection.py uses /dev/serial0. These are not the final architecture, which routes both lidars through Pixhawk and USB MAVLink. Inspect wiring and device selection before running.
