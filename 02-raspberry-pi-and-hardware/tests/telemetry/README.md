# Flight-controller and lidar diagnostics

Run these manual checks on the Raspberry Pi in sequence: heartbeat reception, attitude messages, then both lidar readings. Each test must release the serial connection before the next test or the live sender starts.

The [MAVLink and lidar guide](../../docs/mavlink-and-lidar-testing.md) provides the commands, sensor identities and expected results. These scripts interact with hardware and are not an automated unit-test suite.
