# Shared timing and telemetry utilities

Both the Raspberry Pi sender and laptop receiver import these modules. No separate process needs to be started from this folder.

- `timing.py` provides clock estimation, sensor history and sample selection.
- `telemetry_common.py` provides telemetry packet and range utilities.

Keep this folder beside the Pi and ground-station folders when copying the application to either device. See [timing and sensor association](../docs/timing-and-sensor-association.md) for the underlying behaviour.
