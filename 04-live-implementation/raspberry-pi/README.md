# Raspberry Pi video and telemetry

The onboard Pi captures camera video and forwards flight-controller telemetry to the laptop. SegFormer inference runs on the laptop.

Start MediaMTX with `phase4.yml`, then run `pi_telemetry_sender.py` with the laptop address supplied through `--host`. Preserve the adjacent `shared/` folder because the sender imports its utilities.

Follow the [startup sequence](../docs/startup-sequence.md) for prerequisites, commands and expected results.
