# Earlier Pi dashboard

`app.py`, `app_feedonly.py` and `templates/index.html` run on Raspberry Pi. They combine camera/recording and MAVLink experiments. Inspect device indices, camera settings, calibration, marker size and recording paths before use. They own camera/serial resources and must be stopped before the final live pipeline starts.

This is not the SegFormer ground-station UI. Use [the live implementation](../../04-live-implementation/README.md) for laptop inference. The HTML template remains beside the Flask applications in its original `templates/` directory.
