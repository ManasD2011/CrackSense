# tests

Automated software tests run on a development laptop, not the aircraft. From repository root: `python -m unittest discover -s 04-live-implementation/tests -v`. The integration test needs NumPy, OpenCV and PyAV with libx264. Core/sender tests do not require hardware. Tests do not validate flight, actual sensor acquisition times or CUDA checkpoints.
