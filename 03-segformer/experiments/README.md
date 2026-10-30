# Supporting experiments

| File | Location | Status |
|---|---|---|
| `DL_L1.ipynb` | Learning/training computer | Activation-function plots, not model training |
| `testing.ipynb` | Training computer | Python/PyTorch/CUDA environment check |
| `crack_tester.py` | Laptop | Generic Gradio template; requires adapted checkpoint/output handling and normalization for SegFormer |
| `drone_pipeline_PI.py` | Intended Pi experiment | File-watcher prototype with external inference/checkpoint assumptions; not final onboard inference |

No file here is required for the final live path. Gradio is optional and intentionally omitted from the main requirements. Earlier root script versions are preserved in `../legacy/previous-repository/` for comparison; use the current phase paths for instructions.
