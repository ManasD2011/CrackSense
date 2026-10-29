# Phase 1 — CNN and U-Net

**Runs on:** training/offline inference computer. **Status:** historical baseline, not tested on live drone feed.

The initial build used F450, 1000 KV motors, 40 A ESCs, FS-i6 radio, Pixhawk 2.4.8 and LiPo power, tuned with Mission Planner. Stability and model accuracy motivated the later stages.

| Path | Role |
|---|---|
| `notebooks/01_cnn_classification.ipynb` | Train and evaluate the image-level CNN |
| `notebooks/02_unet_segmentation.ipynb` | Train U-Net using generated pseudo-masks in this historical notebook |
| `src/models/` | TensorFlow/Keras architecture definitions |
| `src/inference/` | Load saved models and predict |
| `src/utils/config.py` | Image sizes, model paths, training settings |
| `api/main.py` and `frontend/` | Flask API and browser demo on the computer |

## Initial aircraft

<img src="../docs/images/project/drone-v1.jpeg" alt="Top view of the initial F450 build before the Raspberry Pi camera integration" width="420">

*The original aircraft build associated with the CNN/U-Net development stage. Those models were not tested on a real drone feed.*

## Run sequence

From the repository root, create an environment dedicated to this TensorFlow phase:

```bash
cd 01-cnn-unet
python -m venv .venv
```

Activate `.venv` (`.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Linux), then:

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyter lab
```

Open the CNN notebook first and U-Net notebook second. Run them with the notebook directory as kernel working directory; their root-resolution cells use its parent. Inspect discovered class folders and class labels before training. Expected artifacts are model files in `saved_models/`, loss/metric plots and predictions. Training has not been rerun as part of this repository organization.

With both expected model files present, run from this phase root:

```bash
python api/main.py
```

Open the local server URL printed by Flask. The demo is an offline computer application, not Pi deployment.

## Historical supervision caveat

The U-Net notebook uses adaptive thresholding, CLAHE and morphological operations to generate targets from positive class images; negative images receive zero masks. This is pseudo-mask supervision. It must not be described as training against manually annotated masks. The notebook's assumptions about class-folder layout do not establish whether every release of the dataset lacks segmentation masks. SegFormer uses paired masks in a different workflow.

Classification confidence and the demo's intensity labels are not validated structural severity measurements. Requirements are the original broad constraints, not a reproduced environment lock.

Next: [Pi and hardware](../02-raspberry-pi-and-hardware/README.md) or [SegFormer](../03-segformer/README.md).
