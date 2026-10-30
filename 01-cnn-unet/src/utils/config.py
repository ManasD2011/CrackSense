"""
Configuration constants for the CrackSense project.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
CNN_MODEL_PATH    = os.path.join(SAVED_MODELS_DIR, "cnn_crack_classifier.h5")
UNET_MODEL_PATH   = os.path.join(SAVED_MODELS_DIR, "unet_crack_segmentor.h5")

# ─── Dataset (Kaggle Hub) ─────────────────────────────────────────────────────
KAGGLE_DATASET_HANDLE = "vangiap/crackvision12k"

# ─── Image settings ───────────────────────────────────────────────────────────
CNN_IMG_SIZE   = (192,192)   # Input size for CNN classifier
UNET_IMG_SIZE  = (192, 192)  # Input size for UNet segmentor (matches saved model)
CHANNELS       = 3

# ─── Classification thresholds (based on softmax probability of crack class) ──
INTENSITY_HIGH   = 0.85   # >= 0.85 → HIGH
INTENSITY_MEDIUM = 0.55   # 0.55–0.85 → MEDIUM
                           # < 0.55  → LOW

# ─── Training hyperparameters ─────────────────────────────────────────────────
CNN_BATCH_SIZE   = 32
CNN_EPOCHS       = 30
UNET_BATCH_SIZE  = 8
UNET_EPOCHS      = 40
LEARNING_RATE    = 1e-4
VALIDATION_SPLIT = 0.2

# ─── Segmentation overlay ─────────────────────────────────────────────────────
OVERLAY_ALPHA  = 0.45        # Overlay transparency
OVERLAY_COLOR  = (255, 0, 0) # Red highlight for cracks (BGR for OpenCV)

# ─── Flask server ─────────────────────────────────────────────────────────────
FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
DEBUG_MODE  = False
MAX_CONTENT_MB = 16
