"""
UNet inference wrapper: loads trained model, generates mask + overlay image.
"""

import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import UNET_MODEL_PATH
from src.utils.image_utils import (
    preprocess_for_unet, generate_overlay,
    ndarray_to_base64, compute_crack_coverage
)

_unet_model = None


def _load_model():
    """
    Load the UNet model by rebuilding the architecture from code and
    loading only the saved weights. This avoids Keras version
    incompatibility issues (e.g. 'batch_shape', 'optional' errors)
    that occur when using tf.keras.models.load_model() on .h5 files
    saved with a different Keras version.
    """
    global _unet_model
    if _unet_model is None:
        if not os.path.exists(UNET_MODEL_PATH):
            raise FileNotFoundError(
                f"UNet model not found at {UNET_MODEL_PATH}. "
                "Please run the training notebook first."
            )
        from src.models.unet_model import build_unet_segmentor
        _unet_model = build_unet_segmentor()
        _unet_model.load_weights(UNET_MODEL_PATH)
        print(f"[CrackSense] UNet weights loaded from {UNET_MODEL_PATH}")
    return _unet_model


def segment(image_bgr: np.ndarray) -> dict:
    """
    Run the trained UNet on a BGR image to produce a segmentation mask
    and a color-overlay image.

    Returns a dict:
    {
        "mask_b64": str,        # base64 JPEG of the binary mask
        "overlay_b64": str,     # base64 JPEG of original + crack overlay
        "crack_coverage": float # percentage of pixels classified as crack
    }
    """
    model = _load_model()
    tensor = preprocess_for_unet(image_bgr)          # (1, 256, 256, 3)
    raw_mask = model.predict(tensor, verbose=0)[0]    # (256, 256, 1)
    mask = raw_mask[:, :, 0]                          # (256, 256)

    # Generate overlay on original image (not resized)
    overlay = generate_overlay(image_bgr, mask)

    # Grayscale mask image for display
    mask_vis = (mask * 255).astype(np.uint8)
    import cv2
    mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

    return {
        "mask_b64": ndarray_to_base64(mask_bgr),
        "overlay_b64": ndarray_to_base64(overlay),
        "crack_coverage": compute_crack_coverage(mask),
    }
