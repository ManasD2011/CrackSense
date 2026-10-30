"""
Image preprocessing and overlay generation utilities.
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import (
    CNN_IMG_SIZE, UNET_IMG_SIZE, OVERLAY_ALPHA, OVERLAY_COLOR
)


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_for_cnn(image: np.ndarray) -> np.ndarray:
    """
    Resize + normalize image for CNN classifier.
    Returns array of shape (1, H, W, 3) with values in [0, 1].
    """
    img = cv2.resize(image, CNN_IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_for_unet(image: np.ndarray) -> np.ndarray:
    """
    Resize + normalize image for UNet segmentor.
    Returns array of shape (1, H, W, 3) with values in [0, 1].
    """
    img = cv2.resize(image, UNET_IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes into a BGR numpy array (OpenCV format).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from bytes.")
    return img


def load_image_from_path(path: str) -> np.ndarray:
    """Load image from file path → BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at path: {path}")
    return img


# ─── Overlay generation ───────────────────────────────────────────────────────

def generate_overlay(original_bgr: np.ndarray,
                     mask: np.ndarray,
                     alpha: float = OVERLAY_ALPHA,
                     color_bgr: tuple = OVERLAY_COLOR) -> np.ndarray:
    """
    Blend segmentation mask onto original image.

    Args:
        original_bgr: H×W×3 uint8 BGR image.
        mask: H×W float32 mask with values in [0, 1].
        alpha: transparency of crack overlay.
        color_bgr: BGR color for crack highlighting (default: red).

    Returns:
        H×W×3 uint8 BGR overlay image.
    """
    h, w = original_bgr.shape[:2]

    # Resize mask to original image dimensions
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Threshold mask
    binary_mask = (mask_resized > 0.5).astype(np.uint8)

    # Create colored overlay
    overlay = original_bgr.copy()
    crack_layer = np.zeros_like(original_bgr)
    crack_layer[:] = color_bgr  # red channel for cracks

    # Apply colored highlight only on crack pixels
    mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    blended = cv2.addWeighted(original_bgr, 1 - alpha, crack_layer, alpha, 0)
    overlay = np.where(mask_3ch, blended, original_bgr)

    # Draw contours for sharper crack outlines
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color_bgr, 1)

    return overlay.astype(np.uint8)


# ─── Encoding helpers ─────────────────────────────────────────────────────────

def ndarray_to_base64(img_bgr: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a BGR numpy image to a base64 string (for JSON transport)."""
    success, buffer = cv2.imencode(fmt, img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image to buffer.")
    return base64.b64encode(buffer).decode("utf-8")


def pil_image_to_base64(pil_img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image to base64 string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def bgr_to_rgb_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to RGB PIL Image."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ─── Crack coverage ───────────────────────────────────────────────────────────

def compute_crack_coverage(mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate the percentage of image pixels classified as crack.

    Args:
        mask: H×W float32 mask (values 0–1).
        threshold: binarization threshold.

    Returns:
        Crack coverage as a percentage (0–100).
    """
    binary = (mask > threshold).astype(np.float32)
    coverage = (binary.sum() / binary.size) * 100.0
    return round(float(coverage), 2)
