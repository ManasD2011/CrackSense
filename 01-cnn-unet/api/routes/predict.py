"""
/api/predict  –  POST endpoint for crack detection.

Accepts:  multipart/form-data with key "file" (image)
Returns:  JSON with UNet segmentation + overlay
"""

import os, sys, traceback
import numpy as np
from flask import Blueprint, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.image_utils import (
    load_image_from_bytes, ndarray_to_base64, bgr_to_rgb_pil, pil_image_to_base64
)
from src.inference.segmentor import segment

predict_bp = Blueprint('predict', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}


def _allowed(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Form field: file (image)

    Response JSON:
    {
        "success": true,
        "label": "crack" | "no_crack",
        "intensity": "HIGH" | "MEDIUM" | "LOW" | "NONE",
        "confidence": 0.923,
        "crack_coverage": 14.3,
        "original_b64": "<base64 JPEG>",
        "overlay_b64":  "<base64 JPEG>"
    }
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not _allowed(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
        }), 415

    try:
        image_bytes = file.read()
        image_bgr   = load_image_from_bytes(image_bytes)

        # ── Segmentation (UNet only) ──────────────────────────────────────────
        seg_result = segment(image_bgr)

        # ── Derive intensity and label from crack coverage ──────────────────
        crack_coverage = seg_result["crack_coverage"]

        if crack_coverage < 0.1:
            label = "no_crack"
            intensity = "NONE"
            confidence = 1.0  # High confidence no crack
        elif crack_coverage < 5:
            label = "crack"
            intensity = "LOW"
            confidence = crack_coverage / 100
        elif crack_coverage < 15:
            label = "crack"
            intensity = "MEDIUM"
            confidence = crack_coverage / 100
        else:
            label = "crack"
            intensity = "HIGH"
            confidence = min(crack_coverage / 100, 1.0)

        # ── Encode original image for frontend display ──────────────────────
        original_b64 = ndarray_to_base64(image_bgr)

        return jsonify({
            "success":        True,
            "label":          label,
            "intensity":      intensity,
            "confidence":     float(confidence),
            "crack_coverage": float(crack_coverage),
            "original_b64":   original_b64,
            "overlay_b64":    seg_result["overlay_b64"],
        }), 200

    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "hint": "Train the models first by running the Jupyter notebooks."
        }), 503

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
