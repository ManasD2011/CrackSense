"""
CNN inference wrapper: loads the trained model and classifies crack intensity.
"""

import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import (
    CNN_MODEL_PATH, INTENSITY_HIGH, INTENSITY_MEDIUM
)
from src.utils.image_utils import preprocess_for_cnn

# Lazy import TensorFlow to speed up non-ML usage
_cnn_model = None


def _load_model():
    """
    Load the CNN model by rebuilding the architecture from code and
    loading only the saved weights. This avoids Keras version
    incompatibility issues that occur with tf.keras.models.load_model().
    """
    global _cnn_model
    if _cnn_model is None:
        if not os.path.exists(CNN_MODEL_PATH):
            raise FileNotFoundError(
                f"CNN model not found at {CNN_MODEL_PATH}. "
                "Please run the training notebook first."
            )
        from src.models.cnn_model import build_cnn_classifier
        _cnn_model = build_cnn_classifier()
        _cnn_model.load_weights(CNN_MODEL_PATH)
        print(f"[CrackSense] CNN weights loaded from {CNN_MODEL_PATH}")
    return _cnn_model


def classify(image_bgr: np.ndarray) -> dict:
    """
    Run the trained CNN classifier on a BGR image.

    Returns a dict:
    {
        "label": "crack" | "no_crack",
        "confidence": float (0–1),
        "intensity": "HIGH" | "MEDIUM" | "LOW" | "NONE"
    }
    """
    model = _load_model()
    tensor = preprocess_for_cnn(image_bgr)          # (1, 224, 224, 3)
    prob = float(model.predict(tensor, verbose=0)[0][0])

    if prob >= 0.5:
        label = "crack"
        if prob >= INTENSITY_HIGH:
            intensity = "HIGH"
        elif prob >= INTENSITY_MEDIUM:
            intensity = "MEDIUM"
        else:
            intensity = "LOW"
    else:
        label = "no_crack"
        intensity = "NONE"

    return {
        "label": label,
        "confidence": round(prob, 4),
        "intensity": intensity,
    }
