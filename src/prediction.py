"""Inference helpers for TensorDigits digit predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tensorflow import keras

from src.preprocessing import BlankDrawingError, InvalidDrawingError, preprocess_image
from src.training import MODEL_PATH, NUM_CLASSES, load_model

ImageLike = np.ndarray


@dataclass(frozen=True)
class PredictionResult:
    """Structured output from a single digit prediction."""

    digit: int
    confidence: float
    probabilities: np.ndarray  # shape (10,), float32

    def as_dict(self) -> dict:
        return {
            "digit": self.digit,
            "confidence": float(self.confidence),
            "probabilities": [float(p) for p in self.probabilities],
        }


def get_model(path=MODEL_PATH) -> keras.Model:
    """Load the trained classifier from disk."""
    return load_model(path)


def predict_digit(image: ImageLike, model: keras.Model) -> PredictionResult:
    """Preprocess a drawing and return the predicted digit with probabilities.

    Raises
    ------
    BlankDrawingError
        If the drawing has no usable content.
    InvalidDrawingError
        If the image cannot be interpreted.
    RuntimeError
        If model inference fails.
    """
    try:
        tensor = preprocess_image(image)
    except (BlankDrawingError, InvalidDrawingError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise InvalidDrawingError(f"Could not process the drawing: {exc}") from exc

    try:
        raw = model.predict(tensor, verbose=0)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Prediction failed: {exc}") from exc

    probabilities = np.asarray(raw[0], dtype=np.float32)
    if probabilities.shape != (NUM_CLASSES,):
        raise RuntimeError(
            f"Unexpected model output shape {probabilities.shape}; expected ({NUM_CLASSES},)."
        )

    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    return PredictionResult(digit=digit, confidence=confidence, probabilities=probabilities)
