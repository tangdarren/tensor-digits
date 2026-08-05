"""Image preprocessing utilities for digit prediction."""

from __future__ import annotations

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] and ensure shape (N, 28, 28, 1)."""
    array = images.astype("float32") / 255.0
    if array.ndim == 3:
        array = np.expand_dims(array, axis=-1)
    return array


def preprocess_image(image):
    """Prepare a drawn image for model inference.

    Placeholder — canvas-to-tensor conversion comes in a later step.
    """
    raise NotImplementedError("Image preprocessing is not implemented yet.")
