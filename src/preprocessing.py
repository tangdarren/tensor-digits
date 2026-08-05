"""Convert user drawings into MNIST-compatible model inputs.

Streamlit canvases are typically dark ink on a light (or transparent)
background. MNIST digits are light strokes on a dark background. This
module bridges that gap and produces a ``(1, 28, 28, 1)`` float32 tensor
with values in ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

ImageInput = Union[np.ndarray, Image.Image, str, Path]

TARGET_SIZE = 28
DIGIT_BOX = 20  # Fit the cropped digit inside this box before centering.
INK_THRESHOLD = 30  # Pixel intensity (0–255) treated as ink after inversion.
MIN_INK_PIXELS = 20
MIN_BBOX_SIDE = 2


class BlankDrawingError(ValueError):
    """Raised when a drawing has no usable digit content."""


class InvalidDrawingError(ValueError):
    """Raised when an input cannot be interpreted as an image."""


def _load_as_pil(image: ImageInput) -> Image.Image:
    """Accept paths, PIL images, or NumPy arrays and return a PIL image."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise InvalidDrawingError(f"Image file not found: {path}")
        return Image.open(path)

    if isinstance(image, Image.Image):
        return image.copy()

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]

        if array.dtype != np.uint8:
            # Float tensors in [0, 1] or [0, 255] → uint8 for PIL.
            max_val = float(np.nanmax(array)) if array.size else 0.0
            if max_val <= 1.0:
                array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            return Image.fromarray(array, mode="L")
        if array.ndim == 3 and array.shape[-1] in (3, 4):
            mode = "RGBA" if array.shape[-1] == 4 else "RGB"
            return Image.fromarray(array, mode=mode)
        raise InvalidDrawingError(
            f"Unsupported array shape for drawing input: {array.shape}"
        )

    raise InvalidDrawingError(f"Unsupported image type: {type(image)!r}")


def to_grayscale(image: ImageInput) -> np.ndarray:
    """Convert an image to an ``uint8`` grayscale array shaped ``(H, W)``.

    Transparent pixels (common on Streamlit canvases) are composited onto a
    white background so strokes remain dark on light.
    """
    pil_image = _load_as_pil(image)

    if pil_image.mode in ("RGBA", "LA") or (
        pil_image.mode == "P" and "transparency" in pil_image.info
    ):
        rgba = pil_image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, rgba)
        gray = composited.convert("L")
    else:
        gray = pil_image.convert("L")

    return np.asarray(gray, dtype=np.uint8)


def invert_to_mnist_polarity(gray: np.ndarray) -> np.ndarray:
    """Ensure light digit strokes on a dark background (MNIST style).

    Canvas drawings are usually dark ink on a light background. If the
    mean intensity is bright, the image is inverted.
    """
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Mostly light background → invert so ink becomes bright.
    if float(np.mean(gray)) > 127.0:
        return 255 - gray
    return gray.copy()


def find_ink_bbox(gray: np.ndarray, threshold: int = INK_THRESHOLD) -> tuple[int, int, int, int]:
    """Return ``(row_min, row_max, col_min, col_max)`` for ink pixels."""
    mask = gray > threshold
    if not np.any(mask):
        raise BlankDrawingError(
            "No digit detected. Draw a clearer number from 0 through 9."
        )

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return int(row_min), int(row_max), int(col_min), int(col_max)


def crop_to_content(gray: np.ndarray, threshold: int = INK_THRESHOLD) -> np.ndarray:
    """Remove empty borders around the drawn digit."""
    row_min, row_max, col_min, col_max = find_ink_bbox(gray, threshold=threshold)
    cropped = gray[row_min : row_max + 1, col_min : col_max + 1]

    height, width = cropped.shape
    if height < MIN_BBOX_SIDE or width < MIN_BBOX_SIDE:
        raise BlankDrawingError(
            "Drawing is too small to recognize. Try drawing a larger digit."
        )

    ink_pixels = int(np.count_nonzero(cropped > threshold))
    if ink_pixels < MIN_INK_PIXELS:
        raise BlankDrawingError(
            "Drawing looks blank or too faint. Try a darker, thicker stroke."
        )

    return cropped


def resize_preserving_aspect(
    digit: np.ndarray,
    max_side: int = DIGIT_BOX,
) -> np.ndarray:
    """Resize so the longer side equals ``max_side``, keeping proportions."""
    height, width = digit.shape
    scale = max_side / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    pil_digit = Image.fromarray(digit, mode="L")
    resized = pil_digit.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def center_on_canvas(
    digit: np.ndarray,
    canvas_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Paste the resized digit into the center of a square black canvas."""
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    height, width = digit.shape

    if height > canvas_size or width > canvas_size:
        digit = resize_preserving_aspect(digit, max_side=canvas_size)
        height, width = digit.shape

    row_offset = (canvas_size - height) // 2
    col_offset = (canvas_size - width) // 2
    canvas[row_offset : row_offset + height, col_offset : col_offset + width] = digit
    return canvas


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale pixel values to ``[0, 1]`` and ensure shape ``(N, 28, 28, 1)``."""
    array = images.astype("float32") / 255.0
    if array.ndim == 3:
        array = np.expand_dims(array, axis=-1)
    elif array.ndim == 2:
        array = array[np.newaxis, ..., np.newaxis]
    elif array.ndim == 4 and array.shape[-1] != 1:
        raise InvalidDrawingError(
            f"Expected a single-channel batch, got shape {array.shape}"
        )
    return array


def preprocess_image(
    image: ImageInput,
    *,
    return_uint8: bool = False,
) -> np.ndarray:
    """Convert a user drawing into a model-ready MNIST-style tensor.

    Parameters
    ----------
    image:
        Path, PIL image, or NumPy array (grayscale / RGB / RGBA).
    return_uint8:
        If ``True``, return the centered ``(28, 28)`` ``uint8`` image
        before normalization (useful for debugging).

    Returns
    -------
    np.ndarray
        ``(1, 28, 28, 1)`` float32 in ``[0, 1]``, unless ``return_uint8``.
    """
    gray = to_grayscale(image)
    if gray.size == 0 or min(gray.shape) < 1:
        raise InvalidDrawingError("Image is empty.")

    mnist_like = invert_to_mnist_polarity(gray)
    cropped = crop_to_content(mnist_like)
    resized = resize_preserving_aspect(cropped, max_side=DIGIT_BOX)
    centered = center_on_canvas(resized, canvas_size=TARGET_SIZE)

    if return_uint8:
        return centered

    return normalize_images(centered)


def save_preprocessed_preview(
    image: ImageInput,
    output_path: str | Path,
    *,
    scale: int = 10,
) -> Path:
    """Run preprocessing and write an upscaled preview PNG for inspection.

    The preview uses the MNIST polarity (light digit on dark background)
    so developers can visually confirm crop, scale, and centering.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    centered = preprocess_image(image, return_uint8=True)
    preview = Image.fromarray(centered, mode="L")
    if scale > 1:
        preview = preview.resize(
            (TARGET_SIZE * scale, TARGET_SIZE * scale),
            Image.Resampling.NEAREST,
        )
    preview.save(output)
    return output


def describe_tensor(tensor: np.ndarray) -> dict:
    """Return a small diagnostic summary of a preprocessed tensor."""
    array = np.asarray(tensor)
    return {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
        "ink_fraction": float(np.mean(array > 0.1)) if array.size else None,
    }
