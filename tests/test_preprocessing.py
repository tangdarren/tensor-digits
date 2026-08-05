"""Tests for handwritten digit preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.preprocessing import (
    BlankDrawingError,
    InvalidDrawingError,
    describe_tensor,
    invert_to_mnist_polarity,
    preprocess_image,
    save_preprocessed_preview,
    to_grayscale,
)


def _draw_black_digit_on_white(size: int = 200) -> Image.Image:
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    # Thick stroke resembling a handwritten "1".
    draw.line((size // 2, 30, size // 2, size - 30), fill=0, width=14)
    draw.line((size // 2 - 25, 50, size // 2, 30), fill=0, width=10)
    return image


def test_to_grayscale_composites_transparent_rgba():
    rgba = np.zeros((40, 40, 4), dtype=np.uint8)
    rgba[10:30, 10:30, 3] = 0  # transparent
    rgba[15:25, 18:22, :3] = 0
    rgba[15:25, 18:22, 3] = 255  # opaque black ink
    gray = to_grayscale(rgba)
    assert gray.shape == (40, 40)
    assert gray.dtype == np.uint8
    # Transparent areas become white; ink stays dark.
    assert gray[0, 0] == 255
    assert gray[20, 20] < 50


def test_invert_canvas_to_mnist_polarity():
    canvas = np.full((28, 28), 255, dtype=np.uint8)
    canvas[10:18, 12:16] = 0
    inverted = invert_to_mnist_polarity(canvas)
    assert inverted[0, 0] == 0
    assert inverted[12, 14] == 255


def test_preprocess_image_shape_range_and_polarity():
    drawing = _draw_black_digit_on_white()
    tensor = preprocess_image(drawing)

    assert tensor.shape == (1, 28, 28, 1)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    # Background should be dark (near 0); strokes bright.
    assert float(tensor.mean()) < 0.35
    assert float(tensor.max()) > 0.5
    # Digit should be roughly centered: middle region has more ink than corners.
    center = tensor[0, 8:20, 8:20, 0].mean()
    corner = tensor[0, 0:4, 0:4, 0].mean()
    assert center > corner


def test_preprocess_accepts_numpy_rgb_and_path(tmp_path):
    drawing = _draw_black_digit_on_white().convert("RGB")
    path = tmp_path / "digit.png"
    drawing.save(path)

    from_path = preprocess_image(path)
    from_array = preprocess_image(np.asarray(drawing))

    assert from_path.shape == (1, 28, 28, 1)
    assert from_array.shape == (1, 28, 28, 1)
    assert np.allclose(from_path, from_array, atol=1e-5)


def test_blank_drawing_raises():
    blank = Image.new("RGB", (200, 200), color=(255, 255, 255))
    with pytest.raises(BlankDrawingError):
        preprocess_image(blank)


def test_invalid_drawing_raises():
    with pytest.raises(InvalidDrawingError):
        preprocess_image(np.zeros((10,), dtype=np.uint8))


def test_save_preprocessed_preview(tmp_path):
    drawing = _draw_black_digit_on_white()
    out = tmp_path / "preview.png"
    path = save_preprocessed_preview(drawing, out, scale=8)
    assert path.exists()
    preview = Image.open(path)
    assert preview.size == (28 * 8, 28 * 8)
    assert preview.mode == "L"


def test_describe_tensor():
    drawing = _draw_black_digit_on_white()
    info = describe_tensor(preprocess_image(drawing))
    assert info["shape"] == (1, 28, 28, 1)
    assert info["dtype"] == "float32"
    assert 0.0 <= info["min"] <= info["max"] <= 1.0
