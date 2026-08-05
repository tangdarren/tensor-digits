"""Tests for handwritten digit preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.preprocessing import (
    TARGET_SIZE,
    BlankDrawingError,
    InvalidDrawingError,
    center_on_canvas,
    describe_tensor,
    invert_to_mnist_polarity,
    normalize_images,
    preprocess_image,
    resize_preserving_aspect,
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


def _draw_white_digit_on_black(size: int = 280) -> np.ndarray:
    """Simulate the Streamlit canvas polarity."""
    image = Image.new("RGBA", (size, size), color=(0, 0, 0, 255))
    draw = ImageDraw.Draw(image)
    draw.line((size // 2, 40, size // 2, size - 40), fill=(255, 255, 255, 255), width=16)
    draw.line((size // 2 - 30, 60, size // 2, 40), fill=(255, 255, 255, 255), width=12)
    return np.asarray(image)


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


def test_invert_skips_already_mnist_like_images():
    mnist_like = np.zeros((28, 28), dtype=np.uint8)
    mnist_like[8:20, 10:18] = 255
    result = invert_to_mnist_polarity(mnist_like)
    assert np.array_equal(result, mnist_like)


def test_preprocess_image_shape_range_and_polarity():
    drawing = _draw_black_digit_on_white()
    tensor = preprocess_image(drawing)

    assert tensor.shape == (1, TARGET_SIZE, TARGET_SIZE, 1)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    # Background should be dark (near 0); strokes bright.
    assert float(tensor.mean()) < 0.35
    assert float(tensor.max()) > 0.5
    # Digit should be roughly centered: middle region has more ink than corners.
    center = tensor[0, 8:20, 8:20, 0].mean()
    corner = tensor[0, 0:4, 0:4, 0].mean()
    assert center > corner


def test_preprocess_white_on_black_canvas_dimensions_and_normalization():
    tensor = preprocess_image(_draw_white_digit_on_black())
    assert tensor.shape == (1, 28, 28, 1)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert float(tensor.max()) > 0.5
    assert float(tensor.mean()) < 0.35


def test_preprocess_centers_offset_digit():
    canvas = Image.new("L", (280, 280), color=255)
    draw = ImageDraw.Draw(canvas)
    # Digit drawn in the upper-left corner rather than the center.
    draw.ellipse((20, 20, 90, 110), outline=0, width=10)
    tensor = preprocess_image(canvas)
    center = float(tensor[0, 8:20, 8:20, 0].mean())
    corners = [
        float(tensor[0, 0:4, 0:4, 0].mean()),
        float(tensor[0, 0:4, 24:28, 0].mean()),
        float(tensor[0, 24:28, 0:4, 0].mean()),
        float(tensor[0, 24:28, 24:28, 0].mean()),
    ]
    assert center > max(corners)


def test_preprocess_accepts_numpy_rgb_and_path(tmp_path):
    drawing = _draw_black_digit_on_white().convert("RGB")
    path = tmp_path / "digit.png"
    drawing.save(path)

    from_path = preprocess_image(path)
    from_array = preprocess_image(np.asarray(drawing))

    assert from_path.shape == (1, 28, 28, 1)
    assert from_array.shape == (1, 28, 28, 1)
    assert np.allclose(from_path, from_array, atol=1e-5)


def test_preprocess_accepts_float_unit_interval_array():
    drawing = np.asarray(_draw_black_digit_on_white(), dtype=np.float32) / 255.0
    tensor = preprocess_image(drawing)
    assert tensor.shape == (1, 28, 28, 1)
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_blank_drawing_raises():
    blank = Image.new("RGB", (200, 200), color=(255, 255, 255))
    with pytest.raises(BlankDrawingError):
        preprocess_image(blank)


def test_blank_black_canvas_raises():
    blank = np.zeros((200, 200, 4), dtype=np.uint8)
    blank[..., 3] = 255
    with pytest.raises(BlankDrawingError):
        preprocess_image(blank)


def test_tiny_drawing_raises():
    image = Image.new("L", (100, 100), color=255)
    draw = ImageDraw.Draw(image)
    draw.point((50, 50), fill=0)
    with pytest.raises(BlankDrawingError):
        preprocess_image(image)


def test_invalid_drawing_raises():
    with pytest.raises(InvalidDrawingError):
        preprocess_image(np.zeros((10,), dtype=np.uint8))


def test_missing_path_raises(tmp_path):
    with pytest.raises(InvalidDrawingError):
        preprocess_image(tmp_path / "does_not_exist.png")


def test_return_uint8_centered_image():
    centered = preprocess_image(_draw_black_digit_on_white(), return_uint8=True)
    assert centered.shape == (28, 28)
    assert centered.dtype == np.uint8
    assert int(centered.max()) > 50


def test_resize_and_center_helpers_preserve_canvas_size():
    digit = np.zeros((40, 10), dtype=np.uint8)
    digit[:, :] = 255
    resized = resize_preserving_aspect(digit, max_side=20)
    assert max(resized.shape) == 20
    canvas = center_on_canvas(resized, canvas_size=28)
    assert canvas.shape == (28, 28)
    # Content should sit near the horizontal center.
    cols = np.where(canvas > 0)[1]
    assert abs(cols.mean() - 13.5) < 3.0


def test_normalize_images_batch_and_single():
    batch = normalize_images(np.full((3, 28, 28), 255, dtype=np.uint8))
    single = normalize_images(np.full((28, 28), 128, dtype=np.uint8))
    assert batch.shape == (3, 28, 28, 1)
    assert single.shape == (1, 28, 28, 1)
    assert batch.dtype == np.float32
    assert single[0, 0, 0, 0] == pytest.approx(128 / 255)


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
