"""Tests for digit prediction wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.preprocessing import BlankDrawingError
from src.prediction import PredictionResult, predict_digit
from src.training import MODEL_PATH, load_model

ROOT = Path(__file__).resolve().parents[1]


def _load_font(size: int = 140):
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        path = Path(name)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_white_on_black_digit(digit: int, size: int = 280) -> np.ndarray:
    """Simulate a Streamlit canvas drawing (white ink on black)."""
    image = Image.new("RGBA", (size, size), color=(0, 0, 0, 255))
    draw = ImageDraw.Draw(image)
    font = _load_font(size=int(size * 0.55))
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2 - bbox[0]
    y = (size - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(image)


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not present — run `python train.py` first.")
    return load_model(MODEL_PATH)


def test_predict_digit_returns_valid_result(model):
    drawing = make_white_on_black_digit(3)
    result = predict_digit(drawing, model)
    assert isinstance(result, PredictionResult)
    assert 0 <= result.digit <= 9
    assert 0.0 <= result.confidence <= 1.0
    assert result.probabilities.shape == (10,)
    assert pytest.approx(float(result.probabilities.sum()), abs=1e-4) == 1.0
    assert result.digit == int(np.argmax(result.probabilities))


def test_predict_several_digits(model):
    """Run several synthetic digits through the full preprocess → predict path."""
    correct = 0
    for digit in (0, 1, 2, 3, 4, 5, 7, 8):
        result = predict_digit(make_white_on_black_digit(digit), model)
        assert result.probabilities.shape == (10,)
        assert result.confidence == pytest.approx(float(result.probabilities[result.digit]))
        if result.digit == digit:
            correct += 1
    # Font glyphs are not MNIST handwriting; require a majority match.
    assert correct >= 5, f"Only {correct}/8 synthetic digits classified correctly"


def test_blank_drawing_raises(model):
    blank = np.zeros((280, 280, 4), dtype=np.uint8)
    blank[..., 3] = 255  # opaque black, no strokes
    with pytest.raises(BlankDrawingError):
        predict_digit(blank, model)


def test_prediction_module_importable_by_app():
    import py_compile

    py_compile.compile(str(ROOT / "app.py"), doraise=True)
    py_compile.compile(str(ROOT / "src" / "prediction.py"), doraise=True)
