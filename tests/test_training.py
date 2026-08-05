"""Tests for the TensorDigits training pipeline."""

import numpy as np
import pytest

from src.preprocessing import normalize_images
from src.training import (
    MODELS_DIR,
    MODEL_PATH,
    build_model,
    load_and_prepare_data,
    load_json,
    load_model,
)


def test_normalize_images_scales_and_adds_channel():
    full = np.zeros((2, 28, 28), dtype=np.uint8)
    full[0, 0, 0] = 255
    result = normalize_images(full)
    assert result.shape == (2, 28, 28, 1)
    assert result.dtype == np.float32
    assert result[0, 0, 0, 0] == pytest.approx(1.0)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_build_model_output_shape():
    model = build_model()
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)


def test_saved_model_artifacts_exist_and_load():
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not present yet — run `python train.py` first.")

    model = load_model(MODEL_PATH)
    history = load_json(MODELS_DIR / "training_history.json")
    evaluation = load_json(MODELS_DIR / "evaluation.json")

    assert model.name == "digit_classifier"
    assert "metrics" in history
    assert "accuracy" in history["metrics"]
    assert "test_accuracy" in evaluation
    assert 0.0 <= evaluation["test_accuracy"] <= 1.0


def test_saved_model_loads_and_predicts_expected_shape():
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not present yet — run `python train.py` first.")

    model = load_model(MODEL_PATH)
    sample = np.zeros((1, 28, 28, 1), dtype=np.float32)
    sample[0, 5:22, 10:18, 0] = 1.0
    output = model.predict(sample, verbose=0)

    assert model.input_shape == (None, 28, 28, 1)
    assert output.shape == (1, 10)
    assert output.dtype == np.float32 or str(output.dtype).startswith("float")
    assert pytest.approx(float(output.sum()), abs=1e-4) == 1.0
    assert 0 <= int(np.argmax(output[0])) <= 9


def test_data_shapes_smoke():
    # Avoid re-downloading in unit tests if Keras cache is cold; skip if unavailable.
    try:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_prepare_data()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MNIST unavailable: {exc}")

    assert x_train.shape[1:] == (28, 28, 1)
    assert x_val.shape[1:] == (28, 28, 1)
    assert x_test.shape == (10000, 28, 28, 1)
    assert len(y_train) == len(x_train)
    assert len(y_val) == len(x_val)
    assert len(y_test) == 10000
