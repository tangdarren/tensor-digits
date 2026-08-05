"""Tests for model insights helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.insights import (
    artifacts_available,
    build_insight_bundle,
    confusion_matrix,
    figure_confusion_matrix,
    figure_misclassified_examples,
    figure_training_curves,
    load_insight_sources,
    select_misclassified,
)
from src.training import EVALUATION_PATH, HISTORY_PATH, MODEL_PATH


def test_artifacts_available_with_saved_files():
    assert artifacts_available() is (
        HISTORY_PATH.exists() and EVALUATION_PATH.exists() and MODEL_PATH.exists()
    )


def test_load_insight_sources_contains_metrics():
    if not (HISTORY_PATH.exists() and EVALUATION_PATH.exists()):
        pytest.skip("Training artifacts not present.")
    history, evaluation = load_insight_sources()
    assert "metrics" in history
    assert "accuracy" in history["metrics"]
    assert "loss" in history["metrics"]
    assert "test_accuracy" in evaluation


def test_confusion_matrix_and_misclassified_selection():
    y_true = np.array([0, 1, 2, 2, 3])
    y_pred = np.array([0, 1, 1, 2, 9])
    images = np.arange(5 * 28 * 28, dtype=np.uint8).reshape(5, 28, 28)

    matrix = confusion_matrix(y_true, y_pred)
    assert matrix.shape == (10, 10)
    assert matrix[0, 0] == 1
    assert matrix[2, 1] == 1
    assert matrix[3, 9] == 1

    errs, truths, preds = select_misclassified(images, y_true, y_pred, count=2, seed=0)
    assert len(errs) == 2
    assert np.all(truths != preds)
    assert set(zip(truths.tolist(), preds.tolist(), strict=True)).issubset({(2, 1), (3, 9)})


def test_figure_builders_return_figures():
    history = {
        "metrics": {
            "accuracy": [0.9, 0.95],
            "val_accuracy": [0.91, 0.94],
            "loss": [0.3, 0.1],
            "val_loss": [0.25, 0.12],
        }
    }
    fig_curves = figure_training_curves(history)
    assert fig_curves is not None
    fig_curves.clf()

    matrix = np.eye(10, dtype=np.int64) * 5
    fig_cm = figure_confusion_matrix(matrix)
    assert fig_cm is not None
    fig_cm.clf()

    images = np.zeros((2, 28, 28), dtype=np.uint8)
    fig_err = figure_misclassified_examples(images, np.array([1, 2]), np.array([7, 3]))
    assert fig_err is not None
    fig_err.clf()


def test_build_insight_bundle_smoke():
    if not artifacts_available():
        pytest.skip("Model/history/evaluation artifacts not present.")
    bundle = build_insight_bundle(error_count=4)
    assert bundle.confusion_matrix.shape == (10, 10)
    assert int(bundle.confusion_matrix.sum()) == 10000
    assert 0.0 <= float(bundle.evaluation["test_accuracy"]) <= 1.0
    assert len(bundle.error_images) <= 4
    if len(bundle.error_images):
        assert bundle.error_images.shape[1:] == (28, 28)
        assert np.all(bundle.error_true != bundle.error_pred)
