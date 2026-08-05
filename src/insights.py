"""Model insight helpers: history charts, confusion matrix, error examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from tensorflow import keras

from src.preprocessing import normalize_images
from src.training import (
    EVALUATION_PATH,
    HISTORY_PATH,
    MODEL_PATH,
    NUM_CLASSES,
    load_json,
    load_model,
)

# Keep misclassified gallery small so insights stay secondary to drawing.
DEFAULT_ERROR_COUNT = 8
INSIGHT_SEED = 42


@dataclass(frozen=True)
class InsightBundle:
    """Cached evaluation artifacts used by the Streamlit insights panel."""

    history: dict
    evaluation: dict
    confusion_matrix: np.ndarray
    error_images: np.ndarray  # (N, 28, 28) uint8-like float/raw
    error_true: np.ndarray
    error_pred: np.ndarray


def artifacts_available(
    history_path: Path = HISTORY_PATH,
    evaluation_path: Path = EVALUATION_PATH,
    model_path: Path = MODEL_PATH,
) -> bool:
    """Return True when all artifacts needed for insights exist."""
    return history_path.exists() and evaluation_path.exists() and model_path.exists()


def load_insight_sources(
    history_path: Path = HISTORY_PATH,
    evaluation_path: Path = EVALUATION_PATH,
) -> tuple[dict, dict]:
    """Load training history and evaluation JSON."""
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {evaluation_path}")
    return load_json(history_path), load_json(evaluation_path)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Build an integer confusion matrix without sklearn."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            matrix[truth, pred] += 1
    return matrix


def select_misclassified(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    count: int = DEFAULT_ERROR_COUNT,
    seed: int = INSIGHT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a small, deterministic sample of incorrect predictions."""
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        empty = np.zeros((0, 28, 28), dtype=images.dtype)
        return empty, np.array([], dtype=int), np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    take = min(count, int(wrong.size))
    chosen = rng.choice(wrong, size=take, replace=False)
    chosen.sort()
    return images[chosen], y_true[chosen].astype(int), y_pred[chosen].astype(int)


def build_insight_bundle(
    model_path: Path = MODEL_PATH,
    history_path: Path = HISTORY_PATH,
    evaluation_path: Path = EVALUATION_PATH,
    error_count: int = DEFAULT_ERROR_COUNT,
    seed: int = INSIGHT_SEED,
) -> InsightBundle:
    """Load artifacts, evaluate MNIST test predictions, and package insights."""
    history, evaluation = load_insight_sources(history_path, evaluation_path)
    model = load_model(model_path)

    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_norm = normalize_images(x_test)
    probabilities = model.predict(x_norm, verbose=0)
    y_pred = np.argmax(probabilities, axis=1).astype(int)
    y_true = y_test.astype(int)

    matrix = confusion_matrix(y_true, y_pred)
    error_images, error_true, error_pred = select_misclassified(
        x_test,
        y_true,
        y_pred,
        count=error_count,
        seed=seed,
    )

    return InsightBundle(
        history=history,
        evaluation=evaluation,
        confusion_matrix=matrix,
        error_images=error_images,
        error_true=error_true,
        error_pred=error_pred,
    )


def _style_axes(ax) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)
    ax.tick_params(colors="#000000")
    ax.yaxis.label.set_color("#000000")
    ax.xaxis.label.set_color("#000000")
    ax.title.set_color("#000000")
    ax.grid(True, color="#DDDDDD", linewidth=0.8, alpha=1.0)


def figure_training_curves(history: dict) -> Figure:
    """Black-and-white accuracy and loss curves from saved training history."""
    metrics = history.get("metrics", {})
    epochs = list(range(1, len(metrics.get("accuracy", [])) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=120)
    fig.patch.set_facecolor("white")

    ax_acc, ax_loss = axes
    ax_acc.plot(epochs, metrics.get("accuracy", []), color="#000000", linewidth=2.0, label="Train")
    ax_acc.plot(
        epochs,
        metrics.get("val_accuracy", []),
        color="#000000",
        linewidth=2.0,
        linestyle="--",
        label="Validation",
    )
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xticks(epochs)
    ax_acc.legend(frameon=False)
    _style_axes(ax_acc)

    ax_loss.plot(epochs, metrics.get("loss", []), color="#000000", linewidth=2.0, label="Train")
    ax_loss.plot(
        epochs,
        metrics.get("val_loss", []),
        color="#000000",
        linewidth=2.0,
        linestyle="--",
        label="Validation",
    )
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xticks(epochs)
    ax_loss.legend(frameon=False)
    _style_axes(ax_loss)

    fig.tight_layout()
    return fig


def figure_confusion_matrix(matrix: np.ndarray) -> Figure:
    """Black-and-white confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    image = ax.imshow(matrix, cmap="gray_r", interpolation="nearest")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#000000")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#000000")

    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ticks = list(range(NUM_CLASSES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # Annotate cells lightly for readability without color clutter.
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "#FFFFFF" if value > threshold else "#000000"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=7)

    for spine in ax.spines.values():
        spine.set_color("#000000")
    ax.tick_params(colors="#000000")
    fig.tight_layout()
    return fig


def figure_misclassified_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Figure | None:
    """Gallery of a few misclassified MNIST test examples."""
    count = int(len(images))
    if count == 0:
        return None

    cols = min(4, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 1.9 * rows), dpi=120)
    fig.patch.set_facecolor("white")
    axes_list = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes_list):
        ax.set_facecolor("white")
        ax.axis("off")
        if idx >= count:
            continue
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(
            f"true {int(y_true[idx])} → pred {int(y_pred[idx])}",
            fontsize=9,
            color="#000000",
        )

    fig.suptitle("Sample incorrect predictions", fontsize=11, color="#000000", y=1.02)
    fig.tight_layout()
    return fig
