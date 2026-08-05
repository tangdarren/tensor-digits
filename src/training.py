"""MNIST training, evaluation, and model persistence for TensorDigits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing import normalize_images

SEED = 42
IMG_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_VALIDATION_SPLIT = 0.1

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "digit_classifier.keras"
HISTORY_PATH = MODELS_DIR / "training_history.json"
EVALUATION_PATH = MODELS_DIR / "evaluation.json"


def set_reproducibility(seed: int = SEED) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for reproducible runs."""
    keras.utils.set_random_seed(seed)


def load_and_prepare_data(
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    seed: int = SEED,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Load MNIST, normalize images, and create train / validation / test splits."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    # Hold out a validation set from the official training split.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    val_count = int(len(x_train) * validation_split)
    val_idx, train_idx = indices[:val_count], indices[val_count:]

    return (
        (x_train[train_idx], y_train[train_idx]),
        (x_train[val_idx], y_train[val_idx]),
        (x_test, y_test),
    )


def build_model(input_shape: tuple[int, int, int] = IMG_SHAPE) -> keras.Model:
    """Build a small CNN for 28×28 grayscale digit classification."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="digit_classifier",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_json(data: dict, path: Path) -> None:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def load_json(path: Path) -> dict:
    """Load a JSON artifact from disk."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_trained_model(model: keras.Model, path: Path = MODEL_PATH) -> Path:
    """Persist the model using Keras's native ``.keras`` format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def load_model(path: str | Path = MODEL_PATH) -> keras.Model:
    """Load a saved TensorFlow / Keras model from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model found at {model_path}")
    return keras.models.load_model(model_path)


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    seed: int = SEED,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """Train, evaluate, and save the digit classifier plus metric artifacts.

    Returns a summary dict with paths and test metrics.
    """
    set_reproducibility(seed)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_prepare_data(
        validation_split=validation_split,
        seed=seed,
    )

    model = build_model()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    model_path = models_dir / "digit_classifier.keras"
    history_path = models_dir / "training_history.json"
    evaluation_path = models_dir / "evaluation.json"

    save_trained_model(model, model_path)

    history_payload = {
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "seed": seed,
        "metrics": {key: [float(v) for v in values] for key, values in history.history.items()},
    }
    save_json(history_payload, history_path)

    evaluation_payload = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "num_test_samples": int(len(x_test)),
        "model_path": str(model_path.relative_to(ROOT_DIR)),
    }
    save_json(evaluation_payload, evaluation_path)

    summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "model_path": model_path,
        "history_path": history_path,
        "evaluation_path": evaluation_path,
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
    }
    return summary


def print_training_summary(summary: dict) -> None:
    """Print a concise report of accuracy and saved artifacts."""
    accuracy_pct = summary["test_accuracy"] * 100
    print("\n" + "=" * 56)
    print("TensorDigits training complete")
    print("=" * 56)
    print(f"Train samples : {summary['train_samples']:,}")
    print(f"Val samples   : {summary['val_samples']:,}")
    print(f"Test samples  : {summary['test_samples']:,}")
    print(f"Test accuracy : {summary['test_accuracy']:.4f} ({accuracy_pct:.2f}%)")
    print(f"Test loss     : {summary['test_loss']:.4f}")
    print("-" * 56)
    print("Saved files:")
    print(f"  Model      : {summary['model_path']}")
    print(f"  History    : {summary['history_path']}")
    print(f"  Evaluation : {summary['evaluation_path']}")
    print("=" * 56)
