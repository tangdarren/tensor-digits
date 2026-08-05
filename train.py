#!/usr/bin/env python3
"""CLI entry point for training the TensorDigits MNIST classifier."""

from __future__ import annotations

import argparse

from src.training import load_model, print_training_summary, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN on MNIST for handwritten digit recognition.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of training data held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    print_training_summary(summary)

    # Confirm the saved model reloads cleanly.
    reloaded = load_model(summary["model_path"])
    print(f"\nReload check : loaded '{reloaded.name}' from {summary['model_path']}")


if __name__ == "__main__":
    main()
