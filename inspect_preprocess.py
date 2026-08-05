#!/usr/bin/env python3
"""Generate sample drawings and inspect preprocessing output.

Usage:
    python inspect_preprocess.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.preprocessing import (
    BlankDrawingError,
    describe_tensor,
    preprocess_image,
    save_preprocessed_preview,
)

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "assets" / "samples"
PREVIEWS_DIR = ROOT / "assets" / "previews"


def _load_font(size: int = 120) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
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


def make_canvas_digit(digit: int, size: int = 280, offset: tuple[int, int] | None = None) -> Image.Image:
    """Create a black-on-white drawing similar to a Streamlit canvas."""
    image = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = _load_font(size=int(size * 0.55))
    text = str(digit)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    if offset is None:
        x = (size - text_w) // 2 - bbox[0]
        y = (size - text_h) // 2 - bbox[1]
    else:
        x, y = offset

    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return image


def make_rgba_canvas_digit(digit: int, size: int = 280) -> np.ndarray:
    """Create an RGBA canvas with transparent background and black ink."""
    image = Image.new("RGBA", (size, size), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _load_font(size=int(size * 0.55))
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2 - bbox[0]
    y = (size - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)
    return np.asarray(image)


def make_blank_canvas(size: int = 280) -> Image.Image:
    return Image.new("RGB", (size, size), color=(255, 255, 255))


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[str, Image.Image | np.ndarray]] = [
        ("digit_3_centered", make_canvas_digit(3)),
        ("digit_7_offset", make_canvas_digit(7, offset=(40, 30))),
        ("digit_0_rgba", make_rgba_canvas_digit(0)),
        ("digit_5_large", make_canvas_digit(5, size=400)),
    ]

    print("TensorDigits preprocessing inspection")
    print("=" * 56)

    for name, sample in samples:
        sample_path = SAMPLES_DIR / f"{name}.png"
        if isinstance(sample, np.ndarray):
            Image.fromarray(sample).save(sample_path)
        else:
            sample.save(sample_path)

        tensor = preprocess_image(sample)
        preview_path = save_preprocessed_preview(
            sample,
            PREVIEWS_DIR / f"{name}_mnist.png",
            scale=10,
        )
        info = describe_tensor(tensor)

        print(f"\n{name}")
        print(f"  source   : {sample_path}")
        print(f"  preview  : {preview_path}")
        print(f"  shape    : {info['shape']}")
        print(f"  dtype    : {info['dtype']}")
        print(f"  range    : [{info['min']:.4f}, {info['max']:.4f}]")
        print(f"  mean     : {info['mean']:.4f}")
        print(f"  ink frac : {info['ink_fraction']:.4f}")

        assert tensor.shape == (1, 28, 28, 1)
        assert tensor.dtype == np.float32
        assert 0.0 <= info["min"] <= info["max"] <= 1.0
        # MNIST-style: background near 0, digit strokes brighter.
        assert info["mean"] < 0.35
        assert info["max"] > 0.5

    print("\nBlank canvas validation")
    try:
        preprocess_image(make_blank_canvas())
        raise SystemExit("Expected BlankDrawingError for empty canvas")
    except BlankDrawingError as exc:
        print(f"  ok: {exc}")

    print("\n" + "=" * 56)
    print("All sample checks passed.")
    print(f"Sample drawings : {SAMPLES_DIR}")
    print(f"MNIST previews  : {PREVIEWS_DIR}")
    print("=" * 56)


if __name__ == "__main__":
    main()
