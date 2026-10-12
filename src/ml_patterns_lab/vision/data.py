"""Create small synthetic grayscale images for two inspection domains."""

from __future__ import annotations

import numpy as np

IMAGE_SIDE = 24
IMAGE_SIZE = IMAGE_SIDE * IMAGE_SIDE


def make_images(
    rows: int = 1200,
    domain: str = "source",
    seed: int = 51,
) -> tuple[np.ndarray, np.ndarray]:
    if domain not in {"source", "target"}:
        raise ValueError("domain must be 'source' or 'target'.")

    rng = np.random.default_rng(seed)
    labels = np.tile(np.array([0, 1], dtype=np.int64), rows // 2)
    if rows % 2:
        labels = np.append(labels, 0)
    rng.shuffle(labels)
    images = np.empty((len(labels), IMAGE_SIDE, IMAGE_SIDE), dtype=np.float32)

    for index, label in enumerate(labels):
        image = rng.normal(0.20, 0.07, size=(IMAGE_SIDE, IMAGE_SIDE))
        if domain == "target":
            illumination = np.linspace(-0.04, 0.10, IMAGE_SIDE)[:, None]
            image = image + illumination
        if label:
            start = int(rng.integers(4, IMAGE_SIDE - 5))
            if domain == "source":
                image[3:21, start : start + 2] += 0.72
            else:
                for row in range(3, 21):
                    column = int(np.clip(start + (row - 12) // 3, 2, IMAGE_SIDE - 3))
                    image[row, column : column + 2] += 0.72
        else:
            for _ in range(3):
                row = int(rng.integers(2, IMAGE_SIDE - 2))
                column = int(rng.integers(2, IMAGE_SIDE - 2))
                image[row, column] += 0.20
        images[index] = np.clip(image, 0.0, 1.0)
    return images, labels


def quantize_images(images: np.ndarray, bits: int = 5) -> np.ndarray:
    """Simulate reduced-precision pixel input before model inference."""
    if not isinstance(bits, int) or not 1 <= bits <= 16:
        raise ValueError("bits must be an integer between 1 and 16.")
    levels = (1 << bits) - 1
    values = np.asarray(images, dtype=np.float32)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Pixels must be finite values between zero and one.")
    return (np.round(values * levels) / levels).astype(np.float32)
