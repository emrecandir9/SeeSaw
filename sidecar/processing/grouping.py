"""Numpy-only helpers that turn patch labels and attention into contract outputs.

Nothing here imports torch, so CI can test it. scikit-learn is imported lazily
inside cluster_patches.
"""

import base64
import io

import numpy as np
from PIL import Image


def normalize_attention(attention: np.ndarray) -> np.ndarray:
    """Divide by the largest group value; all zeros if that value is not positive."""
    values = np.asarray(attention, dtype=np.float64)
    peak = values.max() if values.size else 0.0
    if peak <= 0:
        return np.zeros_like(values)
    return values / peak


def encode_label_png(labels: np.ndarray) -> str:
    """8-bit grayscale PNG of the label map, base64 (ASCII)."""
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(labels, dtype=np.uint8), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_label_png(b64: str) -> np.ndarray:
    image = Image.open(io.BytesIO(base64.b64decode(b64)))
    return np.asarray(image, dtype=np.uint8)
