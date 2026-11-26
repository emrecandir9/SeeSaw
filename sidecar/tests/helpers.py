"""Shared test helpers (not a test module)."""

import io
import json

import numpy as np
from PIL import Image


def png_bytes(array: np.ndarray, mode: str | None = None) -> bytes:
    """Encode an array as PNG bytes, optionally forcing a Pillow mode."""
    image = Image.fromarray(array)
    if mode is not None:
        image = image.convert(mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def rgb_image(size: int = 224, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def metadata(**overrides) -> str:
    body = {
        "schema_version": 1,
        "probe_id": "p-1",
        "session_id": "s-1",
        "models": ["dinov2_vits14"],
        "match_against": {"dinov2_vits14": None},
        "reset_history": False,
    }
    body.update(overrides)
    return json.dumps(body)
