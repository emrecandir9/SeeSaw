"""The /probe v1 contract: request parsing, validation and response bodies.

See docs/api.md for the written contract.
"""

import io
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from processing.grouping import encode_label_png, normalize_attention

SCHEMA_VERSION = 1
K = 8
IMAGE_SIZE = 224
PATCH = 14
GRID = IMAGE_SIZE // PATCH  # 16


class ContractError(Exception):
    """A failure that maps to an error response."""

    def __init__(self, status: int, code: str, message: str, probe_id: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.probe_id = probe_id


def invalid(message: str, probe_id: str | None = None) -> ContractError:
    return ContractError(400, "invalid_input", message, probe_id)


def unavailable(message: str, probe_id: str | None) -> ContractError:
    return ContractError(503, "model_unavailable", message, probe_id)


def processing(message: str, probe_id: str | None) -> ContractError:
    return ContractError(500, "processing_error", message, probe_id)


@dataclass
class ProbeRequest:
    probe_id: str
    session_id: str
    models: list[str]
    match_against: dict[str, str | None]
    reset_history: bool
    image: np.ndarray  # (224, 224, 3) uint8, RGB


def parse_request(image_bytes: bytes, metadata_json: str) -> ProbeRequest:
    meta = _parse_metadata(metadata_json)
    image = _decode_image(image_bytes, meta["probe_id"])
    return ProbeRequest(image=image, **meta)


def _parse_metadata(metadata_json: str) -> dict:
    try:
        meta = json.loads(metadata_json)
    except (TypeError, ValueError) as exc:
        raise invalid(f"metadata is not valid JSON: {exc}") from None
    if not isinstance(meta, dict):
        raise invalid("metadata must be a JSON object")

    probe_id = meta.get("probe_id")
    if not isinstance(probe_id, str) or not probe_id:
        raise invalid("probe_id must be a non-empty string")

    def fail(message: str) -> ContractError:
        return invalid(message, probe_id)

    version = meta.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:  # rejects true and 1.0, which compare equal to 1
        raise fail(f"schema_version must be {SCHEMA_VERSION}")

    session_id = meta.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise fail("session_id must be a non-empty string")

    models = meta.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
        raise fail("models must be a non-empty list of model IDs")
    if len(set(models)) != len(models):
        raise fail("models must not repeat")

    raw_match = meta.get("match_against")
    if raw_match is None:  # absent or null
        raw_match = {}
    if not isinstance(raw_match, dict):
        raise fail("match_against must be an object of model ID -> probe_id or null")
    for value in raw_match.values():
        if value is not None and (not isinstance(value, str) or not value):
            raise fail("match_against values must be non-empty probe_id strings or null")
    match_against = {m: raw_match[m] for m in models if m in raw_match}

    reset_history = meta.get("reset_history", False)
    if not isinstance(reset_history, bool):
        raise fail("reset_history must be true or false")

    return {
        "probe_id": probe_id,
        "session_id": session_id,
        "models": models,
        "match_against": match_against,
        "reset_history": reset_history,
    }


def _decode_image(image_bytes: bytes, probe_id: str) -> np.ndarray:
    unreadable = (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError)
    try:
        image = Image.open(io.BytesIO(image_bytes))  # reads the header only
    except unreadable as exc:
        raise invalid(f"image is not a readable PNG: {exc}", probe_id) from None

    # Check the header before decoding pixels, so a wrong or huge image costs nothing.
    if image.format != "PNG":
        raise invalid(f"image must be a PNG, got {image.format}", probe_id)
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise invalid(f"image must be {IMAGE_SIZE}x{IMAGE_SIZE}, got {image.width}x{image.height}", probe_id)
    if image.mode not in ("RGB", "RGBA"):
        raise invalid(f"image must be RGB or RGBA, got mode {image.mode}", probe_id)

    try:
        image.load()
    except unreadable as exc:
        raise invalid(f"image is not a readable PNG: {exc}", probe_id) from None
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def error_body(probe_id: str | None, code: str, message: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_id": probe_id,
        "status": "error",
        "error": {"code": code, "message": message},
    }


def build_result(
    model_id: str,
    labels: np.ndarray,
    attention: Sequence[float],
    upsampler: str,
    matched_to: str | None,
    timings: dict[str, float | None],
) -> dict:
    values = np.asarray(attention, dtype=np.float64)
    height, width = labels.shape
    return {
        "model_id": model_id,
        "width": int(width),
        "height": int(height),
        "cluster_map_png_base64": encode_label_png(labels),
        "attention": [float(v) for v in values],
        "attention_norm": [float(v) for v in normalize_attention(values)],
        "upsampler": upsampler,
        "matched_to": matched_to,
        "timings_ms": dict(timings),
    }


def ok_body(probe_id: str, mode: str, results: list[dict], server_total_ms: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_id": probe_id,
        "status": "ok",
        "mode": mode,
        "server_total_ms": server_total_ms,
        "results": results,
    }
