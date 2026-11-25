"""Real-model tests. Skipped when torch or scikit-learn is missing (as in CI).

Set SEESAW_DEVICE=cpu to force the CPU path. The first run downloads the
DINOv2 weights (~90 MB) into the torch hub cache.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from contract import ContractError, ProbeRequest  # noqa: E402
from models.dinov2 import Dinov2Encoder, RealBackend, pick_device  # noqa: E402
from processing.grouping import decode_label_png  # noqa: E402
from processing.history import ProbeHistory  # noqa: E402
from server import create_app  # noqa: E402
from tests.helpers import metadata, png_bytes  # noqa: E402

IMAGES = Path(__file__).resolve().parent.parent / "fixtures" / "images"


def load(name: str) -> np.ndarray:
    return np.asarray(Image.open(IMAGES / name).convert("RGB"), dtype=np.uint8)


@pytest.fixture(scope="module")
def encoder():
    return Dinov2Encoder(pick_device(os.environ.get("SEESAW_DEVICE", "auto")))


@pytest.fixture
def backend(encoder):
    return RealBackend(encoder, ProbeHistory())


def request(image, probe_id="p-1", models=("dinov2_vits14",)):
    return ProbeRequest(probe_id, "s-1", list(models), {}, False, image)


def test_pick_device_cpu():
    assert pick_device("cpu") == "cpu"


def test_attention_row_sums_to_one(encoder):
    out = encoder.encode(load("image1.png"))
    assert abs(out.attention_row_sum - 1.0) < 1e-3
    assert out.patch_attention.shape == (16, 16)
    assert (out.patch_attention >= 0).all()


def test_features_normalized(encoder):
    out = encoder.encode(load("image1.png"))
    assert out.features.shape == (256, 384)
    np.testing.assert_allclose(np.linalg.norm(out.features, axis=1), 1.0, atol=1e-4)


def test_deterministic_labels(backend):
    image = load("image1.png")
    first = backend.handle(request(image), "valid")["results"][0]["cluster_map_png_base64"]
    second = backend.handle(request(image, "p-2"), "valid")["results"][0]["cluster_map_png_base64"]
    assert first == second


def test_real_probe_end_to_end(backend):
    client = TestClient(create_app(backend, "real"))
    for name in ("image1.png", "image2.png"):
        response = client.post(
            "/probe",
            files={"image": (name, (IMAGES / name).read_bytes(), "image/png")},
            data={"metadata": metadata()},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["mode"] == "real"
        (result,) = body["results"]
        assert result["upsampler"] == "nearest"
        assert result["matched_to"] is None
        assert result["timings_ms"]["featup"] is None
        assert result["timings_ms"]["match"] is None
        assert result["timings_ms"]["encode"] > 0
        labels = decode_label_png(result["cluster_map_png_base64"])
        assert labels.shape == (224, 224)
        assert labels.max() <= 7
        assert len(result["attention"]) == 8
        assert all(math.isfinite(v) for v in result["attention"])
        assert max(result["attention_norm"]) == 1.0


def test_group_ids_follow_image_height(backend):
    result = backend.handle(request(load("image2.png")), "valid")["results"][0]
    labels = decode_label_png(result["cluster_map_png_base64"])
    rows = np.indices(labels.shape)[0]
    present = [g for g in range(8) if (labels == g).any()]
    mean_rows = [rows[labels == g].mean() for g in present]
    assert present == list(range(len(present)))
    assert mean_rows == sorted(mean_rows)


def test_uniform_image(backend):
    body = backend.handle(request(np.zeros((224, 224, 3), dtype=np.uint8)), "valid")
    (result,) = body["results"]
    assert len(result["attention"]) == 8
    assert all(math.isfinite(v) for v in result["attention"])


def test_unknown_model_503(backend):
    with pytest.raises(ContractError) as info:
        backend.handle(request(load("image1.png"), models=("mock_model_b",)), "valid")
    assert info.value.status == 503


def test_mock_scenario_rejected_in_real_mode(backend):
    with pytest.raises(ContractError) as info:
        backend.handle(request(load("image1.png")), "failure")
    assert info.value.status == 400


def test_reset_history_clears_session(backend):
    image = load("image1.png")
    backend.handle(request(image, "p-0"), "valid")
    assert backend.history.get("s-1", "dinov2_vits14", "p-0") is not None
    req = request(image, "p-1")
    req.reset_history = True
    backend.handle(req, "valid")
    assert backend.history.get("s-1", "dinov2_vits14", "p-0") is None
    assert backend.history.get("s-1", "dinov2_vits14", "p-1").shape == (8, 384)
