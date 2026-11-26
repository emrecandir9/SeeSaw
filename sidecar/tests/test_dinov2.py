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


def test_attention_hook_matches_module(encoder):
    """Rebuild the last block's attention from its weights; it must reproduce the
    module's output, and its CLS row must equal what the encoder reports."""
    import torch

    attn = encoder.model.blocks[-1].attn
    seen = {}
    handle = attn.register_forward_hook(lambda m, inp, out: seen.update(x=inp[0], y=out))
    try:
        out = encoder.encode(load("image1.png"))
    finally:
        handle.remove()
    x, y = seen["x"].float(), seen["y"].float()
    B, N, C = x.shape
    with torch.inference_mode():
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        weights = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
        rebuilt = attn.proj((weights @ v).transpose(1, 2).reshape(B, N, C))
    np.testing.assert_allclose(rebuilt.cpu().numpy(), y.cpu().numpy(), atol=1e-3)
    expected = weights[0, :, 0, 1:].mean(0).reshape(16, 16).cpu().numpy()
    np.testing.assert_allclose(out.patch_attention, expected, atol=1e-5)


def test_stored_centroids_follow_new_ids(backend, encoder):
    image = load("image1.png")
    result = backend.handle(request(image, "p-c"), "valid")["results"][0]
    patch_labels = decode_label_png(result["cluster_map_png_base64"])[::14, ::14].ravel()
    centroids = backend.history.get("s-1", "dinov2_vits14", "p-c")
    features = encoder.encode(image).features
    nearest = ((features[:, None, :] - centroids[None]) ** 2).sum(-1).argmin(1)
    assert (nearest == patch_labels).mean() > 0.95


class ConstantEncoder:
    device = "cpu"

    def encode(self, image):
        from models.dinov2 import EncoderOutput

        return EncoderOutput(np.ones((256, 384), dtype=np.float32), np.full((16, 16), 1 / 257, dtype=np.float32), 1.0)

    def synchronize(self):
        pass


def test_identical_features_leave_empty_groups():
    backend = RealBackend(ConstantEncoder(), ProbeHistory())
    (result,) = backend.handle(request(np.zeros((224, 224, 3), dtype=np.uint8), "p-u"), "valid")["results"]
    assert result["attention"][0] > 0
    assert result["attention"][1:] == [0.0] * 7
    centroids = backend.history.get("s-1", "dinov2_vits14", "p-u")
    assert (centroids[1:] == 0).all()
