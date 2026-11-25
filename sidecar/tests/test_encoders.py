"""Shared encoder base and the model B candidate encoders.

Skipped without torch. Candidate tests also need timm and the CLIP fork, and
download weights on first run. SEESAW_DEVICE=cpu forces the CPU path.
"""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.base import QKVClsAttention, finish  # noqa: E402

IMAGES = Path(__file__).resolve().parent.parent / "fixtures" / "images"


def test_finish_shapes():
    rng = np.random.default_rng(0)
    tokens = torch.tensor(rng.normal(size=(196, 8)), dtype=torch.float32)
    row = torch.softmax(torch.tensor(rng.normal(size=197), dtype=torch.float32), 0)
    out = finish(tokens, row, grid=14)
    assert out.features.shape == (196, 8)
    np.testing.assert_allclose(np.linalg.norm(out.features, axis=1), 1.0, atol=1e-5)
    assert out.patch_attention.shape == (14, 14)
    np.testing.assert_allclose(out.patch_attention.ravel(), row[1:].numpy(), atol=1e-7)
    assert abs(out.attention_row_sum - 1.0) < 1e-5


class ToyAttention(torch.nn.Module):
    def __init__(self, dim=4, heads=2):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = torch.nn.Linear(dim, dim * 3)

    def forward(self, x):
        return x


def test_qkv_hook_on_toy_module():
    torch.manual_seed(0)
    attn = ToyAttention()
    hook = QKVClsAttention(attn)
    x = torch.randn(1, 5, 4)
    attn(x)
    qkv = attn.qkv(x).reshape(1, 5, 3, 2, 2).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    expected = torch.softmax((q[:, :, 0:1] * attn.scale) @ k.transpose(-2, -1), -1)[:, :, 0].mean(1)
    torch.testing.assert_close(hook.cls_row, expected)


CANDIDATES = ["clip_vitb16", "dino_vits16", "vit_s16_sup"]


def load_image(name="image1.png"):
    from PIL import Image

    return np.asarray(Image.open(IMAGES / name).convert("RGB"), dtype=np.uint8)


@pytest.fixture(scope="module")
def encoders():
    pytest.importorskip("timm")
    pytest.importorskip("clip")
    from models.base import pick_device
    from models.candidates import load_encoder

    device = pick_device(os.environ.get("SEESAW_DEVICE", "auto"))
    return {model_id: load_encoder(model_id, device) for model_id in CANDIDATES}


@pytest.mark.parametrize("model_id", CANDIDATES)
def test_shapes_and_norms(encoders, model_id):
    enc = encoders[model_id]
    out = enc.encode(load_image())
    assert (enc.model_id, enc.patch, enc.grid) == (model_id, 16, 14)
    assert out.features.shape[0] == 196
    np.testing.assert_allclose(np.linalg.norm(out.features, axis=1), 1.0, atol=1e-4)
    assert out.patch_attention.shape == (14, 14)
    assert (out.patch_attention >= 0).all()
    assert abs(out.attention_row_sum - 1.0) < 1e-3


@pytest.mark.parametrize("model_id", CANDIDATES)
def test_deterministic(encoders, model_id):
    enc = encoders[model_id]
    first, second = enc.encode(load_image()), enc.encode(load_image())
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.patch_attention, second.patch_attention)


def _reference_cls_row(model_id, attn, x, y):
    """CLS attention row computed independently of the encoder's hook."""
    if model_id == "clip_vitb16":
        # PyTorch's own MultiheadAttention, asked for head-averaged weights.
        _, weights = attn(x, x, x, need_weights=True, average_attn_weights=True)
        return weights[0, 0]
    B, N, C = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    weights = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
    rebuilt = attn.proj((weights @ v).transpose(1, 2).reshape(B, N, C))
    real = y[0] if isinstance(y, tuple) else y  # DINO v1 returns (x, attn)
    torch.testing.assert_close(rebuilt, real, atol=1e-3, rtol=1e-3)
    if isinstance(y, tuple):  # DINO v1 also hands back its own weights
        torch.testing.assert_close(weights, y[1], atol=1e-5, rtol=1e-4)
    return weights[0, :, 0].mean(0)


@pytest.mark.parametrize("model_id", CANDIDATES)
def test_hook_matches_module(encoders, model_id):
    enc = encoders[model_id]
    attn = enc.last_attention_module
    seen = {}
    handle = attn.register_forward_hook(lambda m, inp, out: seen.update(x=inp[0], y=out))
    try:
        out = enc.encode(load_image())
    finally:
        handle.remove()
    with torch.inference_mode():
        row = _reference_cls_row(model_id, attn, seen["x"], seen["y"])
    np.testing.assert_allclose(out.patch_attention.ravel(), row[1:].float().cpu().numpy(), atol=1e-5)


def test_clip_features_follow_featup_path(encoders):
    enc = encoders["clip_vitb16"]
    image = load_image()
    out = enc.encode(image)
    from models.base import IMAGENET_MEAN, IMAGENET_STD, to_input

    with torch.inference_mode():
        spatial = enc.model.get_visual_features(to_input(image, IMAGENET_MEAN, IMAGENET_STD, enc.device), include_cls=False)
    from models.base import channel_norm

    expected = torch.nn.functional.normalize(channel_norm(spatial[0].permute(1, 2, 0).reshape(196, -1).float()), dim=-1)
    np.testing.assert_allclose(out.features, expected.cpu().numpy(), atol=1e-5)


def test_vit_download_skips_xet(encoders):
    # timm imports huggingface_hub first, so the env var alone is read too late.
    import huggingface_hub.constants

    if os.environ.get("HF_HUB_DISABLE_XET", "1").upper() in {"1", "ON", "YES", "TRUE"}:
        assert huggingface_hub.constants.HF_HUB_DISABLE_XET is True


def test_clip_text_tower_stays_on_cpu(encoders):
    # Only the image tower runs; keeping the text tower off the GPU keeps the memory column honest.
    enc = encoders["clip_vitb16"]
    assert all(p.device.type == "cpu" for p in enc.model.transformer.parameters())
    assert enc.model.token_embedding.weight.device.type == "cpu"
    assert all(p.device.type == torch.device(enc.device).type for p in enc.model.visual.parameters())


def test_channel_norm_matches_featup_layernorm():
    # FeatUp's ChannelNorm is LayerNorm(dim); its checkpoint has no weights for it, so default init.
    from models.base import channel_norm

    x = torch.randn(256, 384)
    torch.testing.assert_close(channel_norm(x), torch.nn.LayerNorm(384)(x).detach(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("model_id", CANDIDATES)
def test_features_are_channel_normed(encoders, model_id):
    # FeatUp's featurizers (use_norm=True) apply ChannelNorm for every backbone, as for model A:
    # after LayerNorm each token has zero mean, which L2 normalization keeps.
    out = encoders[model_id].encode(load_image())
    np.testing.assert_allclose(out.features.mean(axis=1), 0.0, atol=1e-5)
