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
