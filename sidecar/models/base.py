"""What every vision encoder shares: output type, device helpers, input
preparation and the CLS-to-patch attention hook for q/k/v attention modules
(DINOv2, DINO v1, timm ViT)."""

from dataclasses import dataclass

import numpy as np
import torch

# FeatUp normalizes every backbone's input with these (featup/util.py `norm`).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EncoderOutput:
    features: np.ndarray  # (grid*grid, dim) float32, L2-normalized rows
    patch_attention: np.ndarray  # (grid, grid) float32, CLS-to-patch, mean over heads
    attention_row_sum: float  # full CLS row incl. CLS itself; ~1.0


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def to_input(image: np.ndarray, mean, std, device: str) -> torch.Tensor:
    """(224, 224, 3) uint8 RGB -> normalized (1, 3, 224, 224) float tensor."""
    x = torch.tensor(image, device=device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def finish(tokens: torch.Tensor, cls_row: torch.Tensor, grid: int) -> EncoderOutput:
    """Patch tokens (grid*grid, dim) and the full CLS attention row (1 + grid*grid,)."""
    features = torch.nn.functional.normalize(tokens.float(), dim=-1)
    return EncoderOutput(
        features=features.cpu().numpy(),
        patch_attention=cls_row[1:].reshape(grid, grid).float().cpu().numpy(),
        attention_row_sum=float(cls_row.sum()),
    )


class QKVClsAttention:
    """Pre-hook on an attention module with `qkv`, `num_heads`, `scale` that
    recomputes softmax(q.k^T * scale) for the CLS query, averaged over heads."""

    def __init__(self, attn: torch.nn.Module):
        self.cls_row: torch.Tensor | None = None  # (B, N)
        attn.register_forward_pre_hook(self._capture)

    def _capture(self, attn: torch.nn.Module, inputs: tuple) -> None:
        x = inputs[0]
        B, N, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]  # (B, heads, N, d)
        if hasattr(attn, "q_norm"):  # timm; Identity for the models we use
            q, k = attn.q_norm(q), attn.k_norm(k)
        weights = ((q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)).softmax(-1)  # (B, heads, 1, N)
        self.cls_row = weights[:, :, 0, :].mean(1)
