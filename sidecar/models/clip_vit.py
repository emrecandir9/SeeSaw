"""Model B candidate: CLIP ViT-B/16 (language-supervised), via Mark Hamilton's
CLIP fork, the one FeatUp uses (`get_visual_features`)."""

import numpy as np
import torch

from models.base import IMAGENET_MEAN, IMAGENET_STD, EncoderOutput, channel_norm, finish, jbu_upsample, load_upsampler, synchronize, to_input


class MHACls:
    """Pre-hook on CLIP's nn.MultiheadAttention (sequence-first input) that
    recomputes the CLS query's attention weights, averaged over heads."""

    def __init__(self, attn: torch.nn.MultiheadAttention):
        self.cls_row: torch.Tensor | None = None  # (B, N)
        attn.register_forward_pre_hook(self._capture)

    def _capture(self, attn: torch.nn.MultiheadAttention, inputs: tuple) -> None:
        x = inputs[0]  # (N, B, C)
        N, B, C = x.shape
        heads, d = attn.num_heads, C // attn.num_heads
        w_q, w_k, _ = attn.in_proj_weight.chunk(3)
        b_q, b_k, _ = attn.in_proj_bias.chunk(3)
        q = torch.nn.functional.linear(x[:1], w_q, b_q).reshape(1, B, heads, d).permute(1, 2, 0, 3)  # (B, h, 1, d)
        k = torch.nn.functional.linear(x, w_k, b_k).reshape(N, B, heads, d).permute(1, 2, 0, 3)  # (B, h, N, d)
        weights = ((q * d**-0.5) @ k.transpose(-2, -1)).softmax(-1)  # (B, h, 1, N)
        self.cls_row = weights[:, :, 0, :].mean(1)


class ClipEncoder:
    model_id = "clip_vitb16"
    patch = 16
    grid = 14

    def __init__(self, device: str, upsampler: str = "nearest"):
        import clip

        self.device = device
        self._jbu, self.execution_path = load_upsampler(device, self.model_id, upsampler)
        model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
        # clip.load gives fp16 weights on CUDA; load on CPU and keep fp32 for a fair comparison.
        # Only the image tower goes to the device: the text tower is never used.
        self.model = model.float().eval()
        self.model.visual.to(device)
        self.last_attention_module = self.model.visual.transformer.resblocks[-1].attn
        self._attention = MHACls(self.last_attention_module)

    def synchronize(self) -> None:
        synchronize(self.device)

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> EncoderOutput:
        # ImageNet mean/std like FeatUp, not CLIP's own (see models/base.py).
        x = to_input(image, IMAGENET_MEAN, IMAGENET_STD, self.device)
        spatial = self.model.get_visual_features(x, include_cls=False)  # (1, 512, 14, 14)
        tokens = spatial[0].permute(1, 2, 0).reshape(self.grid * self.grid, -1)
        low = channel_norm(tokens)  # FeatUp use_norm, as model A
        out = finish(low, self._attention.cls_row[0], self.grid)
        if self._jbu is not None:
            out.upsample = lambda: jbu_upsample(self._jbu, low, x, self.grid)
        return out
