"""Model B candidate: supervised ViT-S/16 (ImageNet-21k labels, fine-tuned on ImageNet-1k)."""

import os

import numpy as np
import torch

from models.base import IMAGENET_MEAN, IMAGENET_STD, EncoderOutput, QKVClsAttention, channel_norm, finish, synchronize, to_input

TIMM_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"


class VitSupEncoder:
    model_id = "vit_s16_sup"
    patch = 16
    grid = 14

    def __init__(self, device: str):
        import huggingface_hub.constants
        import timm

        # Hugging Face's xet download failed on our network; the classic path works.
        # huggingface_hub reads the variable once at import, so set its constant too.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        huggingface_hub.constants.HF_HUB_DISABLE_XET = os.environ["HF_HUB_DISABLE_XET"].upper() in {"1", "ON", "YES", "TRUE"}
        self.device = device
        self.model = timm.create_model(TIMM_NAME, pretrained=True).eval().to(device)
        self.last_attention_module = self.model.blocks[-1].attn
        self._attention = QKVClsAttention(self.last_attention_module)

    def synchronize(self) -> None:
        synchronize(self.device)

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> EncoderOutput:
        # ImageNet mean/std like FeatUp, not timm's 0.5/0.5 (see models/base.py).
        x = to_input(image, IMAGENET_MEAN, IMAGENET_STD, self.device)
        tokens = self.model.forward_features(x)[0, self.model.num_prefix_tokens :]  # after final norm
        return finish(channel_norm(tokens), self._attention.cls_row[0], self.grid)  # FeatUp use_norm, as model A
