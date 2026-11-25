"""Model B candidate: DINO v1 ViT-S/16 (self-supervised, previous generation)."""

import warnings

import numpy as np
import torch

from models.base import IMAGENET_MEAN, IMAGENET_STD, EncoderOutput, QKVClsAttention, finish, synchronize, to_input

HUB_REPO = "facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a"


class DinoV1Encoder:
    model_id = "dino_vits16"
    patch = 16
    grid = 14

    def __init__(self, device: str):
        self.device = device
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # skip_validation: see models/dinov2.py (GitHub API rate limit); the commit is on DINO's main.
            model = torch.hub.load(HUB_REPO, "dino_vits16", trust_repo=True, skip_validation=True, verbose=False)
        self.model = model.eval().to(device)
        self.last_attention_module = self.model.blocks[-1].attn
        self._attention = QKVClsAttention(self.last_attention_module)

    def synchronize(self) -> None:
        synchronize(self.device)

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> EncoderOutput:
        x = to_input(image, IMAGENET_MEAN, IMAGENET_STD, self.device)
        # Last block output after the final norm, as FeatUp's "token" features.
        tokens = self.model.get_intermediate_layers(x, n=1)[0][0, 1:]
        return finish(tokens, self._attention.cls_row[0], self.grid)
