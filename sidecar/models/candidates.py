"""All encoders the model B comparison can load, by model ID."""

from typing import Callable


def _dinov2(device: str, upsampler: str):
    from models.dinov2 import Dinov2Encoder

    encoder = Dinov2Encoder(device, upsampler)
    encoder.model_id, encoder.patch, encoder.grid = "dinov2_vits14", 14, 16
    encoder.last_attention_module = encoder.model.blocks[-1].attn
    return encoder


def _clip(device: str, upsampler: str):
    from models.clip_vit import ClipEncoder

    return ClipEncoder(device, upsampler)


def _dino_v1(device: str, upsampler: str):
    from models.dino_v1 import DinoV1Encoder

    return DinoV1Encoder(device, upsampler)


def _vit_sup(device: str, upsampler: str):
    from models.vit_sup import VitSupEncoder

    return VitSupEncoder(device, upsampler)


ENCODERS: dict[str, Callable] = {
    "dinov2_vits14": _dinov2,
    "clip_vitb16": _clip,
    "dino_vits16": _dino_v1,
    "vit_s16_sup": _vit_sup,
}


def load_encoder(model_id: str, device: str, upsampler: str = "nearest"):
    if model_id not in ENCODERS:
        raise KeyError(f"unknown model '{model_id}', expected one of {', '.join(ENCODERS)}")
    return ENCODERS[model_id](device, upsampler)
