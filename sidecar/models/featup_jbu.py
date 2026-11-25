"""FeatUp's JBU upsamplers, with pinned code and checkpoints, one per backbone.

The backbones stay ours (models/dinov2.py and the candidates, pinned); only the
upsampler comes from FeatUp. Each checkpoint was trained on its own FeatUp
featurizer plus a ChannelNorm (models/base.py: channel_norm); tests/test_featup.py
checks that our backbones give the same output as FeatUp's hub models.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

FEATUP_COMMIT = "6b5a6c0e91f75e69194807128dcbc39c3084a30d"
BASE_URL = "https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/"


@dataclass(frozen=True)
class Checkpoint:
    hub_name: str  # FeatUp's torch hub entry point
    sha256: str
    dim: int

    @property
    def url(self) -> str:
        return f"{BASE_URL}{self.hub_name}_jbu_stack_cocostuff.ckpt"


CHECKPOINTS = {  # by our model ID; SHA-256 of the files downloaded on jessica, 25 Nov 2026
    "dinov2_vits14": Checkpoint("dinov2", "438dad4dedb3c3ba4b7fa13d1c8eb465dafd323dadc039afd04b15178ea7b38c", 384),
    "clip_vitb16": Checkpoint("clip", "e5dfaae2f7409d29cd067a595b1de0009e093d5a1dac9ea93443596eda7f0a49", 512),
    "dino_vits16": Checkpoint("dino16", "407dff491058280abfbf38647e9365bf6397bc53d03b13251ed5a6715ef8d1b2", 384),
    "vit_s16_sup": Checkpoint("vit", "b7de0e4c641cdb180ed9b1e573117f7a98e40deca4055553fc5d59fb35b7c097", 384),
}
CHECKPOINT_URL = CHECKPOINTS["dinov2_vits14"].url  # model A, the one the server logs
CHECKPOINT_SHA256 = CHECKPOINTS["dinov2_vits14"].sha256
INSTALL_HINT = (
    "install it on a CUDA machine with `pip install --no-deps --no-build-isolation -r requirements-featup.txt` "
    "(needs nvcc matching torch's CUDA), or start with --upsampler nearest"
)


class FeatUpUnavailable(RuntimeError):
    pass


def check_available() -> None:
    """Raise FeatUpUnavailable unless FeatUp and its compiled kernels import."""
    try:
        import featup.upsamplers  # noqa: F401  (loads torch first, which the kernels need)
        import adaptive_conv_cuda_impl  # noqa: F401  compiled kernels: a partial install has the package without them
        import adaptive_conv_cpp_impl  # noqa: F401
    except ImportError as exc:
        raise FeatUpUnavailable(f"FeatUp is not available ({exc}); {INSTALL_HINT}") from None


def verify_checkpoint(path: Path, expected: str) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise FeatUpUnavailable(f"{path}: SHA-256 {digest}, expected {expected}; delete it to download again")


def checkpoint_path(checkpoint: Checkpoint) -> Path:
    import torch

    path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint.url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(checkpoint.url, str(path), hash_prefix=checkpoint.sha256)
    verify_checkpoint(path, checkpoint.sha256)
    return path


def load_jbu(device: str, model_id: str = "dinov2_vits14"):
    """FeatUp's JBUStack (torch module) with the model's pinned weights, and the execution path for the A4 record."""
    check_available()
    import torch
    from featup.upsamplers import JBUStack

    checkpoint = CHECKPOINTS[model_id]
    state = torch.load(checkpoint_path(checkpoint), map_location="cpu", weights_only=True)["state_dict"]
    weights = {k.removeprefix("upsampler."): v for k, v in state.items() if k.startswith("upsampler.")}
    jbu = JBUStack(checkpoint.dim)
    jbu.load_state_dict(weights, strict=True)
    return jbu.eval().to(device), "cuda kernel" if device.startswith("cuda") else "cpp fallback"
