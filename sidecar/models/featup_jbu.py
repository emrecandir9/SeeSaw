"""FeatUp's JBU upsampler for DINOv2 ViT-S/14, with pinned code and checkpoint.

The backbone stays ours (models/dinov2.py, pinned hub commit); only the upsampler
comes from FeatUp. FeatUp's own hub model wraps the same dinov2_vits14 features plus
a ChannelNorm (models/base.py: channel_norm); tests/test_featup.py checks that both
paths give the same output.
"""

import hashlib
from pathlib import Path

import torch

FEATUP_COMMIT = "6b5a6c0e91f75e69194807128dcbc39c3084a30d"
CHECKPOINT_URL = (
    "https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dinov2_jbu_stack_cocostuff.ckpt"
)
CHECKPOINT_SHA256 = "438dad4dedb3c3ba4b7fa13d1c8eb465dafd323dadc039afd04b15178ea7b38c"
INSTALL_HINT = (
    "install it on a CUDA machine with `pip install --no-deps --no-build-isolation -r requirements-featup.txt` "
    "(needs nvcc matching torch's CUDA), or start with --upsampler nearest"
)


class FeatUpUnavailable(RuntimeError):
    pass


def check_available() -> None:
    """Raise FeatUpUnavailable unless FeatUp and its compiled kernels import."""
    try:
        import featup.upsamplers  # noqa: F401  (imports adaptive_conv_cuda_impl and adaptive_conv_cpp_impl)
    except ImportError as exc:
        raise FeatUpUnavailable(f"FeatUp is not available ({exc}); {INSTALL_HINT}") from None


def verify_checkpoint(path: Path) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise FeatUpUnavailable(f"{path}: SHA-256 {digest}, expected {CHECKPOINT_SHA256}; delete it to download again")


def checkpoint_path() -> Path:
    path = Path(torch.hub.get_dir()) / "checkpoints" / CHECKPOINT_URL.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(CHECKPOINT_URL, str(path), hash_prefix=CHECKPOINT_SHA256)
    verify_checkpoint(path)
    return path


def load_jbu(device: str, dim: int = 384) -> tuple[torch.nn.Module, str]:
    """FeatUp's JBUStack with the pinned weights, and the execution path for the A4 record."""
    check_available()
    from featup.upsamplers import JBUStack

    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)["state_dict"]
    weights = {k.removeprefix("upsampler."): v for k, v in state.items() if k.startswith("upsampler.")}
    jbu = JBUStack(dim)
    jbu.load_state_dict(weights, strict=True)
    return jbu.eval().to(device), "cuda kernel" if device.startswith("cuda") else "cpp fallback"
