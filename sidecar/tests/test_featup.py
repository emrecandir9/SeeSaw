"""FeatUp JBU pixel labels for model A.

The checksum test needs only torch. The rest need the `featup` package and CUDA
(the lab machine), and download FeatUp's hub code and checkpoint on first run.
"""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from contract import ProbeRequest  # noqa: E402
from processing.grouping import decode_label_png  # noqa: E402
from processing.history import ProbeHistory  # noqa: E402

IMAGES = Path(__file__).resolve().parent.parent / "fixtures" / "images"


def load(name: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(IMAGES / name).convert("RGB"), dtype=np.uint8)


def test_checkpoint_hash_mismatch_refuses(tmp_path):
    from models.featup_jbu import CHECKPOINT_SHA256, FeatUpUnavailable, verify_checkpoint

    bad = tmp_path / "dinov2_jbu_stack_cocostuff.ckpt"
    bad.write_bytes(b"not the checkpoint")
    with pytest.raises(FeatUpUnavailable) as info:
        verify_checkpoint(bad)
    assert CHECKPOINT_SHA256 in str(info.value)


@pytest.fixture(scope="module")
def encoder():
    pytest.importorskip("featup")
    if not torch.cuda.is_available():
        pytest.skip("FeatUp tests run on a CUDA machine")
    from models.dinov2 import Dinov2Encoder

    return Dinov2Encoder(os.environ.get("SEESAW_DEVICE", "cuda"), upsampler="featup")


@pytest.fixture
def backend(encoder):
    from models.dinov2 import RealBackend

    return RealBackend(encoder, ProbeHistory())


def probe(backend, image):
    body = backend.handle(ProbeRequest("p-1", "s-1", ["dinov2_vits14"], {}, False, image), "valid")
    return body["results"][0]


def test_execution_path_is_cuda_kernel(encoder):
    assert encoder.execution_path == "cuda kernel"


def test_matches_featup_hub(encoder):
    from models.base import IMAGENET_MEAN, IMAGENET_STD, to_input  # same mean/std as featup.util.norm
    from models.featup_jbu import FEATUP_COMMIT

    hub = torch.hub.load(f"mhamilton723/FeatUp:{FEATUP_COMMIT}", "dinov2", use_norm=True, trust_repo=True, skip_validation=True)
    hub = hub.to(encoder.device).eval()
    image = load("image1.png")
    out = encoder.encode(image)
    x = to_input(image, IMAGENET_MEAN, IMAGENET_STD, encoder.device)
    with torch.inference_mode():
        low = torch.nn.functional.normalize(hub.model(x), dim=1)[0].permute(1, 2, 0).reshape(256, 384)
        high = torch.nn.functional.interpolate(hub(x), (224, 224), mode="bilinear", align_corners=False)
        high = torch.nn.functional.normalize(high, dim=1)[0].permute(1, 2, 0).reshape(224 * 224, 384)
    np.testing.assert_allclose(out.features, low.cpu().numpy(), atol=1e-4)
    np.testing.assert_allclose(out.upsample(), high.cpu().numpy(), atol=1e-4)


def test_featup_probe(backend):
    result = probe(backend, load("image1.png"))
    assert result["upsampler"] == "featup_jbu"
    assert result["timings_ms"]["featup"] > 0
    labels = decode_label_png(result["cluster_map_png_base64"])
    assert labels.shape == (224, 224)
    assert labels.max() <= 7
    assert np.isfinite(result["attention"]).all()
    blocks = labels.reshape(16, 14, 16, 14).transpose(0, 2, 1, 3).reshape(256, 196)
    assert any(len(np.unique(b)) > 1 for b in blocks), "labels are still patch blocks"


def test_featup_uniform_image(backend):
    result = probe(backend, np.full((224, 224, 3), 128, dtype=np.uint8))
    labels = decode_label_png(result["cluster_map_png_base64"])
    assert labels.shape == (224, 224) and labels.max() <= 7
    assert np.isfinite(result["attention"]).all()
