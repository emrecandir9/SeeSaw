"""FeatUp JBU pixel labels: model A in the server, all four models in the comparison.

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


CANDIDATES = ["clip_vitb16", "dino_vits16", "vit_s16_sup"]


@pytest.mark.parametrize("model_id", ["dinov2_vits14", *CANDIDATES])
def test_checkpoint_hash_mismatch_refuses(tmp_path, model_id):
    from models.featup_jbu import CHECKPOINTS, FeatUpUnavailable, verify_checkpoint

    bad = tmp_path / "checkpoint.ckpt"
    bad.write_bytes(b"not the checkpoint")
    with pytest.raises(FeatUpUnavailable) as info:
        verify_checkpoint(bad, CHECKPOINTS[model_id].sha256)
    assert CHECKPOINTS[model_id].sha256 in str(info.value)


def test_checkpoint_table_covers_every_model():
    from models.candidates import ENCODERS
    from models.featup_jbu import CHECKPOINT_SHA256, CHECKPOINT_URL, CHECKPOINTS

    assert set(CHECKPOINTS) == set(ENCODERS)
    names = {m: c.url.rsplit("/", 1)[1] for m, c in CHECKPOINTS.items()}
    assert names == {
        "dinov2_vits14": "dinov2_jbu_stack_cocostuff.ckpt",
        "clip_vitb16": "clip_jbu_stack_cocostuff.ckpt",
        "dino_vits16": "dino16_jbu_stack_cocostuff.ckpt",
        "vit_s16_sup": "vit_jbu_stack_cocostuff.ckpt",
    }
    assert {m: c.dim for m, c in CHECKPOINTS.items()} == {"dinov2_vits14": 384, "clip_vitb16": 512, "dino_vits16": 384, "vit_s16_sup": 384}
    assert (CHECKPOINT_URL, CHECKPOINT_SHA256) == (CHECKPOINTS["dinov2_vits14"].url, CHECKPOINTS["dinov2_vits14"].sha256)


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


class NoUpsample:
    """The same encoder with FeatUp switched off: the nearest path on identical groups."""

    def __init__(self, encoder):
        self.inner, self.device = encoder, encoder.device

    def encode(self, image):
        out = self.inner.encode(image)
        out.upsample = None
        return out

    def synchronize(self):
        self.inner.synchronize()


@pytest.mark.parametrize("name", ["image1.png", "image2.png"])
def test_pixel_ids_follow_patch_ids(encoder, name):
    # Pixel IDs must use the height-ordered centroids, i.e. mostly agree with the nearest-mode
    # patch labels (the IDs attention and history use); with raw K-means order they would not.
    from models.dinov2 import RealBackend

    image = load(name)
    pixel = decode_label_png(probe(RealBackend(encoder, ProbeHistory()), image)["cluster_map_png_base64"])
    patch = decode_label_png(probe(RealBackend(NoUpsample(encoder), ProbeHistory()), image)["cluster_map_png_base64"])
    agreement = (pixel == patch).mean()
    assert agreement > 0.7, agreement


def test_featup_uniform_image(backend):
    result = probe(backend, np.full((224, 224, 3), 128, dtype=np.uint8))
    labels = decode_label_png(result["cluster_map_png_base64"])
    assert labels.shape == (224, 224) and labels.max() <= 7
    assert np.isfinite(result["attention"]).all()


# Model B candidates (comparison only): each with its own FeatUp checkpoint.
HUB_NAMES = {"clip_vitb16": "clip", "dino_vits16": "dino16", "vit_s16_sup": "vit"}


@pytest.fixture(scope="module", params=CANDIDATES)
def candidate(request):
    pytest.importorskip("featup")
    pytest.importorskip("timm")
    pytest.importorskip("clip")
    if not torch.cuda.is_available():
        pytest.skip("FeatUp tests run on a CUDA machine")
    from models.candidates import load_encoder

    return load_encoder(request.param, os.environ.get("SEESAW_DEVICE", "cuda"), upsampler="featup")


def test_candidate_execution_path_is_cuda_kernel(candidate):
    assert candidate.execution_path == "cuda kernel"


def test_candidate_matches_featup_hub(candidate):
    from models.base import IMAGENET_MEAN, IMAGENET_STD, to_input
    from models.featup_jbu import FEATUP_COMMIT

    hub = torch.hub.load(
        f"mhamilton723/FeatUp:{FEATUP_COMMIT}", HUB_NAMES[candidate.model_id], use_norm=True, trust_repo=True, skip_validation=True
    )
    hub = hub.to(candidate.device).float().eval()  # FeatUp's clip.load gives fp16 on CUDA; ours is fp32
    image = load("image1.png")
    out = candidate.encode(image)
    x = to_input(image, IMAGENET_MEAN, IMAGENET_STD, candidate.device)
    with torch.inference_mode():
        low = torch.nn.functional.normalize(hub.model(x), dim=1)[0]
        dim = low.shape[0]
        high = torch.nn.functional.normalize(hub(x), dim=1)[0]  # 14 x 16 = 224 already
    np.testing.assert_allclose(out.features, low.permute(1, 2, 0).reshape(196, dim).cpu().numpy(), atol=1e-4)
    np.testing.assert_allclose(out.upsample(), high.permute(1, 2, 0).reshape(224 * 224, dim).cpu().numpy(), atol=1e-4)


def test_candidate_nearest_mode_has_no_upsampler():
    pytest.importorskip("timm")
    from models.dino_v1 import DinoV1Encoder

    encoder = DinoV1Encoder("cpu")
    assert encoder.execution_path is None
    assert encoder.encode(load("image1.png")).upsample is None
