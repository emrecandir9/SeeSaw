"""Frozen DINOv2 ViT-S/14: patch features and last-layer CLS-to-patch attention.

The backbone comes from the pinned facebookresearch/dinov2 torch hub repo, the
same backbone FeatUp wraps.
"""

import logging
import time
import warnings

import numpy as np
import torch

from contract import GRID, IMAGE_SIZE, K, ContractError, ProbeRequest, build_result, ok_body, unavailable
from processing.grouping import (
    cluster_patches,
    group_attention,
    order_centroids,
    pixel_labels,
    relabel_by_height,
    upsample_nearest,
)
from processing.history import ProbeHistory

from models.base import (  # noqa: F401
    IMAGENET_MEAN,
    IMAGENET_STD,
    EncoderOutput,
    QKVClsAttention,
    channel_norm,
    finish,
    jbu_upsample,
    pick_device,
    synchronize,
    to_input,
)

HUB_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MODEL_ID = "dinov2_vits14"
log = logging.getLogger("seesaw.dinov2")


class Dinov2Encoder:
    def __init__(self, device: str, upsampler: str = "nearest"):
        if upsampler not in ("nearest", "featup"):
            raise ValueError(f"upsampler must be 'nearest' or 'featup', got {upsampler!r}")
        self.device = device
        self.upsampler = upsampler
        self.execution_path: str | None = None
        self._jbu = None
        if upsampler == "featup":
            # Our pinned backbone + FeatUp's upsampler weights; plan p.7 allows another copy of the
            # backbone if it gives the same features (checked in tests/test_featup.py).
            from models.featup_jbu import load_jbu

            self._jbu, self.execution_path = load_jbu(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # xFormers availability notices
            # skip_validation: the fork check calls the GitHub API (60 requests/hour unauthenticated)
            # and fails when that limit is used up. HUB_REPO pins a commit on the official main branch.
            model = torch.hub.load(HUB_REPO, MODEL_ID, trust_repo=True, skip_validation=True, verbose=False)
        self.model = model.eval().to(device)
        self._attention = QKVClsAttention(self.model.blocks[-1].attn)

    def synchronize(self) -> None:
        synchronize(self.device)

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> EncoderOutput:
        x = to_input(image, IMAGENET_MEAN, IMAGENET_STD, self.device)
        tokens = self.model.forward_features(x)["x_norm_patchtokens"][0]  # (256, 384)
        # FeatUp's featurizer output (use_norm=True): K-means centroids and FeatUp's
        # upsampled features then live in the same space.
        low = channel_norm(tokens)
        out = finish(low, self._attention.cls_row[0], GRID)
        if self._jbu is not None:
            out.upsample = lambda: jbu_upsample(self._jbu, low, x, GRID)  # 16x16 -> 256x256 -> 224x224
        return out


class RealBackend:
    def __init__(self, encoder: Dinov2Encoder, history: ProbeHistory):
        self.encoder = encoder
        self.history = history
        self.models = [MODEL_ID]
        self.device = encoder.device

    def warm_up(self) -> None:
        started = time.perf_counter()
        out = self.encoder.encode(np.zeros((224, 224, 3), dtype=np.uint8))
        if out.upsample is not None:
            out.upsample()
        self.encoder.synchronize()
        log.info("warm-up on %s took %.0f ms", self.device, (time.perf_counter() - started) * 1000)

    def handle(self, req: ProbeRequest, scenario: str) -> dict:
        if scenario != "valid":
            raise ContractError(400, "invalid_input", "X-Mock-Scenario is only available in mock mode", req.probe_id)
        unknown = [m for m in req.models if m not in self.models]
        if unknown:
            raise unavailable(f"model not available: {', '.join(unknown)}", req.probe_id)
        if req.reset_history:
            self.history.reset(req.session_id)

        timings: dict[str, float | None] = {"encode": None, "cluster": None, "featup": None, "assign": None, "match": None}

        t0 = time.perf_counter()
        out = self.encoder.encode(req.image)
        self.encoder.synchronize()
        t1 = time.perf_counter()

        raw_labels, centroids = cluster_patches(out.features, K)
        patch_labels = relabel_by_height(raw_labels.reshape(GRID, GRID), K)
        ordered = order_centroids(raw_labels, patch_labels, centroids)
        t2 = time.perf_counter()

        # Paper and plan p.6: group mean of the patch attention over each group's patches.
        attention = group_attention(patch_labels, out.patch_attention, K)
        featup_ms = 0.0
        if out.upsample is not None:
            f0 = time.perf_counter()
            pixels = out.upsample()  # (224*224, 384)
            featup_ms = (time.perf_counter() - f0) * 1000
            labels = pixel_labels(pixels, patch_labels, ordered, (IMAGE_SIZE, IMAGE_SIZE))
            timings["featup"] = round(featup_ms, 2)
            upsampler = "featup_jbu"
        else:
            labels = upsample_nearest(patch_labels)
            upsampler = "nearest"
        t3 = time.perf_counter()

        timings["encode"] = round((t1 - t0) * 1000, 2)
        timings["cluster"] = round((t2 - t1) * 1000, 2)
        timings["assign"] = round((t3 - t2) * 1000 - featup_ms, 2)

        self.history.store(req.session_id, MODEL_ID, req.probe_id, ordered)
        result = build_result(MODEL_ID, labels, attention, upsampler, None, timings)
        return ok_body(req.probe_id, "real", [result], 0.0)
