"""Frozen DINOv2 ViT-S/14: patch features and last-layer CLS-to-patch attention.

The backbone comes from the pinned facebookresearch/dinov2 torch hub repo, the
same backbone FeatUp wraps.
"""

import logging
import time
import warnings

import numpy as np
import torch

from contract import GRID, K, ContractError, ProbeRequest, build_result, ok_body, unavailable
from processing.grouping import cluster_patches, group_attention, relabel_by_height, upsample_nearest
from processing.history import ProbeHistory

from models.base import (  # noqa: F401
    IMAGENET_MEAN,
    IMAGENET_STD,
    EncoderOutput,
    QKVClsAttention,
    channel_norm,
    finish,
    pick_device,
    synchronize,
    to_input,
)

HUB_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MODEL_ID = "dinov2_vits14"
log = logging.getLogger("seesaw.dinov2")


class Dinov2Encoder:
    def __init__(self, device: str):
        self.device = device
        # TODO(E2): load through FeatUp's hub (plan p.7) and take the backbone from it;
        # keep the attention hook below. Verify FeatUp wraps this same dinov2_vits14.
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
        return finish(channel_norm(tokens), self._attention.cls_row[0], GRID)


class RealBackend:
    def __init__(self, encoder: Dinov2Encoder, history: ProbeHistory):
        self.encoder = encoder
        self.history = history
        self.models = [MODEL_ID]
        self.device = encoder.device

    def warm_up(self) -> None:
        started = time.perf_counter()
        self.encoder.encode(np.zeros((224, 224, 3), dtype=np.uint8))
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
        ordered = np.zeros_like(centroids)
        for old in range(K):  # reorder centroids to the new IDs
            mask = raw_labels.reshape(GRID, GRID) == old
            if mask.any():
                ordered[patch_labels[mask][0]] = centroids[old]
        t2 = time.perf_counter()

        attention = group_attention(patch_labels, out.patch_attention, K)
        labels = upsample_nearest(patch_labels)
        t3 = time.perf_counter()

        timings["encode"] = round((t1 - t0) * 1000, 2)
        timings["cluster"] = round((t2 - t1) * 1000, 2)
        timings["assign"] = round((t3 - t2) * 1000, 2)

        self.history.store(req.session_id, MODEL_ID, req.probe_id, ordered)
        result = build_result(MODEL_ID, labels, attention, "nearest", None, timings)
        return ok_body(req.probe_id, "real", [result], 0.0)
