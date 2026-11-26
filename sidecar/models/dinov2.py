"""Frozen DINOv2 ViT-S/14: patch features and last-layer CLS-to-patch attention.

The backbone comes from the pinned facebookresearch/dinov2 torch hub repo, the
same backbone FeatUp wraps.
"""

import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from contract import GRID, K, ContractError, ProbeRequest, build_result, ok_body, unavailable
from processing.grouping import cluster_patches, group_attention, relabel_by_height, upsample_nearest
from processing.history import ProbeHistory

HUB_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MODEL_ID = "dinov2_vits14"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
log = logging.getLogger("seesaw.dinov2")


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class EncoderOutput:
    features: np.ndarray  # (256, 384) float32, L2-normalized rows
    patch_attention: np.ndarray  # (16, 16) float32, CLS-to-patch, mean over heads
    attention_row_sum: float  # full CLS row incl. CLS itself; ~1.0


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
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self._cls_row: torch.Tensor | None = None
        self.model.blocks[-1].attn.register_forward_pre_hook(self._capture_attention)

    def _capture_attention(self, attn: torch.nn.Module, inputs: tuple) -> None:
        """Recompute the last block's attention weights for the CLS query."""
        x = inputs[0]
        B, N, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]  # (B, heads, N, d)
        weights = ((q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)).softmax(-1)  # (B, heads, 1, N)
        self._cls_row = weights[:, :, 0, :].mean(1)  # (B, N), mean over heads

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> EncoderOutput:
        x = torch.tensor(image, device=self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - self._mean) / self._std
        tokens = self.model.forward_features(x)["x_norm_patchtokens"][0]  # (256, 384)
        features = torch.nn.functional.normalize(tokens, dim=-1)
        cls_row = self._cls_row[0]
        return EncoderOutput(
            features=features.float().cpu().numpy(),
            patch_attention=cls_row[1:].reshape(GRID, GRID).float().cpu().numpy(),
            attention_row_sum=float(cls_row.sum()),
        )


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
