# FeatUp pixel labels for model A (E2) — design

Date: 25 Nov 2026 · Owner: Emre · Branch: `emre/E2-featup` (from `emre/E4-model-b-comparison`)

## Why

Model A's label maps are 16 × 16 patch labels stretched to 224 × 224 (`upsampler: "nearest"`), so groups are blocky and do not follow object edges. The plan (p.2, p.5) and acceptance A4 require pixel labels from FeatUp features assigned to the nearest K-means centroid, with the checkpoint and execution path recorded. Milestone 1 (1 Dec) expects FeatUp operational; FeatUp and matching complete by 3 Dec.

## Goals

- Real mode returns FeatUp pixel labels for `dinov2_vits14` with `upsampler: "featup_jbu"` and a measured `timings_ms.featup`.
- Group attention (brightness) stays the group mean of the last-layer CLS-to-patch attention over each group's patches, as the paper and the plan's contract define it.
- Runs on jessica (RTX 4000 Ada, CUDA kernel); the Mac and CI keep working with nearest labels by an explicit choice.
- Checkpoint (URL + SHA-256) and execution path (CUDA kernel or C++ CPU) recorded in the startup log and README.

## Non-goals

- FeatUp for the model B candidates and in `compare_models.py` (separate E4 task before 30 Nov).
- Matching (`match_against`, E3).
- A pure-PyTorch port of FeatUp's adaptive convolution.
- Contract changes: `upsampler` and `timings_ms.featup` already exist in `/probe` v1.

## Anti-goals

- A lab run silently serving coarse maps because FeatUp failed to import.
- An unpinned backbone or checkpoint.

## Facts that shape it (checked on jessica)

- FeatUp's DINOv2 featurizer returns `forward_features(x)["x_norm_patchtokens"]` of `dinov2_vits14` — the features we already use — but loads DINOv2 from the unpinned `facebookresearch/dinov2` main.
- `use_norm=True` adds `ChannelNorm` = `LayerNorm(384)` over channels. The checkpoint `dinov2_jbu_stack_cocostuff.ckpt` holds only `upsampler.*` weights (42 tensors), so ChannelNorm keeps its default affine (weight 1, bias 0).
- The upsampler is `JBUStack(384)`: four ×2 JBU stages guided by the normalized input image, 16 × 16 → 256 × 256, plus `fixup_proj * 0.1`. FeatUp passes the same normalized image as backbone input and guidance.
- `featup.upsamplers` and `featup.layers` import only torch and the compiled `adaptive_conv_cuda_impl` / `adaptive_conv_cpp_impl`. FeatUp's other requirements (kornia, lightning, timm 0.4.12) are not needed.
- FeatUp at `6b5a6c0e91f75e69194807128dcbc39c3084a30d` builds on jessica with CUDA 13.4 + torch 2.15.0+cu134; backbone + JBU ≈ 20 ms per probe. It cannot be installed on the Mac (setup.py always builds the CUDA extension).

## Decisions (Emre, 25 Nov), checked against the paper and the plan

Rule from Emre: where an answer contradicts the paper (Xie and Lin, SIGGRAPH Posters '26) or the implementation plan v2, the paper or plan wins.

| Question | Emre's answer | Paper / plan | Final |
| --- | --- | --- | --- |
| FeatUp missing | Explicit flag | Plan p.8: a documented FeatUp blocker may permit temporary coarse maps; A4: record the execution path | Explicit flag `--upsampler featup\|nearest`, default `featup`. If FeatUp cannot load, the server refuses to start and says to pass `--upsampler nearest`. |
| Group attention | Mean over pixel labels | Paper §2: "last-layer CLS-to-patch attention averaged per group"; FeatUp "upsamples the grouping". Plan p.6: `attention[8]` = "group mean last-layer CLS-to-patch attention" | **Changed:** group mean over the group's **patches** (patch labels from K-means), as in E1. |
| 256 → 224 | Resize features | Paper §2: "label each upsampled pixel by its nearest patch-level centroid, giving a 224² cluster-id image" | Resize the **features** bilinearly to 224 × 224, then assign each pixel. |
| Loader | Pinned + verify | Plan p.7: "Load the backbone through FeatUp (or verify that another copy of the weights gives the same features)" | Keep our pinned DINOv2 and attention hook; build `ChannelNorm` + `JBUStack` from the `featup` package and load the upsampler weights from the pinned checkpoint. A jessica-only test checks our output against FeatUp's own hub model. |
| K-means features (my call) | – | Paper §2: K-means on "L2-normalized patch features" and nearest "patch-level centroid"; plan p.7: the upsampler was trained on its featurizer's output "with normalization when use_norm=True" | K-means on the L2-normalized **ChannelNorm'd** patch features (FeatUp's `use_norm=True` featurizer output) in **both** modes, so centroids and upsampled features share one space and Mac/lab patch groups are identical. ChannelNorm has no trained weights, so it is `layer_norm` without affine and needs no FeatUp install. |

## Approaches considered

- **Pinned backbone + FeatUp's upsampler weights (chosen):** reproducible, keeps the E1 hook and loader, small change; needs one equivalence test against FeatUp's hub.
- **Load the whole model through FeatUp's hub:** exactly FeatUp's path, but the DINOv2 inside it is unpinned and the hook must reach inside FeatUp's wrapper.
- **Silent fallback / pure-PyTorch port:** rejected (anti-goal; too slow and too much work for the sprint).

## Pipeline (featup mode)

1. `x = to_input(image, ImageNet mean/std)` (unchanged).
2. Backbone: `tokens = x_norm_patchtokens` (256 × 384), attention hook unchanged.
3. `low = ChannelNorm(tokens as 1 × 384 × 16 × 16)`.
4. K-means (K=8, single-threaded) on L2-normalized `low` patch vectors → patch labels, centroids; relabel by height (paper §3) and reorder centroids (unchanged logic).
5. Attention: `group_attention(patch_labels, patch_attention)` (unchanged); empty groups 0.0.
6. `high = JBUStack(low, x)` → 1 × 384 × 256 × 256 → bilinear resize to 224 × 224 → L2-normalize per pixel.
7. Pixel labels = nearest reordered centroid (Euclidean) for each of the 224 × 224 feature vectors.
8. Timings: `encode` = steps 1–3, `cluster` = 4, `featup` = 6, `assign` = 5 and 7.

Nearest mode: steps 1–5, then `upsample_nearest(patch_labels)` (the E1 path, except that K-means now sees the ChannelNorm'd features).

## Interfaces

- `models/featup_jbu.py` (new): pinned `FEATUP_COMMIT`, `CHECKPOINT_URL`, `CHECKPOINT_SHA256 = 438dad4d…b38c`; `load_upsampler(device) -> (channel_norm, jbu)`; raises a clear error if `featup` or its extensions do not import, or the checksum differs. Reports the execution path (`"cuda kernel"` on CUDA, `"cpp"` otherwise).
- `processing/grouping.py` (numpy only, testable in CI): `assign_nearest(features (N, D), centroids (K, D)) -> labels (N,) uint8`.
- `models/base.py`: `channel_norm(tokens)` = `layer_norm` over the feature dim, no affine (FeatUp's ChannelNorm with its untrained default weights).
- `models/dinov2.py`: `Dinov2Encoder(device, upsampler="featup"|"nearest")`; `RealBackend.handle` branches on the mode and sets `upsampler` in the result.
- `server.py`: `--upsampler featup|nearest` (default `featup`, ignored with `--mock`); startup logs the checkpoint and execution path.
- `requirements-featup.txt` (new): the pinned FeatUp git URL, installed with `pip install --no-deps --no-build-isolation -r requirements-featup.txt` on CUDA machines only.

## Tests

- CI / Mac: `assign_nearest` (hand-built centroids, lowest index on ties, uint8 range); `channel_norm` equals `torch.nn.LayerNorm(384)` at default init; the server refuses to start with `--upsampler featup` when FeatUp is absent and names the flag; nearest mode still returns `upsampler: "nearest"` block labels.
- jessica (skipped without `featup` + CUDA): checksum verified; our `low` and `high` match FeatUp hub's `up.model(x)` / `up(x)` on the fixture image (allclose); a real `/probe` returns `upsampler: "featup_jbu"`, a 224 × 224 map with IDs 0–7, finite attention, non-null `featup` timing; labels are not block-constant (some 14 × 14 block contains two IDs).

## Left to the builder

- Exact tolerances for the equivalence test.
- Log wording; README section placement (setup on jessica, flag, A4 record).

## Calls I made (Emre to check)

- K-means runs on the ChannelNorm'd features in both modes (see the decisions table).
- Nearest centroid by Euclidean distance (the K-means objective) on L2-normalized vectors.
- Checkpoint integrity by a SHA-256 check in code.
- No `/health` change before the 27 Nov freeze; the execution path is logged, and each result's `upsampler` field says which path produced it.
