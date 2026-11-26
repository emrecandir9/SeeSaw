# FeatUp for the model B candidates (comparison only) — short design

**Why:** plan p.7, "How to decide": run each candidate on the same captures "with K=8, FeatUp where
available", record "encode + FeatUp time, GPU memory, and whether attention extraction works", and show
side-by-side label maps for A and each B. The team picks model B on 1 Dec, so all three candidates
(CLIP ViT-B/16, DINO v1 ViT-S/16, supervised ViT-S/16) get FeatUp now; `/probe` stays model A only.

**Checked on jessica (25 Nov):**
- FeatUp publishes JBU checkpoints for all three: `clip` (dim 512), `dino16` and `vit` (dim 384). Each
  loads into `JBUStack(dim)` with `strict=True` (42 tensors, no other keys).
- FeatUp's hub featurizers load with timm 1.0.31 (the timm 0.4.12 risk did not happen).
- Our encoders already use FeatUp's featurizer paths. Low-res features vs FeatUp hub (`use_norm=True`):
  DINO v1 identical, ViT 9e-6, CLIP 0.0095 (FeatUp's `clip.load` runs fp16 on CUDA; ours is fp32).

## Decisions (paper and plan first)

| Question | Decision | Source |
| --- | --- | --- |
| Which upsampler per candidate | Its own FeatUp JBU checkpoint (`{clip,dino16,vit}_jbu_stack_cocostuff.ckpt`), SHA-256 pinned | plan p.7: upsampler trained on its own featurizer |
| Features | ChannelNorm (`use_norm=True`), as for model A | plan p.7, FeatUp hub default |
| Backbone copy | Ours, verified equal to FeatUp's hub featurizer (CLIP hub cast to fp32) | plan p.7 "or verify that another copy of the weights gives the same features" |
| Pixel labels | Nearest patch-level centroid per FeatUp pixel, height-ordered IDs, as model A | paper §2 |
| Group attention | Patch-level group mean, unchanged | paper §2, plan p.6 |
| Resize | 14 × 16 = 224, so the JBU output is already 224² (the resize is a no-op) | — |
| Comparison default | `--upsampler featup`; without FeatUp it exits with the `--upsampler nearest` hint, as the server does | E2 rule: no silent fallback |
| Timing | Record encode ms and FeatUp ms separately, plus peak GPU memory | plan p.7 |
| ARI vs A | On the 224² pixel label maps | — |

## Work

1. `models/featup_jbu.py`: per-model checkpoint table (URL, SHA-256, dim); `load_jbu(device, model_id)`.
2. `models/base.py`: shared `jbu_upsample(jbu, low, x)`; each candidate encoder takes `upsampler`.
3. `processing/grouping.py`: `order_centroids` and `pixel_labels`, shared by `RealBackend` and the comparison.
4. `tools/compare_models.py`: `--upsampler`, FeatUp ms column, pixel-label grids, text updated.
5. Tests: hub equivalence and execution path per candidate (CUDA), hash mismatch per model, comparison
   refusal without FeatUp (CI), comparison end to end with FeatUp (CUDA).

Not in scope: paired `/probe` for model B (after the 1 Dec choice), the demo-scene captures (28 Nov).
