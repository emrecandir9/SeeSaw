# Model B comparison: candidate encoders and comparison script

Date: 25 Nov 2026 · Owner: Emre · Issue: #6 (E4), first part
Branch: `emre/E4-model-b-comparison` (from `emre/E1-sidecar-baseline`; PR into `develop` after #5 merges)

## Intent

On 30 Nov Emre shows the team how the three shortlisted model B candidates group and attend to the same captures as model A (DINOv2 ViT-S/14). On 1 Dec the team picks one. The comparison must make "visible, explainable disagreement" and "usable attention channel" easy to judge, and record time and memory per model. The winner's encoder then becomes the real model B for paired inference (rest of E4).

## Goals

- One encoder per candidate in `sidecar/models/`, with the same interface as the DINOv2 encoder (L2-normalized patch features + last-layer CLS-to-patch attention).
- `sidecar/tools/compare_models.py`: runs model A and the candidates on the same images and writes, per image, a PNG grid and an overall metrics table (markdown + JSON).
- Runs on the Mac now (CPU/MPS) and unchanged on the lab PC (CUDA).

## Non-goals

- FeatUp. The comparison is at patch level; the FeatUp column says "not yet (E2)" and timings record `featup: null`. FeatUp is added to the script after E2 or on the lab PC.
- Paired inference in the server (`/probe` with two models): after the 1 Dec choice.
- Choosing the winner. The script reports; the team decides.
- Godot captures: rerun on them when the capture rig works (~28 Nov).

## Anti-goals

- A candidate whose features differ from what its FeatUp upsampler was trained on. Each encoder reproduces the feature path FeatUp uses.
- Timing numbers that look like lab PC results. Every report states the machine and device.

## Decisions taken with Emre

| Topic | Decision |
| --- | --- |
| Candidates | CLIP ViT-B/16, DINO v1 ViT-S/16, supervised ViT-S/16 (issue #6) |
| FeatUp | Patch level now; FeatUp stage later |
| Images | The two paper renders now (`fixtures/images/image1.png`, `image2.png`); Godot captures later |
| Code home | Reusable encoders in `sidecar/models/` + script in `sidecar/tools/` |
| Output | PNG grid per image + metrics table (markdown + JSON), in a git-ignored folder |
| CLIP loading | Mark Hamilton's CLIP fork (the one FeatUp imports), pinned |

## Approaches considered (CLIP loading)

| Option | Trade-off |
| --- | --- |
| **A. Hamilton's CLIP fork, pinned (chosen)** | Exactly FeatUp's CLIP feature path (`get_visual_features`), and FeatUp needs it anyway. Unmaintained; `setup.py` uses `pkg_resources`, so install may need `--no-build-isolation`. |
| B. `open_clip` + reproduce the fork's patch path | Maintained pip package; equivalence to the fork must be proven at E2, which needs the fork anyway. |
| C. Hugging Face `transformers` CLIP | Another large dependency and a different internal layout. |

## Candidates

| Model ID | Source (pinned) | Patch / grid | Dim | Preprocessing | Attention module |
| --- | --- | --- | --- | --- | --- |
| `dinov2_vits14` (A) | existing `models/dinov2.py` | 14 / 16×16 | 384 | ImageNet mean/std | qkv-style |
| `clip_vitb16` | `clip @ git+https://github.com/mhamilton723/CLIP.git@dce9d43cc659cb9a013745b378c78077fde6aac0`, `clip.load("ViT-B/16")`, features from `get_visual_features(img, include_cls=False)` | 16 / 14×14 | 512 | CLIP mean `(0.4815, 0.4578, 0.4082)` / std `(0.2686, 0.2613, 0.2758)` | `nn.MultiheadAttention` |
| `dino_vits16` | torch hub `facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a`, `dino_vits16`, last-layer patch tokens after the final norm | 16 / 14×14 | 384 | ImageNet mean/std | qkv-style |
| `vit_s16_sup` | `timm` `vit_small_patch16_224.augreg_in21k_ft_in1k` (believed identical to FeatUp's timm 0.4.12 weights; verify at E2), last-layer patch tokens after the final norm | 16 / 14×14 | 384 | timm config for this model (mean/std 0.5) | qkv-style |

All run in float32 (CLIP is loaded in fp32 even on CUDA) so the comparison is like for like.

## Shared encoder interface

`sidecar/models/base.py` holds what all encoders share: `EncoderOutput` (moved from `dinov2.py`), `pick_device` (moved), `synchronize(device)`, and the qkv-style CLS attention hook now inside `Dinov2Encoder`, generalized to any module with `qkv`, `num_heads`, `scale`. `dinov2.py` imports them; its behavior and tests stay the same.

Each encoder exposes `model_id`, `patch`, `grid`, `device`, `encode(image: (224, 224, 3) uint8) -> EncoderOutput`, `synchronize()`. `EncoderOutput.features` is `(grid², dim)`, `patch_attention` is `(grid, grid)`.

CLIP attention: a pre-hook on the last residual block's `nn.MultiheadAttention` recomputes softmax(q·kᵀ/√d) for the CLS query from `in_proj_weight`/`in_proj_bias`, mean over heads (CLIP's layout is sequence-first).

## Comparison script

```
python tools/compare_models.py [--images PATH ...] [--models ID ...] [--device auto] [--runs 10] [--out comparison-out]
```

Per image and model: one warm-up, then `--runs` probes. Record:

- encode and cluster time: median and min–max over the runs (ms), GPU work synchronized
- memory: CUDA peak allocated; MPS driver-allocated; CPU "n/a"
- attention check: row sum ≈ 1, all finite, and a peakedness value (max / mean of patch attention) so a flat, useless attention map shows up
- groups: K-means K=8, numbered by height, nearest-upsampled to 224×224 (same helpers as the server)
- agreement with model A: adjusted Rand index between the two 224×224 label maps (1 = same grouping, about 0 = unrelated). This is the "disagreement" number; the images show whether it is explainable.
- FeatUp: `null` / "not yet (E2)"

Outputs in `--out` (default `sidecar/comparison-out/`, git-ignored):

- `<image>_grid.png`: one row per model; columns: input, groups (8 colors over the image), attention heatmap, groups with attention as brightness. Model name and key numbers in each row label.
- `metrics.md`: one table per image plus a header with date, machine, device, torch version and git commit.
- `metrics.json`: the same numbers, for later comparison with lab PC runs.

## Dependencies

`requirements-ml.txt` adds `timm==1.0.31` and the pinned CLIP fork (with its `ftfy`, `regex`). `requirements.txt` (CI) does not change. If the fork does not install with build isolation, the README documents `pip install --no-build-isolation` for that one line.

## Tests (skip without the ML packages)

- Per encoder: features `(196, dim)` with unit-norm rows; attention `(14, 14)`, non-negative, row sum ≈ 1.
- Per encoder: rebuilding the last attention block from its weights reproduces the module's output, and its CLS row equals the encoder's attention (as for DINOv2).
- CLIP features equal the fork's `get_visual_features` output (the FeatUp path) for the same input.
- Script: runs on one image with `--runs 1` and writes the grid PNG, `metrics.md` and `metrics.json` with every model present.
- Existing DINOv2 tests stay green after the refactor.

## Left to the builder

Colors, grid layout details, exact memory API per device, function names inside the script.
