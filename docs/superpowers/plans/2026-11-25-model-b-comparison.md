# Model B comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three candidate encoders (CLIP ViT-B/16, DINO v1 ViT-S/16, supervised ViT-S/16) with the DINOv2 encoder's interface, and `tools/compare_models.py` that writes per-image PNG grids and a metrics table.

**Architecture:** `models/base.py` holds the shared output type, device helpers and the q/k/v CLS-attention hook; each candidate is a small module using it (CLIP has its own hook for `nn.MultiheadAttention`). The script reuses `processing/grouping.py` for K-means, numbering and upsampling.

**Tech Stack:** Python 3.11, torch 2.15, timm 1.0.31, Mark Hamilton's CLIP fork @ `dce9d43`, scikit-learn, Pillow.

**Spec:** `docs/superpowers/specs/2026-11-25-model-b-comparison-design.md`

## Global Constraints

- `requirements.txt` (CI) unchanged. `requirements-ml.txt` adds `timm==1.0.31`, `ftfy`, `regex`, `setuptools<81` (the fork imports `pkg_resources` at runtime) and `clip @ git+https://github.com/mhamilton723/CLIP.git@dce9d43cc659cb9a013745b378c78077fde6aac0`; README says to install with `--no-build-isolation`.
- Model IDs: `clip_vitb16`, `dino_vits16`, `vit_s16_sup`; A stays `dinov2_vits14`.
- DINO v1 hub: `facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a`, `trust_repo=True, skip_validation=True`.
- timm: `vit_small_patch16_224.augreg_in21k_ft_in1k`; the module sets `HF_HUB_DISABLE_XET=1` by `os.environ.setdefault` before loading (xet download failed on this network).
- Everything float32. Grid 14×14 for the candidates, 16×16 for A.
- Output default `sidecar/comparison-out/`, added to root `.gitignore`.
- New tests skip without torch/timm/clip (`pytest.importorskip`).

## Review Focus

1. CLIP on CUDA: `clip.load` returns fp16 weights on CUDA → encoder must call `.float()`. (Task 2 code; untestable on the Mac, noted in README.)
2. A candidate that fails to load (network, missing package) → the script reports it in the table and continues with the others. (Task 3 test with a bogus model ID.)
3. Images that are not 224×224 → script resizes like the capture would not; instead it rejects with a clear message. (Task 3 test.)
4. Encoders called twice in a row give identical output (no stale hook state). (Task 2 test.)
5. MPS memory API missing on older torch → memory reported as `null`, not a crash. (Task 3, by construction: guarded `getattr`.)

---

### Task 1: Shared base, DINOv2 refactor

**Files:** Create `sidecar/models/base.py`; Modify `sidecar/models/dinov2.py`; Test `sidecar/tests/test_dinov2.py` (unchanged, must stay green), `sidecar/tests/test_encoders.py` (new, base part).

**Interfaces — Produces (`models/base.py`):**
- `EncoderOutput` (moved; same fields), `pick_device(pref="auto") -> str` (moved), `synchronize(device: str) -> None`
- `class QKVClsAttention(attn_module)`: registers a forward pre-hook; `.cls_row -> torch.Tensor (B, N)` mean over heads of softmax(q·kᵀ·scale) for query 0; applies `q_norm`/`k_norm` if the module has them; uses `attn.num_heads`, `attn.scale`, `attn.qkv`.
- `to_input(image: np.ndarray, mean, std, device) -> torch.Tensor (1,3,224,224)`
- `finish(tokens: torch.Tensor (N_patches, D), cls_row: torch.Tensor (N,), grid: int) -> EncoderOutput` — L2-normalize tokens, drop CLS from the row, reshape to (grid, grid).

`dinov2.py` keeps `HUB_REPO`, `MODEL_ID`, `Dinov2Encoder`, `RealBackend`, and re-exports `EncoderOutput`, `pick_device` (imports from base) so existing imports keep working.

- [ ] Write `test_encoders.py::test_finish_shapes` (random tokens (196, 8), row (197,) summing to 1 → features unit-norm, attention (14,14), row_sum 1) and `test_qkv_hook_on_toy_module` (a tiny module with `qkv`, `num_heads=2`, `scale` → hook row equals a hand-computed softmax row).
- [ ] Run → fails (no `models.base`).
- [ ] Implement `base.py`; switch `dinov2.py` to it.
- [ ] Run full suite → all green (DINOv2 tests unchanged).
- [ ] Commit `sidecar: share encoder base between models`.

### Task 2: Candidate encoders

**Files:** Create `sidecar/models/dino_v1.py`, `sidecar/models/vit_sup.py`, `sidecar/models/clip_vit.py`, `sidecar/models/candidates.py`; Modify `sidecar/requirements-ml.txt`, root `.gitignore`; Test `sidecar/tests/test_encoders.py`.

**Interfaces — Produces:** each module a class with `model_id`, `patch=16`, `grid=14`, `device`, `encode(image) -> EncoderOutput`, `synchronize()`:
- `DinoV1Encoder(device)`: features = `model.norm(...)` patch tokens from the last block (use `model.get_intermediate_layers(x, n=1)[0][0, 1:]`), attention via `QKVClsAttention(model.blocks[-1].attn)`; ImageNet mean/std.
- `VitSupEncoder(device)`: `model.forward_features(x)[0, model.num_prefix_tokens:]`, hook on `model.blocks[-1].attn`; mean/std from `model.pretrained_cfg`.
- `ClipEncoder(device)`: `clip.load("ViT-B/16", device=device, jit=False)` then `.float()`; features = `get_visual_features(x, include_cls=False)` (1,512,14,14) → (196,512); attention via a pre-hook on `visual.transformer.resblocks[-1].attn` (sequence-first input `(N, B, C)`; q,k from `in_proj_weight`/`in_proj_bias`, 12 heads, scale `head_dim**-0.5`); CLIP mean/std.
- `candidates.py`: `ENCODERS: dict[str, Callable[[str], encoder]]` with `dinov2_vits14` (wrapping `Dinov2Encoder`, `patch=14`, `grid=16`), `clip_vitb16`, `dino_vits16`, `vit_s16_sup`; `load_encoder(model_id, device)`.

- [ ] Add deps to `requirements-ml.txt` and `comparison-out/` to `.gitignore`.
- [ ] Tests, parametrized over the three candidate IDs (module-scope encoder fixture): `test_shapes_and_norms` (features (196, D) unit norm; attention (14,14) ≥ 0; row sum within 1e-3 of 1); `test_deterministic` (two encodes identical); `test_hook_matches_module` — rebuild the last attention block from its weights, compare to the module's real output (atol 1e-3), and its CLS row to the encoder's attention (atol 1e-5); `test_clip_features_match_fork` (encoder features equal `get_visual_features` on the same tensor, normalized).
- [ ] Run → fail (modules missing).
- [ ] Implement the three encoders and `candidates.py`.
- [ ] Run full suite on MPS and with `SEESAW_DEVICE=cpu` → green.
- [ ] Commit `sidecar: add CLIP, DINO v1 and supervised ViT encoders`.

### Task 3: Comparison script

**Files:** Create `sidecar/tools/compare_models.py`; Test `sidecar/tests/test_compare.py`; Modify `sidecar/README.md` (comparison section + fork install note).

**Interfaces — Produces:**
- `analyse(encoder, image, runs: int) -> dict` with keys `model_id, grid, encode_ms {median,min,max}, cluster_ms {...}, memory_mb (float|None), attention {row_sum, finite, peakedness}, labels (224,224 uint8), patch_attention, attention_norm (8,), featup: None`
- `agreement(labels_a, labels_b) -> float` (sklearn `adjusted_rand_score` on flattened maps)
- `render_grid(image, rows: list[dict]) -> PIL.Image` (columns: input, groups, attention heatmap, groups × attention; row label with model ID, ARI, encode median)
- `main(argv)`: `--images` (default `fixtures/images/*.png`), `--models` (default all four), `--device auto`, `--runs 10`, `--out comparison-out`. Rejects non-224×224 images with a message. A model that fails to load becomes a row `{"model_id", "error"}` in the table and is skipped in grids. Writes `<image-stem>_grid.png`, `metrics.md` (header: date, `platform.platform()`, device, torch version, `git rev-parse --short HEAD`), `metrics.json`.

- [ ] Tests: `test_agreement_identical_and_permuted` (same map → 1.0; relabeled map → 1.0; random → < 0.1); `test_render_grid_size` (2 rows → expected pixel size); `test_script_end_to_end` (`main(["--images", image1, "--models", "dinov2_vits14", "vit_s16_sup", "--runs", "1", "--out", tmp])` → grid PNG exists, `metrics.json` has both models with finite numbers); `test_bad_model_reported` (`--models dinov2_vits14 nope` → json has an `error` row for `nope`, still writes A); `test_rejects_wrong_size_image`.
- [ ] Run → fail.
- [ ] Implement.
- [ ] Run full suite → green.
- [ ] Run for real: `python tools/compare_models.py --device mps` on both images, all four models; read the grids and table.
- [ ] Commit `sidecar: add model B comparison script`.

### Task 4: Review and report

- [ ] Gemini whole-branch review (agy, read-only), re-grade, one fix pass with tests.
- [ ] Commit plan + spec. Push and PR only on Emre's OK (the branch is stacked on E1, so the PR waits for #5).
- [ ] Report the comparison results to Emre (grids + table), clearly labeled as Mac numbers.
