# FeatUp pixel labels for model A — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real mode returns FeatUp JBU pixel labels for `dinov2_vits14` (`upsampler: "featup_jbu"`), with an explicit `--upsampler nearest` path where FeatUp is not installed.

**Architecture:** Our pinned DINOv2 + attention hook stay. Patch tokens go through FeatUp's ChannelNorm (a no-affine LayerNorm); K-means runs on those, L2-normalized. In featup mode FeatUp's `JBUStack` (weights from the pinned checkpoint, SHA-256 checked) upsamples them to 256², they are resized to 224², L2-normalized, and each pixel takes its nearest centroid. Group attention stays the per-patch group mean.

**Tech Stack:** Python 3.11 (CI) / 3.13 (jessica), torch 2.15, FeatUp @ `6b5a6c0e91f75e69194807128dcbc39c3084a30d` (CUDA build, jessica only), numpy, scikit-learn.

**Spec:** `docs/superpowers/specs/2026-11-25-featup-model-a-design.md`

## Global Constraints

- Contract unchanged: maps 224 × 224, IDs 0–7, `upsampler` is `"nearest"` or `"featup_jbu"`, `timings_ms` keys `encode, cluster, featup, assign, match`.
- `attention[8]` = group mean last-layer CLS-to-patch attention over each group's patches (paper §2, plan p.6).
- FeatUp commit `6b5a6c0e91f75e69194807128dcbc39c3084a30d`; checkpoint `https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dinov2_jbu_stack_cocostuff.ckpt`, SHA-256 `438dad4dedb3c3ba4b7fa13d1c8eb465dafd323dadc039afd04b15178ea7b38c`.
- `--upsampler` default `featup`; if FeatUp cannot load the server exits and names `--upsampler nearest`. No silent fallback.
- CI installs only `requirements.txt`; torch tests skip without torch, FeatUp tests skip without `featup` + CUDA.
- No `/health` change before the 27 Nov freeze.

## Review Focus

1. Uniform / blank image in featup mode: every pixel ties; must still return valid IDs 0–7 and finite attention (test in Task 2).
2. FeatUp installed but its compiled extension missing (`import adaptive_conv_cuda_impl` fails): same clear refusal as FeatUp absent (test in Task 3).
3. `--mock` together with the default `--upsampler featup` on a machine without FeatUp: mock must still start (test in Task 3).
4. Checkpoint already cached but corrupt / different: SHA-256 mismatch must refuse, not load (test in Task 2).
5. Groups present in patches but not in pixels (or the reverse): attention stays finite, empty groups 0.0 (covered by the uniform-image test and the existing empty-group test).

---

### Task 1: ChannelNorm features and nearest-centroid assignment

**Files:**
- Modify: `sidecar/models/base.py`, `sidecar/models/dinov2.py`, `sidecar/processing/grouping.py`
- Test: `sidecar/tests/test_grouping.py`, `sidecar/tests/test_encoders.py`, `sidecar/tests/test_dinov2.py`

**Interfaces:**
- Produces: `channel_norm(tokens: torch.Tensor) -> torch.Tensor` (layer_norm over the last dim, no affine, eps 1e-5); `assign_nearest(features: np.ndarray (N, D), centroids: np.ndarray (K, D)) -> np.ndarray (N,) uint8` (smallest squared Euclidean distance, lowest index on ties); `Dinov2Encoder.encode` features = L2-normalized `channel_norm(x_norm_patchtokens)`.

- [ ] **Step 1: Failing tests**
  - `test_assign_nearest_picks_closest`: centroids `[[1,0],[0,1],[-1,0]]`, features `[[0.9,0.1],[0.1,0.9],[-1,0.2]]` → `[0,1,2]`, dtype uint8.
  - `test_assign_nearest_ties_take_lowest_index`: two identical centroids → label 0.
  - `test_channel_norm_matches_featup_layernorm`: random (256, 384) → equals `torch.nn.LayerNorm(384)(x)` (default init) within 1e-6.
  - `test_features_follow_channel_norm` (test_dinov2.py, needs torch): `encoder.encode(image1).features` equals L2-normalized `channel_norm(forward_features(x)["x_norm_patchtokens"][0])` within 1e-5.
- [ ] **Step 2:** `cd sidecar && .venv/bin/pytest -q tests/test_grouping.py tests/test_encoders.py tests/test_dinov2.py` → the four new tests FAIL (ImportError / mismatch).
- [ ] **Step 3:** Implement the three interfaces.
- [ ] **Step 4:** Same command → PASS; full `pytest -q` green.
- [ ] **Step 5:** Commit `sidecar: cluster DINOv2 on FeatUp's channel-normalized features`.

### Task 2: FeatUp JBU in the real backend

**Files:**
- Create: `sidecar/models/featup_jbu.py`, `sidecar/requirements-featup.txt`, `sidecar/tests/test_featup.py`
- Modify: `sidecar/models/base.py` (EncoderOutput), `sidecar/models/dinov2.py`

**Interfaces:**
- Consumes: Task 1's `channel_norm`, `assign_nearest`.
- Produces:
  - `featup_jbu.FeatUpUnavailable(RuntimeError)`; `check_available() -> None` (imports `featup.upsamplers`; on any ImportError raises FeatUpUnavailable whose message contains `--upsampler nearest` and the install command); `load_jbu(device: str) -> tuple[torch.nn.Module, str]` (JBUStack(384) in eval mode with `strict=True` upsampler weights; returns `"cuda kernel"` on CUDA, `"cpp fallback"` otherwise); `verify_checkpoint(path) -> None` raises FeatUpUnavailable on SHA mismatch.
  - `EncoderOutput.upsample: Callable[[], np.ndarray] | None = None` → (224·224, 384) float32, L2-normalized rows (row-major pixels).
  - `Dinov2Encoder(device, upsampler="nearest")`; attributes `upsampler`, `execution_path` (None in nearest mode).
  - `RealBackend.handle` in featup mode: `labels = assign_nearest(out.upsample(), ordered).reshape(224, 224)`, result `upsampler="featup_jbu"`, `timings_ms.featup` set.
- `requirements-featup.txt`: `featup @ git+https://github.com/mhamilton723/FeatUp.git@6b5a6c0e91f75e69194807128dcbc39c3084a30d`, with the comment `pip install --no-deps --no-build-isolation -r requirements-featup.txt` (CUDA toolkit with nvcc matching torch's CUDA).

- [ ] **Step 1: Failing tests** in `tests/test_featup.py`
  - CI/Mac (torch only): `test_checkpoint_hash_mismatch_refuses(tmp_path)`: a file with other bytes → `verify_checkpoint` raises FeatUpUnavailable mentioning the expected hash.
  - jessica (`pytest.importorskip("featup")`, skip unless CUDA):
    - `test_matches_featup_hub`: our low-res (ChannelNorm'd, 1×384×16×16) and 256² JBU output equal FeatUp hub `dinov2` (`trust_repo=True`, pinned commit) `up.model(x)` / `up(x)` within atol 1e-4.
    - `test_featup_probe`: `/probe` on image1 → `upsampler == "featup_jbu"`, map 224², IDs ⊂ 0–7, `timings_ms.featup > 0`, attention finite, and some 14×14 block contains ≥ 2 IDs.
    - `test_featup_uniform_image`: a constant gray image → valid map and finite attention.
    - `test_execution_path_is_cuda_kernel`: `encoder.execution_path == "cuda kernel"`.
- [ ] **Step 2:** Mac: `pytest -q tests/test_featup.py` → hash test FAILS (ImportError), the rest SKIP. jessica: all FAIL.
- [ ] **Step 3:** Implement. JBU guidance is the same normalized input `x`; resize with `F.interpolate(..., (224, 224), mode="bilinear", align_corners=False)`; checkpoint downloaded to `$TORCH_HOME/hub/checkpoints/` with `torch.hub.download_url_to_file`, SHA checked before `torch.load(weights_only=False)`. Warm-up also runs the upsampler.
- [ ] **Step 4:** Mac and jessica (`. /local/mucandy/seesaw/env.sh`, rsync the tree): `pytest -q` → all green; FeatUp tests run on jessica.
- [ ] **Step 5:** Commit `sidecar: FeatUp JBU pixel labels for model A`.

### Task 3: `--upsampler` flag, startup record, docs

**Files:**
- Modify: `sidecar/server.py`, `sidecar/README.md`
- Test: `sidecar/tests/test_server.py`

**Interfaces:**
- Consumes: `featup_jbu.check_available`, `Dinov2Encoder(device, upsampler)`, `encoder.execution_path`.
- Produces: `--upsampler {featup,nearest}` (default `featup`); in real mode with `featup`, `check_available()` runs **before** the backbone loads; on FeatUpUnavailable `main` exits via `parser.error(<message>)`; startup log line names upsampler, execution path and checkpoint SHA (A4). `--mock` ignores `--upsampler`.

- [ ] **Step 1: Failing tests**
  - `test_featup_missing_refuses_with_hint(monkeypatch, capsys)`: `monkeypatch.setitem(sys.modules, "featup", None)` → `main(["--upsampler", "featup"])` raises SystemExit; stderr contains `--upsampler nearest`.
  - `test_featup_extension_missing_refuses(monkeypatch, capsys)`: `sys.modules["adaptive_conv_cuda_impl"] = None` (FeatUp package importable or not) → same.
  - `test_mock_ignores_upsampler(monkeypatch)`: patch `uvicorn.run` to record; `main(["--mock"])` with `sys.modules["featup"] = None` starts (run called).
- [ ] **Step 2:** `pytest -q tests/test_server.py` → FAIL (unknown argument).
- [ ] **Step 3:** Implement; README: run section (`--upsampler`), FeatUp install on jessica (env.sh, CUDA 13.4, `requirements-featup.txt`), A4 record (checkpoint URL + SHA, execution path from the log), Mac uses `--upsampler nearest`.
- [ ] **Step 4:** Full suite green on Mac and jessica; on jessica `python server.py --port 8766` starts and logs `featup_jbu … cuda kernel`; one curl `/probe` returns `featup_jbu`; stop the server.
- [ ] **Step 5:** Commit `sidecar: --upsampler flag and FeatUp startup record`.

### Task 4: Review and hand-off

- [ ] Gemini whole-branch review (read-only; inputs copied to the scratchpad), re-grade, one fix pass with tests.
- [ ] Commit spec and plan (`docs: add FeatUp model A design and plan`).
- [ ] Report to Emre: rulings, deferred minors, jessica timings (encode / cluster / featup / assign). Push and PR only on Emre's OK.
