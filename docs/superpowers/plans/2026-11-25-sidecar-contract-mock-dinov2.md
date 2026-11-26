# Sidecar day 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI sidecar that answers `POST /probe` from fixtures (`--mock`) or from a frozen DINOv2 ViT-S/14 (real mode), plus the v1 contract draft in `docs/api.md`.

**Architecture:** `contract.py` parses and validates requests and builds response bodies. `processing/` holds numpy-only grouping helpers and the centroid history, so CI can test them. `mock.py` and `models/dinov2.py` each provide a backend with the same `handle(req, scenario) -> dict` method; `server.py` wires one of them into FastAPI behind a single-call lock.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, python-multipart, numpy, Pillow, pytest, httpx (test client); ML extras: torch, torchvision, scikit-learn.

**Spec:** `docs/superpowers/specs/2026-11-25-sidecar-contract-mock-dinov2-design.md`

## Global Constraints

- Python 3.11 (CI). numpy is capped at `2.4.x` because numpy 2.5 requires Python ≥ 3.12.
- `sidecar/requirements.txt` must stay light (no torch, no scikit-learn); CI runs `pip install -r requirements.txt pytest` then `pytest -q` in `sidecar/`.
- `requirements-ml.txt` starts with `-r requirements.txt`.
- `SCHEMA_VERSION = 1`, `K = 8`, `IMAGE_SIZE = 224`, `PATCH = 14`, `GRID = 16`.
- Model IDs: `dinov2_vits14` (real and mock), `mock_model_b` (mock only).
- Hub: `facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8`, entry `dinov2_vits14`. Loader carries `# TODO(E2): load through FeatUp's hub (plan p.7)`.
- Default bind `127.0.0.1:8765`. Status codes: 400 `invalid_input`, 503 `model_unavailable`, 500 `processing_error`.
- Cluster map: 8-bit grayscale PNG, values 0–7, top-left origin.
- `upsampler` is `"nearest"` in real mode today; `featup`/`match` timings are `null`; `matched_to` is always `null` in real mode today.
- Nothing under `godot/` is touched. No weights committed.

## Review Focus

1. Bytes that are not a PNG, or a truncated PNG → 400 `invalid_input`, never 500. (Task 1 test.)
2. Wrong image size, grayscale or palette PNG → 400; RGBA accepted with alpha dropped. (Task 1 tests.)
3. Missing `image` or `metadata` part → 400 with the contract error body, not FastAPI's default 422. (Task 3 test.)
4. Two requests at once → both answered, model calls serialized, no crash. (Task 3 test.)
5. A uniform image (fewer than 8 distinct patch features) → still 8 attention values, empty groups get 0, no crash. (Task 2 test for the helpers, Task 4 test end to end.)

---

### Task 1: Package skeleton and contract

**Files:**
- Create: `sidecar/requirements.txt`, `sidecar/requirements-ml.txt`, `sidecar/pytest.ini`, `sidecar/contract.py`, `sidecar/processing/__init__.py`, `sidecar/models/__init__.py`
- Test: `sidecar/tests/test_contract.py`

**Interfaces:**
- Produces (`contract.py`):
  - constants `SCHEMA_VERSION, K, IMAGE_SIZE, PATCH, GRID`
  - `class ContractError(Exception)` with attributes `status: int, code: str, message: str, probe_id: str | None`; helpers `invalid(msg, probe_id=None)`, `unavailable(msg, probe_id)`, `processing(msg, probe_id)` returning `ContractError`
  - `@dataclass ProbeRequest(probe_id: str, session_id: str, models: list[str], match_against: dict[str, str | None], reset_history: bool, image: np.ndarray)`; `image` is `(224, 224, 3) uint8`
  - `parse_request(image_bytes: bytes, metadata_json: str) -> ProbeRequest` (raises `ContractError` 400)
  - `error_body(probe_id: str | None, code: str, message: str) -> dict`
  - `build_result(model_id: str, labels: np.ndarray, attention: Sequence[float], upsampler: str, matched_to: str | None, timings: dict[str, float | None]) -> dict` (uses Task 2's `normalize_attention` and `encode_label_png`; implement those two in this task's commit if you prefer, Task 2 owns their tests)
  - `ok_body(probe_id: str, mode: str, results: list[dict], server_total_ms: float) -> dict`

Rules `parse_request` enforces: metadata is a JSON object; `schema_version == 1`; `probe_id`, `session_id` non-empty strings; `models` a non-empty list of unique strings; `match_against` optional, object of string → string|null (keys not in `models` are ignored); `reset_history` optional bool, default false. Image: decodes with Pillow as PNG, size exactly 224×224, mode `RGB` or `RGBA` (RGBA → drop alpha); anything else → 400 with a message naming the problem. `probe_id` is attached to the error when metadata parsed far enough to read it.

- [ ] **Step 1: requirements.** `requirements.txt`: `fastapi==0.165.0`, `uvicorn==0.54.0`, `python-multipart==0.0.34`, `numpy==2.4.6`, `pillow==13.0.0`, `httpx==0.28.1`. `requirements-ml.txt`: `-r requirements.txt`, `torch==2.15.0`, `torchvision==0.29.2`, `scikit-learn==1.9.1`. `pytest.ini`: `[pytest]` with `pythonpath = .` and `testpaths = tests`. Create a local 3.11 venv (`uv venv -p 3.11 sidecar/.venv`, `uv pip install -r requirements.txt pytest`); fix pins if resolution fails.
- [ ] **Step 2: failing tests** in `test_contract.py`, using a `png_bytes(array, mode)` helper:
  - `test_parses_valid_request`: RGB 224×224 + full metadata → fields match, `image.shape == (224, 224, 3)`, `image.dtype == np.uint8`
  - `test_rgba_drops_alpha`: RGBA input → shape `(224, 224, 3)`, RGB values unchanged
  - `test_defaults`: no `match_against`/`reset_history` → `{}` and `False`
  - `test_rejects` parametrized over: not-PNG bytes, truncated PNG, 256×256, grayscale `L`, palette `P`, metadata not JSON, metadata a list, `schema_version: 2`, empty `probe_id`, `models: []`, duplicate models, `match_against` value is int → each raises `ContractError` with `status == 400` and `code == "invalid_input"`
  - `test_error_keeps_probe_id`: valid metadata + bad image → `err.probe_id == "p-1"`
  - `test_build_result_shape`: labels all 3, attention `[0,0,0,2,0,0,0,0]` → keys exactly `model_id,width,height,cluster_map_png_base64,attention,attention_norm,upsampler,matched_to,timings_ms`; `attention_norm[3] == 1.0`; decoding the PNG gives the labels back
- [ ] **Step 3: run** `cd sidecar && .venv/bin/pytest tests/test_contract.py -q` → fails (module missing).
- [ ] **Step 4: implement** `contract.py` per the interface above.
- [ ] **Step 5: run** the same command → all pass.
- [ ] **Step 6: commit** `sidecar: add contract parsing and response builders`.

### Task 2: Grouping helpers and history

**Files:**
- Create: `sidecar/processing/grouping.py`, `sidecar/processing/history.py`
- Test: `sidecar/tests/test_grouping.py`, `sidecar/tests/test_history.py`

**Interfaces:**
- Produces (`processing/grouping.py`, numpy only at import time):
  - `relabel_by_height(patch_labels: np.ndarray, k: int = K) -> np.ndarray` — input `(16,16)` ints in `[0,k)`, output `(16,16) uint8` where new ID 0 is the group with the smallest mean row; ties by mean column, then old ID; empty groups take the highest IDs
  - `group_attention(patch_labels: np.ndarray, patch_attention: np.ndarray, k: int = K) -> np.ndarray` — `(k,) float64` mean per group, empty → 0.0
  - `normalize_attention(attention: np.ndarray) -> np.ndarray` — divide by max; all zeros if max ≤ 0
  - `upsample_nearest(patch_labels: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray` — each patch repeated over a 14×14 block, `(224,224) uint8`
  - `encode_label_png(labels: np.ndarray) -> str` / `decode_label_png(b64: str) -> np.ndarray` — 8-bit `L` PNG, base64 ASCII
  - `cluster_patches(features: np.ndarray, k: int = K, seed: int = 0) -> tuple[np.ndarray, np.ndarray]` — `(256,)` labels and `(k, D)` centroids; imports scikit-learn inside the function; `KMeans(n_clusters=k, n_init=10, random_state=seed)`
- Produces (`processing/history.py`):
  - `class ProbeHistory(max_per_key: int = 10)` with `store(session_id, model_id, probe_id, centroids: np.ndarray) -> None`, `get(session_id, model_id, probe_id) -> np.ndarray | None`, `reset(session_id) -> None`. Oldest entry per `(session_id, model_id)` drops after 10.

- [ ] **Step 1: failing tests.**
  - `test_relabel_top_group_is_zero`: rows 0–7 old label 5, rows 8–15 old label 2 → rows 0–7 become 0, rows 8–15 become 1
  - `test_relabel_empty_groups_last`: only old labels {4, 6} present → output values ⊆ {0, 1}
  - `test_group_attention_means_and_empty`: two groups with known values → exact means; absent groups 0.0; result length 8
  - `test_normalize_zero_safe`: all zeros → all zeros; `[1, 2, 4]` → `[0.25, 0.5, 1.0]`
  - `test_upsample_blocks`: label at patch (row 0, col 15) = 7 → pixels `[0:14, 210:224] == 7`, shape `(224,224)`, dtype uint8
  - `test_png_roundtrip`: random labels 0–7 → identical after encode/decode
  - `test_cluster_patches` (`pytest.importorskip("sklearn")`): 256 points in 8 well-separated blobs → 8 distinct labels, centroids shape `(8, D)`, same result on a second call
  - `test_history_cap_and_reset`: store 11 probes → first gone, last present; `reset` clears only that session
- [ ] **Step 2: run** `.venv/bin/pytest tests/test_grouping.py tests/test_history.py -q` → fails.
- [ ] **Step 3: implement** both modules.
- [ ] **Step 4: run** → pass (cluster test skipped without sklearn).
- [ ] **Step 5: commit** `sidecar: add grouping helpers and probe history`.

### Task 3: Test images, fixtures, mock backend and server

**Files:**
- Create: `sidecar/fixtures/images/image1.png`, `image2.png`, `ATTRIBUTION.md`; `sidecar/tools/make_fixtures.py`; `sidecar/fixtures/contract/{one_model_valid,paired_valid,invalid_labels,wrong_id,stale_match_against,service_failure}.json`; `sidecar/mock.py`; `sidecar/server.py`
- Test: `sidecar/tests/test_fixtures.py`, `sidecar/tests/test_mock.py`, `sidecar/tests/test_server.py`

**Interfaces:**
- Consumes: Task 1 contract, Task 2 grouping/history.
- Produces:
  - `mock.py`: `SCENARIOS = ("valid", "invalid_labels", "wrong_id", "stale", "failure", "unavailable")`, `MOCK_MODELS = ("dinov2_vits14", "mock_model_b")`, `class MockBackend(fixtures_dir: Path, history: ProbeHistory)` with `models: list[str]`, `device = "none"`, `handle(req: ProbeRequest, scenario: str) -> dict`
  - `server.py`: `create_app(backend, mode: str, default_scenario: str = "valid", delay_ms: int = 0) -> FastAPI`; `main(argv: list[str] | None = None) -> None` with flags `--mock`, `--scenario`, `--delay-ms`, `--device` (default `auto`), `--host` (default `127.0.0.1`), `--port` (default `8765`)
  - `tools/make_fixtures.py`: `synthetic_map() -> np.ndarray` (224×224 uint8, asymmetric) and CLI `python tools/make_fixtures.py [--with-real]`

Images: center-crop `~/Downloads/image1.png` and `image2.png` to a square, resize to 224×224 (Lanczos), save RGB PNG. `ATTRIBUTION.md` text as in the spec's "Image attribution" section.

Synthetic map: eight groups laid out so no flip or rotation maps it onto itself, e.g. a 3×3-ish block layout with group 0 a small square marker at the top-left corner and group 7 only along the bottom-right; attention values fixed (e.g. `[0.009, 0.002, ..., 0.004]`).

Fixtures (probe_id `"p-fixture"`, session `"s-fixture"`, `mode: "mock"`):
- `one_model_valid`: `dinov2_vits14` with the synthetic map, `upsampler: "nearest"`, `matched_to: null`
- `paired_valid`: `dinov2_vits14` = real DINOv2 on `image1` (only written with `--with-real`; Task 4 generates it), `mock_model_b` = synthetic map flipped left-right
- `invalid_labels`: as one_model_valid but the map contains value 9
- `wrong_id`: as one_model_valid with `probe_id: "p-fixture-WRONG"`
- `stale_match_against`: as one_model_valid with `matched_to: null` and a `"_note"` explaining the request had sent `match_against`
- `service_failure`: `error_body("p-fixture", "processing_error", "simulated failure")`, plus `"_status": 500`

Mock `handle` rules: unknown model → `unavailable` (503). Scenario `failure` → `processing` (500); `unavailable` → 503. Otherwise pick results by requested models from `paired_valid` (or `one_model_valid` when only `dinov2_vits14` is requested), using the scenario's fixture for `invalid_labels`/`wrong_id`/`stale`; set `probe_id` to the request's (`wrong_id`: request id + `"-WRONG"`); in `valid`, `matched_to = match_against[model]` if `history.get(...)` knows it, else null; store each served probe in history (zero centroids); honor `reset_history`. Until Task 4 writes `paired_valid.json`, Task 3 generates it with the synthetic map for both models so the mock works; Task 4 overwrites model A with real output.

Server rules: scenario from `X-Mock-Scenario` header, else `default_scenario`; unknown scenario → 400. `delay_ms` sleeps (in the worker thread) before handling. Model call runs via `run_in_threadpool` under a `threading.Lock`. `ContractError` → JSON `error_body` with its status; any other exception → 500 `processing_error`; `RequestValidationError` (missing parts) → 400 `invalid_input`. Sets `server_total_ms` on success. `GET /health` → `{"status": "ok", "mode", "models", "device"}`.

- [ ] **Step 1: images.** Write the two cropped images and `ATTRIBUTION.md`; open both to check the crop.
- [ ] **Step 2: failing tests.**
  - `test_fixtures.py::test_fixture_files_valid` parametrized over the five result fixtures: each result decodes to `(224,224)`, 8 finite `attention`, `attention_norm` max is 1.0 or all zero, values ≤ 7 except in `invalid_labels` where `9` is present
  - `test_fixtures.py::test_synthetic_map_asymmetric`: map differs from its left-right flip, up-down flip and 180° rotation; contains all 8 IDs; `map[0, 0] == 0`
  - `test_mock.py`: `valid` one-model echoes probe_id and returns 1 result; paired returns 2 results in request order; `wrong_id` changes probe_id; `invalid_labels` map contains 9; `stale` gives `matched_to` null; `failure` raises status 500; `unknown_model` raises 503; second valid probe with `match_against` = first probe_id → `matched_to` = first probe_id; after `reset_history` → null
  - `test_server.py` (FastAPI `TestClient` over `create_app(MockBackend(...), "mock")`): `/health` reports mock and both models; POST valid → 200 with `server_total_ms`; header `X-Mock-Scenario: failure` → 500 body with `status: "error"`; missing `metadata` part → 400 `invalid_input`; bad image → 400 with probe_id; unknown scenario header → 400; two concurrent POSTs from threads with `delay_ms=200` → both 200
- [ ] **Step 3: run** `.venv/bin/pytest -q` → new tests fail.
- [ ] **Step 4: implement** `tools/make_fixtures.py`, run it (`python tools/make_fixtures.py`), then `mock.py`, `server.py`.
- [ ] **Step 5: run** `.venv/bin/pytest -q` → all pass. Smoke test: `python server.py --mock` and `curl` `/health` and one `/probe` with `image1.png`.
- [ ] **Step 6: commit** `sidecar: add mock server, fixtures and test images`.

### Task 4: Real DINOv2 backend

**Files:**
- Create: `sidecar/models/dinov2.py`
- Modify: `sidecar/server.py` (real mode wiring), `sidecar/fixtures/contract/paired_valid.json` (regenerate)
- Test: `sidecar/tests/test_dinov2.py` (module-level `pytest.importorskip("torch")` and `("sklearn")`)

**Interfaces:**
- Consumes: Task 1 `ProbeRequest`, `build_result`, `ok_body`, `unavailable`; Task 2 helpers and `ProbeHistory`.
- Produces:
  - `HUB_REPO`, `MODEL_ID = "dinov2_vits14"`
  - `pick_device(pref: str = "auto") -> str` — cuda, then mps, then cpu
  - `@dataclass EncoderOutput(features: np.ndarray, patch_attention: np.ndarray, attention_row_sum: float)` — features `(256, 384) float32` L2-normalized rows; patch_attention `(16,16) float32`
  - `class Dinov2Encoder(device: str)` with `encode(image: np.ndarray) -> EncoderOutput` and `synchronize() -> None`
  - `class RealBackend(encoder: Dinov2Encoder, history: ProbeHistory)` with `models = ["dinov2_vits14"]`, `device`, `handle(req, scenario) -> dict`

Attention: a forward pre-hook on `model.blocks[-1].attn` captures its input `x` (B, N, C); recompute with the module's own weights:

```python
B, N, C = x.shape
qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
q, k = qkv[0], qkv[1]                                  # (B, heads, N, d)
weights = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
cls_row = weights[:, :, 0, :].mean(1)                  # (B, N), mean over heads
patch_attention = cls_row[0, 1:]                       # drop CLS itself; 256 values
attention_row_sum = cls_row[0].sum()                   # ≈ 1.0, checked in tests
```

Input: uint8 → float in [0,1] → ImageNet mean `(0.485, 0.456, 0.406)` / std `(0.229, 0.224, 0.225)`. Features from `forward_features(x)["x_norm_patchtokens"]`, then L2-normalized. `model.eval()`, `torch.inference_mode()`.

`handle`: requested model not `dinov2_vits14` → 503. `reset_history` → `history.reset`. Timings (synchronize before each stop): `encode` (encode), `cluster` (`cluster_patches` + `relabel_by_height` + reorder centroids to new IDs), `assign` (`group_attention` + `upsample_nearest`), `featup: None`, `match: None`. Store centroids in history; `matched_to = None`. `scenario` other than `valid` → 400 (`X-Mock-Scenario` is mock only).

`server.py main`: without `--mock`, build `RealBackend(Dinov2Encoder(pick_device(args.device)), ProbeHistory())`, warm it up with one encode of a black image before serving, log device and warm-up time.

- [ ] **Step 1: install ML deps** into the venv (`uv pip install -r requirements-ml.txt`).
- [ ] **Step 2: failing tests.**
  - `test_attention_row_sums_to_one`: encode `image1` → `abs(attention_row_sum - 1) < 1e-3`; `patch_attention.shape == (16,16)`; all ≥ 0
  - `test_features_normalized`: row norms within `1e-4` of 1, shape `(256, 384)`
  - `test_deterministic_labels`: `handle` twice on `image1` → identical cluster maps
  - `test_real_probe_end_to_end`: `TestClient` over real backend, POST `image1` and `image2` → 200, one result, `upsampler == "nearest"`, `timings_ms["featup"] is None`, map values ⊆ 0–7, 8 finite attention values
  - `test_uniform_image`: all-black image → 200, 8 attention values, no exception
  - `test_unknown_model_503`: models `["mock_model_b"]` → 503
- [ ] **Step 3: run** `.venv/bin/pytest tests/test_dinov2.py -q` → fails.
- [ ] **Step 4: implement** `models/dinov2.py` and real-mode wiring.
- [ ] **Step 5: run** the full suite on MPS and with `--device cpu` (env `SEESAW_DEVICE=cpu` read by the test fixture) → pass. Record warm per-stage timings from one run for the PR description.
- [ ] **Step 6: regenerate** `python tools/make_fixtures.py --with-real` → `paired_valid.json` model A is real; look at the map overlaid on `image1` (save a preview PNG to the scratchpad, not the repo) to check it looks like the paper's groups.
- [ ] **Step 7: commit** `sidecar: add DINOv2 ViT-S/14 backend with CLS attention`.

### Task 5: Documentation and pull request

**Files:**
- Create: `docs/api.md`, `sidecar/README.md`
- Add: the spec and this plan (already written)

- [ ] **Step 1: `docs/api.md`** — the contract section of the spec, written for Gloria: request, response, errors, `/health`, mock scenarios and `--delay-ms`, conventions (origin, nearest sampling, no color conversion), and the six open points for 27 Nov. State "Draft — freeze as v1 on 27 Nov".
- [ ] **Step 2: `sidecar/README.md`** — setup (Python 3.11 venv, light vs ML requirements, Windows activate command), run mock, run real, run tests, where weights are cached, the `TODO(E2)` FeatUp note.
- [ ] **Step 3: verify** clean install in a fresh venv with only `requirements.txt` + pytest → `pytest -q` passes with ML tests skipped (what CI does).
- [ ] **Step 4: commit** `docs: add /probe v1 draft, sidecar README, design and plan`.
- [ ] **Step 5: push** `git push -u origin emre/E1-sidecar-baseline`; open a PR into `develop` with the template filled: "Part of #<E1>", scenes changed: none, tested on Mac M5 (MPS + CPU, mock), measured timings, acceptance: A3 (partial), reviewer request to Gloria for the contract. Wait for CI green.
