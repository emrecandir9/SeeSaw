# Sidecar

Python server that runs the vision models for See/Saw. The Godot client posts a 224 × 224 capture to `POST /probe` and gets back an 8-group label map and per-group attention for each requested model. The contract is in [docs/api.md](../docs/api.md).

Owner: Emre. Status: contract draft, mock mode, DINOv2 ViT-S/14 with FeatUp JBU pixel labels (E2; nearest labels where FeatUp is not installed), model B comparison script (E4). Matching (E3) and paired inference (E4) come next.

## Setup

Python **3.11** (same as CI). From this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Then pick one:

```bash
pip install -r requirements.txt     # mock mode and tests only (Godot developers)
```

```bash
pip install -r requirements-ml.txt  # real model too (lab PC, Emre)
```

`requirements-ml.txt` includes Mark Hamilton's CLIP fork (the one FeatUp uses), whose `setup.py` needs `pkg_resources`. Install it with build isolation off:

```bash
pip install "setuptools<81" wheel
pip install --no-build-isolation -r requirements-ml.txt
```

On the lab PC, install the CUDA build of torch 2.15 first, following pytorch.org for the installed CUDA version, then the two commands above. If Hugging Face downloads fail (seen on some networks), set `HF_HUB_DISABLE_XET=1`.

## Run

Mock mode, no model or GPU needed:

```bash
python server.py --mock
```

Real model with FeatUp pixel labels (lab machine; downloads the DINOv2 weights, about 90 MB, and FeatUp's 2 MB checkpoint into the torch hub cache on first start, then warms up):

```bash
python server.py
```

Without FeatUp (Mac, Windows laptops): the default `--upsampler featup` refuses to start, so choose the coarse labels explicitly:

```bash
python server.py --upsampler nearest
```

Useful flags: `--host 0.0.0.0` to accept other machines, `--port 8765`, `--device auto|cuda|mps|cpu`, `--upsampler featup|nearest`, and for mock mode `--scenario NAME` and `--delay-ms N`. See `python server.py --help` and the mock section of [docs/api.md](../docs/api.md).

Check it is up:

```bash
curl http://127.0.0.1:8765/health
```

## FeatUp (E2, CUDA machines only)

FeatUp's JBU upsampler needs its compiled CUDA kernel, so it installs only where `nvcc` matches torch's CUDA version. After `requirements-ml.txt`:

```bash
pip install --no-deps --no-build-isolation -r requirements-featup.txt
```

On the CREMI machine jessica (RTX 4000 Ada) the environment lives on the local disk: `. /local/mucandy/seesaw/env.sh` activates it and sets `CUDA_HOME=/usr/local/cuda-13.4`, the only CUDA install there with `nvcc`; torch is `2.15.0+cu134` from `https://download.pytorch.org/whl/cu134`.

How a probe is computed (paper §2, plan p.5): DINOv2 patch tokens go through FeatUp's ChannelNorm (a LayerNorm without trained weights); K-means (K=8) runs on those, L2-normalized; FeatUp's JBU upsamples them 16×16 → 256×256 guided by the input image; they are resized to 224×224, L2-normalized, and each pixel takes its nearest patch-level centroid. Group attention stays the mean of the patch attention over each group's patches.

A4 record, printed at startup: `upsampler=featup_jbu execution=cuda kernel checkpoint=<url> sha256=<hash>`. Pinned: FeatUp commit `6b5a6c0e91f75e69194807128dcbc39c3084a30d`, checkpoint `dinov2_jbu_stack_cocostuff.ckpt` (SHA-256 checked before loading, see `models/featup_jbu.py`). Our pinned DINOv2 gives the same features and upsampled output as FeatUp's own hub model (`tests/test_featup.py`, to 1e-4).

## Tests

```bash
pytest -q
```

CI installs only `requirements.txt`, so the real-model tests in `tests/test_dinov2.py` are skipped there. Run them locally with `requirements-ml.txt` installed; `SEESAW_DEVICE=cpu pytest -q tests/test_dinov2.py` forces the CPU path.

## Model B comparison (E4)

```bash
python tools/compare_models.py --device auto
```

Runs DINOv2 (model A) and the candidates CLIP ViT-B/16 (`clip_vitb16`), DINO v1 ViT-S/16 (`dino_vits16`) and supervised ViT-S/16 (`vit_s16_sup`) on `fixtures/images/*.png` (or `--images ...`, each 224 × 224), 10 probes each (`--runs`). Writes `comparison-out/` (git-ignored): one grid per image (input, groups, attention heatmap, groups × attention) and `metrics.md` / `metrics.json` with encode and K-means time, memory, attention checks and agreement with model A (adjusted Rand index). Patch level only; FeatUp comes with E2. All candidates use ImageNet normalization and the token path FeatUp's upsamplers were trained on.

## Fixtures

`fixtures/contract/` holds the contract fixtures used by mock mode and client tests. Regenerate with:

```bash
python tools/make_fixtures.py               # synthetic maps; keeps the real model A in paired_valid.json
```

```bash
python tools/make_fixtures.py --with-real   # also recompute model A with DINOv2 (needs requirements-ml.txt)
```

`fixtures/images/` holds the two test images from the paper's scenes; see `fixtures/images/ATTRIBUTION.md`.

## Layout

| Path | Purpose |
| --- | --- |
| `server.py` | FastAPI app and command line |
| `contract.py` | Request parsing and validation, response bodies |
| `mock.py` | Mock backend and scenarios |
| `models/base.py` | Shared encoder pieces: output type, devices, input normalization, CLS attention hook |
| `models/dinov2.py` | DINOv2 loading and the real backend |
| `models/featup_jbu.py` | FeatUp JBU upsampler: pinned checkpoint, availability check |
| `models/clip_vit.py`, `dino_v1.py`, `vit_sup.py`, `candidates.py` | Model B candidate encoders and the registry |
| `tools/compare_models.py` | Model B comparison script |
| `processing/grouping.py` | K-means, group numbering, group attention, upsampling, PNG (numpy only) |
| `processing/history.py` | Centroids of recent probes per session and model |
| `tools/make_fixtures.py` | Fixture generator |

## Notes

- DINOv2 comes from the `facebookresearch/dinov2` torch hub at a pinned commit (`HUB_REPO` in `models/dinov2.py`); only the upsampler comes from FeatUp. Plan p.7 allows another copy of the backbone if it gives the same features, which `tests/test_featup.py` checks.
- Groups are numbered by image-space height (top group = 0) until matching lands.
- Timings measured on a laptop are for debugging only. Reported latencies come from the lab PC.
