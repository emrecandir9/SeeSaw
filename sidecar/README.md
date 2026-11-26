# Sidecar

Python server that runs the vision models for See/Saw. The Godot client posts a 224 × 224 capture to `POST /probe` and gets back an 8-group label map and per-group attention for each requested model. The contract is in [docs/api.md](../docs/api.md).

Owner: Emre. Status: contract draft, mock mode, and DINOv2 ViT-S/14 with nearest-upsampled labels. FeatUp (E2), matching (E3) and model B (E4) come next.

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

On the lab PC, install the CUDA build of torch 2.15 first, following pytorch.org for the installed CUDA version, then `pip install -r requirements-ml.txt`.

## Run

Mock mode, no model or GPU needed:

```bash
python server.py --mock
```

Real model (downloads the DINOv2 weights, about 90 MB, into the torch hub cache on first start, then warms up):

```bash
python server.py
```

Useful flags: `--host 0.0.0.0` to accept other machines, `--port 8765`, `--device auto|cuda|mps|cpu`, and for mock mode `--scenario NAME` and `--delay-ms N`. See `python server.py --help` and the mock section of [docs/api.md](../docs/api.md).

Check it is up:

```bash
curl http://127.0.0.1:8765/health
```

## Tests

```bash
pytest -q
```

CI installs only `requirements.txt`, so the real-model tests in `tests/test_dinov2.py` are skipped there. Run them locally with `requirements-ml.txt` installed; `SEESAW_DEVICE=cpu pytest -q tests/test_dinov2.py` forces the CPU path.

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
| `models/dinov2.py` | DINOv2 loading, CLS attention hook, real backend |
| `processing/grouping.py` | K-means, group numbering, group attention, upsampling, PNG (numpy only) |
| `processing/history.py` | Centroids of recent probes per session and model |
| `tools/make_fixtures.py` | Fixture generator |

## Notes

- DINOv2 comes from the `facebookresearch/dinov2` torch hub at a pinned commit (`HUB_REPO` in `models/dinov2.py`). At E2 the loader switches to FeatUp's hub so the backbone matches FeatUp's upsampler (plan p.7); see the `TODO(E2)` there.
- Groups are numbered by image-space height (top group = 0) until matching lands.
- Timings measured on a laptop are for debugging only. Reported latencies come from the lab PC.
