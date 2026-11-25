# `/probe` API — v1 draft

**Status: draft, 25 Nov 2026.** Gloria and Emre freeze it as v1 on 27 Nov (plan p.6). The open points are at the end. Change requests go to Emre (owner of `sidecar/`).

The Godot client sends one captured view per request. The sidecar answers with, per requested model, an 8-group label map and the attention of each group.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/probe` | Run the requested models on one capture |
| `GET` | `/health` | Connection check: mode, models, device |

Default address: `http://127.0.0.1:8765` (Link mode, everything on the lab PC). Start the server with `--host 0.0.0.0` to accept laptops or a native Quest build.

## Request: `POST /probe`

`multipart/form-data` with two parts:

| Part | Content |
| --- | --- |
| `image` | PNG, exactly 224 × 224, RGB or RGBA (alpha is dropped). The capture camera's view, top row first. |
| `metadata` | JSON string, fields below |

```json
{
  "schema_version": 1,
  "probe_id": "p-000042",
  "session_id": "s-7f3a",
  "models": ["dinov2_vits14"],
  "match_against": {"dinov2_vits14": "p-000041"},
  "reset_history": false
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | Must be `1` |
| `probe_id` | non-empty string | Unique per probe; echoed unchanged in the response |
| `session_id` | non-empty string | Matching history key. New ID or `reset_history` at scene change |
| `models` | list of unique model IDs | `dinov2_vits14`; add model B only while the lens is on |
| `match_against` | object, optional | Model ID → `probe_id` of the last result the client **applied** for that model, or `null`. Keys for models not requested are ignored |
| `reset_history` | bool, optional (default `false`) | Forget this session's history before processing |

K is fixed at 8 in v1. The client keeps the capture's view/projection matrices and capture depth keyed by `probe_id`; they are not sent.

### Why `match_against`

If the server matched against its own latest output and the client then discarded a late result, the next probe would be matched against colors the user never saw, so colors would break exactly when the system is slow. With `match_against`, matching follows what is on screen. The server keeps centroids for the last 10 probes per session and model; an unknown ID is treated as a reset.

## Response: 200

```json
{
  "schema_version": 1,
  "probe_id": "p-000042",
  "status": "ok",
  "mode": "real",
  "server_total_ms": 61.2,
  "results": [
    {
      "model_id": "dinov2_vits14",
      "width": 224,
      "height": 224,
      "cluster_map_png_base64": "iVBORw0KGgo...",
      "attention": [0.0041, 0.0032, 0.0127, 0.0027, 0.0065, 0.007, 0.0022, 0.001],
      "attention_norm": [0.32, 0.25, 1.0, 0.21, 0.51, 0.55, 0.17, 0.08],
      "upsampler": "nearest",
      "matched_to": null,
      "timings_ms": {"encode": 22.1, "cluster": 9.1, "featup": null, "assign": 0.03, "match": null}
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `mode` | `"real"` or `"mock"`. Mock output must never be presented as real. |
| `server_total_ms` | Time inside the server for this request, upload and network excluded |
| `results` | One entry per requested model, in request order |
| `cluster_map_png_base64` | 8-bit grayscale PNG, 224 × 224. Pixel value = group ID 0–7 (not a color). Top-left origin, x right, y down. |
| `attention[i]` | Last-layer CLS-to-patch attention, averaged over heads, averaged over the patches of group i. Empty group → 0. Raw values (they sum to roughly 1 over all patches, so a group is around 0.004). |
| `attention_norm[i]` | `attention[i] / max(attention)`, in 0..1; all zeros if the max is 0. Ready for brightness. |
| `upsampler` | `"nearest"`: patch labels repeated over 14 × 14 blocks (coarse, does **not** satisfy A4). `"featup_jbu"`: FeatUp pixel labels (from E2). |
| `matched_to` | The `probe_id` these groups were matched against, or `null` when this result is a reset (no `match_against`, unknown ID or `reset_history`). Lets the client detect that colors may change. Always `null` until matching lands (E3, 3 Dec). |
| `timings_ms` | Per-model stage times. `null` = stage did not run. GPU work is synchronized before each time is taken. |

Group IDs before matching exists: groups are numbered by image-space height, as in the paper. Group 0 is the topmost group on screen (smallest mean row index, since rows count downward), so the same image always gives the same IDs.

## Errors

No partial results: either every requested model succeeded, or the response is an error.

```json
{
  "schema_version": 1,
  "probe_id": "p-000042",
  "status": "error",
  "error": {"code": "invalid_input", "message": "image must be 224x224, got 256x256"}
}
```

| HTTP | `code` | When |
| --- | --- | --- |
| 400 | `invalid_input` | Bad metadata, image not a 224 × 224 RGB/RGBA PNG, missing form part, malformed multipart body, unknown mock scenario, `X-Mock-Scenario` sent to a real server |
| 503 | `model_unavailable` | A requested model is not loaded |
| 500 | `processing_error` | Anything that failed while processing |

`probe_id` is `null` when the metadata could not be read.

## `GET /health`

```json
{"status": "ok", "mode": "mock", "models": ["dinov2_vits14", "mock_model_b"], "device": "none"}
```

`device` is `cuda`, `mps` or `cpu` in real mode, `none` in mock mode.

## Mock mode (for Godot work without a GPU)

```bash
python server.py --mock [--scenario NAME] [--delay-ms N]
```

- Validates requests exactly like real mode, then answers from `sidecar/fixtures/contract/`.
- Models: `dinov2_vits14` and `mock_model_b` (a stand-in until model B is chosen on 1 Dec).
- `matched_to` behaves like the real server will: it echoes `match_against` if the mock served that probe in this session, else `null`.
- `--delay-ms` waits before answering, inside the same one-at-a-time lock as a real model call, to test busy state, late results and timeouts. `--scenario` and `--delay-ms` are refused without `--mock`.
- Force a case per request with the header `X-Mock-Scenario` (or for every request with `--scenario`):

| Scenario | Response | What the client should do |
| --- | --- | --- |
| `valid` (default) | 200 | Apply it |
| `invalid_labels` | 200, the first requested model's map contains value 9 | Reject the result, keep the previous overlay |
| `wrong_id` | 200, `probe_id` ≠ request | Discard it |
| `stale` | 200, `matched_to: null` although `match_against` was sent | Apply; colors may change |
| `failure` | 500 `processing_error` | Show the error, keep the previous overlay, allow retry |
| `unavailable` | 503 `model_unavailable` | Same as failure |

The fixtures in `sidecar/fixtures/contract/` are also example responses for client tests. `one_model_valid.json` uses a deliberately asymmetric map (group 0 is a small square in the **top-left** corner) to check image origin and texture Y flip in Godot. `paired_valid.json` holds real DINOv2 output on the saloon test image (model A) and a mirrored synthetic map (model B).

## Conventions

- Image origin top-left, x right, y down, for both the capture PNG and the label map. Document the Godot texture and projection convention against `one_model_valid.json`.
- Sample the label texture with nearest filtering and no color conversion; the value read as 0..1 times 255, rounded, is the group ID.
- The client measures its own end-to-end time separately; `server_total_ms` excludes upload, network and rendering.
- One probe in flight per client. Never apply a response after its request has expired; a response always belongs to its saved capture, never to the headset's current pose.

## Open points for the 27 Nov freeze

1. Metadata as a JSON form field (proposed) or as headers.
2. RGBA accepted with alpha dropped (proposed), or RGB only.
3. `attention_norm` rule (proposed: divide by the max group) and whether the shader uses it directly.
4. Client timeout, set after measuring cold and warm inference on the lab PC.
5. Port (proposed 8765) and host binding for each deployment mode.
6. Godot texture Y convention, checked with the asymmetric fixture.
