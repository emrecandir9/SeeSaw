# G2 async client: first increment

Owner: Gloria. Plan v2 (25 Nov), pages 5–6, 8, 10–11. Target: 1 Dec real single-model loop; A10/A11 remain pending integrated headset/lab-PC evidence.

This branch starts from develop after E1 and the Godot CI pin. It adds isolated network scripts, tests and documentation. No scene, G1 file, overlay, UI or sidecar implementation is edited. The runner creates a temporary Godot project, so this can be reviewed independently of G1 PR #11. E0/E1/G1/B1 are still prerequisites for the integrated loop.

## Consumer interface

Instantiate `godot/scripts/network/probe_client.gd` as a Node. Configure `endpoint`, `timeout_seconds` and `allow_mock` before submission. Mock acceptance defaults to false; explicitly enable it for desktop development. The 15-second timeout is provisional, not a lab measurement.

- `submit(rgb_png, capture_context, models, mock_scenario="") -> String`: returns a new transport probe ID or an empty string if rejected. Supply a 224×224 RGB8/RGBA8 PNG and one or two distinct model IDs. Request model B only while the lens is enabled. Both requested models receive the same image part.
- `result_ready(probe_id, response, capture_context)`: emits a fully validated response, including decoded L8 `labels` images and measured `client_total_ms` (submit through validation; excludes application/render time). Keep `mode`, `upsampler`, model IDs and server timings visible in downstream UI. Nearest upsampling never satisfies A4.
- `acknowledge(probe_id, applied)`: call after the consumer applies all requested model results, or pass false to discard. Only a true acknowledgment advances matching history for the included models. Requests remain blocked until this decision. Do not acknowledge texture receipt as application.
- `reset_session()`: call on scene changes, backend changes or switching mock/real mode. Cancels transport and invalidates pending callbacks/acknowledgments, starts a new session and clears per-model matching history. The consumer must clear its old overlay too.
- `state_changed(state, detail)`: `idle`, `requesting`, `awaiting_apply`. Yernur can use this for Busy/error text. Failures return to idle with a reason, preserve matching history and leave the consumer's last valid overlay alone. Retry is explicit: submit again with a fresh transport ID; there is no automatic retry storm.

G1's capture ID and the transport probe ID are separate: include the capture ID in `capture_context`, along with saved camera transform, view/projection, depth Image and capture timestamp. Values and Resources are deep-copied at submission. Bind replies to this saved context, never the live headset. This separation gives every retry a unique server identity while preserving the original capture identity.

The client can fetch maps for RGB-only desktop development. **B1 must reject surface application when actual capture depth is missing** and call `acknowledge(id, false)` if no result was applied. This client does not pretend to implement capture-depth rendering or overlay acceptance. Scene integration is a follow-up on Gloria's branch after G1/B1 interfaces are ready.

## Validation and lifecycle

The HTTP request is asynchronous and bounded by a deadline and 2 MiB response limit. A generation token rejects callbacks after timeout/reset. Replies must match schema/status/probe ID, requested model order/count, dimensions, mode, matching reference, attention ranges, timing shapes and upsampler. Labels must be 224×224 8-bit grayscale PNG with values 0–7. IHDR is checked before decode to bound image allocations. Validation is all-or-nothing across models. `matched_to: null` is valid and means history reset; it is not a stale transport response.

Run from the repository root with Python dependencies from `sidecar/requirements.txt` installed:

```text
python tools/test_g2.py --godot /path/to/Godot-4.7.2
```

The runner starts two loopback-only mock servers, runs headless Godot in a temporary project, then terminates its servers. Tests cover multipart exchange, asymmetric label origin, frozen nested/image context, busy rejection, apply/discard bookkeeping, per-model history, wrong IDs/labels/HTTP errors, malformed second-model data, reset/late callbacks, timeout with a continuing main loop, explicit retry, mock rejection in real mode, disconnect and reconnect. The existing required Godot CI job now runs these checks even before G1's project is merged.

27 Nov local evidence: Windows laptop, Python 3.12, Godot 4.7.2.stable.official.ed1daf0bf, mock backend: lifecycle suite PASS. This is not measured headset frame-rate or lab performance evidence. CI uses Python 3.11.

## API freeze and outstanding gates

`docs/api.md` remains a draft. This increment uses its current multipart image + metadata form field, port 8765, top-left grayscale labels and per-model max-normalized attention. It does not unilaterally declare the 27 Nov freeze complete. Emre/Gloria must confirm the open contract decisions; Biswarup must verify texture/projection conventions; timeout needs cold/warm lab measurements. Model B is still the 1 Dec team decision.

Before G2 completion: connect G1 captures and B1's depth-aware application; connect Y1 status and end-to-end timing through actual display; test a real backend on the lab PC; measure headset responsiveness during slow inference; record A10/A11 with commit, machine, tester and evidence.
