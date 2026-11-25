"""See/Saw sidecar: POST /probe and GET /health.

    python server.py --mock                 # fixtures, no model needed
    python server.py                        # real DINOv2 ViT-S/14 (needs requirements-ml.txt)
    python server.py --help

See docs/api.md for the contract.
"""

import argparse
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from contract import ContractError, error_body, parse_request
from mock import SCENARIOS, check_scenario

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "contract"
log = logging.getLogger("seesaw.sidecar")


def _error(err: ContractError) -> JSONResponse:
    return JSONResponse(status_code=err.status, content=error_body(err.probe_id, err.code, err.message))


def _probe_id_from(metadata: str | None) -> str | None:
    try:
        value = json.loads(metadata or "").get("probe_id")
    except (ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def create_app(backend, mode: str, default_scenario: str = "valid", delay_ms: int = 0) -> FastAPI:
    app = FastAPI(title="See/Saw sidecar", version="1")
    model_lock = threading.Lock()  # one model call at a time (one GPU)

    @app.exception_handler(RequestValidationError)
    async def _missing_parts(request: Request, exc: RequestValidationError) -> JSONResponse:
        missing = sorted({str(e["loc"][-1]) for e in exc.errors()})
        return _error(ContractError(400, "invalid_input", f"missing or invalid form fields: {', '.join(missing)}"))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "mode": mode, "models": list(backend.models), "device": backend.device}

    @app.post("/probe")
    async def probe(
        image: UploadFile = File(...),
        metadata: str = Form(...),
        x_mock_scenario: str | None = Header(default=None),
    ) -> JSONResponse:
        started = time.perf_counter()
        probe_id = _probe_id_from(metadata)
        try:
            scenario = check_scenario(x_mock_scenario or default_scenario)
            req = parse_request(await image.read(), metadata)
            body = await run_in_threadpool(_run, req, scenario)
        except ContractError as err:
            return _error(err)
        except Exception:  # anything unexpected is a processing error, never a crash
            log.exception("probe %s failed", probe_id)
            return _error(ContractError(500, "processing_error", "unexpected server error", probe_id))
        body["server_total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return JSONResponse(content=body)

    def _run(req, scenario):
        if delay_ms:
            time.sleep(delay_ms / 1000)
        with model_lock:
            return backend.handle(req, scenario)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="See/Saw sidecar")
    parser.add_argument("--mock", action="store_true", help="answer from fixtures instead of a real model")
    parser.add_argument("--scenario", default="valid", choices=SCENARIOS, help="default mock scenario")
    parser.add_argument("--delay-ms", type=int, default=0, help="extra delay before each answer (mock testing)")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu (real mode)")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept other machines")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from processing.history import ProbeHistory

    if args.mock:
        from mock import MockBackend

        backend, mode = MockBackend(FIXTURES_DIR, ProbeHistory()), "mock"
    else:
        from models.dinov2 import Dinov2Encoder, RealBackend, pick_device

        backend, mode = RealBackend(Dinov2Encoder(pick_device(args.device)), ProbeHistory()), "real"
        backend.warm_up()

    log.info("mode=%s device=%s models=%s", mode, backend.device, backend.models)
    import uvicorn

    uvicorn.run(create_app(backend, mode, args.scenario, args.delay_ms), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
