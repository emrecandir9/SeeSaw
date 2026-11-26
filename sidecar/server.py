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
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from contract import ContractError, error_body, invalid, parse_request
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
    return value if isinstance(value, str) and value else None


def create_app(backend, mode: str, default_scenario: str = "valid", delay_ms: int = 0) -> FastAPI:
    app = FastAPI(title="See/Saw sidecar", version="1")
    model_lock = threading.Lock()  # one model call at a time (one GPU)

    @app.exception_handler(RequestValidationError)
    async def _missing_parts(request: Request, exc: RequestValidationError) -> JSONResponse:
        missing = sorted({str(e["loc"][-1]) for e in exc.errors()})
        return _error(ContractError(400, "invalid_input", f"missing or invalid form fields: {', '.join(missing)}"))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code in (400, 422):  # e.g. a malformed multipart body
            return _error(ContractError(400, "invalid_input", f"malformed request: {exc.detail}"))
        return await http_exception_handler(request, exc)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "mode": mode, "models": list(backend.models), "device": backend.device}

    @app.post("/probe")
    async def probe(
        image: UploadFile | None = File(default=None),
        metadata: str | None = Form(default=None),
        x_mock_scenario: str | None = Header(default=None),
    ) -> JSONResponse:
        started = time.perf_counter()
        probe_id = _probe_id_from(metadata)
        try:
            missing = [name for name, part in (("image", image), ("metadata", metadata)) if part is None]
            if missing:
                raise invalid(f"missing form fields: {', '.join(missing)}", probe_id)
            if mode == "mock":
                scenario = check_scenario(x_mock_scenario or default_scenario)
            elif x_mock_scenario is not None:
                raise invalid("X-Mock-Scenario is only available in mock mode", probe_id)
            else:
                scenario = "valid"
            req = parse_request(await image.read(), metadata)
            body = await run_in_threadpool(_run, req, scenario)
        except ContractError as err:
            err.probe_id = err.probe_id or probe_id
            return _error(err)
        except Exception:  # anything unexpected is a processing error, never a crash
            log.exception("probe %s failed", probe_id)
            return _error(ContractError(500, "processing_error", "unexpected server error", probe_id))
        body["server_total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return JSONResponse(content=body)

    def _run(req, scenario):
        with model_lock:  # the mock delay queues like a real model call
            if delay_ms:
                time.sleep(delay_ms / 1000)
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
    if not args.mock and (args.scenario != "valid" or args.delay_ms):
        parser.error("--scenario and --delay-ms only work with --mock")
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
