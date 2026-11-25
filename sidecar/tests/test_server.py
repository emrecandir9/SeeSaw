import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock import MockBackend
from processing.history import ProbeHistory
from server import create_app
from tests.helpers import metadata, png_bytes, rgb_image

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "contract"
IMAGE = png_bytes(rgb_image())


def make_client(**kwargs) -> TestClient:
    return TestClient(create_app(MockBackend(FIXTURES, ProbeHistory()), "mock", **kwargs))


def post(client, image=IMAGE, meta=None, headers=None):
    files = {"image": ("capture.png", image, "image/png")}
    data = {"metadata": meta or metadata()}
    return client.post("/probe", files=files, data=data, headers=headers or {})


def test_health():
    body = make_client().get("/health").json()
    assert body == {"status": "ok", "mode": "mock", "models": ["dinov2_vits14", "mock_model_b"], "device": "none"}


def test_probe_ok():
    response = post(make_client())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["probe_id"] == "p-1"
    assert body["server_total_ms"] >= 0


def test_scenario_header_failure():
    response = post(make_client(), headers={"X-Mock-Scenario": "failure"})
    assert response.status_code == 500
    assert response.json()["status"] == "error"
    assert response.json()["error"]["code"] == "processing_error"


def test_default_scenario_from_cli():
    response = post(make_client(default_scenario="unavailable"))
    assert response.status_code == 503


def test_unknown_scenario_is_400():
    response = post(make_client(), headers={"X-Mock-Scenario": "explode"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_input"


def test_missing_metadata_part_is_400():
    client = make_client()
    response = client.post("/probe", files={"image": ("capture.png", IMAGE, "image/png")})
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "invalid_input"
    assert body["probe_id"] is None


def test_bad_image_keeps_probe_id():
    response = post(make_client(), image=b"not a png")
    assert response.status_code == 400
    assert response.json()["probe_id"] == "p-1"


def test_unexpected_exception_is_500():
    class Broken:
        models = ["dinov2_vits14"]
        device = "none"

        def handle(self, req, scenario):
            raise RuntimeError("boom")

    client = TestClient(create_app(Broken(), "real"))
    response = post(client)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "processing_error"
    assert response.json()["probe_id"] == "p-1"


def test_concurrent_requests_both_answered():
    client = make_client(delay_ms=200)
    statuses = []

    def worker(i):
        statuses.append(post(client, meta=metadata(probe_id=f"p-{i}")).status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert statuses == [200, 200]
