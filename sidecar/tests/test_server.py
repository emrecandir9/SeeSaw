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


class CountingBackend:
    """Records how many handle() calls overlap."""

    models = ["dinov2_vits14"]
    device = "none"

    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0
        self.guard = threading.Lock()

    def handle(self, req, scenario):
        import time

        with self.guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(0.1)
        with self.guard:
            self.in_flight -= 1
        return {"schema_version": 1, "probe_id": req.probe_id, "status": "ok", "mode": "real",
                "server_total_ms": 0.0, "results": []}


def _post_concurrently(client, n=3):
    statuses = []
    threads = [
        threading.Thread(target=lambda i=i: statuses.append(post(client, meta=metadata(probe_id=f"p-{i}")).status_code))
        for i in range(n)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return statuses


def test_model_calls_are_serialized():
    backend = CountingBackend()
    statuses = _post_concurrently(TestClient(create_app(backend, "real")))
    assert statuses == [200, 200, 200]
    assert backend.max_in_flight == 1


def test_mock_delay_queues_like_a_model():
    import time

    client = make_client(delay_ms=200)
    started = time.perf_counter()
    assert _post_concurrently(client, 2) == [200, 200]
    assert time.perf_counter() - started >= 0.4


def test_malformed_multipart_is_contract_400():
    response = make_client().post(
        "/probe", content=b"garbage", headers={"Content-Type": "multipart/form-data"}
    )
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "invalid_input"


def test_unknown_scenario_keeps_probe_id():
    response = post(make_client(), headers={"X-Mock-Scenario": "explode"})
    assert response.json()["probe_id"] == "p-1"


def test_missing_image_keeps_probe_id():
    response = make_client().post("/probe", data={"metadata": metadata()})
    assert response.status_code == 400
    assert response.json()["probe_id"] == "p-1"


def test_scenario_header_rejected_in_real_mode():
    client = TestClient(create_app(CountingBackend(), "real"))
    response = post(client, headers={"X-Mock-Scenario": "valid"})
    assert response.status_code == 400
    assert "mock mode" in response.json()["error"]["message"]
    assert response.json()["probe_id"] == "p-1"


def test_default_scenario_ignored_in_real_mode():
    client = TestClient(create_app(CountingBackend(), "real", default_scenario="failure"))
    assert post(client).status_code == 200


@pytest.mark.parametrize("flags", [["--scenario", "failure"], ["--delay-ms", "100"]])
def test_mock_flags_require_mock(flags):
    from server import main

    with pytest.raises(SystemExit) as info:
        main(flags)
    assert info.value.code == 2  # argparse usage error, before any model is loaded


def _hide_featup(monkeypatch):
    import sys

    for name in list(sys.modules):
        if name == "featup" or name.startswith("featup."):
            monkeypatch.delitem(sys.modules, name)


def test_featup_missing_refuses_with_hint(monkeypatch, capsys):
    import sys

    from server import main

    _hide_featup(monkeypatch)
    monkeypatch.setitem(sys.modules, "featup", None)  # import featup -> ImportError
    with pytest.raises(SystemExit):
        main(["--upsampler", "featup"])
    assert "--upsampler nearest" in capsys.readouterr().err


def test_featup_kernel_missing_refuses(monkeypatch, capsys):
    import sys

    from server import main

    import types

    _hide_featup(monkeypatch)
    # A FeatUp package that imports, without its compiled kernel (a partial install).
    monkeypatch.setitem(sys.modules, "featup", types.ModuleType("featup"))
    monkeypatch.setitem(sys.modules, "featup.upsamplers", types.ModuleType("featup.upsamplers"))
    monkeypatch.setitem(sys.modules, "adaptive_conv_cuda_impl", None)
    with pytest.raises(SystemExit):
        main([])  # featup is the default
    err = capsys.readouterr().err
    assert "--upsampler nearest" in err
    assert "adaptive_conv_cuda_impl" in err


def test_mock_ignores_upsampler(monkeypatch):
    import sys

    import uvicorn

    from server import main

    calls = []
    monkeypatch.setitem(sys.modules, "featup", None)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
    main(["--mock"])
    assert calls and calls[0]["port"] == 8765
