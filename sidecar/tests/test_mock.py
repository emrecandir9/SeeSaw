from pathlib import Path

import numpy as np
import pytest

from contract import ContractError, ProbeRequest
from mock import MockBackend
from processing.grouping import decode_label_png
from processing.history import ProbeHistory

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "contract"


def request(probe_id="p-1", models=("dinov2_vits14",), match_against=None, reset=False, session="s-1"):
    return ProbeRequest(
        probe_id=probe_id,
        session_id=session,
        models=list(models),
        match_against=match_against or {},
        reset_history=reset,
        image=np.zeros((224, 224, 3), dtype=np.uint8),
    )


@pytest.fixture
def backend():
    return MockBackend(FIXTURES, ProbeHistory())


def test_valid_one_model(backend):
    body = backend.handle(request(), "valid")
    assert body["probe_id"] == "p-1"
    assert body["mode"] == "mock"
    assert [r["model_id"] for r in body["results"]] == ["dinov2_vits14"]


def test_valid_paired_in_request_order(backend):
    body = backend.handle(request(models=("mock_model_b", "dinov2_vits14")), "valid")
    assert [r["model_id"] for r in body["results"]] == ["mock_model_b", "dinov2_vits14"]


def test_wrong_id(backend):
    assert backend.handle(request(), "wrong_id")["probe_id"] != "p-1"


def test_invalid_labels(backend):
    body = backend.handle(request(), "invalid_labels")
    assert 9 in decode_label_png(body["results"][0]["cluster_map_png_base64"])


def test_stale(backend):
    backend.handle(request("p-0"), "valid")
    body = backend.handle(request("p-1", match_against={"dinov2_vits14": "p-0"}), "stale")
    assert body["results"][0]["matched_to"] is None


def test_failure(backend):
    with pytest.raises(ContractError) as info:
        backend.handle(request(), "failure")
    assert info.value.status == 500
    assert info.value.probe_id == "p-1"


def test_unavailable_scenario(backend):
    with pytest.raises(ContractError) as info:
        backend.handle(request(), "unavailable")
    assert info.value.status == 503


def test_unknown_model(backend):
    with pytest.raises(ContractError) as info:
        backend.handle(request(models=("clip",)), "valid")
    assert info.value.status == 503
    assert info.value.code == "model_unavailable"


def test_matched_to_follows_served_probes(backend):
    backend.handle(request("p-0"), "valid")
    body = backend.handle(request("p-1", match_against={"dinov2_vits14": "p-0"}), "valid")
    assert body["results"][0]["matched_to"] == "p-0"
    unknown = backend.handle(request("p-2", match_against={"dinov2_vits14": "never-served"}), "valid")
    assert unknown["results"][0]["matched_to"] is None


def test_reset_history_forgets(backend):
    backend.handle(request("p-0"), "valid")
    body = backend.handle(request("p-1", match_against={"dinov2_vits14": "p-0"}, reset=True), "valid")
    assert body["results"][0]["matched_to"] is None


def test_history_is_per_session(backend):
    backend.handle(request("p-0", session="a"), "valid")
    body = backend.handle(request("p-1", match_against={"dinov2_vits14": "p-0"}, session="b"), "valid")
    assert body["results"][0]["matched_to"] is None
