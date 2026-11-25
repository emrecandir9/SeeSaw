"""Mock backend: answers /probe from the committed contract fixtures.

Lets the Godot side work without a GPU or a model. Every response says
"mode": "mock".
"""

import copy
import json
from pathlib import Path

import numpy as np

from contract import ContractError, ProbeRequest, ok_body, processing, unavailable
from processing.history import ProbeHistory

SCENARIOS = ("valid", "invalid_labels", "wrong_id", "stale", "failure", "unavailable")
MOCK_MODELS = ("dinov2_vits14", "mock_model_b")

class MockBackend:
    device = "none"

    def __init__(self, fixtures_dir: Path, history: ProbeHistory):
        self.models = list(MOCK_MODELS)
        self.history = history
        self._fixtures = {
            path.stem: json.loads(path.read_text()) for path in Path(fixtures_dir).glob("*.json")
        }

    def handle(self, req: ProbeRequest, scenario: str) -> dict:
        unknown = [m for m in req.models if m not in MOCK_MODELS]
        if unknown:
            raise unavailable(f"model not available in mock mode: {', '.join(unknown)}", req.probe_id)
        if scenario == "failure":
            raise processing("simulated failure (mock scenario 'failure')", req.probe_id)
        if scenario == "unavailable":
            raise unavailable("simulated unavailable model (mock scenario 'unavailable')", req.probe_id)
        if req.reset_history:
            self.history.reset(req.session_id)

        results = [self._result_for(model, req, scenario) for model in req.models]
        for model in req.models:
            self.history.store(req.session_id, model, req.probe_id, np.zeros((8, 1), dtype=np.float32))

        probe_id = req.probe_id + "-WRONG" if scenario == "wrong_id" else req.probe_id
        return ok_body(probe_id, "mock", results, 0.0)

    def _result_for(self, model: str, req: ProbeRequest, scenario: str) -> dict:
        source = self._fixtures["one_model_valid" if req.models == ["dinov2_vits14"] else "paired_valid"]
        result = copy.deepcopy(next(r for r in source["results"] if r["model_id"] == model))
        if scenario == "invalid_labels" and model == req.models[0]:
            bad_map = self._fixtures["invalid_labels"]["results"][0]["cluster_map_png_base64"]
            result["cluster_map_png_base64"] = bad_map  # contains label 9

        wanted = req.match_against.get(model)
        known = wanted is not None and self.history.get(req.session_id, model, wanted) is not None
        result["matched_to"] = wanted if known and scenario != "stale" else None
        return result


def check_scenario(name: str) -> str:
    if name not in SCENARIOS:
        raise ContractError(400, "invalid_input", f"unknown mock scenario '{name}', expected one of {', '.join(SCENARIOS)}")
    return name
