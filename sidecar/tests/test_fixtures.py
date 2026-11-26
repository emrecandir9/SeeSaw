import json
import math
from pathlib import Path

import numpy as np
import pytest

from processing.grouping import decode_label_png
from tools.make_fixtures import synthetic_map

CONTRACT = Path(__file__).resolve().parent.parent / "fixtures" / "contract"
RESULT_FIXTURES = ["one_model_valid", "paired_valid", "invalid_labels", "wrong_id", "stale_match_against"]


@pytest.mark.parametrize("name", RESULT_FIXTURES)
def test_fixture_files_valid(name):
    body = json.loads((CONTRACT / f"{name}.json").read_text())
    assert body["schema_version"] == 1
    assert body["status"] == "ok"
    assert body["mode"] == "mock"
    assert body["results"]
    for result in body["results"]:
        labels = decode_label_png(result["cluster_map_png_base64"])
        assert labels.shape == (224, 224)
        assert (result["width"], result["height"]) == (224, 224)
        assert len(result["attention"]) == 8
        assert all(math.isfinite(v) for v in result["attention"])
        norm = result["attention_norm"]
        assert max(norm) == 1.0 or max(norm) == 0.0
        if name == "invalid_labels":
            assert 9 in labels
        else:
            assert labels.max() <= 7


def test_paired_has_both_models():
    body = json.loads((CONTRACT / "paired_valid.json").read_text())
    assert [r["model_id"] for r in body["results"]] == ["dinov2_vits14", "mock_model_b"]


def test_wrong_id_fixture_differs():
    body = json.loads((CONTRACT / "wrong_id.json").read_text())
    assert body["probe_id"] != "p-fixture"


def test_service_failure_fixture():
    body = json.loads((CONTRACT / "service_failure.json").read_text())
    assert body["_status"] == 500
    assert body["status"] == "error"
    assert body["error"]["code"] == "processing_error"


def test_synthetic_map_asymmetric():
    labels = synthetic_map()
    assert labels.shape == (224, 224)
    assert labels.dtype == np.uint8
    assert set(np.unique(labels)) == set(range(8))
    assert labels[0, 0] == 0
    for variant in (np.fliplr(labels), np.flipud(labels), np.rot90(labels, 2)):
        assert not np.array_equal(variant, labels)
