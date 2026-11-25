"""Generate the contract fixtures in fixtures/contract/.

    python tools/make_fixtures.py              # synthetic maps only (no torch needed)
    python tools/make_fixtures.py --with-real  # paired fixture's model A from real DINOv2 on image1

Without --with-real, an existing real model A in paired_valid.json is kept.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIDECAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIDECAR))

from contract import build_result, error_body, ok_body  # noqa: E402

CONTRACT_DIR = SIDECAR / "fixtures" / "contract"
IMAGES_DIR = SIDECAR / "fixtures" / "images"
PROBE_ID = "p-fixture"
MOCK_TIMINGS = {"encode": 0.0, "cluster": 0.0, "featup": None, "assign": 0.0, "match": None}
SYNTHETIC_ATTENTION = [0.009, 0.002, 0.003, 0.006, 0.008, 0.004, 0.001, 0.005]


def synthetic_map() -> np.ndarray:
    """224x224 map with no flip or 180-degree symmetry; group 0 marks the top-left corner."""
    labels = np.zeros((224, 224), dtype=np.uint8)
    labels[:56, :112] = 1
    labels[:56, 112:] = 2
    labels[56:140, :75] = 3
    labels[56:140, 75:150] = 4
    labels[56:140, 150:] = 5
    labels[140:, :150] = 6
    labels[140:, 150:] = 7
    labels[:28, :28] = 0  # top-left marker
    return labels


def synthetic_result(model_id: str = "dinov2_vits14") -> dict:
    return build_result(model_id, synthetic_map(), SYNTHETIC_ATTENTION, "nearest", None, MOCK_TIMINGS)


def model_b_result() -> dict:
    return build_result(
        "mock_model_b", np.fliplr(synthetic_map()).copy(), SYNTHETIC_ATTENTION[::-1], "nearest", None, MOCK_TIMINGS
    )


def real_result() -> dict:
    """Model A from real DINOv2 on image1, relabeled as a mock fixture."""
    from PIL import Image

    from contract import ProbeRequest
    from models.dinov2 import Dinov2Encoder, RealBackend, pick_device
    from processing.history import ProbeHistory

    image = np.asarray(Image.open(IMAGES_DIR / "image1.png").convert("RGB"), dtype=np.uint8)
    backend = RealBackend(Dinov2Encoder(pick_device()), ProbeHistory())
    req = ProbeRequest(PROBE_ID, "s-fixture", ["dinov2_vits14"], {}, False, image)
    result = backend.handle(req, "valid")["results"][0]
    result["timings_ms"] = dict(MOCK_TIMINGS)
    return result


def body(results: list[dict], probe_id: str = PROBE_ID) -> dict:
    return ok_body(probe_id, "mock", results, 0.0)


def write(name: str, data: dict) -> None:
    (CONTRACT_DIR / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote fixtures/contract/{name}.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-real", action="store_true", help="compute paired model A with real DINOv2")
    args = parser.parse_args(argv)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    write("one_model_valid", body([synthetic_result()]))

    paired_path = CONTRACT_DIR / "paired_valid.json"
    if args.with_real:
        model_a = real_result()
    elif paired_path.exists():
        model_a = json.loads(paired_path.read_text())["results"][0]
    else:
        model_a = synthetic_result()
    write("paired_valid", body([model_a, model_b_result()]))

    bad = synthetic_map()
    bad[100:120, 100:120] = 9
    invalid = build_result("dinov2_vits14", bad, SYNTHETIC_ATTENTION, "nearest", None, MOCK_TIMINGS)
    write("invalid_labels", body([invalid]))

    write("wrong_id", body([synthetic_result()], probe_id=PROBE_ID + "-WRONG"))

    stale = body([synthetic_result()])
    stale["_note"] = (
        "The request sent match_against {'dinov2_vits14': 'p-unknown'}; the server did not know that probe, "
        "so it treated the result as a reset: matched_to is null and colors may change."
    )
    write("stale_match_against", stale)

    failure = error_body(PROBE_ID, "processing_error", "simulated failure")
    failure["_status"] = 500
    write("service_failure", failure)


if __name__ == "__main__":
    main()
