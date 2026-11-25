"""tools/compare_models.py. Needs the ML packages (skipped in CI)."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from tests.helpers import png_bytes, rgb_image  # noqa: E402
from tools.compare_models import agreement, main, render_grid  # noqa: E402

IMAGE1 = str(Path(__file__).resolve().parent.parent / "fixtures" / "images" / "image1.png")


def test_agreement_identical_and_permuted():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 8, (224, 224)).astype(np.uint8)
    assert agreement(labels, labels) == pytest.approx(1.0)
    permuted = ((labels.astype(int) + 3) % 8).astype(np.uint8)  # same groups, other IDs
    assert agreement(labels, permuted) == pytest.approx(1.0)
    unrelated = rng.integers(0, 8, (224, 224)).astype(np.uint8)
    assert agreement(labels, unrelated) < 0.1


def test_render_grid_size():
    image = rgb_image()
    row = {
        "model_id": "x",
        "labels": np.zeros((224, 224), dtype=np.uint8),
        "patch_attention": np.ones((14, 14), dtype=np.float32) / 197,
        "attention_norm": np.ones(8),
        "ari_vs_a": None,
        "encode_ms": {"median": 1.0},
    }
    grid = render_grid(image, [row, dict(row, model_id="y")])
    cell, label = 224, 28
    assert grid.size == (4 * cell, 2 * (cell + label))


def test_script_end_to_end(tmp_path):
    pytest.importorskip("timm")
    main(["--images", IMAGE1, "--models", "dinov2_vits14", "vit_s16_sup", "--runs", "1", "--out", str(tmp_path)])
    assert (tmp_path / "image1_grid.png").exists()
    assert "vit_s16_sup" in (tmp_path / "metrics.md").read_text()
    data = json.loads((tmp_path / "metrics.json").read_text())
    rows = {r["model_id"]: r for r in data["images"]["image1"]}
    assert set(rows) == {"dinov2_vits14", "vit_s16_sup"}
    assert rows["dinov2_vits14"]["ari_vs_a"] == pytest.approx(1.0)
    assert 0.0 <= rows["vit_s16_sup"]["ari_vs_a"] <= 1.0
    for row in rows.values():
        assert row["encode_ms"]["median"] > 0
        assert row["attention"]["finite"] is True
        assert abs(row["attention"]["row_sum"] - 1) < 1e-3
        assert row["attention"]["peakedness"] >= 1.0
        assert 3 / 196 <= row["attention"]["top3_share"] <= 1.0
        assert row["featup"] is None
    assert data["environment"]["device"]
    assert "Top-3 share" in (tmp_path / "metrics.md").read_text()


def test_bad_model_reported(tmp_path):
    main(["--images", IMAGE1, "--models", "dinov2_vits14", "nope", "--runs", "1", "--out", str(tmp_path)])
    data = json.loads((tmp_path / "metrics.json").read_text())
    rows = {r["model_id"]: r for r in data["images"]["image1"]}
    assert "error" in rows["nope"]
    assert "encode_ms" in rows["dinov2_vits14"]
    assert (tmp_path / "image1_grid.png").exists()


def test_rejects_wrong_size_image(tmp_path):
    bad = tmp_path / "big.png"
    bad.write_bytes(png_bytes(rgb_image(256)))
    with pytest.raises(SystemExit) as info:
        main(["--images", str(bad), "--models", "dinov2_vits14", "--out", str(tmp_path / "out")])
    assert "224x224" in str(info.value)
