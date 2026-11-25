import numpy as np
import pytest

from processing.grouping import (
    decode_label_png,
    encode_label_png,
    group_attention,
    normalize_attention,
    relabel_by_height,
    upsample_nearest,
)


def test_relabel_top_group_is_zero():
    labels = np.zeros((16, 16), dtype=int)
    labels[:8] = 5
    labels[8:] = 2
    out = relabel_by_height(labels)
    assert out.dtype == np.uint8
    assert (out[:8] == 0).all()
    assert (out[8:] == 1).all()


def test_relabel_ties_broken_by_column():
    labels = np.zeros((16, 16), dtype=int)
    labels[:, :8] = 6  # left half
    labels[:, 8:] = 1  # right half, same mean row
    out = relabel_by_height(labels)
    assert (out[:, :8] == 0).all()
    assert (out[:, 8:] == 1).all()


def test_relabel_empty_groups_last():
    labels = np.full((16, 16), 4, dtype=int)
    labels[10:] = 6
    out = relabel_by_height(labels)
    assert set(np.unique(out)) <= {0, 1}


def test_group_attention_means_and_empty():
    labels = np.zeros((16, 16), dtype=int)
    labels[8:] = 3
    attention = np.zeros((16, 16))
    attention[:8] = 0.5
    attention[8:12] = 1.0
    attention[12:] = 3.0
    out = group_attention(labels, attention)
    assert out.shape == (8,)
    assert out[0] == pytest.approx(0.5)
    assert out[3] == pytest.approx(2.0)
    assert out[[1, 2, 4, 5, 6, 7]].tolist() == [0.0] * 6


def test_normalize_zero_safe():
    assert normalize_attention(np.zeros(8)).tolist() == [0.0] * 8
    assert normalize_attention(np.array([1.0, 2.0, 4.0])).tolist() == [0.25, 0.5, 1.0]


def test_upsample_blocks():
    labels = np.zeros((16, 16), dtype=int)
    labels[0, 15] = 7
    out = upsample_nearest(labels)
    assert out.shape == (224, 224)
    assert out.dtype == np.uint8
    assert (out[0:14, 210:224] == 7).all()
    assert out.sum() == 7 * 14 * 14


def test_png_roundtrip():
    labels = np.random.default_rng(1).integers(0, 8, (224, 224), dtype=np.uint8)
    np.testing.assert_array_equal(decode_label_png(encode_label_png(labels)), labels)


def test_cluster_patches():
    pytest.importorskip("sklearn")
    from processing.grouping import cluster_patches

    rng = np.random.default_rng(0)
    centers = np.eye(8, 32) * 10
    features = (centers[np.arange(256) % 8] + rng.normal(0, 0.1, (256, 32))).astype(np.float32)
    labels, centroids = cluster_patches(features)
    assert labels.shape == (256,)
    assert centroids.shape == (8, 32)
    assert len(np.unique(labels)) == 8
    labels2, _ = cluster_patches(features)
    np.testing.assert_array_equal(labels, labels2)


def test_cluster_patches_uniform_input():
    pytest.importorskip("sklearn")
    from processing.grouping import cluster_patches

    labels, centroids = cluster_patches(np.ones((256, 32), dtype=np.float32))
    assert labels.shape == (256,)
    assert centroids.shape == (8, 32)
    assert labels.min() >= 0 and labels.max() < 8
