import numpy as np

from processing.history import ProbeHistory


def test_store_and_get():
    history = ProbeHistory()
    centroids = np.ones((8, 4))
    history.store("s", "m", "p-0", centroids)
    np.testing.assert_array_equal(history.get("s", "m", "p-0"), centroids)
    assert history.get("s", "other", "p-0") is None
    assert history.get("s", "m", "unknown") is None


def test_history_cap_and_reset():
    history = ProbeHistory(max_per_key=10)
    for i in range(11):
        history.store("s", "m", f"p-{i}", np.full((8, 4), i))
        history.store("t", "m", f"p-{i}", np.full((8, 4), i))
    assert history.get("s", "m", "p-0") is None
    assert history.get("s", "m", "p-1") is not None
    assert history.get("s", "m", "p-10") is not None
    history.reset("s")
    assert history.get("s", "m", "p-10") is None
    assert history.get("t", "m", "p-10") is not None
