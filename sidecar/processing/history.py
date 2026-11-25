"""Centroids of recent probes, per session and model, for temporal matching (E3)."""

from collections import OrderedDict

import numpy as np


class ProbeHistory:
    def __init__(self, max_per_key: int = 10):
        self.max_per_key = max_per_key
        self._entries: dict[tuple[str, str], OrderedDict[str, np.ndarray]] = {}

    def store(self, session_id: str, model_id: str, probe_id: str, centroids: np.ndarray) -> None:
        entries = self._entries.setdefault((session_id, model_id), OrderedDict())
        entries[probe_id] = centroids
        entries.move_to_end(probe_id)
        while len(entries) > self.max_per_key:
            entries.popitem(last=False)

    def get(self, session_id: str, model_id: str, probe_id: str) -> np.ndarray | None:
        return self._entries.get((session_id, model_id), {}).get(probe_id)

    def reset(self, session_id: str) -> None:
        for key in [key for key in self._entries if key[0] == session_id]:
            del self._entries[key]
