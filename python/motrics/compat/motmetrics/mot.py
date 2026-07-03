"""``MOTAccumulator``: a drop-in for ``motmetrics.MOTAccumulator``, backed by motrics.

Frame-by-frame ``update(oids, hids, dists, frameid)`` calls are stored (with ids
remapped to the dense integers motrics' core requires) and handed to
:func:`motrics.compute_clear_from_similarity` /
:func:`motrics.compute_identity_from_similarity` on demand, in one call rather
than a per-frame event log.

Not a full reimplementation: unlike real motmetrics, per-frame continuity
("carry forward an established track when possible") is a preference inside
one optimal assignment rather than a hard first pass, though the two agree in
practice. There is also no ``events``/``mot_events`` DataFrame; use
:func:`motrics.compat.motmetrics.metrics.create` for summary metrics.
"""

from __future__ import annotations

import math
from typing import Any, SupportsFloat

import motrics

# motrics wants a similarity with a threshold; motmetrics distances have none
# (only NaN means "do not pair"), so distances are negated and disallowed
# pairs get -inf, with the threshold set to exclude only that.
_DISALLOWED = float("-inf")
_THRESHOLD = -1e300


def _pair_similarity(d: SupportsFloat) -> float:
    v = float(d)
    return -v if v == v else _DISALLOWED  # v == v is False only for NaN


def _to_similarity(dists: Any, n_rows: int, n_cols: int) -> list[list[float]]:
    if n_rows == 0:
        return []
    if n_cols == 0:
        return [[] for _ in range(n_rows)]
    return [[_pair_similarity(d) for d in row] for row in dists]


def _intern(index: dict[Any, int], key: Any) -> int:
    if key not in index:
        index[key] = len(index)
    return index[key]


class MOTAccumulator:
    """Accumulate tracking events frame by frame (see module docstring)."""

    def __init__(
        self, auto_id: bool = False, max_switch_time: float = math.inf
    ) -> None:
        if max_switch_time != math.inf:
            raise NotImplementedError(
                "motrics.compat.motmetrics.MOTAccumulator does not support "
                "max_switch_time yet; open an issue if you need it"
            )
        self.auto_id = auto_id
        self.max_switch_time = max_switch_time
        self._oid_index: dict[Any, int] = {}
        self._hid_index: dict[Any, int] = {}
        self._gt_ids: list[list[int]] = []
        self._pred_ids: list[list[int]] = []
        self._similarity: list[list[list[float]]] = []
        self._frame_ids: list[Any] = []

    def reset(self) -> None:
        """Reset the accumulator to empty state."""
        self._oid_index.clear()
        self._hid_index.clear()
        self._gt_ids.clear()
        self._pred_ids.clear()
        self._similarity.clear()
        self._frame_ids.clear()

    def update(
        self,
        oids: Any,
        hids: Any,
        dists: Any,
        frameid: Any = None,
        vf: str = "",
    ) -> Any:
        """Record one frame's objects/hypotheses and their distance matrix.

        Same contract as ``motmetrics.MOTAccumulator.update``: ``dists`` is an
        ``NxM`` matrix (``N = len(oids)``, ``M = len(hids)``) with ``NaN`` for
        pairs that must never be matched. ``vf`` (per-event debug logging) is
        accepted for signature compatibility but ignored.
        """
        del vf
        oids = list(oids)
        hids = list(hids)

        if frameid is None:
            if not self.auto_id:
                raise AssertionError("auto-id is not enabled")
            frameid = len(self._frame_ids)
        elif self.auto_id:
            raise AssertionError("Cannot provide frame id when auto-id is enabled")

        self._gt_ids.append([_intern(self._oid_index, o) for o in oids])
        self._pred_ids.append([_intern(self._hid_index, h) for h in hids])
        self._similarity.append(_to_similarity(dists, len(oids), len(hids)))
        self._frame_ids.append(frameid)
        return frameid

    def _clear(self) -> motrics.ClearMetrics:
        return motrics.compute_clear_from_similarity(
            self._gt_ids, self._pred_ids, self._similarity, threshold=_THRESHOLD
        )

    def _identity(self) -> motrics.IdentityMetrics:
        return motrics.compute_identity_from_similarity(
            self._gt_ids, self._pred_ids, self._similarity, threshold=_THRESHOLD
        )
