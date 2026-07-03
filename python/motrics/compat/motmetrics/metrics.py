"""Drop-in for ``motmetrics.metrics``, backed by motrics' Rust core.

Only a subset of ``motmetrics.metrics.motchallenge_metrics`` is implemented so
far (the CLEAR/Identity metrics motrics already computes). The rest —
mostly_tracked/partially_tracked/mostly_lost, num_fragmentations, and the
transfer/ascend/migrate switch subtypes — need per-trajectory bookkeeping
motrics' core doesn't do yet; requesting them raises ``NotImplementedError``
rather than silently returning nothing.
"""

from __future__ import annotations

from typing import Any

from motrics.compat.motmetrics.mot import MOTAccumulator

#: Full MOTChallenge metric list, kept identical to ``motmetrics.metrics`` so
#: existing code referencing it doesn't break on import; only :data:`SUPPORTED`
#: can actually be computed today.
motchallenge_metrics = [
    "idf1",
    "idp",
    "idr",
    "recall",
    "precision",
    "num_unique_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "mota",
    "motp",
    "num_transfer",
    "num_ascend",
    "num_migrate",
]

SUPPORTED = (
    "mota",
    "motp",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "recall",
    "precision",
    "num_unique_objects",
    "idf1",
    "idp",
    "idr",
)


def _quiet_divide(a: float, b: float) -> float:
    """``a / b``, returning NaN (not raising) on ``0 / 0``, +-inf on ``x / 0``."""
    if b != 0:
        return a / b
    if a > 0:
        return float("inf")
    if a < 0:
        return float("-inf")
    return float("nan")


def _summarize(acc: MOTAccumulator) -> dict[str, float]:
    clear = acc._clear()
    identity = acc._identity()
    num_detections = clear.num_matches  # motmetrics counts switches as detected too
    errors = clear.num_misses + clear.num_switches + clear.num_false_positives

    num_precision_denom = clear.num_false_positives + num_detections
    num_idf1_denom = identity.num_gt + identity.num_pred
    return {
        "mota": 1.0 - _quiet_divide(errors, clear.num_gt),
        "motp": -clear.motp if num_detections else float("nan"),
        "num_false_positives": clear.num_false_positives,
        "num_misses": clear.num_misses,
        "num_switches": clear.num_switches,
        "recall": _quiet_divide(num_detections, clear.num_gt),
        "precision": _quiet_divide(num_detections, num_precision_denom),
        "num_unique_objects": len(acc._oid_index),
        "idf1": identity.idf1 if num_idf1_denom else float("nan"),
        "idp": identity.idp if (identity.idtp + identity.idfp) else float("nan"),
        "idr": identity.idr if (identity.idtp + identity.idfn) else float("nan"),
    }


class MetricsHost:
    """Compute summary metrics for one or more accumulators.

    Mirrors ``motmetrics.metrics.MetricsHost.compute``'s common usage
    (``mh.compute(acc, metrics=..., name=...)``); the more exotic
    ``compute_overall``/``compute_many``/custom-metric-registration surface of
    the original isn't reproduced.
    """

    def compute(
        self,
        acc: MOTAccumulator,
        metrics: str | list[str] | None = None,
        name: Any = None,
        return_dataframe: bool = True,
    ) -> Any:
        if metrics is None:
            metrics = list(SUPPORTED)
        elif isinstance(metrics, str):
            metrics = [metrics]

        unsupported = [m for m in metrics if m not in SUPPORTED]
        if unsupported:
            raise NotImplementedError(
                f"motrics.compat.motmetrics supports {SUPPORTED}; "
                f"not yet implemented: {unsupported}"
            )

        values = _summarize(acc)
        data = {k: values[k] for k in metrics}
        if not return_dataframe:
            return data

        import pandas as pd

        return pd.DataFrame(data, index=[0 if name is None else name])


def create() -> MetricsHost:
    """Return a new :class:`MetricsHost`."""
    return MetricsHost()
