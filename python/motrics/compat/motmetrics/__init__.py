"""Drop-in for `py-motmetrics <https://github.com/cheind/py-motmetrics>`_, backed
by motrics' Rust core.

Swap the import and keep the rest of your evaluation code as-is::

    import motrics.compat.motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for gt_boxes, pred_boxes, gt_ids, pred_ids in sequence:
        dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.SUPPORTED, name="acc")

See :mod:`motrics.compat.motmetrics.metrics` for which metrics are supported
today, and :mod:`motrics.compat.motmetrics.mot` for how ``MOTAccumulator``
differs from the original internally. ``metrics.create().compute()`` returns a
``pandas.DataFrame`` when ``return_dataframe=True`` (the default, matching
motmetrics); install the ``motrics[compat]`` extra for pandas.
"""

from __future__ import annotations

from motrics.compat.motmetrics import distances, metrics
from motrics.compat.motmetrics.mot import MOTAccumulator

__all__ = ["MOTAccumulator", "distances", "metrics"]
