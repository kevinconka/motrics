"""Drop-in for ``motmetrics.distances``, backed by motrics' Rust IoU kernel."""

from __future__ import annotations

from collections.abc import Sequence

import motrics

Bbox = Sequence[float]


def _xywh_to_xyxy(boxes: Sequence[Bbox]) -> list[tuple[float, float, float, float]]:
    out = []
    for b in boxes:
        x, y, w, h = (float(v) for v in b)
        out.append((x, y, x + w, y + h))
    return out


def iou_matrix(
    objs: Sequence[Bbox], hyps: Sequence[Bbox], max_iou: float = 1.0
) -> list[list[float]]:
    """IoU distance matrix (``1 - IoU``) between ``xywh`` rectangles.

    Same contract as ``motmetrics.distances.iou_matrix``: pairs with distance
    above ``max_iou`` (i.e. IoU below ``1 - max_iou``) are ``NaN``.
    """
    objs = list(objs)
    hyps = list(hyps)
    if not objs or not hyps:
        return []
    sims = motrics.iou_matrix(_xywh_to_xyxy(objs), _xywh_to_xyxy(hyps))
    return [[_dist(s, max_iou) for s in row] for row in sims]


def _dist(sim: float, max_iou: float) -> float:
    d = 1.0 - sim
    return d if d <= max_iou else float("nan")


def norm2squared_matrix(
    objs: Sequence[Sequence[float]],
    hyps: Sequence[Sequence[float]],
    max_d2: float = float("inf"),
) -> list[list[float]]:
    """Squared Euclidean distance matrix, pure Python (no motrics core involved).

    Same contract as ``motmetrics.distances.norm2squared_matrix``: pairs
    farther apart than ``max_d2`` are ``NaN``.
    """
    objs = list(objs)
    hyps = list(hyps)
    if not objs or not hyps:
        return []
    out = []
    for o in objs:
        row = []
        for h in hyps:
            d2 = sum((float(oc) - float(hc)) ** 2 for oc, hc in zip(o, h, strict=True))
            row.append(d2 if d2 <= max_d2 else float("nan"))
        out.append(row)
    return out
