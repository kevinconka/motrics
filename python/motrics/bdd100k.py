"""BDD100K tracking ingest helpers.

Parse the BDD100K tracking JSON format: one file per sequence, a list of
frame objects sorted by ``index``, each with a ``labels`` array of
``{"id", "category", "box2d": {"x1", "y1", "x2", "y2"}, "attributes": {...}}``
entries. Boxes are already ``xyxy``.

BDD100K is box-based in TrackEval (its ``BDD100K`` dataset scores with box
IoU, not the mask kernel), so this reuses the existing box-IoU path. Two
ingest paths mirror ``motrics.kitti``:

- ``load_bdd100k`` — plain box/id/class parsing for tracker results.
- ``load_bdd100k_gt`` + ``preprocess_bdd100k`` — replicates TrackEval's
  ``BDD100K`` preprocessing (eight classes evaluated separately, crowd/
  distractor ground truth pulled out as ignore regions, and unmatched
  predictions mostly inside a crowd-ignore region dropped), for numbers
  matching TrackEval's own reported values.
"""

from __future__ import annotations

import json
from os import PathLike
from typing import Any

from motrics._motrics import match_boxes
from motrics._types import Bbox

# Per-frame detections: frame index -> (ids, boxes, class ids).
_ByFrame = dict[int, tuple[list[int], list[Bbox], list[int]]]
# Per-frame crowd-ignore regions: frame index -> boxes.
_IgnoreByFrame = dict[int, list[Bbox]]

#: BDD100K category -> class id, as used by TrackEval's ``BDD100K``. Includes
#: the three distractor categories (ids 3, 8, 9), which are never evaluated but
#: must be recognised so they can be routed to the ignore regions.
_CLASS_NAME_TO_ID = {
    "pedestrian": 1,
    "rider": 2,
    "other person": 3,
    "car": 4,
    "bus": 5,
    "truck": 6,
    "train": 7,
    "trailer": 8,
    "other vehicle": 9,
    "motorcycle": 10,
    "bicycle": 11,
}
#: The eight evaluated classes (everything except the distractor categories).
_VALID_CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "bus",
    "truck",
    "train",
    "motorcycle",
    "bicycle",
)
#: Ground-truth categories that mark a crowd-ignore region rather than an
#: object to evaluate.
_DISTRACTOR_CLASSES = frozenset({"other person", "trailer", "other vehicle"})
#: Unmatched predictions overlapping a crowd-ignore region by more than this
#: fraction of their own area are dropped.
_IGNORE_REGION_OVERLAP = 0.5


def _read_frames(path: str | PathLike[str]) -> list[dict[str, Any]]:
    """Load a BDD100K sequence JSON and return its frames sorted by ``index``."""
    with open(path, encoding="utf-8") as handle:
        frames: list[dict[str, Any]] = json.load(handle)
    return sorted(frames, key=lambda frame: frame["index"])


def _box2d(ann: dict[str, Any]) -> Bbox:
    box = ann["box2d"]
    return (box["x1"], box["y1"], box["x2"], box["y2"])


def _is_crowd(ann: dict[str, Any]) -> bool:
    return bool(ann.get("attributes", {}).get("Crowd"))


def load_bdd100k(path: str | PathLike[str]) -> _ByFrame:
    """Parse a BDD100K tracking result JSON into per-frame detections.

    Returns:
        A mapping from frame ``index`` to ``(ids, boxes, class ids)``, boxes
        in ``xyxy``. Labels whose category is not a known BDD100K class are
        dropped; every frame present in the file gets an entry even if empty.
    """
    frames: _ByFrame = {}
    for frame in _read_frames(path):
        ids, boxes, classes = frames.setdefault(int(frame["index"]), ([], [], []))
        for ann in frame.get("labels", []):
            class_id = _CLASS_NAME_TO_ID.get(ann["category"])
            if class_id is None:
                continue
            ids.append(int(ann["id"]))
            boxes.append(_box2d(ann))
            classes.append(class_id)
    return frames


def load_bdd100k_gt(
    path: str | PathLike[str],
) -> tuple[_ByFrame, _IgnoreByFrame]:
    """Parse a BDD100K tracking ground-truth JSON.

    Crowd-marked labels (``attributes.Crowd``) and the distractor categories
    (``other person``, ``trailer``, ``other vehicle``) are crowd-ignore
    regions, not objects: they are returned separately rather than mixed into
    the per-frame ground truth, matching TrackEval's ``BDD100K`` loading.
    """
    gt: _ByFrame = {}
    ignore: _IgnoreByFrame = {}
    for frame in _read_frames(path):
        index = int(frame["index"])
        ids, boxes, classes = gt.setdefault(index, ([], [], []))
        regions = ignore.setdefault(index, [])
        for ann in frame.get("labels", []):
            if ann["category"] in _DISTRACTOR_CLASSES or _is_crowd(ann):
                regions.append(_box2d(ann))
                continue
            class_id = _CLASS_NAME_TO_ID.get(ann["category"])
            if class_id is None:
                continue
            ids.append(int(ann["id"]))
            boxes.append(_box2d(ann))
            classes.append(class_id)
    return gt, ignore


def _ioa(box: Bbox, region: Bbox) -> float:
    """Intersection of ``box`` and ``region`` over the area of ``box``."""
    iw = max(min(box[2], region[2]) - max(box[0], region[0]), 0.0)
    ih = max(min(box[3], region[3]) - max(box[1], region[1]), 0.0)
    area = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    return (iw * ih) / area if area > 0 else 0.0


def preprocess_bdd100k(
    gt: _ByFrame,
    pred: _ByFrame,
    cls: str,
    *,
    ignore_regions: _IgnoreByFrame | None = None,
    iou_threshold: float = 0.5,
) -> tuple[list[list[int]], list[list[Bbox]], list[list[int]], list[list[Bbox]]]:
    """Align and filter ground truth/predictions exactly as TrackEval's
    ``BDD100K`` does before scoring a single class.

    Per frame: keep only ``cls`` ground truth and ``cls`` predictions, match
    every kept ground-truth box against every kept prediction, and drop
    unmatched predictions that lie mostly inside a crowd-ignore region.
    Unlike KITTI 2D-box there is no distractor-match removal and no ground-truth
    removal — the eight BDD100K classes are simply evaluated independently.

    Args:
        gt: Ground truth from :func:`load_bdd100k_gt` (crowd/distractor labels
            already routed to ``ignore_regions``).
        pred: Predictions from :func:`load_bdd100k`.
        cls: Class to evaluate, one of the eight BDD100K classes.
        ignore_regions: Crowd-ignore regions from :func:`load_bdd100k_gt`.
        iou_threshold: Threshold for the gt/prediction match.

    Returns:
        ``(gt_ids, gt_boxes, pred_ids, pred_boxes)``, ready for
        ``compute_clear`` / ``compute_identity`` / ``compute_hota``.
    """
    if cls not in _VALID_CLASSES:
        raise ValueError(f"unknown class {cls!r}, expected one of {_VALID_CLASSES}")
    cls_id = _CLASS_NAME_TO_ID[cls]
    ignore_regions = ignore_regions or {}

    gt_ids: list[list[int]] = []
    gt_boxes: list[list[Bbox]] = []
    pred_ids: list[list[int]] = []
    pred_boxes: list[list[Bbox]] = []
    for frame in sorted(set(gt) | set(pred) | set(ignore_regions)):
        g_ids, g_boxes, g_classes = gt.get(frame, ([], [], []))
        p_ids, p_boxes, p_classes = pred.get(frame, ([], [], []))
        regions = ignore_regions.get(frame, [])

        g_idx = [i for i, c in enumerate(g_classes) if c == cls_id]
        p_idx = [i for i, c in enumerate(p_classes) if c == cls_id]
        g_ids_c = [g_ids[i] for i in g_idx]
        g_boxes_c = [g_boxes[i] for i in g_idx]
        p_ids_c = [p_ids[i] for i in p_idx]
        p_boxes_c = [p_boxes[i] for i in p_idx]

        matched_pred: set[int] = set()
        if g_ids_c and p_ids_c:
            m = match_boxes(g_boxes_c, p_boxes_c, iou_threshold, "hungarian")
            matched_pred = {pi for _, pi in m.matches}

        drop_pred = {
            pi
            for pi in range(len(p_ids_c))
            if pi not in matched_pred
            and any(
                _ioa(p_boxes_c[pi], region) > _IGNORE_REGION_OVERLAP
                for region in regions
            )
        }

        gt_ids.append(list(g_ids_c))
        gt_boxes.append(list(g_boxes_c))
        pred_ids.append([p_ids_c[i] for i in range(len(p_ids_c)) if i not in drop_pred])
        pred_boxes.append(
            [p_boxes_c[i] for i in range(len(p_boxes_c)) if i not in drop_pred]
        )
    return gt_ids, gt_boxes, pred_ids, pred_boxes
