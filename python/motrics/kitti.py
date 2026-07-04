"""KITTI 2D-box tracking ingest helpers.

Parse the KITTI tracking label/result format: one space-separated row per
object per frame, columns ``frame, track_id, type, truncated, occluded,
alpha, left, top, right, bottom, height, width, length, x, y, z, ry,
[score]``.

Boxes (``left, top, right, bottom``) are already ``xyxy`` — no conversion
needed, unlike MOTChallenge's ``xywh``.

Two ingest paths mirroring ``motrics.motchallenge``:

- ``load_kitti`` — plain box/id/class parsing for tracker results.
- ``load_kitti_gt`` + ``preprocess_kitti`` — replicates TrackEval's
  ``Kitti2DBox`` preprocessing (per-class evaluation, distractor-class and
  occluded/truncated ground truth dropped, small or ignore-region-covered
  unmatched predictions dropped), for numbers matching TrackEval's own
  reported values.
"""

from __future__ import annotations

from os import PathLike

from motrics._motrics import match_boxes

# A bounding box in xyxy format: (x1, y1, x2, y2).
Bbox = tuple[float, float, float, float]
# Per-frame tracker detections: frame number -> (ids, boxes, class ids).
_ByFrame = dict[int, tuple[list[int], list[Bbox], list[int]]]
# Per-frame ground truth with the extra columns preprocessing needs:
# frame number -> (ids, boxes, class ids, truncation, occlusion).
_GtByFrame = dict[int, tuple[list[int], list[Bbox], list[int], list[int], list[int]]]
# Per-frame "DontCare" crowd-ignore regions.
_IgnoreByFrame = dict[int, list[Bbox]]

#: KITTI class name -> class id, as used by TrackEval's ``Kitti2DBox``.
_CLASS_NAME_TO_ID = {
    "car": 1,
    "van": 2,
    "truck": 3,
    "pedestrian": 4,
    "person": 5,
    "cyclist": 6,
    "tram": 7,
    "misc": 8,
    "dontcare": 9,
}
#: For each evaluated class, the class that is a distractor for it: `person`
#: (a sitting person) for `pedestrian`, `van` for `car`.
_DISTRACTOR_FOR = {"pedestrian": "person", "car": "van"}
#: GT boxes with occlusion beyond this level are treated as distractors.
_MAX_OCCLUSION = 2
#: GT boxes with truncation beyond this level are treated as distractors.
_MAX_TRUNCATION = 0
#: Unmatched predictions this short (or shorter) are dropped.
_MIN_HEIGHT = 25.0
#: Unmatched predictions overlapping a "DontCare" region by more than this
#: fraction of their own area are dropped.
_IGNORE_REGION_OVERLAP = 0.5


def _read_rows(path: str | PathLike[str]) -> list[list[str]]:
    with open(path, encoding="utf-8") as handle:
        return [line.split() for raw in handle if (line := raw.strip())]


def _box(parts: list[str]) -> Bbox:
    left, top, right, bottom = (float(parts[i]) for i in range(6, 10))
    return (left, top, right, bottom)


def load_kitti(path: str | PathLike[str]) -> _ByFrame:
    """Parse a KITTI tracking result file into per-frame detections.

    Returns:
        A mapping from frame number to ``(ids, boxes, class ids)``, boxes in
        ``xyxy``. Rows with a negative track id are dropped.
    """
    frames: _ByFrame = {}
    for parts in _read_rows(path):
        track_id = int(float(parts[1]))
        if track_id < 0:
            continue
        class_id = _CLASS_NAME_TO_ID.get(parts[2].lower())
        if class_id is None:
            continue
        ids, boxes, classes = frames.setdefault(int(float(parts[0])), ([], [], []))
        ids.append(track_id)
        boxes.append(_box(parts))
        classes.append(class_id)
    return frames


def load_kitti_gt(
    path: str | PathLike[str],
) -> tuple[_GtByFrame, _IgnoreByFrame]:
    """Parse a KITTI tracking ``label_02`` ground-truth file, keeping the
    class, truncation and occlusion columns :func:`preprocess_kitti` needs.

    "DontCare" rows are crowd-ignore regions, not objects: they are returned
    separately rather than mixed into the per-frame ground truth.
    """
    gt: _GtByFrame = {}
    ignore: _IgnoreByFrame = {}
    for parts in _read_rows(path):
        frame = int(float(parts[0]))
        if parts[2].lower() == "dontcare":
            ignore.setdefault(frame, []).append(_box(parts))
            continue
        track_id = int(float(parts[1]))
        if track_id < 0:
            continue
        class_id = _CLASS_NAME_TO_ID.get(parts[2].lower())
        if class_id is None:
            continue
        ids, boxes, classes, truncation, occlusion = gt.setdefault(
            frame, ([], [], [], [], [])
        )
        ids.append(track_id)
        boxes.append(_box(parts))
        classes.append(class_id)
        truncation.append(int(float(parts[3])))
        occlusion.append(int(float(parts[4])))
    return gt, ignore


def _ioa(box: Bbox, region: Bbox) -> float:
    """Intersection of ``box`` and ``region`` over the area of ``box``."""
    iw = max(min(box[2], region[2]) - max(box[0], region[0]), 0.0)
    ih = max(min(box[3], region[3]) - max(box[1], region[1]), 0.0)
    area = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    return (iw * ih) / area if area > 0 else 0.0


def preprocess_kitti(
    gt: _GtByFrame,
    ignore_regions: _IgnoreByFrame,
    pred: _ByFrame,
    cls: str,
    *,
    iou_threshold: float = 0.5,
) -> tuple[list[list[int]], list[list[Bbox]], list[list[int]], list[list[Bbox]]]:
    """Align and filter ground truth/predictions exactly as TrackEval's
    ``Kitti2DBox`` does before scoring a single class.

    Per frame: keep only ``cls`` and its distractor-class ground truth (plus
    ``cls`` predictions), match every kept ground-truth box against every
    kept prediction, and drop predictions matched to a distractor-class,
    occluded, or truncated ground-truth box. Remaining unmatched predictions
    are dropped if they are too short or mostly covered by a "DontCare"
    region. Ground truth is then reduced to ``cls`` boxes that are neither
    occluded nor truncated.

    Args:
        gt: Ground truth from :func:`load_kitti_gt` (unfiltered — every class
            and occlusion/truncation level must still be present).
        ignore_regions: "DontCare" regions from :func:`load_kitti_gt`.
        pred: Predictions from :func:`load_kitti`.
        cls: Class to evaluate, ``"pedestrian"`` or ``"car"``.
        iou_threshold: Threshold for the gt/prediction match.

    Returns:
        ``(gt_ids, gt_boxes, pred_ids, pred_boxes)``, ready for
        ``compute_clear`` / ``compute_identity`` / ``compute_hota``.
    """
    if cls not in _DISTRACTOR_FOR:
        raise ValueError(f'unknown class {cls!r}, expected "pedestrian" or "car"')
    cls_id = _CLASS_NAME_TO_ID[cls]
    distractor_id = _CLASS_NAME_TO_ID[_DISTRACTOR_FOR[cls]]

    gt_ids: list[list[int]] = []
    gt_boxes: list[list[Bbox]] = []
    pred_ids: list[list[int]] = []
    pred_boxes: list[list[Bbox]] = []
    for frame in sorted(set(gt) | set(pred) | set(ignore_regions)):
        g_ids, g_boxes, g_classes, g_trunc, g_occ = gt.get(frame, ([], [], [], [], []))
        p_ids, p_boxes, p_classes = pred.get(frame, ([], [], []))
        regions = ignore_regions.get(frame, [])

        g_idx = [i for i, c in enumerate(g_classes) if c in (cls_id, distractor_id)]
        p_idx = [i for i, c in enumerate(p_classes) if c == cls_id]
        g_ids_c = [g_ids[i] for i in g_idx]
        g_boxes_c = [g_boxes[i] for i in g_idx]
        g_classes_c = [g_classes[i] for i in g_idx]
        g_trunc_c = [g_trunc[i] for i in g_idx]
        g_occ_c = [g_occ[i] for i in g_idx]
        p_ids_c = [p_ids[i] for i in p_idx]
        p_boxes_c = [p_boxes[i] for i in p_idx]

        drop_pred = set()
        matched_pred = set()
        if g_ids_c and p_ids_c:
            m = match_boxes(g_boxes_c, p_boxes_c, iou_threshold, "hungarian")
            for gi, pi in m.matches:
                matched_pred.add(pi)
                if (
                    g_classes_c[gi] == distractor_id
                    or g_occ_c[gi] > _MAX_OCCLUSION
                    or g_trunc_c[gi] > _MAX_TRUNCATION
                ):
                    drop_pred.add(pi)

        for pi in range(len(p_ids_c)):
            if pi in matched_pred:
                continue
            box = p_boxes_c[pi]
            if box[3] - box[1] <= _MIN_HEIGHT or any(
                _ioa(box, region) > _IGNORE_REGION_OVERLAP for region in regions
            ):
                drop_pred.add(pi)

        keep_gt = [
            i
            for i in range(len(g_ids_c))
            if g_classes_c[i] == cls_id
            and g_occ_c[i] <= _MAX_OCCLUSION
            and g_trunc_c[i] <= _MAX_TRUNCATION
        ]
        gt_ids.append([g_ids_c[i] for i in keep_gt])
        gt_boxes.append([g_boxes_c[i] for i in keep_gt])
        pred_ids.append([p_ids_c[i] for i in range(len(p_ids_c)) if i not in drop_pred])
        pred_boxes.append(
            [p_boxes_c[i] for i in range(len(p_boxes_c)) if i not in drop_pred]
        )
    return gt_ids, gt_boxes, pred_ids, pred_boxes
