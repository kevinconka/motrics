"""KITTI-3D (AB3DMOT) tracking ingest helpers.

KITTI's tracking label/result format already carries a 3D box (dimensions,
location, rotation_y) on the same rows the 2D adapter reads for its 2D box.
AB3DMOT's KITTI-3D-MOT extension scores those rows with 3D IoU instead of 2D
IoU; everything else is the official KITTI devkit convention the 2D adapter
already implements.

TrackEval has no KITTI-3D dataset class, so there is nothing to build a
parity test against the way the other adapters do. Instead, this reuses
``motrics.kitti``'s own TrackEval-validated preprocessing rules (per-class
evaluation, person/van distractors, occlusion/truncation thresholds,
min-height and DontCare-region filtering for unmatched predictions) and only
changes the box representation and similarity kernel: 3D box, 3D IoU
(:func:`motrics.iou_3d`) instead of 2D IoU. See ``tests/test_kitti_3d_preprocessing.py``
for a scenario built to reduce to the exact same keep/drop decisions as the
TrackEval-validated 2D case.

Two ingest paths mirroring ``motrics.kitti``:

- ``load_kitti_3d`` -- plain box/id/class parsing for tracker results.
- ``load_kitti_3d_gt`` + ``preprocess_kitti_3d`` -- the same filtering as
  ``preprocess_kitti``, scored by 3D IoU. Like ``motrics.kitti_mots``, returns
  a precomputed similarity matrix rather than raw boxes: the core
  ``compute_*`` functions have no 3D-box overload, only the
  ``compute_*_from_similarity`` ones.
"""

from __future__ import annotations

from os import PathLike

from motrics._motrics import iou_3d_matrix, match_boxes_3d
from motrics._types import Bbox, Box3d

# Per-frame tracker detections: frame number -> (ids, 3D boxes, 2D boxes,
# class ids). The 2D box is kept alongside the 3D one for the min-height /
# DontCare-overlap filtering below, which stays in 2D box space.
_ByFrame = dict[int, tuple[list[int], list[Box3d], list[Bbox], list[int]]]
# Per-frame ground truth with the extra columns preprocessing needs: frame
# number -> (ids, 3D boxes, 2D boxes, class ids, truncation, occlusion).
_GtByFrame = dict[
    int,
    tuple[list[int], list[Box3d], list[Bbox], list[int], list[int], list[int]],
]
# Per-frame "DontCare" crowd-ignore regions. 2D only: DontCare rows carry no
# meaningful 3D box in the KITTI format.
_IgnoreByFrame = dict[int, list[Bbox]]

#: KITTI class name -> class id, as used by TrackEval's ``Kitti2DBox`` (and
#: therefore this adapter too -- see the module docstring).
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
#: Unmatched predictions this short (2D box height) or shorter are dropped.
_MIN_HEIGHT = 25.0
#: Unmatched predictions overlapping a "DontCare" region (2D IoA) by more
#: than this fraction of their own area are dropped.
_IGNORE_REGION_OVERLAP = 0.5


def _read_rows(path: str | PathLike[str]) -> list[list[str]]:
    with open(path, encoding="utf-8") as handle:
        return [line.split() for raw in handle if (line := raw.strip())]


def _box2d(parts: list[str]) -> Bbox:
    left, top, right, bottom = (float(parts[i]) for i in range(6, 10))
    return (left, top, right, bottom)


def _box3d(parts: list[str]) -> Box3d:
    """3D box in :func:`motrics.iou_3d`'s convention: ``(x, y, z, l, w, h, yaw)``.

    KITTI's columns are ``h, w, l`` (dimensions), then ``x, y, z``
    (location), then ``ry`` (rotation_y). ``(x, y, z)`` is the *bottom*
    centre of the box (the ground-contact point), not its centroid, so ``y``
    is shifted up by half the height to get the true vertical centre
    ``iou_3d`` expects.
    """
    h, w, length = (float(parts[i]) for i in range(10, 13))
    x, y, z = (float(parts[i]) for i in range(13, 16))
    ry = float(parts[16])
    return (x, y - h / 2.0, z, length, w, h, ry)


def load_kitti_3d(path: str | PathLike[str]) -> _ByFrame:
    """Parse a KITTI tracking result file into per-frame 3D detections.

    Returns:
        A mapping from frame number to ``(ids, 3D boxes, 2D boxes, class
        ids)``. Rows with a negative track id are dropped.
    """
    frames: _ByFrame = {}
    for parts in _read_rows(path):
        track_id = int(float(parts[1]))
        if track_id < 0:
            continue
        class_id = _CLASS_NAME_TO_ID.get(parts[2].lower())
        if class_id is None:
            continue
        ids, boxes3d, boxes2d, classes = frames.setdefault(
            int(float(parts[0])), ([], [], [], [])
        )
        ids.append(track_id)
        boxes3d.append(_box3d(parts))
        boxes2d.append(_box2d(parts))
        classes.append(class_id)
    return frames


def load_kitti_3d_gt(
    path: str | PathLike[str],
) -> tuple[_GtByFrame, _IgnoreByFrame]:
    """Parse a KITTI tracking ``label_02`` ground-truth file for 3D scoring.

    "DontCare" rows are crowd-ignore regions, not objects, and carry no
    meaningful 3D box: they are returned separately as a 2D box, exactly
    like :func:`motrics.kitti.load_kitti_gt`.
    """
    gt: _GtByFrame = {}
    ignore: _IgnoreByFrame = {}
    for parts in _read_rows(path):
        frame = int(float(parts[0]))
        if parts[2].lower() == "dontcare":
            ignore.setdefault(frame, []).append(_box2d(parts))
            continue
        track_id = int(float(parts[1]))
        if track_id < 0:
            continue
        class_id = _CLASS_NAME_TO_ID.get(parts[2].lower())
        if class_id is None:
            continue
        ids, boxes3d, boxes2d, classes, truncation, occlusion = gt.setdefault(
            frame, ([], [], [], [], [], [])
        )
        ids.append(track_id)
        boxes3d.append(_box3d(parts))
        boxes2d.append(_box2d(parts))
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


def preprocess_kitti_3d(
    gt: _GtByFrame,
    pred: _ByFrame,
    cls: str,
    *,
    ignore_regions: _IgnoreByFrame | None = None,
    iou_threshold: float = 0.5,
) -> tuple[list[list[int]], list[list[int]], list[list[list[float]]]]:
    """Align and filter ground truth/predictions for KITTI-3D-MOT scoring.

    Applies the same rules as :func:`motrics.kitti.preprocess_kitti` (see the
    module docstring for why): keep only ``cls`` and its distractor class,
    match every kept ground-truth box against every kept prediction by 3D
    IoU, drop predictions matched to a distractor, occluded, or truncated
    ground-truth box, and drop remaining unmatched predictions that are too
    short (2D box height) or mostly covered by a DontCare region (2D IoA).
    Ground truth is reduced to non-occluded, non-truncated ``cls`` boxes.

    Args:
        gt: Ground truth from :func:`load_kitti_3d_gt` (unfiltered -- every
            class and occlusion/truncation level must still be present).
        pred: Predictions from :func:`load_kitti_3d`.
        cls: Class to evaluate, ``"pedestrian"`` or ``"car"``.
        ignore_regions: "DontCare" regions from :func:`load_kitti_3d_gt`.
        iou_threshold: 3D IoU threshold for the gt/prediction match.

    Returns:
        ``(gt_ids, pred_ids, similarity)`` ready for
        ``compute_clear_from_similarity`` / ``compute_identity_from_similarity``
        / ``compute_hota_from_similarity``: 3D-box similarity has no core
        ``compute_*`` overload of its own, only the precomputed-similarity
        one. ``similarity[t]`` is the 3D IoU matrix between ``gt_ids[t]`` and
        ``pred_ids[t]``.
    """
    if cls not in _DISTRACTOR_FOR:
        raise ValueError(f'unknown class {cls!r}, expected "pedestrian" or "car"')
    cls_id = _CLASS_NAME_TO_ID[cls]
    distractor_id = _CLASS_NAME_TO_ID[_DISTRACTOR_FOR[cls]]
    ignore_regions = ignore_regions or {}

    gt_ids: list[list[int]] = []
    pred_ids: list[list[int]] = []
    similarity: list[list[list[float]]] = []
    for frame in sorted(set(gt) | set(pred) | set(ignore_regions)):
        g_ids, g_boxes3d, _g_boxes2d, g_classes, g_trunc, g_occ = gt.get(
            frame, ([], [], [], [], [], [])
        )
        p_ids, p_boxes3d, p_boxes2d, p_classes = pred.get(frame, ([], [], [], []))
        regions = ignore_regions.get(frame, [])

        g_idx = [i for i, c in enumerate(g_classes) if c in (cls_id, distractor_id)]
        p_idx = [i for i, c in enumerate(p_classes) if c == cls_id]
        g_ids_c = [g_ids[i] for i in g_idx]
        g_boxes3d_c = [g_boxes3d[i] for i in g_idx]
        g_classes_c = [g_classes[i] for i in g_idx]
        g_trunc_c = [g_trunc[i] for i in g_idx]
        g_occ_c = [g_occ[i] for i in g_idx]
        p_ids_c = [p_ids[i] for i in p_idx]
        p_boxes3d_c = [p_boxes3d[i] for i in p_idx]
        p_boxes2d_c = [p_boxes2d[i] for i in p_idx]

        drop_pred = set()
        matched_pred = set()
        if g_ids_c and p_ids_c:
            m = match_boxes_3d(g_boxes3d_c, p_boxes3d_c, iou_threshold, "hungarian")
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
            box2d = p_boxes2d_c[pi]
            if box2d[3] - box2d[1] <= _MIN_HEIGHT or any(
                _ioa(box2d, region) > _IGNORE_REGION_OVERLAP for region in regions
            ):
                drop_pred.add(pi)

        keep_gt = [
            i
            for i in range(len(g_ids_c))
            if g_classes_c[i] == cls_id
            and g_occ_c[i] <= _MAX_OCCLUSION
            and g_trunc_c[i] <= _MAX_TRUNCATION
        ]
        kept_pred = [i for i in range(len(p_ids_c)) if i not in drop_pred]

        gt_ids.append([g_ids_c[i] for i in keep_gt])
        pred_ids.append([p_ids_c[i] for i in kept_pred])
        similarity.append(
            iou_3d_matrix(
                [g_boxes3d_c[i] for i in keep_gt],
                [p_boxes3d_c[i] for i in kept_pred],
            )
        )
    return gt_ids, pred_ids, similarity
