"""KITTI-MOTS tracking ingest helpers.

Parse the KITTI-MOTS format: one space-separated row per object per frame,
columns ``frame, id, class_id, img_height, img_width, rle_counts``, where
``rle_counts`` is pycocotools' compressed-string RLE. ``class_id`` is ``1``
(car), ``2`` (pedestrian), or ``10`` (an "ignore"/DontCare region, not an
object).

Two ingest paths mirroring ``motrics.kitti``:

- ``load_kitti_mots`` — plain mask/id/class parsing for tracker results.
- ``load_kitti_mots_gt`` + ``preprocess_kitti_mots`` — replicates TrackEval's
  ``KittiMOTS`` preprocessing (per-class evaluation, matching ground truth to
  predictions by mask IoU, dropping unmatched predictions mostly covered by
  an ignore region), for numbers matching TrackEval's own reported values.
"""

from __future__ import annotations

from os import PathLike

from motrics._motrics import Mask, mask_iou_matrix, mask_merge, match_masks

# Per-frame tracker detections: frame number -> (ids, masks, class ids).
_ByFrame = dict[int, tuple[list[int], list[Mask], list[int]]]
# Per-frame ground truth, "ignore" rows excluded (see load_kitti_mots_gt).
_GtByFrame = dict[int, tuple[list[int], list[Mask], list[int]]]
# Per-frame "ignore" (class id 10 / DontCare) regions.
_IgnoreByFrame = dict[int, list[Mask]]

#: KITTI-MOTS class name -> class id, as used by TrackEval's ``KittiMOTS``.
_CLASS_NAME_TO_ID = {"car": 1, "pedestrian": 2}
#: The class id marking a row as a crowd-ignore region rather than an object.
_IGNORE_CLASS_ID = 10
#: Unmatched predictions overlapping an ignore region by more than this
#: fraction of their own area are dropped.
_IGNORE_REGION_OVERLAP = 0.5


def _read_rows(path: str | PathLike[str]) -> list[list[str]]:
    with open(path, encoding="utf-8") as handle:
        return [line.split() for raw in handle if (line := raw.strip())]


def _mask(parts: list[str]) -> Mask:
    height, width = int(parts[3]), int(parts[4])
    return Mask.from_coco((height, width), parts[5])


def load_kitti_mots(path: str | PathLike[str]) -> _ByFrame:
    """Parse a KITTI-MOTS tracking result file into per-frame detections.

    Returns:
        A mapping from frame number to ``(ids, masks, class ids)``.
    """
    frames: _ByFrame = {}
    for parts in _read_rows(path):
        ids, masks, classes = frames.setdefault(int(parts[0]), ([], [], []))
        ids.append(int(parts[1]))
        masks.append(_mask(parts))
        classes.append(int(parts[2]))
    return frames


def load_kitti_mots_gt(
    path: str | PathLike[str],
) -> tuple[_GtByFrame, _IgnoreByFrame]:
    """Parse a KITTI-MOTS ground-truth file, separating "ignore" regions
    (class id ``10``) from real objects.
    """
    gt: _GtByFrame = {}
    ignore: _IgnoreByFrame = {}
    for parts in _read_rows(path):
        frame = int(parts[0])
        class_id = int(parts[2])
        if class_id == _IGNORE_CLASS_ID:
            ignore.setdefault(frame, []).append(_mask(parts))
            continue
        ids, masks, classes = gt.setdefault(frame, ([], [], []))
        ids.append(int(parts[1]))
        masks.append(_mask(parts))
        classes.append(class_id)
    return gt, ignore


def preprocess_kitti_mots(
    gt: _GtByFrame,
    pred: _ByFrame,
    cls: str,
    *,
    ignore_regions: _IgnoreByFrame | None = None,
    iou_threshold: float = 0.5,
) -> tuple[list[list[int]], list[list[int]], list[list[list[float]]]]:
    """Align and filter ground truth/predictions exactly as TrackEval's
    ``KittiMOTS`` does before scoring a single class.

    Per frame: keep only ``cls`` ground truth and predictions, match every
    kept ground-truth mask against every kept prediction by mask IoU. Unlike
    :func:`preprocess_kitti`, KITTI-MOTS has no distractor classes, so a
    match never causes a prediction to be dropped — matched predictions are
    always kept. Remaining unmatched predictions are dropped if more than
    ``_IGNORE_REGION_OVERLAP`` of their own area falls inside an ignore
    region. All ground truth of ``cls`` is kept.

    Args:
        gt: Ground truth from :func:`load_kitti_mots_gt` (unfiltered — every
            class must still be present).
        pred: Predictions from :func:`load_kitti_mots`.
        cls: Class to evaluate, ``"pedestrian"`` or ``"car"``.
        ignore_regions: Ignore regions from :func:`load_kitti_mots_gt`, if any.
        iou_threshold: Threshold for the gt/prediction match.

    Returns:
        ``(gt_ids, pred_ids, similarity)`` ready for
        ``compute_clear_from_similarity`` / ``compute_identity_from_similarity``
        / ``compute_hota_from_similarity``: mask similarity has no core
        ``compute_*`` overload of its own, only the precomputed-similarity
        one. ``similarity[t]`` is the mask IoU matrix between ``gt_ids[t]``
        and ``pred_ids[t]``.
    """
    if cls not in _CLASS_NAME_TO_ID:
        raise ValueError(f'unknown class {cls!r}, expected "pedestrian" or "car"')
    cls_id = _CLASS_NAME_TO_ID[cls]
    ignore_regions = ignore_regions or {}

    gt_ids: list[list[int]] = []
    pred_ids: list[list[int]] = []
    similarity: list[list[list[float]]] = []
    for frame in sorted(set(gt) | set(pred) | set(ignore_regions)):
        g_ids, g_masks, g_classes = gt.get(frame, ([], [], []))
        p_ids, p_masks, p_classes = pred.get(frame, ([], [], []))
        regions = ignore_regions.get(frame, [])

        g_idx = [i for i, c in enumerate(g_classes) if c == cls_id]
        p_idx = [i for i, c in enumerate(p_classes) if c == cls_id]
        g_ids_c = [g_ids[i] for i in g_idx]
        g_masks_c = [g_masks[i] for i in g_idx]
        p_ids_c = [p_ids[i] for i in p_idx]
        p_masks_c = [p_masks[i] for i in p_idx]

        matched_pred = set()
        if g_masks_c and p_masks_c:
            m = match_masks(g_masks_c, p_masks_c, iou_threshold, "hungarian")
            matched_pred = {pi for _, pi in m.matches}

        unmatched = [i for i in range(len(p_ids_c)) if i not in matched_pred]
        drop_pred = set()
        if regions and unmatched:
            ignore_region = mask_merge(regions)
            unmatched_masks = [p_masks_c[i] for i in unmatched]
            ioa = mask_iou_matrix(unmatched_masks, [ignore_region], is_crowd=[True])
            drop_pred = {
                unmatched[i]
                for i, row in enumerate(ioa)
                if row[0] > _IGNORE_REGION_OVERLAP
            }

        gt_ids.append(g_ids_c)
        kept_pred = [i for i in range(len(p_ids_c)) if i not in drop_pred]
        pred_ids.append([p_ids_c[i] for i in kept_pred])
        similarity.append(mask_iou_matrix(g_masks_c, [p_masks_c[i] for i in kept_pred]))
    return gt_ids, pred_ids, similarity
