"""MOTChallenge ingest helpers.

Parse MOTChallenge-format CSV files (ground truth or tracker results) into the
frame-aligned inputs consumed by the metric functions.

The MOTChallenge line format is::

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

Boxes are given as top-left corner plus width/height (1-based pixels) and are
converted here to the ``xyxy`` convention ``(x1, y1, x2, y2)`` used throughout
``motrics``.

Two ingest paths:

- ``load_motchallenge`` + ``align_frames`` — plain box/id parsing, no
  filtering beyond an optional confidence cutoff.
- ``load_motchallenge_gt`` + ``preprocess_motchallenge`` — replicates
  TrackEval's ``MotChallenge2DBox`` preprocessing (distractor-class removal,
  pedestrian-only, "do not consider" rows dropped), for numbers matching
  TrackEval's own reported values.
"""

from __future__ import annotations

from os import PathLike

from motrics._motrics import match_boxes

# A bounding box in xyxy format: (x1, y1, x2, y2).
Bbox = tuple[float, float, float, float]
# Per-frame detections: frame number -> (object ids, boxes).
Frames = dict[int, tuple[list[int], list[Bbox]]]
# Per-frame ground truth with the extra columns preprocessing needs:
# frame number -> (ids, boxes, class ids, "consider this box" flags).
GtFrames = dict[int, tuple[list[int], list[Bbox], list[int], list[bool]]]

#: MOTChallenge class id for the evaluated class; every other class exists
#: only to drive preprocessing (see :func:`preprocess_motchallenge`).
_PEDESTRIAN_CLASS = 1
#: Class ids TrackEval's ``MotChallenge2DBox`` treats as distractors: a
#: detection is a distractor if it looks enough like one of these (not a
#: pedestrian) that a tracker firing on it shouldn't be penalised.
#: ``person_on_vehicle``, ``static_person``, ``distractor``, ``reflection``.
_DISTRACTOR_CLASSES = frozenset({2, 7, 8, 12})
#: MOT20 additionally treats ``non_mot_vehicle`` as a distractor.
_MOT20_EXTRA_DISTRACTOR_CLASS = 6


def _read_rows(path: str | PathLike[str]) -> list[list[str]]:
    with open(path, encoding="utf-8") as handle:
        return [line.split(",") for raw in handle if (line := raw.strip())]


def _box(parts: list[str]) -> Bbox:
    left, top, width, height = (float(parts[i]) for i in range(2, 6))
    return (left, top, left + width, top + height)


def load_motchallenge(
    path: str | PathLike[str], *, min_confidence: float | None = None
) -> Frames:
    """Parse a MOTChallenge CSV file into per-frame detections.

    Args:
        path: Path to a MOTChallenge ``gt.txt`` or results file.
        min_confidence: If given, drop rows whose confidence (7th column) is
            below this value. Typical for filtering tracker results; leave as
            ``None`` for ground truth.

    Returns:
        A mapping from frame number to ``(ids, boxes)``, boxes in ``xyxy``.
    """
    frames: Frames = {}
    for parts in _read_rows(path):
        if (
            min_confidence is not None
            and len(parts) > 6
            and (float(parts[6]) < min_confidence)
        ):
            continue
        ids, boxes = frames.setdefault(int(float(parts[0])), ([], []))
        ids.append(int(float(parts[1])))
        boxes.append(_box(parts))
    return frames


def load_motchallenge_gt(path: str | PathLike[str]) -> GtFrames:
    """Parse a MOTChallenge ``gt.txt``, keeping the class and "consider this
    box" columns :func:`preprocess_motchallenge` needs.

    Unlike :func:`load_motchallenge`, nothing is filtered here — every row
    (every class, including ones marked "do not consider") is kept, since
    :func:`preprocess_motchallenge` needs the full picture to replicate
    TrackEval's preprocessing.
    """
    frames: GtFrames = {}
    for parts in _read_rows(path):
        ids, boxes, classes, keep = frames.setdefault(
            int(float(parts[0])), ([], [], [], [])
        )
        ids.append(int(float(parts[1])))
        boxes.append(_box(parts))
        classes.append(int(float(parts[7])) if len(parts) > 7 else _PEDESTRIAN_CLASS)
        keep.append(float(parts[6]) != 0 if len(parts) > 6 else True)
    return frames


def align_frames(
    gt: Frames, pred: Frames
) -> tuple[list[list[int]], list[list[Bbox]], list[list[int]], list[list[Bbox]]]:
    """Align two parsed sequences onto a shared, ordered frame timeline.

    Frames present in only one sequence are padded with empty detections in the
    other, so the returned lists all have the same length and can be passed
    straight to ``compute_clear`` / ``compute_identity`` / ``compute_hota``.

    Returns:
        ``(gt_ids, gt_boxes, pred_ids, pred_boxes)``.
    """
    gt_ids: list[list[int]] = []
    gt_boxes: list[list[Bbox]] = []
    pred_ids: list[list[int]] = []
    pred_boxes: list[list[Bbox]] = []
    for frame in sorted(set(gt) | set(pred)):
        g_ids, g_boxes = gt.get(frame, ([], []))
        p_ids, p_boxes = pred.get(frame, ([], []))
        gt_ids.append(g_ids)
        gt_boxes.append(g_boxes)
        pred_ids.append(p_ids)
        pred_boxes.append(p_boxes)
    return gt_ids, gt_boxes, pred_ids, pred_boxes


def preprocess_motchallenge(
    gt: GtFrames,
    pred: Frames,
    *,
    iou_threshold: float = 0.5,
    benchmark: str = "MOT17",
) -> tuple[list[list[int]], list[list[Bbox]], list[list[int]], list[list[Bbox]]]:
    """Align and filter ground truth/predictions exactly as TrackEval's
    ``MotChallenge2DBox`` does before scoring.

    Per frame: match *every* ground-truth box (any class) against every
    prediction, and drop predictions matched to a distractor-class box —
    a tracker shouldn't be penalised for firing on something that merely
    looks like a pedestrian. Ground truth is then reduced to pedestrian
    boxes not marked "do not consider" (the ``conf`` column in ``gt.txt``).
    MOTChallenge has no separate crowd-ignore regions; distractor classes
    serve that role.

    Args:
        gt: Ground truth from :func:`load_motchallenge_gt` (unfiltered — every
            class and every "do not consider" row must still be present).
        pred: Predictions from :func:`load_motchallenge`.
        iou_threshold: Threshold for the gt/prediction distractor match.
        benchmark: ``"MOT20"`` additionally treats ``non_mot_vehicle`` as a
            distractor class; every other benchmark uses the same set.

    Returns:
        ``(gt_ids, gt_boxes, pred_ids, pred_boxes)``, ready for
        ``compute_clear`` / ``compute_identity`` / ``compute_hota``.
    """
    distractor_classes = _DISTRACTOR_CLASSES
    if benchmark == "MOT20":
        distractor_classes |= {_MOT20_EXTRA_DISTRACTOR_CLASS}

    gt_ids: list[list[int]] = []
    gt_boxes: list[list[Bbox]] = []
    pred_ids: list[list[int]] = []
    pred_boxes: list[list[Bbox]] = []
    for frame in sorted(set(gt) | set(pred)):
        g_ids, g_boxes, g_classes, g_keep = gt.get(frame, ([], [], [], []))
        p_ids, p_boxes = pred.get(frame, ([], []))

        drop_pred = set()
        if g_ids and p_ids:
            m = match_boxes(g_boxes, p_boxes, iou_threshold, "hungarian")
            drop_pred = {
                pi for gi, pi in m.matches if g_classes[gi] in distractor_classes
            }

        keep_gt = [
            i
            for i in range(len(g_ids))
            if g_classes[i] == _PEDESTRIAN_CLASS and g_keep[i]
        ]
        gt_ids.append([g_ids[i] for i in keep_gt])
        gt_boxes.append([g_boxes[i] for i in keep_gt])
        pred_ids.append([p_ids[i] for i in range(len(p_ids)) if i not in drop_pred])
        pred_boxes.append(
            [p_boxes[i] for i in range(len(p_boxes)) if i not in drop_pred]
        )
    return gt_ids, gt_boxes, pred_ids, pred_boxes
