"""MOTChallenge ingest helpers.

Parse MOTChallenge-format CSV files (ground truth or tracker results) into the
frame-aligned inputs consumed by the metric functions.

The MOTChallenge line format is::

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

Boxes are given as top-left corner plus width/height (1-based pixels) and are
converted here to the ``xyxy`` convention ``(x1, y1, x2, y2)`` used throughout
``motrics``.
"""

from __future__ import annotations

from os import PathLike

# A bounding box in xyxy format: (x1, y1, x2, y2).
Bbox = tuple[float, float, float, float]
# Per-frame detections: frame number -> (object ids, boxes).
Frames = dict[int, tuple[list[int], list[Bbox]]]


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
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            left, top, width, height = (float(parts[i]) for i in range(2, 6))
            if (
                min_confidence is not None
                and len(parts) > 6
                and (float(parts[6]) < min_confidence)
            ):
                continue
            ids, boxes = frames.setdefault(frame, ([], []))
            ids.append(obj_id)
            boxes.append((left, top, left + width, top + height))
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
    timeline = sorted(set(gt) | set(pred))
    gt_ids: list[list[int]] = []
    gt_boxes: list[list[Bbox]] = []
    pred_ids: list[list[int]] = []
    pred_boxes: list[list[Bbox]] = []
    for frame in timeline:
        g_ids, g_boxes = gt.get(frame, ([], []))
        p_ids, p_boxes = pred.get(frame, ([], []))
        gt_ids.append(g_ids)
        gt_boxes.append(g_boxes)
        pred_ids.append(p_ids)
        pred_boxes.append(p_boxes)
    return gt_ids, gt_boxes, pred_ids, pred_boxes
