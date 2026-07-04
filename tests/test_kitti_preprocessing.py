"""Parity: ``motrics.preprocess_kitti`` vs TrackEval's real preprocessing.

Builds a small KITTI-format scenario covering every case TrackEval's
``Kitti2DBox.get_preprocessed_seq_data`` handles for a single evaluated class
(a plain pedestrian, a distractor ("person") matched by a prediction, an
occluded pedestrian matched by a prediction, a truncated pedestrian matched by
a prediction, an irrelevant "car" detection, an unmatched real false positive,
an unmatched too-short prediction, and an unmatched prediction mostly inside a
"DontCare" region), then runs it through both:

- ``motrics.preprocess_kitti``, fed by ``motrics.load_kitti_gt`` /
  ``motrics.load_kitti``.
- TrackEval's own (unmodified) ``get_preprocessed_seq_data``, called directly
  via a bare instance (``object.__new__``) so no real dataset directory/seqmap
  layout is needed — only the four attributes that method actually reads.

Both are built independently from the same raw rows so the comparison is
genuine, not circular.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import motrics
import numpy as np
import pytest

trackeval_datasets = pytest.importorskip("trackeval.datasets")

# Validated against trackeval==1.3.0 (see tests/test_motchallenge_preprocessing.py
# for the same reach-into-internals caveat).
Kitti2DBox = trackeval_datasets.kitti_2d_box.Kitti2DBox

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

# frame, id, type, truncated, occluded, alpha, left, top, right, bottom,
# h, w, l, x, y, z, ry.
GT_ROWS = [
    (1, 1, "Pedestrian", 0, 0, 0, 0, 0, 10, 30, 0, 0, 0, 0, 0, 0, 0),  # plain
    (1, 2, "Person", 0, 0, 0, 20, 0, 30, 30, 0, 0, 0, 0, 0, 0, 0),  # distractor
    (1, 3, "Pedestrian", 0, 3, 0, 40, 0, 50, 30, 0, 0, 0, 0, 0, 0, 0),  # occluded
    (1, 4, "Pedestrian", 1, 0, 0, 60, 0, 70, 30, 0, 0, 0, 0, 0, 0, 0),  # truncated
    (1, 5, "Car", 0, 0, 0, 80, 0, 90, 30, 0, 0, 0, 0, 0, 0, 0),  # irrelevant class
    (1, -1, "DontCare", 0, 0, 0, 100, 0, 120, 30, 0, 0, 0, 0, 0, 0, 0),  # ignore
    (2, 1, "Pedestrian", 0, 0, 0, 1, 1, 11, 31, 0, 0, 0, 0, 0, 0, 0),
]
PRED_ROWS = [
    (1, 10, "Pedestrian", -1, -1, -1, 0, 0, 10, 30, 0, 0, 0, 0, 0, 0, 0, 0.9),
    (1, 20, "Pedestrian", -1, -1, -1, 20, 0, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0.9),
    (1, 30, "Pedestrian", -1, -1, -1, 40, 0, 50, 30, 0, 0, 0, 0, 0, 0, 0, 0.9),
    (1, 40, "Pedestrian", -1, -1, -1, 60, 0, 70, 30, 0, 0, 0, 0, 0, 0, 0, 0.9),
    (1, 50, "Pedestrian", -1, -1, -1, 200, 0, 230, 30, 0, 0, 0, 0, 0, 0, 0, 0.9),
    (1, 60, "Pedestrian", -1, -1, -1, 300, 0, 315, 20, 0, 0, 0, 0, 0, 0, 0, 0.9),
    (1, 70, "Pedestrian", -1, -1, -1, 102, 2, 118, 28, 0, 0, 0, 0, 0, 0, 0, 0.9),
    (2, 10, "Pedestrian", -1, -1, -1, 1, 1, 11, 31, 0, 0, 0, 0, 0, 0, 0, 0.9),
]
NUM_TIMESTEPS = 2


def _write_rows(path: Path, rows: list[tuple]) -> None:
    path.write_text(
        "\n".join(" ".join(str(v) for v in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _motrics_result(
    tmp_path: Path,
) -> tuple[list[list[int]], list[list[tuple]], list[list[int]], list[list[tuple]]]:
    gt_path, pred_path = tmp_path / "gt.txt", tmp_path / "pred.txt"
    _write_rows(gt_path, GT_ROWS)
    _write_rows(pred_path, PRED_ROWS)
    gt, ignore = motrics.load_kitti_gt(gt_path)
    pred = motrics.load_kitti(pred_path)
    return motrics.preprocess_kitti(gt, pred, "pedestrian", ignore_regions=ignore)


def _trackeval_result() -> dict[str, Any]:
    """Build TrackEval's raw_data straight from the row tuples above (an
    independent parse, not routed through motrics' own ingest) and run its
    real (unmodified) preprocessing."""
    by_frame_gt = {t: [r for r in GT_ROWS if r[0] == t] for t in (1, 2)}
    by_frame_pred = {t: [r for r in PRED_ROWS if r[0] == t] for t in (1, 2)}

    gt_ids, gt_dets, gt_classes, gt_extras, gt_ignore = [], [], [], [], []
    tracker_ids, tracker_dets, tracker_classes, tracker_confidences = [], [], [], []
    similarity_scores = []
    for t in range(1, NUM_TIMESTEPS + 1):
        g = [r for r in by_frame_gt.get(t, []) if r[2].lower() != "dontcare"]
        ign = [r for r in by_frame_gt.get(t, []) if r[2].lower() == "dontcare"]
        p = by_frame_pred.get(t, [])
        g_xyxy = np.array([row[6:10] for row in g], dtype=float).reshape(-1, 4)
        p_xyxy = np.array([row[6:10] for row in p], dtype=float).reshape(-1, 4)
        gt_ids.append(np.array([row[1] for row in g], dtype=int))
        gt_dets.append(g_xyxy)
        gt_classes.append(
            np.array([_CLASS_NAME_TO_ID[row[2].lower()] for row in g], dtype=int)
        )
        gt_extras.append(
            {
                "truncation": np.array([row[3] for row in g], dtype=int),
                "occlusion": np.array([row[4] for row in g], dtype=int),
            }
        )
        ign_xyxy = np.array([row[6:10] for row in ign], dtype=float).reshape(-1, 4)
        gt_ignore.append(ign_xyxy)
        tracker_ids.append(np.array([row[1] for row in p], dtype=int))
        tracker_dets.append(p_xyxy)
        tracker_classes.append(
            np.array([_CLASS_NAME_TO_ID[row[2].lower()] for row in p], dtype=int)
        )
        tracker_confidences.append(np.array([row[17] for row in p], dtype=float))
        similarity_scores.append(
            Kitti2DBox._calculate_box_ious(g_xyxy, p_xyxy, box_format="x0y0x1y1")
        )

    raw_data = {
        "num_timesteps": NUM_TIMESTEPS,
        "seq": "test-seq",
        "gt_ids": gt_ids,
        "gt_dets": gt_dets,
        "gt_classes": gt_classes,
        "gt_extras": gt_extras,
        "gt_crowd_ignore_regions": gt_ignore,
        "tracker_ids": tracker_ids,
        "tracker_dets": tracker_dets,
        "tracker_classes": tracker_classes,
        "tracker_confidences": tracker_confidences,
        "similarity_scores": similarity_scores,
    }

    dataset = object.__new__(Kitti2DBox)
    dataset.class_name_to_class_id = _CLASS_NAME_TO_ID
    dataset.max_occlusion = 2
    dataset.max_truncation = 0
    dataset.min_height = 25
    return dataset.get_preprocessed_seq_data(raw_data, "pedestrian")


def _xyxy_set(boxes: np.ndarray) -> set[tuple[float, float, float, float]]:
    return {tuple(box) for box in boxes.tolist()}


def test_preprocess_kitti_matches_trackeval(tmp_path: Path) -> None:
    m_gt_ids, m_gt_boxes, m_pred_ids, m_pred_boxes = _motrics_result(tmp_path)
    te = _trackeval_result()

    assert te["num_gt_dets"] == sum(len(f) for f in m_gt_ids)
    assert te["num_tracker_dets"] == sum(len(f) for f in m_pred_ids)

    for t in range(NUM_TIMESTEPS):
        assert set(m_gt_boxes[t]) == _xyxy_set(te["gt_dets"][t])
        assert set(m_pred_boxes[t]) == _xyxy_set(te["tracker_dets"][t])

    # Sanity check the scenario actually exercises every case.
    assert m_gt_ids == [[1], [1]]  # distractor/occluded/truncated/car gt gone
    assert m_pred_ids == [[10, 50], [10]]  # 20/30/40 dropped, 60/70 dropped
