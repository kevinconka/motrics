"""Parity: ``motrics.preprocess_motchallenge`` vs TrackEval's real preprocessing.

Builds a small MOTChallenge-format scenario covering every case TrackEval's
``MotChallenge2DBox.get_preprocessed_seq_data`` handles (a plain pedestrian, a
distractor matched by a prediction, a "do not consider" pedestrian, and an
unmatched false positive), then runs it through both:

- ``motrics.preprocess_motchallenge``, fed by ``motrics.load_motchallenge_gt``.
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

# Validated against trackeval==1.3.0. This test reaches into private internals
# (_calculate_box_ious, a bare instance via object.__new__) to call TrackEval's
# real preprocessing without a full dataset directory/seqmap layout; a future
# trackeval release could rename/restructure these and break this test loudly
# (ImportError/AttributeError) rather than silently.
MotChallenge2DBox = trackeval_datasets.mot_challenge_2d_box.MotChallenge2DBox

_CLASS_NAME_TO_ID = {
    "pedestrian": 1,
    "person_on_vehicle": 2,
    "car": 3,
    "bicycle": 4,
    "motorbike": 5,
    "non_mot_vehicle": 6,
    "static_person": 7,
    "distractor": 8,
    "occluder": 9,
    "occluder_on_ground": 10,
    "occluder_full": 11,
    "reflection": 12,
    "crowd": 13,
}

# frame, id, left, top, w, h, consider(gt) / conf(pred), class(gt), visibility(gt)
GT_ROWS = [
    (1, 1, 0, 0, 10, 10, 1, 1, 1.0),  # plain pedestrian
    (1, 2, 20, 20, 10, 10, 1, 8, 1.0),  # distractor, matched by pred below
    (1, 3, 40, 40, 10, 10, 0, 1, 1.0),  # pedestrian, "do not consider"
    (2, 1, 1, 1, 10, 10, 1, 1, 1.0),
]
PRED_ROWS = [
    (1, 10, 0, 0, 10, 10, 0.9, -1, -1, -1),  # matches pedestrian 1
    (1, 20, 20, 20, 10, 10, 0.9, -1, -1, -1),  # matches distractor 2 -> dropped
    (1, 30, 90, 90, 10, 10, 0.9, -1, -1, -1),  # unmatched -> real false positive
    (2, 10, 1, 1, 10, 10, 0.9, -1, -1, -1),
]
NUM_TIMESTEPS = 2


def _write_rows(path: Path, rows: list[tuple]) -> None:
    path.write_text(
        "\n".join(",".join(str(v) for v in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _motrics_result(
    tmp_path: Path,
) -> tuple[list[list[int]], list[list[tuple]], list[list[int]], list[list[tuple]]]:
    gt_path, pred_path = tmp_path / "gt.txt", tmp_path / "pred.txt"
    _write_rows(gt_path, GT_ROWS)
    _write_rows(pred_path, PRED_ROWS)
    gt = motrics.load_motchallenge_gt(gt_path)
    pred = motrics.load_motchallenge(pred_path)
    return motrics.preprocess_motchallenge(gt, pred)


def _trackeval_result() -> dict[str, Any]:
    """Build TrackEval's raw_data straight from the row tuples above (an
    independent parse, not routed through motrics' own ingest) and run its
    real (unmodified) preprocessing."""
    by_frame_gt = {t: [r for r in GT_ROWS if r[0] == t] for t in (1, 2)}
    by_frame_pred = {t: [r for r in PRED_ROWS if r[0] == t] for t in (1, 2)}

    gt_ids, gt_dets, gt_classes, gt_extras = [], [], [], []
    tracker_ids, tracker_dets, tracker_classes, tracker_confidences = [], [], [], []
    similarity_scores = []
    for t in range(1, NUM_TIMESTEPS + 1):
        g = by_frame_gt.get(t, [])
        p = by_frame_pred.get(t, [])
        g_xywh = np.array([row[2:6] for row in g], dtype=float).reshape(-1, 4)
        p_xywh = np.array([row[2:6] for row in p], dtype=float).reshape(-1, 4)
        gt_ids.append(np.array([row[1] for row in g], dtype=int))
        gt_dets.append(g_xywh)
        gt_classes.append(np.array([row[7] for row in g], dtype=int))
        gt_extras.append({"zero_marked": np.array([row[6] for row in g], dtype=int)})
        tracker_ids.append(np.array([row[1] for row in p], dtype=int))
        tracker_dets.append(p_xywh)
        tracker_classes.append(np.ones(len(p), dtype=int))
        tracker_confidences.append(np.array([row[6] for row in p], dtype=float))
        similarity_scores.append(
            MotChallenge2DBox._calculate_box_ious(g_xywh, p_xywh, box_format="xywh")
        )

    raw_data = {
        "num_timesteps": NUM_TIMESTEPS,
        "seq": "test-seq",
        "gt_ids": gt_ids,
        "gt_dets": gt_dets,
        "gt_classes": gt_classes,
        "gt_extras": gt_extras,
        "tracker_ids": tracker_ids,
        "tracker_dets": tracker_dets,
        "tracker_classes": tracker_classes,
        "tracker_confidences": tracker_confidences,
        "similarity_scores": similarity_scores,
    }

    dataset = object.__new__(MotChallenge2DBox)
    dataset.benchmark = "MOT17"
    dataset.do_preproc = True
    dataset.class_name_to_class_id = _CLASS_NAME_TO_ID
    dataset.valid_class_numbers = list(_CLASS_NAME_TO_ID.values())
    return dataset.get_preprocessed_seq_data(raw_data, "pedestrian")


def _xyxy_set(boxes_xywh: np.ndarray) -> set[tuple[float, float, float, float]]:
    return {(x, y, x + w, y + h) for x, y, w, h in boxes_xywh.tolist()}


def test_preprocess_motchallenge_matches_trackeval(tmp_path: Path) -> None:
    m_gt_ids, m_gt_boxes, m_pred_ids, m_pred_boxes = _motrics_result(tmp_path)
    te = _trackeval_result()

    assert te["num_gt_dets"] == sum(len(f) for f in m_gt_ids)
    assert te["num_tracker_dets"] == sum(len(f) for f in m_pred_ids)

    for t in range(NUM_TIMESTEPS):
        assert set(m_gt_boxes[t]) == _xyxy_set(te["gt_dets"][t])
        assert set(m_pred_boxes[t]) == _xyxy_set(te["tracker_dets"][t])

    # Sanity check the scenario actually exercises every case.
    assert m_gt_ids == [[1], [1]]  # distractor (2) and "do not consider" (3) gone
    assert m_pred_ids == [[10, 30], [10]]  # 20 dropped (matched the distractor)
