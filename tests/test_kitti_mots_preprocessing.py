"""Parity: ``motrics.preprocess_kitti_mots`` vs TrackEval's real preprocessing.

Builds a small KITTI-MOTS-format scenario covering every case TrackEval's
``KittiMOTS.get_preprocessed_seq_data`` handles for a single evaluated class
(a plain pedestrian matched by a prediction, a pedestrian with no matching
prediction, an irrelevant "car" detection on both sides, an unmatched
prediction mostly inside an ignore region, an unmatched real false positive,
and a second frame with no ignore region at all), then runs it through both:

- ``motrics.preprocess_kitti_mots``, fed by ``motrics.load_kitti_mots_gt`` /
  ``motrics.load_kitti_mots``.
- TrackEval's own (unmodified) ``get_preprocessed_seq_data``, called directly
  via a bare instance (``object.__new__``) so no real dataset directory/seqmap
  layout is needed — only the one attribute that method actually reads.

Both are built independently from the same raw pycocotools-encoded RLEs (not
routed through motrics' own codec) so the comparison is genuine, not
circular — see ``tests/test_mask.py`` for the codec's own parity coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import motrics
import numpy as np
import pytest

pycocotools_mask = pytest.importorskip("pycocotools.mask")
trackeval_datasets = pytest.importorskip("trackeval.datasets")

# Validated against trackeval==1.3.0 (see tests/test_motchallenge_preprocessing.py
# for the same reach-into-internals caveat).
KittiMOTS = trackeval_datasets.kitti_mots.KittiMOTS

_CLASS_NAME_TO_ID = {"car": "1", "pedestrian": "2", "ignore": "10"}
_H, _W = 8, 8


def _rle(row_start: int, row_end: int, col_start: int, col_end: int) -> bytes:
    bitmap = np.zeros((_H, _W), dtype=np.uint8)
    bitmap[row_start:row_end, col_start:col_end] = 1
    return pycocotools_mask.encode(np.asfortranarray(bitmap))["counts"]


# Disjoint 2x2 blocks tiling an 8x8 grid, named by scenario role.
_MASK_PED = _rle(0, 2, 0, 2)  # matched pedestrian, both frames
_MASK_CAR_GT = _rle(0, 2, 2, 4)  # irrelevant class, gt side
_MASK_PED_MISS = _rle(0, 2, 4, 6)  # pedestrian with no matching prediction
_MASK_IGNORE = _rle(0, 2, 6, 8)  # ignore/DontCare region, frame 1 only
_MASK_FP = _rle(2, 4, 0, 2)  # real false positive, no overlap with anything
_MASK_CAR_PRED = _rle(2, 4, 2, 4)  # irrelevant class, prediction side

# (frame, id, class_id, mask) rows, mirroring a KITTI-MOTS label file.
GT_ROWS = [
    (1, 1, "2", _MASK_PED),
    (1, 2, "1", _MASK_CAR_GT),
    (1, 3, "2", _MASK_PED_MISS),
    (1, 10000, "10", _MASK_IGNORE),
    (2, 1, "2", _MASK_PED),
]
PRED_ROWS = [
    (1, 10, "2", _MASK_PED),
    (1, 20, "2", _MASK_IGNORE),
    (1, 30, "2", _MASK_FP),
    (1, 40, "1", _MASK_CAR_PRED),
    (2, 10, "2", _MASK_PED),
]
NUM_TIMESTEPS = 2


def _write_rows(path: Path, rows: list[tuple]) -> None:
    lines = [
        f"{frame} {obj_id} {class_id} {_H} {_W} {counts.decode()}"
        for frame, obj_id, class_id, counts in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _motrics_result(
    tmp_path: Path,
) -> tuple[list[list[int]], list[list[int]], list[list[list[float]]]]:
    gt_path, pred_path = tmp_path / "gt.txt", tmp_path / "pred.txt"
    _write_rows(gt_path, GT_ROWS)
    _write_rows(pred_path, PRED_ROWS)
    gt, ignore = motrics.load_kitti_mots_gt(gt_path)
    pred = motrics.load_kitti_mots(pred_path)
    return motrics.preprocess_kitti_mots(gt, pred, "pedestrian", ignore_regions=ignore)


def _rle_dict(counts: bytes) -> dict[str, Any]:
    return {"size": [_H, _W], "counts": counts}


def _trackeval_result() -> dict[str, Any]:
    """Build TrackEval's raw_data straight from the row tuples above (an
    independent parse, not routed through motrics' own ingest) and run its
    real (unmodified) preprocessing."""
    by_frame_gt = {t: [r for r in GT_ROWS if r[0] == t] for t in (1, 2)}
    by_frame_pred = {t: [r for r in PRED_ROWS if r[0] == t] for t in (1, 2)}

    dataset = object.__new__(KittiMOTS)
    dataset.class_name_to_class_id = _CLASS_NAME_TO_ID

    gt_ids, gt_dets, gt_classes, gt_ignore = [], [], [], []
    tracker_ids, tracker_dets, tracker_classes = [], [], []
    similarity_scores = []
    for t in range(1, NUM_TIMESTEPS + 1):
        g = [r for r in by_frame_gt.get(t, []) if r[2] != "10"]
        ign = [r for r in by_frame_gt.get(t, []) if r[2] == "10"]
        p = by_frame_pred.get(t, [])

        g_dets = [_rle_dict(r[3]) for r in g]
        p_dets = [_rle_dict(r[3]) for r in p]
        gt_ids.append(np.array([r[1] for r in g], dtype=int))
        gt_dets.append(g_dets)
        gt_classes.append(np.array([int(r[2]) for r in g], dtype=int))
        gt_ignore.append(
            pycocotools_mask.merge([_rle_dict(r[3]) for r in ign], intersect=False)
        )
        tracker_ids.append(np.array([r[1] for r in p], dtype=int))
        tracker_dets.append(p_dets)
        tracker_classes.append(np.array([int(r[2]) for r in p], dtype=int))
        similarity_scores.append(dataset._calculate_similarities(g_dets, p_dets))

    raw_data = {
        "num_timesteps": NUM_TIMESTEPS,
        "seq": "test-seq",
        "gt_ids": gt_ids,
        "gt_dets": gt_dets,
        "gt_classes": gt_classes,
        "gt_ignore_region": gt_ignore,
        "tracker_ids": tracker_ids,
        "tracker_dets": tracker_dets,
        "tracker_classes": tracker_classes,
        "similarity_scores": similarity_scores,
    }
    return dataset.get_preprocessed_seq_data(raw_data, "pedestrian")


def _counts_set(dets: list[dict[str, Any]]) -> set[bytes]:
    return {d["counts"] for d in dets}


def test_preprocess_kitti_mots_matches_trackeval(tmp_path: Path) -> None:
    m_gt_ids, m_pred_ids, _similarity = _motrics_result(tmp_path)
    te = _trackeval_result()

    assert te["num_gt_dets"] == sum(len(f) for f in m_gt_ids)
    assert te["num_tracker_dets"] == sum(len(f) for f in m_pred_ids)

    gt_by_id = {(r[0], r[1]): r[3] for r in GT_ROWS}
    pred_by_id = {(r[0], r[1]): r[3] for r in PRED_ROWS}
    for t in range(NUM_TIMESTEPS):
        frame = t + 1
        m_gt_counts = {gt_by_id[(frame, i)] for i in m_gt_ids[t]}
        m_pred_counts = {pred_by_id[(frame, i)] for i in m_pred_ids[t]}
        assert m_gt_counts == _counts_set(te["gt_dets"][t])
        assert m_pred_counts == _counts_set(te["tracker_dets"][t])

    # Sanity check the scenario actually exercises every case.
    assert m_gt_ids == [[1, 3], [1]]  # car (2) excluded, both peds kept
    assert m_pred_ids == [[10, 30], [10]]  # 20 dropped (ignore), 40 excluded (car)
