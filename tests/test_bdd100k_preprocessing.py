"""Parity: ``motrics.preprocess_bdd100k`` vs TrackEval's real preprocessing.

Builds a small BDD100K-format scenario covering every case TrackEval's
``BDD100K.get_preprocessed_seq_data`` handles for a single evaluated class
(a plain pedestrian matched by a prediction, a crowd-marked pedestrian and a
distractor-category object that become ignore regions, an irrelevant "car"
detection, an unmatched real false positive, and an unmatched prediction mostly
inside a crowd-ignore region), then runs it through both:

- ``motrics.preprocess_bdd100k``, fed by ``motrics.load_bdd100k_gt`` /
  ``motrics.load_bdd100k``.
- TrackEval's own (unmodified) ``get_preprocessed_seq_data``, called directly
  via a bare instance (``object.__new__``) so no real dataset directory layout
  is needed — only the one attribute that method actually reads.

Both are built independently from the same raw labels so the comparison is
genuine, not circular.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import motrics
import numpy as np
import pytest

trackeval_datasets = pytest.importorskip("trackeval.datasets")

# Validated against trackeval==1.3.0 (see tests/test_motchallenge_preprocessing.py
# for the same reach-into-internals caveat).
BDD100K = trackeval_datasets.bdd100k.BDD100K

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
_DISTRACTOR_CLASSES = ["other person", "trailer", "other vehicle"]


def _box(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _label(
    id_: int, category: str, box: dict[str, float], *, crowd: bool = False
) -> dict[str, Any]:
    return {
        "id": id_,
        "category": category,
        "box2d": box,
        "attributes": {"Crowd": crowd},
    }


# index -> labels. Two frames; evaluated class is "pedestrian".
GT_FRAMES: list[dict[str, Any]] = [
    {
        "index": 0,
        "labels": [
            _label(1, "pedestrian", _box(0, 0, 10, 30)),  # plain, matched
            _label(2, "pedestrian", _box(20, 0, 30, 30), crowd=True),  # crowd -> ignore
            _label(3, "other person", _box(40, 0, 50, 30)),  # distractor -> ignore
            _label(5, "car", _box(80, 0, 90, 30)),  # irrelevant class
        ],
    },
    {"index": 1, "labels": [_label(1, "pedestrian", _box(1, 1, 11, 31))]},
]
PRED_FRAMES: list[dict[str, Any]] = [
    {
        "index": 0,
        "labels": [
            _label(10, "pedestrian", _box(0, 0, 10, 30)),  # matches gt 1
            _label(20, "pedestrian", _box(20, 0, 30, 30)),  # unmatched, in crowd region
            _label(50, "pedestrian", _box(200, 0, 230, 30)),  # unmatched FP, kept
            _label(60, "car", _box(80, 0, 90, 30)),  # irrelevant class
        ],
    },
    {"index": 1, "labels": [_label(10, "pedestrian", _box(1, 1, 11, 31))]},
]
NUM_TIMESTEPS = 2


def _motrics_result(
    tmp_path: Path,
) -> tuple[list[list[int]], list[list[tuple]], list[list[int]], list[list[tuple]]]:
    gt_path, pred_path = tmp_path / "gt.json", tmp_path / "pred.json"
    gt_path.write_text(json.dumps(GT_FRAMES), encoding="utf-8")
    pred_path.write_text(json.dumps(PRED_FRAMES), encoding="utf-8")
    gt, ignore = motrics.load_bdd100k_gt(gt_path)
    pred = motrics.load_bdd100k(pred_path)
    return motrics.preprocess_bdd100k(gt, pred, "pedestrian", ignore_regions=ignore)


def _dets(labels: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[la["box2d"][k] for k in ("x1", "y1", "x2", "y2")] for la in labels],
        dtype=float,
    ).reshape(-1, 4)


def _trackeval_result() -> dict[str, Any]:
    """Build TrackEval's raw_data straight from the labels above (an independent
    parse, not routed through motrics' own ingest) and run its real (unmodified)
    preprocessing."""
    gt_ids, gt_dets, gt_classes, gt_ignore = [], [], [], []
    tracker_ids, tracker_dets, tracker_classes, similarity_scores = [], [], [], []
    for t in range(NUM_TIMESTEPS):
        g = GT_FRAMES[t]["labels"]
        keep = [
            la
            for la in g
            if la["category"] not in _DISTRACTOR_CLASSES
            and not la["attributes"]["Crowd"]
        ]
        ign = [
            la
            for la in g
            if la["category"] in _DISTRACTOR_CLASSES or la["attributes"]["Crowd"]
        ]
        p = PRED_FRAMES[t]["labels"]

        g_xyxy, p_xyxy = _dets(keep), _dets(p)
        gt_ids.append(np.array([la["id"] for la in keep], dtype=int))
        gt_dets.append(g_xyxy)
        gt_classes.append(
            np.array([_CLASS_NAME_TO_ID[la["category"]] for la in keep], dtype=int)
        )
        gt_ignore.append(_dets(ign))
        tracker_ids.append(np.array([la["id"] for la in p], dtype=int))
        tracker_dets.append(p_xyxy)
        tracker_classes.append(
            np.array([_CLASS_NAME_TO_ID[la["category"]] for la in p], dtype=int)
        )
        similarity_scores.append(
            BDD100K._calculate_box_ious(g_xyxy, p_xyxy, box_format="x0y0x1y1")
        )

    raw_data = {
        "num_timesteps": NUM_TIMESTEPS,
        "gt_ids": gt_ids,
        "gt_dets": gt_dets,
        "gt_classes": gt_classes,
        "gt_crowd_ignore_regions": gt_ignore,
        "tracker_ids": tracker_ids,
        "tracker_dets": tracker_dets,
        "tracker_classes": tracker_classes,
        "similarity_scores": similarity_scores,
    }

    dataset = object.__new__(BDD100K)
    dataset.class_name_to_class_id = _CLASS_NAME_TO_ID
    return dataset.get_preprocessed_seq_data(raw_data, "pedestrian")


def _xyxy_set(boxes: np.ndarray) -> set[tuple[float, float, float, float]]:
    return {tuple(box) for box in boxes.tolist()}


def test_preprocess_bdd100k_matches_trackeval(tmp_path: Path) -> None:
    m_gt_ids, m_gt_boxes, m_pred_ids, m_pred_boxes = _motrics_result(tmp_path)
    te = _trackeval_result()

    assert te["num_gt_dets"] == sum(len(f) for f in m_gt_ids)
    assert te["num_tracker_dets"] == sum(len(f) for f in m_pred_ids)

    for t in range(NUM_TIMESTEPS):
        assert set(m_gt_boxes[t]) == _xyxy_set(te["gt_dets"][t])
        assert set(m_pred_boxes[t]) == _xyxy_set(te["tracker_dets"][t])

    # Sanity check the scenario actually exercises every case.
    assert m_gt_ids == [[1], [1]]  # crowd/distractor/car gt gone
    assert m_pred_ids == [[10, 50], [10]]  # 20 dropped (crowd region), 60 is a car


def test_preprocess_bdd100k_rejects_invalid_class() -> None:
    with pytest.raises(ValueError, match="unknown class"):
        motrics.preprocess_bdd100k({}, {}, "trailer")
