"""DanceTrack support: confirmed via the existing MOTChallenge ingest, no new
adapter code needed — its gt.txt/results format is byte-for-byte MOTChallenge's
(always class=1/consider=1), and TrackEval evaluates it via plain
MotChallenge2DBox with no DanceTrack-specific preprocessing branch.

Round-trips a synthetic sequence through DanceTrack-formatted files and checks
it against the original sequence, TrackEval's real (unmodified) preprocessing,
and TrackEval's real metric classes — not just self-consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import motrics
import pytest

from benchmarks.fixtures import Sequence, make_synthetic

np = pytest.importorskip("numpy")
trackeval_datasets = pytest.importorskip("trackeval.datasets")
trackeval_metrics = pytest.importorskip("trackeval.metrics")

# Validated against trackeval==1.3.0 (see tests/test_motchallenge_preprocessing.py
# for the same reach-into-internals caveat).
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

SEQ = next(s for s in make_synthetic() if s.name == "synth-medium")


def _write_dancetrack_files(tmp_path: Path, seq: Sequence) -> tuple[Path, Path]:
    gt_path, pred_path = tmp_path / "gt.txt", tmp_path / "pred.txt"
    gt_lines = [
        f"{t + 1},{i},{x1},{y1},{x2 - x1},{y2 - y1},1,1,1"
        for t in range(seq.num_frames)
        for i, (x1, y1, x2, y2) in zip(seq.gt_ids[t], seq.gt_boxes[t], strict=True)
    ]
    pred_lines = [
        f"{t + 1},{i},{x1},{y1},{x2 - x1},{y2 - y1},1.0,-1,-1,-1"
        for t in range(seq.num_frames)
        for i, (x1, y1, x2, y2) in zip(seq.pred_ids[t], seq.pred_boxes[t], strict=True)
    ]
    gt_path.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    pred_path.write_text("\n".join(pred_lines) + "\n", encoding="utf-8")
    return gt_path, pred_path


def _motrics_preprocessed(
    gt_path: Path, pred_path: Path
) -> tuple[list[list[int]], list[list[tuple]], list[list[int]], list[list[tuple]]]:
    gt = motrics.load_motchallenge_gt(gt_path)
    pred = motrics.load_motchallenge(pred_path)
    return motrics.preprocess_motchallenge(gt, pred, benchmark="DanceTrack")


def _trackeval_preprocessed(seq: Sequence) -> dict[str, Any]:
    """TrackEval's real (unmodified) preprocessing, fed the same sequence
    independently (not routed through motrics' ingest)."""
    gt_ids = [np.array(f, dtype=int) for f in seq.gt_ids]
    tracker_ids = [np.array(f, dtype=int) for f in seq.pred_ids]
    gt_dets = [
        np.array(
            [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in f], dtype=float
        ).reshape(-1, 4)
        for f in seq.gt_boxes
    ]
    tracker_dets = [
        np.array(
            [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in f], dtype=float
        ).reshape(-1, 4)
        for f in seq.pred_boxes
    ]
    raw_data = {
        "num_timesteps": seq.num_frames,
        "seq": seq.name,
        "gt_ids": gt_ids,
        "gt_dets": gt_dets,
        "gt_classes": [np.ones(len(f), dtype=int) for f in seq.gt_ids],
        "gt_extras": [{"zero_marked": np.ones(len(f), dtype=int)} for f in seq.gt_ids],
        "tracker_ids": tracker_ids,
        "tracker_dets": tracker_dets,
        "tracker_classes": [np.ones(len(f), dtype=int) for f in seq.pred_ids],
        "tracker_confidences": [np.ones(len(f), dtype=float) for f in seq.pred_ids],
        "similarity_scores": [
            MotChallenge2DBox._calculate_box_ious(g, p, box_format="xywh")
            for g, p in zip(gt_dets, tracker_dets, strict=True)
        ],
    }
    dataset = object.__new__(MotChallenge2DBox)
    dataset.benchmark = "DanceTrack"
    dataset.do_preproc = True
    dataset.class_name_to_class_id = _CLASS_NAME_TO_ID
    dataset.valid_class_numbers = list(_CLASS_NAME_TO_ID.values())
    return dataset.get_preprocessed_seq_data(raw_data, "pedestrian")


def _xyxy_set(boxes_xywh: np.ndarray) -> set[tuple[float, float, float, float]]:
    return {(x, y, x + w, y + h) for x, y, w, h in boxes_xywh.tolist()}


def test_dancetrack_preprocessing_is_a_no_op(tmp_path: Path) -> None:
    gt_path, pred_path = _write_dancetrack_files(tmp_path, SEQ)
    gt_ids, gt_boxes, pred_ids, pred_boxes = _motrics_preprocessed(gt_path, pred_path)

    # Nothing dropped: DanceTrack has no distractor classes and every row is
    # always "consider" — round-trips back to exactly the original sequence.
    assert gt_ids == SEQ.gt_ids
    assert pred_ids == SEQ.pred_ids
    assert gt_boxes == SEQ.gt_boxes
    assert pred_boxes == SEQ.pred_boxes


def test_dancetrack_preprocessing_matches_trackeval(tmp_path: Path) -> None:
    gt_path, pred_path = _write_dancetrack_files(tmp_path, SEQ)
    m_gt_ids, m_gt_boxes, m_pred_ids, m_pred_boxes = _motrics_preprocessed(
        gt_path, pred_path
    )
    te = _trackeval_preprocessed(SEQ)

    assert te["num_gt_dets"] == sum(len(f) for f in m_gt_ids)
    assert te["num_tracker_dets"] == sum(len(f) for f in m_pred_ids)
    for t in range(SEQ.num_frames):
        assert set(m_gt_boxes[t]) == _xyxy_set(te["gt_dets"][t])
        assert set(m_pred_boxes[t]) == _xyxy_set(te["tracker_dets"][t])


def test_dancetrack_metrics_match_trackeval(tmp_path: Path) -> None:
    gt_path, pred_path = _write_dancetrack_files(tmp_path, SEQ)
    gt_ids, gt_boxes, pred_ids, pred_boxes = _motrics_preprocessed(gt_path, pred_path)
    args = (gt_ids, gt_boxes, pred_ids, pred_boxes)

    gt_map = {v: k for k, v in enumerate(sorted({i for f in gt_ids for i in f}))}
    tr_map = {v: k for k, v in enumerate(sorted({i for f in pred_ids for i in f}))}
    te_data = {
        "num_timesteps": len(gt_ids),
        "num_gt_ids": len(gt_map),
        "num_tracker_ids": len(tr_map),
        "num_gt_dets": sum(len(f) for f in gt_ids),
        "num_tracker_dets": sum(len(f) for f in pred_ids),
        "gt_ids": [np.array([gt_map[i] for i in f], dtype=int) for f in gt_ids],
        "tracker_ids": [np.array([tr_map[i] for i in f], dtype=int) for f in pred_ids],
        "similarity_scores": [
            np.array(motrics.iou_matrix(g, p), dtype=float).reshape(len(g), len(p))
            for g, p in zip(gt_boxes, pred_boxes, strict=True)
        ],
    }

    clear_ref = trackeval_metrics.CLEAR(
        {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    ).eval_sequence(te_data)
    identity_ref = trackeval_metrics.Identity(
        {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    ).eval_sequence(te_data)
    hota_ref = trackeval_metrics.HOTA({"PRINT_CONFIG": False}).eval_sequence(te_data)

    clear = motrics.compute_clear(*args)
    identity = motrics.compute_identity(*args)
    hota = motrics.compute_hota(*args)

    assert clear.mota == pytest.approx(clear_ref["MOTA"], abs=1e-9)
    assert clear.motp == pytest.approx(clear_ref["MOTP"], abs=1e-9)
    assert identity.idf1 == pytest.approx(identity_ref["IDF1"], abs=1e-9)
    assert np.allclose(hota.hota_alphas, hota_ref["HOTA"], atol=1e-9)
