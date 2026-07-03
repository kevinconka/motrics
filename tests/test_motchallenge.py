"""Tests for MOTChallenge ingest and end-to-end metric integration."""

from pathlib import Path

import motrics
import pytest

# A tiny two-frame sequence, one object, in MOTChallenge format
# (frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z).
GT_TEXT = """\
1,1,10,10,20,20,1,-1,-1,-1
2,1,10,10,20,20,1,-1,-1,-1
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_converts_xywh_to_xyxy(tmp_path: Path) -> None:
    path = _write(tmp_path, "gt.txt", "1,7,10,20,30,40,1,-1,-1,-1\n")
    frames = motrics.load_motchallenge(path)
    ids, boxes = frames[1]
    assert ids == [7]
    # left=10, top=20, width=30, height=40 -> (10, 20, 40, 60)
    assert boxes == [(10.0, 20.0, 40.0, 60.0)]


def test_min_confidence_filters_rows(tmp_path: Path) -> None:
    text = "1,1,0,0,10,10,0.9,-1,-1,-1\n1,2,0,0,10,10,0.2,-1,-1,-1\n"
    path = _write(tmp_path, "res.txt", text)
    frames = motrics.load_motchallenge(path, min_confidence=0.5)
    ids, _ = frames[1]
    assert ids == [1]


def test_align_frames_pads_missing(tmp_path: Path) -> None:
    gt = motrics.load_motchallenge(_write(tmp_path, "gt.txt", GT_TEXT))
    # Prediction only present in frame 2.
    pred = motrics.load_motchallenge(
        _write(tmp_path, "res.txt", "2,1,10,10,20,20,1,-1,-1,-1\n")
    )
    gt_ids, gt_boxes, pred_ids, pred_boxes = motrics.align_frames(gt, pred)
    assert gt_ids == [[1], [1]]
    assert pred_ids == [[], [1]]  # frame 1 padded empty for pred
    assert len(gt_boxes) == len(pred_boxes) == 2


def test_load_motchallenge_gt_reads_class_and_consider_flag(tmp_path: Path) -> None:
    # frame, id, left, top, w, h, consider, class, visibility.
    text = "1,1,0,0,10,10,1,1,1.0\n1,2,20,20,10,10,0,8,1.0\n"
    path = _write(tmp_path, "gt.txt", text)
    ids, _boxes, classes, keep = motrics.load_motchallenge_gt(path)[1]
    assert ids == [1, 2]
    assert classes == [1, 8]
    assert keep == [True, False]


def test_preprocess_motchallenge_drops_pred_matched_to_distractor(
    tmp_path: Path,
) -> None:
    gt = {
        1: (
            [1, 2],
            [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)],
            [1, 8],
            [True, True],
        )
    }
    pred = {
        1: (
            [10, 20, 30],
            [
                (0.0, 0.0, 10.0, 10.0),
                (20.0, 20.0, 30.0, 30.0),
                (100.0, 100.0, 110.0, 110.0),
            ],
        )
    }
    gt_ids, _gt_boxes, pred_ids, _pred_boxes = motrics.preprocess_motchallenge(gt, pred)
    assert gt_ids == [[1]]  # distractor (2) dropped
    assert pred_ids == [[10, 30]]  # pred matched to the distractor (20) dropped


def test_preprocess_motchallenge_drops_do_not_consider_gt(tmp_path: Path) -> None:
    gt = {1: ([1], [(0.0, 0.0, 10.0, 10.0)], [1], [False])}
    pred = {1: ([10], [(0.0, 0.0, 10.0, 10.0)])}
    gt_ids, _gt_boxes, pred_ids, _pred_boxes = motrics.preprocess_motchallenge(gt, pred)
    assert gt_ids == [[]]
    assert pred_ids == [[10]]  # not a distractor match, so pred is kept


def test_preprocess_motchallenge_mot20_extra_distractor_class(tmp_path: Path) -> None:
    gt = {
        1: (
            [1, 2],
            [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)],
            [1, 6],
            [True, True],
        )
    }
    pred = {1: ([10, 20], [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)])}
    _, _, mot17_pred_ids, _ = motrics.preprocess_motchallenge(
        gt, pred, benchmark="MOT17"
    )
    _, _, mot20_pred_ids, _ = motrics.preprocess_motchallenge(
        gt, pred, benchmark="MOT20"
    )
    assert mot17_pred_ids == [[10, 20]]  # class 6 isn't a distractor for MOT17
    assert mot20_pred_ids == [[10]]  # ... but is for MOT20


def test_end_to_end_perfect_sequence(tmp_path: Path) -> None:
    # Identical gt and prediction -> perfect scores across every metric.
    gt = motrics.load_motchallenge(_write(tmp_path, "gt.txt", GT_TEXT))
    pred = motrics.load_motchallenge(_write(tmp_path, "res.txt", GT_TEXT))
    args = motrics.align_frames(gt, pred)

    clear = motrics.compute_clear(*args)
    identity = motrics.compute_identity(*args)
    hota = motrics.compute_hota(*args)

    assert clear.mota == pytest.approx(1.0)
    assert clear.num_switches == 0
    assert identity.idf1 == pytest.approx(1.0)
    assert hota.hota == pytest.approx(1.0)
