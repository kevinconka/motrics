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
