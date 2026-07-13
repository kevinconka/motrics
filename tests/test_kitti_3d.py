"""Tests for KITTI-3D ingest and end-to-end metric integration."""

from pathlib import Path

import motrics
import pytest

# frame, track_id, type, truncated, occluded, alpha, left, top, right, bottom,
# h, w, l, x, y, z, ry. Location (x, y, z) is the box's bottom centre; y is
# shifted by h/2 on load to get the true vertical centre iou_3d expects.
GT_TEXT = """\
1 1 Pedestrian 0 0 0 0 0 10 30 1.8 0.6 0.6 0.0 1.8 10.0 0.0
2 1 Pedestrian 0 0 0 1 1 11 31 1.8 0.6 0.6 0.0 1.8 10.0 0.0
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_kitti_3d_parses_boxes_and_shifts_bottom_to_centre(
    tmp_path: Path,
) -> None:
    text = "1 7 Pedestrian 0 0 0 10 20 40 60 1.8 0.6 0.6 5.0 1.8 12.0 0.3\n"
    frames = motrics.load_kitti_3d(_write(tmp_path, "res.txt", text))
    ids, boxes3d, boxes2d, classes = frames[1]
    assert ids == [7]
    assert boxes2d == [(10.0, 20.0, 40.0, 60.0)]
    # y: 1.8 (bottom) - 1.8/2 (half-height) = 0.9 (centre).
    assert boxes3d == [(5.0, 0.9, 12.0, 0.6, 0.6, 1.8, 0.3)]
    assert classes == [4]  # pedestrian


def test_load_kitti_3d_drops_negative_ids(tmp_path: Path) -> None:
    text = "1 -1 Pedestrian 0 0 0 0 0 10 10 1.8 0.6 0.6 0 1.8 10 0\n"
    frames = motrics.load_kitti_3d(_write(tmp_path, "res.txt", text))
    assert frames == {}


def test_load_kitti_3d_drops_unknown_class(tmp_path: Path) -> None:
    text = "1 7 Misc-not-a-real-class 0 0 0 0 0 10 10 1.8 0.6 0.6 0 1.8 10 0\n"
    frames = motrics.load_kitti_3d(_write(tmp_path, "res.txt", text))
    assert frames == {}


def test_load_kitti_3d_gt_separates_dontcare_regions(tmp_path: Path) -> None:
    text = (
        "1 1 Pedestrian 0 0 0 0 0 10 10 1.8 0.6 0.6 0 1.8 10 0\n"
        "1 -1 DontCare 0 0 0 20 20 30 30 0 0 0 0 0 0 0\n"
    )
    gt, ignore = motrics.load_kitti_3d_gt(_write(tmp_path, "gt.txt", text))
    ids, boxes3d, boxes2d, classes, truncation, occlusion = gt[1]
    assert ids == [1]
    assert boxes2d == [(0.0, 0.0, 10.0, 10.0)]
    assert boxes3d == [(0.0, 0.9, 10.0, 0.6, 0.6, 1.8, 0.0)]
    assert classes == [4]
    assert truncation == [0]
    assert occlusion == [0]
    # DontCare rows carry no meaningful 3D box: only their 2D box is kept.
    assert ignore == {1: [(20.0, 20.0, 30.0, 30.0)]}


def test_load_kitti_3d_gt_reads_truncation_and_occlusion(tmp_path: Path) -> None:
    text = "1 1 Pedestrian 1 3 0 0 0 10 10 1.8 0.6 0.6 0 1.8 10 0\n"
    gt, _ignore = motrics.load_kitti_3d_gt(_write(tmp_path, "gt.txt", text))
    _ids, _boxes3d, _boxes2d, _classes, truncation, occlusion = gt[1]
    assert truncation == [1]
    assert occlusion == [3]


def test_preprocess_kitti_3d_rejects_unknown_class() -> None:
    with pytest.raises(ValueError, match="unknown class"):
        motrics.preprocess_kitti_3d({}, {}, "bicycle")


def test_preprocess_kitti_3d_keeps_real_false_positive() -> None:
    box3d = (0.0, 1.5, 10.0, 0.6, 0.6, 1.8, 0.0)
    box2d = (0.0, 0.0, 10.0, 30.0)  # height 30, above min_height
    pred = {1: ([10], [box3d], [box2d], [4])}
    _gt_ids, pred_ids, _sim = motrics.preprocess_kitti_3d({}, pred, "pedestrian")
    assert pred_ids == [[10]]


def test_preprocess_kitti_3d_drops_short_unmatched_pred() -> None:
    box3d = (0.0, 1.0, 10.0, 0.6, 0.6, 2.0, 0.0)
    box2d = (0.0, 0.0, 10.0, 20.0)  # height 20 <= min_height 25
    pred = {1: ([10], [box3d], [box2d], [4])}
    _gt_ids, pred_ids, _sim = motrics.preprocess_kitti_3d({}, pred, "pedestrian")
    assert pred_ids == [[]]


def test_end_to_end_perfect_sequence(tmp_path: Path) -> None:
    # Identical gt and prediction -> perfect scores across every metric.
    gt, ignore = motrics.load_kitti_3d_gt(_write(tmp_path, "gt.txt", GT_TEXT))
    pred_text = GT_TEXT.replace("1 1 Pedestrian", "1 10 Pedestrian").replace(
        "2 1 Pedestrian", "2 10 Pedestrian"
    )
    pred = motrics.load_kitti_3d(_write(tmp_path, "res.txt", pred_text))
    args = motrics.preprocess_kitti_3d(gt, pred, "pedestrian", ignore_regions=ignore)

    clear = motrics.compute_clear_from_similarity(*args)
    identity = motrics.compute_identity_from_similarity(*args)
    hota = motrics.compute_hota_from_similarity(args[0], args[1], args[2])

    assert clear.mota == pytest.approx(1.0)
    assert clear.num_switches == 0
    assert identity.idf1 == pytest.approx(1.0)
    assert hota.hota == pytest.approx(1.0)


def test_end_to_end_identity_switch(tmp_path: Path) -> None:
    # Same gt object matched to two different predicted ids across frames.
    gt, ignore = motrics.load_kitti_3d_gt(_write(tmp_path, "gt.txt", GT_TEXT))
    pred_text = GT_TEXT.replace("1 1 Pedestrian", "1 10 Pedestrian").replace(
        "2 1 Pedestrian", "2 20 Pedestrian"
    )
    pred = motrics.load_kitti_3d(_write(tmp_path, "res.txt", pred_text))
    args = motrics.preprocess_kitti_3d(gt, pred, "pedestrian", ignore_regions=ignore)

    clear = motrics.compute_clear_from_similarity(*args)
    assert clear.num_matches == 2
    assert clear.num_switches == 1
