"""Decision-equivalence: ``motrics.preprocess_kitti_3d`` vs the TrackEval-validated
``motrics.preprocess_kitti`` (2D) on the same scenario.

TrackEval has no KITTI-3D dataset class, so there is no reference framework
to run this adapter's preprocessing against directly (see the module
docstring in ``motrics/kitti_3d.py``). Instead, this reuses the exact
scenario from ``tests/test_kitti_preprocessing.py`` (already parity-tested
against TrackEval's real ``Kitti2DBox.get_preprocessed_seq_data``) and gives
each object a 3D box: identical for a ground truth and its "designated"
matching prediction, zeroed out (zero-volume, so its 3D IoU with anything is
always 0) for rows that never depend on 3D matching. Since
``preprocess_kitti_3d`` applies the exact same distractor/occlusion/
truncation/min-height/DontCare rules as ``preprocess_kitti`` and only swaps
2D IoU for 3D IoU, the keep/drop decision on this scenario must come out
identical to the already-validated 2D case.
"""

from __future__ import annotations

from pathlib import Path

import motrics
import pytest

# A box (h, w, l, x, y, z, ry): non-degenerate, distinct per matching gt/pred
# pair (spaced apart in x so pairs can't cross-match).
_BOX = (1.8, 0.6, 0.6)  # h, w, l shared by every non-degenerate object below
_ZERO3D = (0, 0, 0, 0, 0, 0, 0)  # zero volume -> zero IoU with anything


def _box_at(x: float) -> tuple:
    h, w, length = _BOX
    return (h, w, length, x, h, 10, 0)


# frame, id, type, truncated, occluded, alpha, left, top, right, bottom,
# then a 3D box (h, w, l, x, y, z, ry), then [score] for predictions.
GT_ROWS = [
    (1, 1, "Pedestrian", 0, 0, 0, 0, 0, 10, 30, *_box_at(0)),  # plain
    (1, 2, "Person", 0, 0, 0, 20, 0, 30, 30, *_box_at(5)),  # distractor
    (1, 3, "Pedestrian", 0, 3, 0, 40, 0, 50, 30, *_box_at(10)),  # occluded
    (1, 4, "Pedestrian", 1, 0, 0, 60, 0, 70, 30, *_box_at(15)),  # truncated
    (1, 5, "Car", 0, 0, 0, 80, 0, 90, 30, *_ZERO3D),  # irrelevant class
    (1, -1, "DontCare", 0, 0, 0, 100, 0, 120, 30, *_ZERO3D),  # ignore region
    (2, 1, "Pedestrian", 0, 0, 0, 1, 1, 11, 31, *_box_at(0)),
]
PRED_ROWS = [
    (1, 10, "Pedestrian", -1, -1, -1, 0, 0, 10, 30, *_box_at(0), 0.9),
    (1, 20, "Pedestrian", -1, -1, -1, 20, 0, 30, 30, *_box_at(5), 0.9),
    (1, 30, "Pedestrian", -1, -1, -1, 40, 0, 50, 30, *_box_at(10), 0.9),
    (1, 40, "Pedestrian", -1, -1, -1, 60, 0, 70, 30, *_box_at(15), 0.9),
    (1, 50, "Pedestrian", -1, -1, -1, 200, 0, 230, 30, *_ZERO3D, 0.9),  # real FP
    (1, 60, "Pedestrian", -1, -1, -1, 300, 0, 315, 20, *_ZERO3D, 0.9),  # too short
    (1, 70, "Pedestrian", -1, -1, -1, 102, 2, 118, 28, *_ZERO3D, 0.9),  # in DontCare
    (2, 10, "Pedestrian", -1, -1, -1, 1, 1, 11, 31, *_box_at(0), 0.9),
]
NUM_TIMESTEPS = 2


def _write_rows(path: Path, rows: list[tuple]) -> None:
    path.write_text(
        "\n".join(" ".join(str(v) for v in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_preprocess_kitti_3d_matches_2d_decisions(tmp_path: Path) -> None:
    gt_path, pred_path = tmp_path / "gt.txt", tmp_path / "pred.txt"
    _write_rows(gt_path, GT_ROWS)
    _write_rows(pred_path, PRED_ROWS)

    gt, ignore = motrics.load_kitti_3d_gt(gt_path)
    pred = motrics.load_kitti_3d(pred_path)
    gt_ids, pred_ids, similarity = motrics.preprocess_kitti_3d(
        gt, pred, "pedestrian", ignore_regions=ignore
    )

    # Same keep/drop decisions as test_kitti_preprocessing.py's TrackEval-validated
    # 2D scenario: distractor/occluded/truncated/car gt gone; 20/30/40 dropped
    # (matched to a distractor/occluded/truncated gt); 60 dropped (too short);
    # 70 dropped (mostly inside the DontCare region).
    assert gt_ids == [[1], [1]]
    assert pred_ids == [[10, 50], [10]]

    # gt 1 vs (pred 10, pred 50): 1.0 for the identical box, 0.0 against the
    # kept-but-unmatched FP (pred 50 has a zero-volume box).
    assert similarity[0][0] == pytest.approx([1.0, 0.0], abs=1e-9)
    assert similarity[1][0] == pytest.approx([1.0], abs=1e-9)


def test_preprocess_kitti_3d_rejects_invalid_class() -> None:
    with pytest.raises(ValueError, match='"pedestrian" or "car"'):
        motrics.preprocess_kitti_3d({}, {}, "cyclist")
