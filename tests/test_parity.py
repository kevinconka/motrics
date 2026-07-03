"""Numeric parity against TrackEval.

TrackEval is fed the same similarities motrics computes (``motrics.iou_matrix``),
so any difference is a genuine metric-math discrepancy. Runs on the same
in-memory sequences the benchmark uses, and skips when TrackEval isn't installed
(the optional ``parity`` dependency group).
"""

from __future__ import annotations

import motrics
import pytest

from benchmarks.fixtures import Sequence, make_synthetic

np = pytest.importorskip("numpy")
te = pytest.importorskip("trackeval.metrics")


def _box(x: float, y: float, w: float = 10.0, h: float = 10.0) -> tuple:
    return (float(x), float(y), float(x + w), float(y + h))


# A hand-built sequence exercising an id switch, a miss and a false positive.
_HANDBUILT = Sequence(
    name="handbuilt",
    gt_ids=[[0, 1], [0, 1], [0, 1], [0]],
    gt_boxes=[
        [_box(0, 0), _box(50, 50)],
        [_box(0, 0), _box(50, 50)],
        [_box(0, 0), _box(50, 50)],
        [_box(0, 0)],
    ],
    pred_ids=[[0, 1], [0, 9], [0], [0, 5]],
    pred_boxes=[
        [_box(0, 0), _box(50, 51)],
        [_box(0, 1), _box(50, 50)],
        [_box(0, 0)],
        [_box(0, 0), _box(200, 200)],
    ],
)

SEQUENCES = [_HANDBUILT, *make_synthetic()]


def _trackeval_data(seq: Sequence) -> dict:
    """TrackEval's per-sequence data dict, with similarities from motrics."""
    gt_map = {v: k for k, v in enumerate(sorted({i for f in seq.gt_ids for i in f}))}
    tr_map = {v: k for k, v in enumerate(sorted({i for f in seq.pred_ids for i in f}))}
    gt_ids, tracker_ids, sims = [], [], []
    for t in range(seq.num_frames):
        gt_ids.append(np.array([gt_map[i] for i in seq.gt_ids[t]], dtype=int))
        tracker_ids.append(np.array([tr_map[i] for i in seq.pred_ids[t]], dtype=int))
        arr = np.array(
            motrics.iou_matrix(seq.gt_boxes[t], seq.pred_boxes[t]), dtype=float
        )
        sims.append(arr.reshape(len(seq.gt_boxes[t]), len(seq.pred_boxes[t])))
    return {
        "num_timesteps": seq.num_frames,
        "num_gt_ids": len(gt_map),
        "num_tracker_ids": len(tr_map),
        "num_gt_dets": seq.num_gt_dets,
        "num_tracker_dets": seq.num_pred_dets,
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": sims,
    }


def _args(seq: Sequence) -> tuple:
    return seq.gt_ids, seq.gt_boxes, seq.pred_ids, seq.pred_boxes


@pytest.mark.parametrize("seq", SEQUENCES, ids=lambda s: s.name)
def test_clear_parity(seq: Sequence) -> None:
    ref = te.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(
        _trackeval_data(seq)
    )
    m = motrics.compute_clear(*_args(seq))
    assert m.num_matches == ref["CLR_TP"]
    assert m.num_false_positives == ref["CLR_FP"]
    assert m.num_misses == ref["CLR_FN"]
    assert m.num_switches == ref["IDSW"]
    assert m.mota == pytest.approx(ref["MOTA"], abs=1e-9)
    assert m.motp == pytest.approx(ref["MOTP"], abs=1e-9)


@pytest.mark.parametrize("seq", SEQUENCES, ids=lambda s: s.name)
def test_identity_parity(seq: Sequence) -> None:
    ref = te.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(
        _trackeval_data(seq)
    )
    m = motrics.compute_identity(*_args(seq))
    assert m.idtp == ref["IDTP"]
    assert m.idfp == ref["IDFP"]
    assert m.idfn == ref["IDFN"]
    assert m.idf1 == pytest.approx(ref["IDF1"], abs=1e-9)
    assert m.idp == pytest.approx(ref["IDP"], abs=1e-9)
    assert m.idr == pytest.approx(ref["IDR"], abs=1e-9)


@pytest.mark.parametrize("seq", SEQUENCES, ids=lambda s: s.name)
def test_hota_parity(seq: Sequence) -> None:
    ref = te.HOTA({"PRINT_CONFIG": False}).eval_sequence(_trackeval_data(seq))
    m = motrics.compute_hota(*_args(seq))
    assert np.allclose(m.hota_alphas, ref["HOTA"], atol=1e-9)
    assert np.allclose(m.deta_alphas, ref["DetA"], atol=1e-9)
    assert np.allclose(m.assa_alphas, ref["AssA"], atol=1e-9)
    assert m.loca == pytest.approx(float(np.mean(ref["LocA"])), abs=1e-9)
