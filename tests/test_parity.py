"""Numeric parity tests against TrackEval's reference metric implementations.

Both engines are fed identical inputs (the IoU/similarity is computed once with
``motrics.iou_matrix`` and handed to TrackEval), so any difference is a genuine
metric-math discrepancy. Skipped automatically where TrackEval isn't installed
(it lives in the optional ``parity`` dependency group).
"""

from __future__ import annotations

import random
from pathlib import Path

import motrics
import pytest

np = pytest.importorskip("numpy")
te_clear = pytest.importorskip("trackeval.metrics").CLEAR
te_identity = pytest.importorskip("trackeval.metrics").Identity
te_hota = pytest.importorskip("trackeval.metrics").HOTA

Frame = tuple[list[int], list[tuple[float, float, float, float]], list[int], list]


def _box(x: float, y: float, w: float = 10.0, h: float = 10.0) -> tuple:
    return (float(x), float(y), float(x + w), float(y + h))


def _build_trackeval_data(seq: list[Frame]) -> dict:
    """Convert a sequence into TrackEval's per-sequence data dict."""
    gt_id_set = sorted({i for f in seq for i in f[0]})
    tr_id_set = sorted({i for f in seq for i in f[2]})
    gt_map = {v: k for k, v in enumerate(gt_id_set)}
    tr_map = {v: k for k, v in enumerate(tr_id_set)}

    gt_ids, tracker_ids, sims = [], [], []
    for g_ids, g_boxes, p_ids, p_boxes in seq:
        gt_ids.append(np.array([gt_map[i] for i in g_ids], dtype=int))
        tracker_ids.append(np.array([tr_map[i] for i in p_ids], dtype=int))
        arr = np.array(motrics.iou_matrix(g_boxes, p_boxes), dtype=float)
        if arr.size == 0:
            arr = arr.reshape(len(g_boxes), len(p_boxes))
        sims.append(arr)

    return {
        "num_timesteps": len(seq),
        "num_gt_ids": len(gt_id_set),
        "num_tracker_ids": len(tr_id_set),
        "num_gt_dets": sum(len(f[0]) for f in seq),
        "num_tracker_dets": sum(len(f[2]) for f in seq),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": sims,
    }


def _motrics_args(seq: list[Frame]):
    return (
        [list(f[0]) for f in seq],
        [list(f[1]) for f in seq],
        [list(f[2]) for f in seq],
        [list(f[3]) for f in seq],
    )


def _random_sequence(seed: int, n_frames: int = 30, n_obj: int = 6) -> list[Frame]:
    rng = random.Random(seed)
    seq: list[Frame] = []
    for t in range(n_frames):
        g_ids, g_boxes, p_ids, p_boxes = [], [], [], []
        for o in range(n_obj):
            if rng.random() < 0.9:  # gt object present
                x, y = 30.0 * o + 1.5 * t, 10.0
                g_ids.append(o)
                g_boxes.append(_box(x, y))
                r = rng.random()
                if r < 0.85:  # detected (with jitter -> IoU varies across thresholds)
                    jx, jy = rng.uniform(-4, 4), rng.uniform(-4, 4)
                    pid = o if rng.random() > 0.05 else o + 100  # rare id switch
                    p_ids.append(pid)
                    p_boxes.append(_box(x + jx, y + jy))
        if rng.random() < 0.3:  # spurious false positive
            p_ids.append(900 + t)
            p_boxes.append(_box(rng.uniform(0, 300), 200.0))
        seq.append((g_ids, g_boxes, p_ids, p_boxes))
    return seq


# A couple of hand-built sequences plus several seeded random ones.
_HANDBUILT: list[list[Frame]] = [
    [
        ([0, 1], [_box(0, 0), _box(50, 50)], [0, 1], [_box(0, 0), _box(50, 51)]),
        ([0, 1], [_box(0, 0), _box(50, 50)], [0, 9], [_box(0, 1), _box(50, 50)]),
        ([0, 1], [_box(0, 0), _box(50, 50)], [0], [_box(0, 0)]),
        ([0], [_box(0, 0)], [0, 5], [_box(0, 0), _box(200, 200)]),
    ],
]

# Shared MOTChallenge-format fixtures — the *same* committed sequences the
# benchmark suite measures (benchmarks/data). Loading them here through the
# public reader means parity is validated on the exact inputs we benchmark,
# rather than on a separate synthetic generator.
_FIXTURES_DIR = Path(__file__).parents[1] / "benchmarks" / "data"


def _load_fixture(seq_dir: Path) -> list[Frame]:
    gt = motrics.load_motchallenge(seq_dir / "gt" / "gt.txt")
    pred = motrics.load_motchallenge(seq_dir / "pred.txt")
    gt_ids, gt_boxes, pred_ids, pred_boxes = motrics.align_frames(gt, pred)
    return [
        (gt_ids[t], gt_boxes[t], pred_ids[t], pred_boxes[t]) for t in range(len(gt_ids))
    ]


_FIXTURE_DIRS = (
    sorted(p for p in _FIXTURES_DIR.iterdir() if (p / "gt" / "gt.txt").is_file())
    if _FIXTURES_DIR.exists()
    else []
)

SEQUENCES = (
    _HANDBUILT
    + [_random_sequence(s) for s in range(6)]
    + [_load_fixture(p) for p in _FIXTURE_DIRS]
)
SEQUENCE_IDS = (
    [f"handbuilt{i}" for i in range(len(_HANDBUILT))]
    + [f"random{i}" for i in range(6)]
    + [f"fixture-{p.name}" for p in _FIXTURE_DIRS]
)


@pytest.mark.parametrize("seq", SEQUENCES, ids=SEQUENCE_IDS)
def test_clear_parity(seq: list[Frame]) -> None:
    ref = te_clear({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(
        _build_trackeval_data(seq)
    )
    m = motrics.compute_clear(*_motrics_args(seq))
    assert m.num_matches == ref["CLR_TP"]
    assert m.num_false_positives == ref["CLR_FP"]
    assert m.num_misses == ref["CLR_FN"]
    assert m.num_switches == ref["IDSW"]
    assert m.mota == pytest.approx(ref["MOTA"], abs=1e-9)
    assert m.motp == pytest.approx(ref["MOTP"], abs=1e-9)


@pytest.mark.parametrize("seq", SEQUENCES, ids=SEQUENCE_IDS)
def test_identity_parity(seq: list[Frame]) -> None:
    ref = te_identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(
        _build_trackeval_data(seq)
    )
    m = motrics.compute_identity(*_motrics_args(seq))
    assert m.idtp == ref["IDTP"]
    assert m.idfp == ref["IDFP"]
    assert m.idfn == ref["IDFN"]
    assert m.idf1 == pytest.approx(ref["IDF1"], abs=1e-9)
    assert m.idp == pytest.approx(ref["IDP"], abs=1e-9)
    assert m.idr == pytest.approx(ref["IDR"], abs=1e-9)


@pytest.mark.parametrize("seq", SEQUENCES, ids=SEQUENCE_IDS)
def test_hota_parity(seq: list[Frame]) -> None:
    ref = te_hota({"PRINT_CONFIG": False}).eval_sequence(_build_trackeval_data(seq))
    m = motrics.compute_hota(*_motrics_args(seq))
    # Per-alpha curves must match, not just the summary means.
    assert np.allclose(m.hota_alphas, ref["HOTA"], atol=1e-9)
    assert np.allclose(m.deta_alphas, ref["DetA"], atol=1e-9)
    assert np.allclose(m.assa_alphas, ref["AssA"], atol=1e-9)
    assert m.hota == pytest.approx(float(np.mean(ref["HOTA"])), abs=1e-9)
    assert m.deta == pytest.approx(float(np.mean(ref["DetA"])), abs=1e-9)
    assert m.assa == pytest.approx(float(np.mean(ref["AssA"])), abs=1e-9)
    assert m.loca == pytest.approx(float(np.mean(ref["LocA"])), abs=1e-9)
