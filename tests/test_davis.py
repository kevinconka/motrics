"""Parity: ``motrics`` DAVIS ingest vs TrackEval's real ``DAVIS`` pipeline.

Writes a small DAVIS-format scenario (indexed-PNG masks, one per frame) and
runs it through both:

- ``motrics.load_davis`` + ``motrics.preprocess_davis``, then the
  ``compute_*_from_similarity`` metrics.
- TrackEval's own (unmodified) ``DAVIS.get_raw_seq_data`` /
  ``get_preprocessed_seq_data`` and its real ``CLEAR``/``Identity``/``HOTA``
  metric classes.

The scenario covers matched objects, an id swap between frames (association),
a ground-truth object the tracker misses, and a tracker false positive that
lands entirely inside a void region — confirming, per TrackEval, that void
regions do not rescue such a detection from counting as a false positive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import motrics
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
pytest.importorskip("pycocotools")
trackeval_datasets = pytest.importorskip("trackeval.datasets")
trackeval_metrics = pytest.importorskip("trackeval.metrics")

# Validated against trackeval==1.3.0 (see tests/test_motchallenge_preprocessing.py
# for the same reach-into-internals caveat).
DAVIS = trackeval_datasets.davis.DAVIS

_H, _W = 8, 8
_VOID = 255
_SEQ = "seq1"


def _frame(objects: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    """Build an index array; ``objects`` maps a pixel value to an
    ``(r0, r1, c0, c1)`` block."""
    frame = np.zeros((_H, _W), dtype=np.uint8)
    for value, (r0, r1, c0, c1) in objects.items():
        frame[r0:r1, c0:c1] = value
    return frame


# Two frames. gt objects 1 and 2 sit in fixed blocks; the tracker matches them
# but swaps their ids between frames (an association error), misses nothing in
# frame 0 but object 2 in frame 1, and adds object 3 sitting inside the void.
_A = (0, 2, 0, 2)  # object 1's block
_B = (0, 2, 2, 4)  # object 2's block
_V = (6, 8, 6, 8)  # void region (frame 0 only)

GT_FRAMES = [
    _frame({1: _A, 2: _B, _VOID: _V}),
    _frame({1: _A, 2: _B}),
]
PRED_FRAMES = [
    _frame({1: _A, 2: _B, 3: _V}),  # 3 lands entirely in the void region
    _frame({2: _A, 1: _B}),  # ids swapped vs gt -> id switches
]
NUM_FRAMES = len(GT_FRAMES)


def _write_seq(seq_dir: Path, frames: list[np.ndarray]) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for i, frame in enumerate(frames):
        Image.fromarray(frame, mode="L").save(seq_dir / f"{i:05d}.png")


def _motrics_preprocessed(
    tmp_path: Path,
) -> tuple[list[list[int]], list[list[int]], list[list[list[float]]]]:
    gt_dir = tmp_path / "gt" / _SEQ
    pred_dir = tmp_path / "tr" / "mytr" / "data" / _SEQ
    _write_seq(gt_dir, GT_FRAMES)
    _write_seq(pred_dir, PRED_FRAMES)
    gt = motrics.load_davis(gt_dir)
    pred = motrics.load_davis(pred_dir)
    return motrics.preprocess_davis(gt, pred)


def _trackeval_data(tmp_path: Path) -> dict[str, Any]:
    """TrackEval's own DAVIS pipeline, reading the same PNG files via a bare
    instance so no real dataset/seqmap layout is needed."""
    _write_seq(tmp_path / "gt" / _SEQ, GT_FRAMES)
    _write_seq(tmp_path / "tr" / "mytr" / "data" / _SEQ, PRED_FRAMES)

    dataset = object.__new__(DAVIS)
    dataset.gt_fol = str(tmp_path / "gt")
    dataset.tracker_fol = str(tmp_path / "tr")
    dataset.tracker_sub_fol = "data"
    dataset.seq_lengths = {_SEQ: NUM_FRAMES}
    dataset.max_det = 0

    raw = dataset.get_raw_seq_data("mytr", _SEQ)
    return dataset.get_preprocessed_seq_data(raw, "general")


def test_preprocess_davis_matches_trackeval(tmp_path: Path) -> None:
    gt_ids, pred_ids, similarity = _motrics_preprocessed(tmp_path)
    te = _trackeval_data(tmp_path / "te")

    assert te["num_gt_dets"] == sum(len(f) for f in gt_ids)
    assert te["num_tracker_dets"] == sum(len(f) for f in pred_ids)
    for t in range(NUM_FRAMES):
        assert len(gt_ids[t]) == len(te["gt_ids"][t])
        assert len(pred_ids[t]) == len(te["tracker_ids"][t])
        m_sim = np.array(similarity[t], dtype=float).reshape(
            len(gt_ids[t]), len(pred_ids[t])
        )
        assert np.allclose(m_sim, te["similarity_scores"][t], atol=1e-9)

    # The void-region false positive is kept, not dropped (object 3, frame 0).
    assert [len(f) for f in pred_ids] == [3, 2]


def test_load_davis_rejects_non_indexed_png(tmp_path: Path) -> None:
    from PIL import Image

    seq_dir = tmp_path / _SEQ
    seq_dir.mkdir(parents=True)
    seq_dir.joinpath("notes.txt").write_text("stray file, must be ignored")
    Image.new("RGB", (_W, _H)).save(seq_dir / "00000.png")
    with pytest.raises(ValueError, match="single-channel"):
        motrics.load_davis(seq_dir)


def test_preprocess_davis_frame_count_mismatch() -> None:
    gt = [([1], [motrics.Mask((1, 1), [0, 1])])]
    with pytest.raises(ValueError, match="frame count mismatch"):
        motrics.preprocess_davis(gt, [])


def test_davis_metrics_match_trackeval(tmp_path: Path) -> None:
    gt_ids, pred_ids, similarity = _motrics_preprocessed(tmp_path)
    te_data = _trackeval_data(tmp_path / "te")

    clear_ref = trackeval_metrics.CLEAR(
        {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    ).eval_sequence(te_data)
    identity_ref = trackeval_metrics.Identity(
        {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    ).eval_sequence(te_data)
    hota_ref = trackeval_metrics.HOTA({"PRINT_CONFIG": False}).eval_sequence(te_data)

    clear = motrics.compute_clear_from_similarity(gt_ids, pred_ids, similarity)
    identity = motrics.compute_identity_from_similarity(gt_ids, pred_ids, similarity)
    hota = motrics.compute_hota_from_similarity(gt_ids, pred_ids, similarity)

    assert clear.mota == pytest.approx(clear_ref["MOTA"], abs=1e-9)
    assert clear.motp == pytest.approx(clear_ref["MOTP"], abs=1e-9)
    assert clear.num_switches == clear_ref["IDSW"]
    assert identity.idf1 == pytest.approx(identity_ref["IDF1"], abs=1e-9)
    assert np.allclose(hota.hota_alphas, hota_ref["HOTA"], atol=1e-9)
