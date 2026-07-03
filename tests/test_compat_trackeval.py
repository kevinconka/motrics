"""Tests for ``motrics.compat.trackeval.evaluate_mot_challenge``."""

from __future__ import annotations

from pathlib import Path

import motrics
import motrics.compat.trackeval as trackeval
import pytest

# Used as both gt and pred; as gt, columns 7-8 are class=pedestrian(1),
# visibility=1.0 (frame, id, left, top, w, h, consider/conf, class, visibility).
PERFECT_TEXT = "1,1,0,0,10,10,1,1,1\n2,1,0,0,10,10,1,1,1\n"
# One miss (frame 2 has no prediction) and one false positive (frame 1).
IMPERFECT_TEXT_GT = "1,1,0,0,10,10,1,1,1\n2,1,0,0,10,10,1,1,1\n"
IMPERFECT_TEXT_PRED = "1,1,0,0,10,10,1,-1,-1,-1\n1,2,90,90,10,10,1,-1,-1,-1\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_matches_direct_compute_for_a_single_sequence(tmp_path: Path) -> None:
    gt_path = _write(tmp_path, "gt.txt", PERFECT_TEXT)
    pred_path = _write(tmp_path, "pred.txt", PERFECT_TEXT)

    results = trackeval.evaluate_mot_challenge({"seq": (gt_path, pred_path)})

    gt = motrics.load_motchallenge_gt(gt_path)
    pred = motrics.load_motchallenge(pred_path)
    args = motrics.preprocess_motchallenge(gt, pred)
    clear, hota = motrics.compute_clear(*args), motrics.compute_hota(*args)
    assert results["seq"]["clear"]["mota"] == pytest.approx(clear.mota)
    assert results["seq"]["hota"]["hota"] == pytest.approx(hota.hota)


def test_combined_seq_recomputes_from_summed_counts(tmp_path: Path) -> None:
    perfect_gt = _write(tmp_path, "perfect_gt.txt", PERFECT_TEXT)
    perfect_pred = _write(tmp_path, "perfect_pred.txt", PERFECT_TEXT)
    flawed_gt = _write(tmp_path, "flawed_gt.txt", IMPERFECT_TEXT_GT)
    flawed_pred = _write(tmp_path, "flawed_pred.txt", IMPERFECT_TEXT_PRED)

    results = trackeval.evaluate_mot_challenge(
        {
            "perfect": (perfect_gt, perfect_pred),
            "flawed": (flawed_gt, flawed_pred),
        }
    )

    combined = results["COMBINED_SEQ"]["clear"]
    flawed_clear = results["flawed"]["clear"]
    perfect_clear = results["perfect"]["clear"]

    # A naive average of the per-sequence MOTA scores would differ from this.
    total_gt = flawed_clear["num_gt"] + perfect_clear["num_gt"]
    total_bad = (
        flawed_clear["num_misses"]
        + flawed_clear["num_false_positives"]
        + flawed_clear["num_switches"]
        + perfect_clear["num_misses"]
        + perfect_clear["num_false_positives"]
        + perfect_clear["num_switches"]
    )
    assert combined["mota"] == pytest.approx(1.0 - total_bad / total_gt)
    assert combined["num_gt"] == total_gt


def test_perfect_sequences_combine_to_perfect_scores(tmp_path: Path) -> None:
    gt_path = _write(tmp_path, "gt.txt", PERFECT_TEXT)
    pred_path = _write(tmp_path, "pred.txt", PERFECT_TEXT)

    results = trackeval.evaluate_mot_challenge(
        {"a": (gt_path, pred_path), "b": (gt_path, pred_path)}
    )

    combined = results["COMBINED_SEQ"]
    assert combined["clear"]["mota"] == pytest.approx(1.0)
    assert combined["clear"]["motp"] == pytest.approx(1.0)
    assert combined["identity"]["idf1"] == pytest.approx(1.0)
    assert combined["hota"]["hota"] == pytest.approx(1.0)
