"""Tests for the raw `motrics.compute_motmetrics_switch_events` API."""

import motrics
import pytest


def test_steady_state_has_no_events() -> None:
    gt_ids = [[1], [1], [1]]
    pred_ids = [[10], [10], [10]]
    similarity = [[[1.0]], [[1.0]], [[1.0]]]
    e = motrics.compute_motmetrics_switch_events(gt_ids, pred_ids, similarity)
    assert (e.num_transfer, e.num_ascend, e.num_migrate) == (0, 0, 0)


def test_migrate_and_transfer() -> None:
    # gt 1 <-> hyp 10 (established), then gt 1 vanishes and gt 2 (brand new)
    # takes hyp 10: a MIGRATE, which is also counted as a TRANSFER.
    gt_ids = [[1], [2]]
    pred_ids = [[10], [10]]
    similarity = [[[1.0]], [[1.0]]]
    e = motrics.compute_motmetrics_switch_events(gt_ids, pred_ids, similarity)
    assert e.num_migrate == 1
    assert e.num_transfer == 1
    assert e.num_ascend == 0


def test_empty_sequence() -> None:
    e = motrics.compute_motmetrics_switch_events([], [], [])
    assert (e.num_transfer, e.num_ascend, e.num_migrate) == (0, 0, 0)


def test_frame_with_only_gt_or_only_pred_is_a_no_op() -> None:
    # A frame with objects but no predictions (or vice versa) can't match
    # anything; it must not disturb the surrounding established track.
    gt_ids = [[1], [1], [], [1]]
    pred_ids = [[10], [], [10], [10]]
    similarity = [[[1.0]], [[]], [], [[1.0]]]
    e = motrics.compute_motmetrics_switch_events(gt_ids, pred_ids, similarity)
    assert (e.num_transfer, e.num_ascend, e.num_migrate) == (0, 0, 0)


def test_stage_one_skips_a_continuation_that_dropped_below_threshold() -> None:
    # gt 1 <-> hyp 10 is established; next frame the same pair is still
    # present but its similarity drops below threshold, so stage 1 must not
    # carry it forward (and stage 2 has nothing better to offer either).
    gt_ids = [[1], [1]]
    pred_ids = [[10], [10]]
    similarity = [[[1.0]], [[0.0]]]
    e = motrics.compute_motmetrics_switch_events(gt_ids, pred_ids, similarity)
    assert (e.num_transfer, e.num_ascend, e.num_migrate) == (0, 0, 0)


def test_stage_two_skips_a_forced_pair_below_threshold() -> None:
    # One object, two predictions, both below threshold: the LAP still has
    # to return a pair (more columns than rows), but it must be discarded
    # rather than treated as a match.
    gt_ids = [[1]]
    pred_ids = [[10, 20]]
    similarity = [[[0.0, 0.0]]]
    e = motrics.compute_motmetrics_switch_events(gt_ids, pred_ids, similarity)
    assert (e.num_transfer, e.num_ascend, e.num_migrate) == (0, 0, 0)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same number of frames"):
        motrics.compute_motmetrics_switch_events([[1]], [[10]], [])


def test_similarity_row_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="similarity has"):
        motrics.compute_motmetrics_switch_events([[1, 2]], [[10]], [[[1.0]]])


def test_similarity_column_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="similarity row 0 has"):
        motrics.compute_motmetrics_switch_events([[1]], [[10, 20]], [[[1.0]]])
