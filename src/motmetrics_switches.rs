//! `py-motmetrics`' switch-subtype events (`TRANSFER`/`ASCEND`/`MIGRATE`).
//!
//! Unlike TrackEval's single continuity-biased assignment (see [`crate::clear`]),
//! `motmetrics.mot.MOTAccumulator.update` matches each frame in two strict
//! stages: first it re-establishes every existing (object, hypothesis) pair
//! that's still available and uncontested, then it runs the optimal
//! assignment on whatever is left. It also keeps separate object- and
//! hypothesis-side match histories, which is what lets it distinguish "this
//! object changed hypothesis id" (`SWITCH`, already covered by
//! [`crate::clear`]'s `num_switches`) from "this hypothesis id changed which
//! object it covers" (`TRANSFER`), and flag the first-ever appearance of
//! either side (`ASCEND`/`MIGRATE`).
//!
//! Ported from `motmetrics.mot.MOTAccumulator.update`, assuming the default
//! `max_switch_time=inf` — the only mode `motrics.compat.motmetrics` supports.

use std::collections::{HashMap, HashSet};

// Filler for disallowed pairs in the LAP cost matrix; mirrors `clear.rs`.
const REJECT: f64 = -1e18;

/// One frame's ground-truth/hypothesis ids and their pairwise similarity, in
/// the same convention as [`crate::clear::SimFrame`] (higher is better, pairs
/// below `threshold` are never matched).
pub struct SimFrame<'a> {
    pub gt_ids: &'a [i64],
    pub pred_ids: &'a [i64],
    pub similarity: &'a [Vec<f64>],
}

/// Counts of `py-motmetrics`' switch-subtype events over a sequence.
#[derive(Debug, Default, Clone, Copy)]
pub struct SwitchEvents {
    /// A hypothesis id switched which object it covers.
    pub num_transfer: usize,
    /// A `TRANSFER` where the new hypothesis id had never matched anything before.
    pub num_ascend: usize,
    /// A `TRANSFER` where the object had never been matched before.
    pub num_migrate: usize,
}

#[derive(Debug, Default)]
struct MatcherState {
    // Most recent hypothesis id each object was matched to (any stage).
    m: HashMap<i64, i64>,
    // Most recent object id each hypothesis was matched to, updated only by
    // stage-2 (LAP) matches — a stage-1 carry-forward leaves it untouched,
    // exactly like `motmetrics`' `res_m`.
    res_m: HashMap<i64, i64>,
    // Objects / hypotheses that have ever been matched (any stage).
    ever_matched_obj: HashSet<i64>,
    ever_matched_hyp: HashSet<i64>,
}

fn accumulate_frame(
    gt_ids: &[i64],
    pred_ids: &[i64],
    sim: &[Vec<f64>],
    threshold: f64,
    state: &mut MatcherState,
    events: &mut SwitchEvents,
) {
    let n_gt = gt_ids.len();
    let n_pred = pred_ids.len();
    if n_gt == 0 || n_pred == 0 {
        return;
    }

    let mut row_masked = vec![false; n_gt];
    let mut col_masked = vec![false; n_pred];

    // Stage 1: re-establish existing (object, hypothesis) pairs, greedily, in
    // object order — not an optimal assignment.
    for i in 0..n_gt {
        let Some(&hprev) = state.m.get(&gt_ids[i]) else {
            continue;
        };
        let Some(j) = pred_ids.iter().position(|&h| h == hprev) else {
            continue;
        };
        if col_masked[j] || sim[i][j] < threshold {
            continue;
        }
        row_masked[i] = true;
        col_masked[j] = true;
        let (o, h) = (gt_ids[i], pred_ids[j]);
        state.m.insert(o, h);
        state.ever_matched_obj.insert(o);
        state.ever_matched_hyp.insert(h);
    }

    // Stage 2: optimal assignment over what's left.
    let remaining_rows: Vec<usize> = (0..n_gt).filter(|&i| !row_masked[i]).collect();
    let remaining_cols: Vec<usize> = (0..n_pred).filter(|&j| !col_masked[j]).collect();
    if remaining_rows.is_empty() || remaining_cols.is_empty() {
        return;
    }

    let n_rows = remaining_rows.len();
    let n_cols = remaining_cols.len();
    let mut flat = vec![REJECT; n_rows * n_cols];
    for (ri, &i) in remaining_rows.iter().enumerate() {
        for (rj, &j) in remaining_cols.iter().enumerate() {
            if sim[i][j] >= threshold {
                flat[ri * n_cols + rj] = sim[i][j];
            }
        }
    }
    let (rows, cols) =
        lsap::solve(n_rows, n_cols, &flat, true).expect("finite score matrix is solvable");

    for (ri, rj) in rows.into_iter().zip(cols) {
        let i = remaining_rows[ri];
        let j = remaining_cols[rj];
        if sim[i][j] < threshold {
            continue; // disallowed pair picked only to fill the assignment
        }
        let (o, h) = (gt_ids[i], pred_ids[j]);

        let is_switch = state.m.get(&o).is_some_and(|&prev| prev != h);
        if is_switch && !state.ever_matched_hyp.contains(&h) {
            events.num_ascend += 1;
        }

        let is_transfer = state.res_m.get(&h).is_some_and(|&prev| prev != o);
        if is_transfer {
            if !state.ever_matched_obj.contains(&o) {
                events.num_migrate += 1;
            }
            events.num_transfer += 1;
        }

        state.m.insert(o, h);
        state.res_m.insert(h, o);
        state.ever_matched_obj.insert(o);
        state.ever_matched_hyp.insert(h);
    }
}

/// Streaming accumulator: fold frames in one at a time with
/// [`SwitchEventsAccumulator::update`], then read the totals with
/// [`SwitchEventsAccumulator::compute`]. State is bounded by the number of
/// distinct objects/hypotheses, not the number of frames.
#[derive(Debug, Default)]
pub struct SwitchEventsAccumulator {
    events: SwitchEvents,
    state: MatcherState,
}

impl SwitchEventsAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one frame's precomputed similarity matrix into the running state.
    pub fn update(&mut self, gt_ids: &[i64], pred_ids: &[i64], sim: &[Vec<f64>], threshold: f64) {
        accumulate_frame(
            gt_ids,
            pred_ids,
            sim,
            threshold,
            &mut self.state,
            &mut self.events,
        );
    }

    pub fn compute(&self) -> SwitchEvents {
        self.events
    }
}

/// Compute `py-motmetrics`' switch-subtype events from precomputed per-frame
/// similarity matrices.
pub fn compute_switch_events_from_similarity(frames: &[SimFrame], threshold: f64) -> SwitchEvents {
    let mut acc = SwitchEventsAccumulator::new();
    for frame in frames {
        acc.update(frame.gt_ids, frame.pred_ids, frame.similarity, threshold);
    }
    acc.compute()
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestFrame = (Vec<i64>, Vec<i64>, Vec<Vec<f64>>);

    fn events(frames: &[TestFrame], threshold: f64) -> SwitchEvents {
        let sim_frames: Vec<SimFrame> = frames
            .iter()
            .map(|(gt_ids, pred_ids, similarity)| SimFrame {
                gt_ids,
                pred_ids,
                similarity,
            })
            .collect();
        compute_switch_events_from_similarity(&sim_frames, threshold)
    }

    #[test]
    fn steady_state_has_no_events() {
        let sim = vec![vec![1.0]];
        let frames: Vec<_> = (0..3).map(|_| (vec![1], vec![10], sim.clone())).collect();
        let e = events(&frames, 0.5);
        assert_eq!((e.num_transfer, e.num_ascend, e.num_migrate), (0, 0, 0));
    }

    #[test]
    fn ascend_new_hypothesis_takes_over_established_object() {
        // Frame 0: gt 1 <-> hyp 10 (established).
        // Frame 1: gt 1 present alone with a brand new hyp 20 -> forced switch
        // to a hypothesis id that has never matched before -> ASCEND (+TRANSFER,
        // since res_m has no entry for 20 yet the object side already has
        // history so it's not a MIGRATE).
        let frames = [
            (vec![1], vec![10], vec![vec![1.0]]),
            (vec![1], vec![20], vec![vec![1.0]]),
        ];
        let e = events(&frames, 0.5);
        assert_eq!(e.num_ascend, 1);
        assert_eq!(e.num_transfer, 0); // hyp 20 has no res_m history yet either
        assert_eq!(e.num_migrate, 0);
    }

    #[test]
    fn migrate_established_hypothesis_takes_new_object() {
        // Frame 0: gt 1 <-> hyp 10 (established, goes through stage 2).
        // Frame 1: gt 1 vanishes, gt 2 (brand new object) takes hyp 10 ->
        // MIGRATE (+TRANSFER), since 10 has res_m history but gt 2 doesn't.
        let frames = [
            (vec![1], vec![10], vec![vec![1.0]]),
            (vec![2], vec![10], vec![vec![1.0]]),
        ];
        let e = events(&frames, 0.5);
        assert_eq!(e.num_migrate, 1);
        assert_eq!(e.num_transfer, 1);
        assert_eq!(e.num_ascend, 0);
    }

    #[test]
    fn transfer_swaps_two_established_pairs() {
        // Frame 0: gt 1 <-> hyp 10, gt 2 <-> hyp 20 (both via stage 2).
        // Frame 1: hyp 10 only overlaps gt 2, hyp 20 only overlaps gt 1 -> a
        // forced swap; both sides have prior history, so it's TRANSFER only
        // (no ASCEND, no MIGRATE) on each hypothesis.
        let frames = [
            (
                vec![1, 2],
                vec![10, 20],
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            ),
            (
                vec![1, 2],
                vec![10, 20],
                vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            ),
        ];
        let e = events(&frames, 0.5);
        assert_eq!(e.num_transfer, 2);
        assert_eq!(e.num_ascend, 0);
        assert_eq!(e.num_migrate, 0);
    }

    #[test]
    fn stage_one_carries_forward_without_events() {
        // gt 1 stays on hyp 10 across three frames purely via stage-1
        // carry-forward; no LAP-driven event should ever fire.
        let sim = vec![vec![1.0]];
        let frames: Vec<_> = (0..3).map(|_| (vec![1], vec![10], sim.clone())).collect();
        let e = events(&frames, 0.5);
        assert_eq!((e.num_transfer, e.num_ascend, e.num_migrate), (0, 0, 0));
    }

    #[test]
    fn empty_frames_produce_no_events() {
        let e = events(&[], 0.5);
        assert_eq!((e.num_transfer, e.num_ascend, e.num_migrate), (0, 0, 0));
    }

    #[test]
    fn frame_with_only_gt_or_only_pred_is_a_no_op() {
        // A frame with objects but no predictions (or vice versa) can't match
        // anything; the matcher must bail out without touching its state.
        let frames = [
            (vec![1], vec![10], vec![vec![1.0]]),
            (vec![1], vec![], vec![]),
            (vec![], vec![10], vec![]),
            (vec![1], vec![10], vec![vec![1.0]]),
        ];
        let e = events(&frames, 0.5);
        assert_eq!((e.num_transfer, e.num_ascend, e.num_migrate), (0, 0, 0));
    }

    #[test]
    fn stage_two_skips_a_forced_pair_below_threshold() {
        // One object, two predictions, both below threshold: the LAP still
        // has to return a pair (more columns than rows), but it must be
        // discarded rather than treated as a match.
        let frames = [(vec![1], vec![10, 20], vec![vec![0.0, 0.0]])];
        let e = events(&frames, 0.5);
        assert_eq!((e.num_transfer, e.num_ascend, e.num_migrate), (0, 0, 0));
    }
}
