//! CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches).
//!
//! Implements the classic CLEAR MOT evaluation (Bernardin & Stiefelhagen).
//! For each frame, ground-truth boxes are matched to tracker boxes with an
//! optimal linear-sum assignment (via [`lsap`]), preferring to keep a
//! ground-truth object on the hypothesis id it was matched to previously so
//! that identity switches are counted correctly.
//!
//! Numbers are bit-exact with TrackEval, including the per-trajectory
//! sub-metrics MT/PT/ML and Frag and the derived MODA/sMOTA/MOTAL scores.

use std::collections::{HashMap, HashSet};

use crate::iou::{iou_matrix, Bbox};

/// A single frame of ground-truth and tracker detections.
///
/// `gt_ids` is aligned with `gt_boxes`, and `pred_ids` with `pred_boxes`.
pub struct Frame<'a> {
    pub gt_ids: &'a [i64],
    pub gt_boxes: &'a [Bbox],
    pub pred_ids: &'a [i64],
    pub pred_boxes: &'a [Bbox],
}

/// A single frame with a precomputed similarity matrix, for callers that
/// already hold pairwise scores (e.g. a `motmetrics`-style distance matrix
/// converted to similarity) instead of boxes.
///
/// `similarity[i][j]` scores `gt_ids[i]` against `pred_ids[j]`; higher is
/// better, matching the IoU convention used everywhere else in this crate.
pub struct SimFrame<'a> {
    pub gt_ids: &'a [i64],
    pub pred_ids: &'a [i64],
    pub similarity: &'a [Vec<f64>],
}

/// Accumulated CLEAR MOT metrics over a sequence.
#[derive(Debug, Default, Clone)]
pub struct ClearMetrics {
    /// Multiple Object Tracking Accuracy: `1 - (FN + FP + IDSW) / num_gt`.
    pub mota: f64,
    /// Multiple Object Tracking Precision: mean IoU over matched pairs.
    pub motp: f64,
    /// Number of frames processed.
    pub num_frames: usize,
    /// Total ground-truth detections across all frames.
    pub num_gt: usize,
    /// True positives: matched (gt, pred) pairs.
    pub num_matches: usize,
    /// False positives: tracker detections with no match.
    pub num_false_positives: usize,
    /// Misses: ground-truth detections with no match.
    pub num_misses: usize,
    /// Identity switches: a gt object matched to a different hypothesis id than
    /// the last time it was matched.
    pub num_switches: usize,
    /// Mostly tracked: gt trajectories matched in more than 80% of the frames
    /// they appear in.
    pub mt: usize,
    /// Partially tracked: gt trajectories matched in 20–80% of their frames.
    pub pt: usize,
    /// Mostly lost: gt trajectories matched in less than 20% of their frames.
    pub ml: usize,
    /// Fragmentations: total number of times a gt trajectory resumes after an
    /// interruption (transitions from untracked to tracked, minus the first).
    pub frag: usize,
    /// Multiple Object Detection Accuracy: `(TP - FP) / max(1, TP + FN)`.
    pub moda: f64,
    /// MOTA using overlap instead of a count of true positives:
    /// `(MOTP_sum - FP - IDSW) / max(1, TP + FN)`.
    pub smota: f64,
    /// MOTA with a log-scaled ID-switch penalty:
    /// `(TP - FP - log10(IDSW)) / max(1, TP + FN)`.
    pub motal: f64,
    /// CLEAR recall: `TP / max(1, TP + FN)`.
    pub clr_re: f64,
    /// CLEAR precision: `TP / max(1, TP + FP)`.
    pub clr_pr: f64,
    /// Per ground-truth trajectory, the fraction of its frames that were matched
    /// (counting id switches as matched). Sorted descending. This is the basis
    /// for MT/PT/ML under any threshold convention — TrackEval's `>0.8`/`<0.2`
    /// as used for `mt`/`pt`/`ml` here, or py-motmetrics' inclusive `>=0.8`.
    pub track_ratios: Vec<f64>,
}

/// Per-gt-trajectory bookkeeping needed for MT/PT/ML and Frag, mirroring
/// TrackEval's `gt_id_count`/`gt_matched_count`/`gt_frag_count` arrays. All keyed
/// by gt id, so memory grows with the number of distinct objects, not frames.
#[derive(Debug, Default)]
struct TrackState {
    // Last hypothesis id each gt object was matched to (persists across gaps);
    // drives IDSW scoring and the matching continuity bonus.
    last_pred_of_gt: HashMap<i64, i64>,
    // Frames each gt id appears in, and frames it was matched in.
    gt_id_count: HashMap<i64, u32>,
    gt_matched_count: HashMap<i64, u32>,
    // Number of untracked -> tracked transitions per gt id.
    gt_frag_count: HashMap<i64, u32>,
    // gt ids matched in the immediately previous *non-empty* frame; TrackEval
    // only resets this on frames that have both gt and predictions, so a gap
    // caused by an empty frame does not itself count as a fragmentation.
    prev_timestep_matched: HashSet<i64>,
    motp_sum: f64,
}

// Scaled by the frame's max score so continuation always wins; mirrors TrackEval's fixed `1000 * score_mat + similarity`.
const CONTINUITY_BONUS_MULTIPLIER: f64 = 1000.0;

// Filler for disallowed pairs; 0.0 only works while every real score stays non-negative.
const REJECT: f64 = -1e18;

/// Match one frame against a precomputed similarity matrix and fold the
/// result into `m` and `state`. Shared by the boxes-based and
/// precomputed-similarity entry points below.
fn accumulate_frame(
    gt_ids: &[i64],
    pred_ids: &[i64],
    sim: &[Vec<f64>],
    threshold: f64,
    state: &mut TrackState,
    m: &mut ClearMetrics,
) {
    let n_gt = gt_ids.len();
    let n_pred = pred_ids.len();
    m.num_gt += n_gt;

    // TrackEval skips these frames without touching `prev_timestep_matched`, so
    // a gap they introduce does not, on its own, register as a fragmentation.
    if n_gt == 0 {
        m.num_false_positives += n_pred;
        return;
    }
    for &g in gt_ids {
        *state.gt_id_count.entry(g).or_insert(0) += 1;
    }
    if n_pred == 0 {
        m.num_misses += n_gt;
        return;
    }

    let max_eligible = sim
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite() && *v >= threshold)
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let continuity_bonus = CONTINUITY_BONUS_MULTIPLIER * (1.0 + max_eligible);

    // Maximisation target: allowed pairs score `sim (+ bonus if continuing)`, disallowed pairs score REJECT.
    let mut flat = vec![REJECT; n_gt * n_pred];
    for i in 0..n_gt {
        let continues = state.last_pred_of_gt.get(&gt_ids[i]);
        for j in 0..n_pred {
            if sim[i][j] >= threshold {
                let bonus = match continues {
                    Some(&prev) if prev == pred_ids[j] => continuity_bonus,
                    _ => 0.0,
                };
                flat[i * n_pred + j] = bonus + sim[i][j];
            }
        }
    }

    let (rows, cols) =
        lsap::solve(n_gt, n_pred, &flat, true).expect("finite score matrix is solvable");

    let mut matched = 0;
    let mut matched_gts = HashSet::with_capacity(n_gt);
    for (i, j) in rows.into_iter().zip(cols) {
        if sim[i][j] < threshold {
            continue; // disallowed pair picked only to fill the assignment
        }
        matched += 1;
        state.motp_sum += sim[i][j];

        let gt_id = gt_ids[i];
        let pred_id = pred_ids[j];
        if let Some(&prev) = state.last_pred_of_gt.get(&gt_id) {
            if prev != pred_id {
                m.num_switches += 1;
            }
        }
        state.last_pred_of_gt.insert(gt_id, pred_id);

        *state.gt_matched_count.entry(gt_id).or_insert(0) += 1;
        // Untracked in the previous non-empty frame but tracked now: a resume.
        if !state.prev_timestep_matched.contains(&gt_id) {
            *state.gt_frag_count.entry(gt_id).or_insert(0) += 1;
        }
        matched_gts.insert(gt_id);
    }
    state.prev_timestep_matched = matched_gts;

    m.num_matches += matched;
    m.num_false_positives += n_pred - matched;
    m.num_misses += n_gt - matched;
}

fn finalize(mut m: ClearMetrics, state: &TrackState) -> ClearMetrics {
    if m.num_gt > 0 {
        let errors = (m.num_misses + m.num_false_positives + m.num_switches) as f64;
        m.mota = 1.0 - errors / m.num_gt as f64;
    }
    if m.num_matches > 0 {
        m.motp = state.motp_sum / m.num_matches as f64;
    }

    // Each trajectory's matched-frame ratio, the basis for MT/PT/ML.
    let mut track_ratios: Vec<f64> = state
        .gt_id_count
        .iter()
        .map(|(gt_id, &present)| {
            let matched = state.gt_matched_count.get(gt_id).copied().unwrap_or(0);
            matched as f64 / present as f64
        })
        .collect();
    track_ratios.sort_unstable_by(|a, b| b.total_cmp(a));

    // TrackEval's bounds: MT strictly above 0.8, ML strictly below 0.2, PT in
    // between inclusive.
    for &ratio in &track_ratios {
        if ratio > 0.8 {
            m.mt += 1;
        } else if ratio >= 0.2 {
            m.pt += 1;
        } else {
            m.ml += 1;
        }
    }
    m.track_ratios = track_ratios;
    m.frag = state
        .gt_frag_count
        .values()
        .map(|&c| c.saturating_sub(1) as usize)
        .sum();

    let (tp, fp, fnn, idsw) = (
        m.num_matches as f64,
        m.num_false_positives as f64,
        m.num_misses as f64,
        m.num_switches as f64,
    );
    let gt_denom = (tp + fnn).max(1.0);
    m.moda = (tp - fp) / gt_denom;
    m.smota = (state.motp_sum - fp - idsw) / gt_denom;
    let safe_log_idsw = if idsw > 0.0 { idsw.log10() } else { 0.0 };
    m.motal = (tp - fp - safe_log_idsw) / gt_denom;
    m.clr_re = tp / gt_denom;
    m.clr_pr = tp / (tp + fp).max(1.0);
    m
}

/// Streaming CLEAR accumulator: fold frames in one at a time with
/// [`ClearAccumulator::update`], then read MOTA/MOTP with
/// [`ClearAccumulator::compute`]. Only bounded per-object state is kept (the
/// last hypothesis id per gt object plus running counts), so memory does not
/// grow with the number of frames. The batch [`compute_clear`] is a thin
/// wrapper over this, so streaming and batch results are identical.
#[derive(Debug, Default)]
pub struct ClearAccumulator {
    metrics: ClearMetrics,
    state: TrackState,
}

impl ClearAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of frames folded in so far.
    pub fn num_frames(&self) -> usize {
        self.metrics.num_frames
    }

    /// Fold one frame's precomputed similarity matrix into the running state.
    pub fn update(&mut self, gt_ids: &[i64], pred_ids: &[i64], sim: &[Vec<f64>], threshold: f64) {
        self.metrics.num_frames += 1;
        accumulate_frame(
            gt_ids,
            pred_ids,
            sim,
            threshold,
            &mut self.state,
            &mut self.metrics,
        );
    }

    /// Finalize all CLEAR fields from the accumulated counts.
    pub fn compute(&self) -> ClearMetrics {
        finalize(self.metrics.clone(), &self.state)
    }
}

/// Compute CLEAR MOT metrics over a sequence of frames.
pub fn compute_clear(frames: &[Frame], threshold: f64) -> ClearMetrics {
    let mut acc = ClearAccumulator::new();
    for frame in frames {
        let sim = iou_matrix(frame.gt_boxes, frame.pred_boxes);
        acc.update(frame.gt_ids, frame.pred_ids, &sim, threshold);
    }
    acc.compute()
}

/// Compute CLEAR MOT metrics from precomputed per-frame similarity matrices
/// instead of boxes (e.g. for callers migrating from a distance-matrix API).
pub fn compute_clear_from_similarity(frames: &[SimFrame], threshold: f64) -> ClearMetrics {
    let mut acc = ClearAccumulator::new();
    for frame in frames {
        acc.update(frame.gt_ids, frame.pred_ids, frame.similarity, threshold);
    }
    acc.compute()
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Bbox = [0.0, 0.0, 10.0, 10.0];
    const B: Bbox = [20.0, 20.0, 30.0, 30.0];

    fn frame<'a>(
        gt_ids: &'a [i64],
        gt_boxes: &'a [Bbox],
        pred_ids: &'a [i64],
        pred_boxes: &'a [Bbox],
    ) -> Frame<'a> {
        Frame {
            gt_ids,
            gt_boxes,
            pred_ids,
            pred_boxes,
        }
    }

    #[test]
    fn perfect_tracking() {
        let frames = [
            frame(&[1, 2], &[A, B], &[1, 2], &[A, B]),
            frame(&[1, 2], &[A, B], &[1, 2], &[A, B]),
        ];
        let m = compute_clear(&frames, 0.5);
        assert_eq!(m.num_matches, 4);
        assert_eq!(m.num_false_positives, 0);
        assert_eq!(m.num_misses, 0);
        assert_eq!(m.num_switches, 0);
        assert!((m.mota - 1.0).abs() < 1e-9);
        assert!((m.motp - 1.0).abs() < 1e-9);
    }

    #[test]
    fn missed_detection() {
        let frames = [frame(&[1], &[A], &[], &[])];
        let m = compute_clear(&frames, 0.5);
        assert_eq!(m.num_misses, 1);
        assert_eq!(m.num_matches, 0);
        assert!((m.mota - 0.0).abs() < 1e-9); // 1 - 1/1
    }

    #[test]
    fn false_positive() {
        let frames = [frame(&[], &[], &[9], &[A])];
        let m = compute_clear(&frames, 0.5);
        assert_eq!(m.num_false_positives, 1);
        assert_eq!(m.num_gt, 0);
    }

    #[test]
    fn identity_switch() {
        // Same gt object (id 1), matched to pred 10 then pred 20.
        let frames = [
            frame(&[1], &[A], &[10], &[A]),
            frame(&[1], &[A], &[20], &[A]),
        ];
        let m = compute_clear(&frames, 0.5);
        assert_eq!(m.num_matches, 2);
        assert_eq!(m.num_switches, 1);
        assert_eq!(m.num_false_positives, 0);
        assert_eq!(m.num_misses, 0);
        assert!((m.mota - 0.5).abs() < 1e-9); // 1 - 1/2
    }

    #[test]
    fn continuity_preferred_over_iou() {
        // Two identical boxes; keeping gt 1 on pred 10 avoids a switch even
        // though the assignment is otherwise symmetric.
        let frames = [
            frame(&[1], &[A], &[10], &[A]),
            frame(&[1, 2], &[A, B], &[10, 20], &[A, B]),
        ];
        let m = compute_clear(&frames, 0.5);
        assert_eq!(m.num_switches, 0);
        assert_eq!(m.num_matches, 3);
    }

    #[test]
    fn empty_sequence() {
        let m = compute_clear(&[], 0.5);
        assert_eq!(m.num_frames, 0);
        assert_eq!(m.mota, 0.0);
        assert_eq!(m.motp, 0.0);
        assert_eq!((m.mt, m.pt, m.ml, m.frag), (0, 0, 0, 0));
    }

    #[test]
    fn mostly_tracked_and_mostly_lost() {
        // gt 1 is matched in every frame (MT); gt 2 is never matched (ML).
        let frames: Vec<_> = (0..5)
            .map(|_| frame(&[1, 2], &[A, B], &[10], &[A]))
            .collect();
        let m = compute_clear(&frames, 0.5);
        assert_eq!((m.mt, m.pt, m.ml), (1, 0, 1));
        assert_eq!(m.frag, 0); // gt 1 tracked without interruption
        assert_eq!(m.track_ratios, vec![1.0, 0.0]); // sorted descending
    }

    #[test]
    fn fragmentation_counts_resumes() {
        // gt 1 matched, matched, lost (present but no overlap), matched, matched:
        // one interruption -> Frag == 1, tracked ratio 4/5 -> partially tracked.
        let frames = [
            frame(&[1], &[A], &[10], &[A]),
            frame(&[1], &[A], &[10], &[A]),
            frame(&[1], &[A], &[10], &[B]),
            frame(&[1], &[A], &[10], &[A]),
            frame(&[1], &[A], &[10], &[A]),
        ];
        let m = compute_clear(&frames, 0.5);
        assert_eq!(m.num_matches, 4);
        assert_eq!((m.mt, m.pt, m.ml), (0, 1, 0));
        assert_eq!(m.frag, 1);
    }
}
