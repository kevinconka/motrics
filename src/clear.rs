//! CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches).
//!
//! Implements the classic CLEAR MOT evaluation (Bernardin & Stiefelhagen).
//! For each frame, ground-truth boxes are matched to tracker boxes with an
//! optimal linear-sum assignment (via [`lsap`]), preferring to keep a
//! ground-truth object on the hypothesis id it was matched to previously so
//! that identity switches are counted correctly.
//!
//! Bit-exact parity with TrackEval (and the extra sub-metrics MT/ML/Frag) is
//! deferred to the dedicated parity-tests milestone; this module implements the
//! core CLEAR counts and the two headline scores.

use std::collections::HashMap;

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
}

// A previously matched (gt, pred) pair gets this bonus so the optimal assignment
// keeps continuing tracks together; far larger than the max IoU of 1.0.
const CONTINUITY_BONUS: f64 = 1000.0;

// Filler score for pairs below `threshold`, so the maximising solve never
// prefers a disallowed pair over an allowed one. Must be lower than any real
// score: `0.0` would work for IoU (always >= 0), but callers that pass a
// negated distance as similarity (`compute_clear_from_similarity`) can have
// every real score negative, in which case `0.0` would look best of all.
const REJECT: f64 = -1e18;

/// Match one frame against a precomputed similarity matrix and fold the
/// result into `m`, `last_pred_of_gt`, and `motp_sum`. Shared by the
/// boxes-based and precomputed-similarity entry points below.
fn accumulate_frame(
    gt_ids: &[i64],
    pred_ids: &[i64],
    sim: &[Vec<f64>],
    threshold: f64,
    last_pred_of_gt: &mut HashMap<i64, i64>,
    motp_sum: &mut f64,
    m: &mut ClearMetrics,
) {
    let n_gt = gt_ids.len();
    let n_pred = pred_ids.len();
    m.num_gt += n_gt;

    if n_gt == 0 {
        m.num_false_positives += n_pred;
        return;
    }
    if n_pred == 0 {
        m.num_misses += n_gt;
        return;
    }

    // Score matrix for maximisation: allowed pairs (similarity >= threshold)
    // score `sim (+ bonus if continuing a track)`; disallowed pairs score
    // `REJECT` and are filtered out after solving.
    let mut flat = vec![REJECT; n_gt * n_pred];
    for i in 0..n_gt {
        let continues = last_pred_of_gt.get(&gt_ids[i]);
        for j in 0..n_pred {
            if sim[i][j] >= threshold {
                let bonus = match continues {
                    Some(&prev) if prev == pred_ids[j] => CONTINUITY_BONUS,
                    _ => 0.0,
                };
                flat[i * n_pred + j] = bonus + sim[i][j];
            }
        }
    }

    let (rows, cols) =
        lsap::solve(n_gt, n_pred, &flat, true).expect("finite score matrix is solvable");

    let mut matched = 0;
    for (i, j) in rows.into_iter().zip(cols) {
        if sim[i][j] < threshold {
            continue; // disallowed pair picked only to fill the assignment
        }
        matched += 1;
        *motp_sum += sim[i][j];

        let gt_id = gt_ids[i];
        let pred_id = pred_ids[j];
        if let Some(&prev) = last_pred_of_gt.get(&gt_id) {
            if prev != pred_id {
                m.num_switches += 1;
            }
        }
        last_pred_of_gt.insert(gt_id, pred_id);
    }

    m.num_matches += matched;
    m.num_false_positives += n_pred - matched;
    m.num_misses += n_gt - matched;
}

fn finalize(mut m: ClearMetrics, motp_sum: f64) -> ClearMetrics {
    if m.num_gt > 0 {
        let errors = (m.num_misses + m.num_false_positives + m.num_switches) as f64;
        m.mota = 1.0 - errors / m.num_gt as f64;
    }
    if m.num_matches > 0 {
        m.motp = motp_sum / m.num_matches as f64;
    }
    m
}

/// Compute CLEAR MOT metrics over a sequence of frames.
pub fn compute_clear(frames: &[Frame], threshold: f64) -> ClearMetrics {
    let mut m = ClearMetrics {
        num_frames: frames.len(),
        ..Default::default()
    };
    // Last hypothesis id each gt object was matched to (persists across gaps).
    let mut last_pred_of_gt: HashMap<i64, i64> = HashMap::new();
    let mut motp_sum = 0.0;

    for frame in frames {
        let sim = iou_matrix(frame.gt_boxes, frame.pred_boxes);
        accumulate_frame(
            frame.gt_ids,
            frame.pred_ids,
            &sim,
            threshold,
            &mut last_pred_of_gt,
            &mut motp_sum,
            &mut m,
        );
    }

    finalize(m, motp_sum)
}

/// Compute CLEAR MOT metrics from precomputed per-frame similarity matrices
/// instead of boxes (e.g. for callers migrating from a distance-matrix API).
pub fn compute_clear_from_similarity(frames: &[SimFrame], threshold: f64) -> ClearMetrics {
    let mut m = ClearMetrics {
        num_frames: frames.len(),
        ..Default::default()
    };
    let mut last_pred_of_gt: HashMap<i64, i64> = HashMap::new();
    let mut motp_sum = 0.0;

    for frame in frames {
        accumulate_frame(
            frame.gt_ids,
            frame.pred_ids,
            frame.similarity,
            threshold,
            &mut last_pred_of_gt,
            &mut motp_sum,
            &mut m,
        );
    }

    finalize(m, motp_sum)
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
    }
}
