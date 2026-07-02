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
        let n_gt = frame.gt_ids.len();
        let n_pred = frame.pred_ids.len();
        m.num_gt += n_gt;

        if n_gt == 0 {
            m.num_false_positives += n_pred;
            continue;
        }
        if n_pred == 0 {
            m.num_misses += n_gt;
            continue;
        }

        let iou = iou_matrix(frame.gt_boxes, frame.pred_boxes);

        // Score matrix for maximisation: allowed pairs (IoU >= threshold) score
        // `iou (+ bonus if continuing a track)`; disallowed pairs score 0 and are
        // filtered out after solving.
        let mut flat = vec![0.0f64; n_gt * n_pred];
        for i in 0..n_gt {
            let continues = last_pred_of_gt.get(&frame.gt_ids[i]);
            for j in 0..n_pred {
                if iou[i][j] >= threshold {
                    let bonus = match continues {
                        Some(&prev) if prev == frame.pred_ids[j] => CONTINUITY_BONUS,
                        _ => 0.0,
                    };
                    flat[i * n_pred + j] = bonus + iou[i][j];
                }
            }
        }

        let (rows, cols) =
            lsap::solve(n_gt, n_pred, &flat, true).expect("finite score matrix is solvable");

        let mut matched = 0;
        for (i, j) in rows.into_iter().zip(cols) {
            if iou[i][j] < threshold {
                continue; // disallowed pair picked only to fill the assignment
            }
            matched += 1;
            motp_sum += iou[i][j];

            let gt_id = frame.gt_ids[i];
            let pred_id = frame.pred_ids[j];
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

    if m.num_gt > 0 {
        let errors = (m.num_misses + m.num_false_positives + m.num_switches) as f64;
        m.mota = 1.0 - errors / m.num_gt as f64;
    }
    if m.num_matches > 0 {
        m.motp = motp_sum / m.num_matches as f64;
    }
    m
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
