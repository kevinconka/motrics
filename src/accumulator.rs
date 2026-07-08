//! Streaming accumulator combining CLEAR and Identity.
//!
//! Both metrics fold frame-by-frame into bounded per-object state (see
//! [`crate::clear::ClearAccumulator`] and [`crate::identity::IdentityAccumulator`]),
//! so an online tracker — or a sequence too large to hold in memory — can be
//! evaluated with `update()` per frame and `compute()` at the end, without ever
//! materialising the whole sequence. HOTA is deliberately excluded: its alpha
//! sweep is a whole-sequence batch computation, so it stays on the batch
//! [`crate::hota::compute_hota`] path.

use crate::clear::{ClearAccumulator, ClearMetrics};
use crate::identity::{IdentityAccumulator, IdentityMetrics};
use crate::iou::{iou_matrix, Bbox};

/// A streaming CLEAR + Identity accumulator. Feeds one shared per-frame
/// similarity matrix to both metrics, mirroring [`crate::evaluate`].
#[derive(Debug, Default)]
pub struct Accumulator {
    threshold: f64,
    clear: ClearAccumulator,
    identity: IdentityAccumulator,
}

impl Accumulator {
    pub fn new(threshold: f64) -> Self {
        Accumulator {
            threshold,
            ..Default::default()
        }
    }

    /// Number of frames folded in so far.
    pub fn num_frames(&self) -> usize {
        self.clear.num_frames()
    }

    /// Fold one frame's precomputed similarity matrix into both metrics.
    pub fn update_from_similarity(&mut self, gt_ids: &[i64], pred_ids: &[i64], sim: &[Vec<f64>]) {
        self.clear.update(gt_ids, pred_ids, sim, self.threshold);
        self.identity.update(gt_ids, pred_ids, sim, self.threshold);
    }

    /// Fold one frame of boxes in, computing the gt/pred IoU matrix once and
    /// sharing it between both metrics.
    pub fn update(
        &mut self,
        gt_ids: &[i64],
        gt_boxes: &[Bbox],
        pred_ids: &[i64],
        pred_boxes: &[Bbox],
    ) {
        let sim = iou_matrix(gt_boxes, pred_boxes);
        self.update_from_similarity(gt_ids, pred_ids, &sim);
    }

    /// Finalize CLEAR and Identity from the accumulated state.
    pub fn compute(&self) -> (ClearMetrics, IdentityMetrics) {
        (self.clear.compute(), self.identity.compute())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clear::{compute_clear, Frame};
    use crate::identity::compute_identity;

    const A: Bbox = [0.0, 0.0, 10.0, 10.0];
    const B: Bbox = [20.0, 20.0, 30.0, 30.0];

    // Feeding frames one at a time must match the batch functions exactly.
    #[test]
    fn streaming_matches_batch() {
        let gt_ids = [vec![1, 2], vec![1, 2], vec![1], vec![1, 2]];
        let gt_boxes = [vec![A, B], vec![A, B], vec![A], vec![A, B]];
        let pred_ids = [vec![10, 20], vec![10, 20], vec![10], vec![20, 10]];
        let pred_boxes = [vec![A, B], vec![A, B], vec![A], vec![B, A]];

        let frames: Vec<Frame> = (0..gt_ids.len())
            .map(|t| Frame {
                gt_ids: &gt_ids[t],
                gt_boxes: &gt_boxes[t],
                pred_ids: &pred_ids[t],
                pred_boxes: &pred_boxes[t],
            })
            .collect();
        let batch_clear = compute_clear(&frames, 0.5);
        let batch_identity = compute_identity(&frames, 0.5);

        let mut acc = Accumulator::new(0.5);
        for t in 0..gt_ids.len() {
            acc.update(&gt_ids[t], &gt_boxes[t], &pred_ids[t], &pred_boxes[t]);
        }
        let (clear, identity) = acc.compute();

        assert_eq!(acc.num_frames(), 4);
        assert_eq!(clear.num_matches, batch_clear.num_matches);
        assert_eq!(clear.num_switches, batch_clear.num_switches);
        assert!((clear.mota - batch_clear.mota).abs() < 1e-12);
        assert!((clear.motp - batch_clear.motp).abs() < 1e-12);
        assert_eq!(identity.idtp, batch_identity.idtp);
        assert_eq!(identity.idfp, batch_identity.idfp);
        assert_eq!(identity.idfn, batch_identity.idfn);
        assert!((identity.idf1 - batch_identity.idf1).abs() < 1e-12);
    }

    #[test]
    fn empty_accumulator() {
        let acc = Accumulator::new(0.5);
        let (clear, identity) = acc.compute();
        assert_eq!(acc.num_frames(), 0);
        assert_eq!(clear.num_frames, 0);
        assert_eq!(clear.mota, 0.0);
        assert_eq!(identity.idf1, 0.0);
    }
}
