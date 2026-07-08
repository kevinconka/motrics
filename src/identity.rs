//! Identity metrics (IDF1, IDP, IDR).
//!
//! Unlike CLEAR (which matches per frame), the Identity metrics of Ristani et
//! al. perform a single global bipartite matching between whole ground-truth
//! trajectories and whole predicted trajectories, chosen to maximise the number
//! of correctly identified detections (IDTP). The assignment is solved with
//! [`lsap`] over the standard fp/fn cost formulation used by TrackEval.
//!
//! Bit-exact TrackEval parity is validated in the parity-tests milestone.

use std::collections::HashMap;

use crate::clear::{Frame, SimFrame};
use crate::iou::iou_matrix;

/// Accumulated Identity metrics over a sequence.
#[derive(Debug, Default, Clone)]
pub struct IdentityMetrics {
    /// Identity F1: `IDTP / (IDTP + 0.5 IDFP + 0.5 IDFN)`.
    pub idf1: f64,
    /// Identity precision: `IDTP / (IDTP + IDFP)`.
    pub idp: f64,
    /// Identity recall: `IDTP / (IDTP + IDFN)`.
    pub idr: f64,
    /// Identity true positives.
    pub idtp: usize,
    /// Identity false positives.
    pub idfp: usize,
    /// Identity false negatives.
    pub idfn: usize,
    /// Number of frames processed.
    pub num_frames: usize,
    /// Total ground-truth detections across all frames.
    pub num_gt: usize,
    /// Total predicted detections across all frames.
    pub num_pred: usize,
}

// Large finite sentinel forbidding a real track from taking another track's
// "dummy" (unmatched) slot. Kept well below f64::MAX so internal solver
// arithmetic cannot overflow.
const FORBIDDEN: f64 = 1e18;

/// Intern an id into `index`, assigning the next dense index on first sight
/// (in order of appearance) and growing `count` to match. Returns the id's
/// dense index.
fn intern(index: &mut HashMap<i64, usize>, count: &mut Vec<usize>, id: i64) -> usize {
    let next = index.len();
    let idx = *index.entry(id).or_insert(next);
    if idx == count.len() {
        count.push(0);
    }
    idx
}

/// Streaming Identity accumulator: fold frames in one at a time with
/// [`IdentityAccumulator::update`], then solve the global assignment with
/// [`IdentityAccumulator::compute`]. Only per-track counts and per-pair
/// co-occurrence ("potential" IDTP) are kept — both bounded by the number of
/// distinct ids, not the number of frames — so memory does not grow with
/// sequence length. The batch [`compute_identity`] is a thin wrapper over
/// this, so streaming and batch results are identical.
#[derive(Debug, Default)]
pub struct IdentityAccumulator {
    num_frames: usize,
    gt_index: HashMap<i64, usize>,
    pred_index: HashMap<i64, usize>,
    gt_count: Vec<usize>,
    pred_count: Vec<usize>,
    // Frames where gt `i` and pred `j` co-occur with similarity >= threshold.
    potential: HashMap<(usize, usize), usize>,
}

impl IdentityAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one frame's precomputed similarity matrix into the running state.
    pub fn update(&mut self, gt_ids: &[i64], pred_ids: &[i64], sim: &[Vec<f64>], threshold: f64) {
        self.num_frames += 1;
        for &id in gt_ids {
            let i = intern(&mut self.gt_index, &mut self.gt_count, id);
            self.gt_count[i] += 1;
        }
        for &id in pred_ids {
            let j = intern(&mut self.pred_index, &mut self.pred_count, id);
            self.pred_count[j] += 1;
        }
        if gt_ids.is_empty() || pred_ids.is_empty() {
            return;
        }
        for (gi, &gid) in gt_ids.iter().enumerate() {
            let i = self.gt_index[&gid];
            for (pj, &pid) in pred_ids.iter().enumerate() {
                if sim[gi][pj] >= threshold {
                    let j = self.pred_index[&pid];
                    *self.potential.entry((i, j)).or_insert(0) += 1;
                }
            }
        }
    }

    /// Solve the global assignment and derive IDF1/IDP/IDR.
    pub fn compute(&self) -> IdentityMetrics {
        let n_g = self.gt_index.len();
        let n_t = self.pred_index.len();
        let mut potential = vec![vec![0usize; n_t]; n_g];
        for (&(i, j), &c) in &self.potential {
            potential[i][j] = c;
        }
        finalize(
            self.num_frames,
            n_g,
            n_t,
            &self.gt_count,
            &self.pred_count,
            &potential,
        )
    }
}

/// Solve the global bipartite assignment and derive IDF1/IDP/IDR from the
/// per-track counts and co-occurrence potential. Shared by both entry points.
fn finalize(
    num_frames: usize,
    n_g: usize,
    n_t: usize,
    gt_count: &[usize],
    pred_count: &[usize],
    potential: &[Vec<usize>],
) -> IdentityMetrics {
    let total_gt: usize = gt_count.iter().sum();
    let total_pred: usize = pred_count.iter().sum();
    let mut m = IdentityMetrics {
        num_frames,
        num_gt: total_gt,
        num_pred: total_pred,
        ..Default::default()
    };

    if n_g == 0 && n_t == 0 {
        return m;
    }

    // Square cost matrix of size N = n_g + n_t. Rows/cols [.., real, .., dummy]:
    //   rows 0..n_g       = gt tracks,      rows n_g..     = tracker dummy slots
    //   cols 0..n_t       = tracker tracks, cols n_t..     = gt dummy slots
    let n = n_g + n_t;
    let mut fn_mat = vec![vec![0.0f64; n]; n];
    let mut fp_mat = vec![vec![0.0f64; n]; n];

    // Forbid off-diagonal dummy matches.
    for row in fp_mat.iter_mut().take(n).skip(n_g) {
        for cell in row.iter_mut().take(n_t) {
            *cell = FORBIDDEN;
        }
    }
    for row in fn_mat.iter_mut().take(n_g) {
        for cell in row.iter_mut().take(n).skip(n_t) {
            *cell = FORBIDDEN;
        }
    }

    for i in 0..n_g {
        for cell in fn_mat[i].iter_mut().take(n_t) {
            *cell = gt_count[i] as f64;
        }
        fn_mat[i][n_t + i] = gt_count[i] as f64; // gt i left unmatched
    }
    for j in 0..n_t {
        for row in fp_mat.iter_mut().take(n_g) {
            row[j] = pred_count[j] as f64;
        }
        fp_mat[n_g + j][j] = pred_count[j] as f64; // tracker j left unmatched
    }
    for i in 0..n_g {
        for j in 0..n_t {
            fn_mat[i][j] -= potential[i][j] as f64;
            fp_mat[i][j] -= potential[i][j] as f64;
        }
    }

    let mut flat = vec![0.0f64; n * n];
    for r in 0..n {
        for c in 0..n {
            flat[r * n + c] = fn_mat[r][c] + fp_mat[r][c];
        }
    }

    let (rows, cols) = lsap::solve(n, n, &flat, false).expect("finite cost matrix is solvable");

    let mut idfn = 0.0f64;
    let mut idfp = 0.0f64;
    for (r, c) in rows.into_iter().zip(cols) {
        idfn += fn_mat[r][c];
        idfp += fp_mat[r][c];
    }
    let idfn = idfn.round() as usize;
    let idfp = idfp.round() as usize;
    let idtp = total_gt - idfn; // == sum of potential over matched real pairs

    m.idtp = idtp;
    m.idfp = idfp;
    m.idfn = idfn;

    let denom_f1 = idtp as f64 + 0.5 * idfp as f64 + 0.5 * idfn as f64;
    m.idf1 = if denom_f1 > 0.0 {
        idtp as f64 / denom_f1
    } else {
        0.0
    };
    m.idp = if idtp + idfp > 0 {
        idtp as f64 / (idtp + idfp) as f64
    } else {
        0.0
    };
    m.idr = if idtp + idfn > 0 {
        idtp as f64 / (idtp + idfn) as f64
    } else {
        0.0
    };
    m
}

/// Compute Identity metrics (IDF1/IDP/IDR) over a sequence of frames.
pub fn compute_identity(frames: &[Frame], threshold: f64) -> IdentityMetrics {
    let mut acc = IdentityAccumulator::new();
    for f in frames {
        let sim = if f.gt_ids.is_empty() || f.pred_ids.is_empty() {
            Vec::new()
        } else {
            iou_matrix(f.gt_boxes, f.pred_boxes)
        };
        acc.update(f.gt_ids, f.pred_ids, &sim, threshold);
    }
    acc.compute()
}

/// Compute Identity metrics from precomputed per-frame similarity matrices
/// instead of boxes (e.g. for callers migrating from a distance-matrix API).
pub fn compute_identity_from_similarity(frames: &[SimFrame], threshold: f64) -> IdentityMetrics {
    let mut acc = IdentityAccumulator::new();
    for f in frames {
        acc.update(f.gt_ids, f.pred_ids, f.similarity, threshold);
    }
    acc.compute()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iou::Bbox;

    const A: Bbox = [0.0, 0.0, 10.0, 10.0];

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
    fn perfect_identity() {
        let frames = [frame(&[1], &[A], &[1], &[A]), frame(&[1], &[A], &[1], &[A])];
        let m = compute_identity(&frames, 0.5);
        assert_eq!(m.idtp, 2);
        assert_eq!(m.idfp, 0);
        assert_eq!(m.idfn, 0);
        assert!((m.idf1 - 1.0).abs() < 1e-9);
        assert!((m.idp - 1.0).abs() < 1e-9);
        assert!((m.idr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn id_split_halves_idf1() {
        // One gt object tracked by two different predicted ids across 4 frames.
        // The global match can only keep one id -> IDTP=2, IDFP=2, IDFN=2.
        let frames = [
            frame(&[1], &[A], &[10], &[A]),
            frame(&[1], &[A], &[10], &[A]),
            frame(&[1], &[A], &[20], &[A]),
            frame(&[1], &[A], &[20], &[A]),
        ];
        let m = compute_identity(&frames, 0.5);
        assert_eq!(m.idtp, 2);
        assert_eq!(m.idfp, 2);
        assert_eq!(m.idfn, 2);
        assert!((m.idf1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn all_false_positives() {
        let frames = [frame(&[], &[], &[9], &[A])];
        let m = compute_identity(&frames, 0.5);
        assert_eq!(m.idtp, 0);
        assert_eq!(m.idfp, 1);
        assert_eq!(m.idfn, 0);
        assert!((m.idf1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn all_misses() {
        let frames = [frame(&[1], &[A], &[], &[])];
        let m = compute_identity(&frames, 0.5);
        assert_eq!(m.idtp, 0);
        assert_eq!(m.idfn, 1);
        assert_eq!(m.idfp, 0);
        assert!((m.idr - 0.0).abs() < 1e-9);
    }

    #[test]
    fn empty_sequence() {
        let m = compute_identity(&[], 0.5);
        assert_eq!(m.num_frames, 0);
        assert_eq!(m.idf1, 0.0);
    }
}
