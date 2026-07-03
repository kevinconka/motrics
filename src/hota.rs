//! HOTA metrics (Higher Order Tracking Accuracy).
//!
//! Implements HOTA (Luiten et al., IJCV 2021) following TrackEval's formulation:
//!
//! 1. A global alignment score is computed between every gt/pred id pair from
//!    their co-occurrence, weighted by per-frame IoU.
//! 2. Each frame is matched with an optimal assignment on
//!    `global_alignment * IoU` (via [`lsap`]).
//! 3. Over a sweep of localization thresholds `alpha` (0.05..=0.95), detection
//!    accuracy (DetA), association accuracy (AssA), and localization (LocA) are
//!    accumulated; `HOTA(alpha) = sqrt(DetA * AssA)`.
//!
//! The reported scalars are the mean over the alpha thresholds. Bit-exact
//! TrackEval parity is validated in the parity-tests milestone.

use std::collections::HashMap;

use crate::clear::Frame;
use crate::iou::iou_matrix;

/// HOTA metrics over a sequence, summarised (mean over alpha) with the per-alpha
/// curves retained.
#[derive(Debug, Default, Clone)]
pub struct HotaMetrics {
    /// HOTA score: mean over alpha of `sqrt(DetA * AssA)`.
    pub hota: f64,
    /// Detection accuracy: mean over alpha of `TP / (TP + FN + FP)`.
    pub deta: f64,
    /// Association accuracy: mean over alpha.
    pub assa: f64,
    /// Localization accuracy: mean over alpha of the mean matched IoU.
    pub loca: f64,
    /// The alpha (localization) thresholds swept.
    pub alphas: Vec<f64>,
    /// Per-alpha HOTA scores, parallel to `alphas`.
    pub hota_alphas: Vec<f64>,
    /// Per-alpha DetA scores, parallel to `alphas`.
    pub deta_alphas: Vec<f64>,
    /// Per-alpha AssA scores, parallel to `alphas`.
    pub assa_alphas: Vec<f64>,
    /// Per-alpha LocA scores, parallel to `alphas`.
    pub loca_alphas: Vec<f64>,
    /// Per-alpha true positive counts, parallel to `alphas`.
    pub hota_tp_alphas: Vec<f64>,
    /// Per-alpha false negative counts, parallel to `alphas`.
    pub hota_fn_alphas: Vec<f64>,
    /// Per-alpha false positive counts, parallel to `alphas`.
    pub hota_fp_alphas: Vec<f64>,
    /// Per-alpha association recall, parallel to `alphas`.
    pub ass_re_alphas: Vec<f64>,
    /// Per-alpha association precision, parallel to `alphas`.
    pub ass_pr_alphas: Vec<f64>,
    /// Number of frames processed.
    pub num_frames: usize,
    /// Total ground-truth detections across all frames.
    pub num_gt: usize,
    /// Total predicted detections across all frames.
    pub num_pred: usize,
}

/// Localization thresholds: 0.05, 0.10, ..., 0.95 (19 values), as in TrackEval.
fn alpha_thresholds() -> Vec<f64> {
    (1..=19).map(|i| f64::from(i) * 0.05).collect()
}

/// Per-frame precomputed dense id indices and IoU matrix.
struct FrameData {
    gt_idx: Vec<usize>,
    pred_idx: Vec<usize>,
    iou: Vec<Vec<f64>>,
}

/// Compute HOTA metrics over a sequence of frames.
pub fn compute_hota(frames: &[Frame]) -> HotaMetrics {
    let mut gt_index: HashMap<i64, usize> = HashMap::new();
    let mut pred_index: HashMap<i64, usize> = HashMap::new();
    for f in frames {
        for &id in f.gt_ids {
            let next = gt_index.len();
            gt_index.entry(id).or_insert(next);
        }
        for &id in f.pred_ids {
            let next = pred_index.len();
            pred_index.entry(id).or_insert(next);
        }
    }
    let n_g = gt_index.len();
    let n_t = pred_index.len();
    let alphas = alpha_thresholds();
    let n_a = alphas.len();

    let mut result = HotaMetrics {
        num_frames: frames.len(),
        alphas: alphas.clone(),
        hota_alphas: vec![0.0; n_a],
        deta_alphas: vec![0.0; n_a],
        assa_alphas: vec![0.0; n_a],
        loca_alphas: vec![0.0; n_a],
        hota_tp_alphas: vec![0.0; n_a],
        hota_fn_alphas: vec![0.0; n_a],
        hota_fp_alphas: vec![0.0; n_a],
        ass_re_alphas: vec![0.0; n_a],
        ass_pr_alphas: vec![0.0; n_a],
        ..Default::default()
    };

    // Precompute per-frame indices, IoU matrices, and id detection counts.
    let mut fds: Vec<FrameData> = Vec::with_capacity(frames.len());
    let mut gt_count = vec![0.0f64; n_g];
    let mut pred_count = vec![0.0f64; n_t];
    for f in frames {
        let gt_idx: Vec<usize> = f.gt_ids.iter().map(|id| gt_index[id]).collect();
        let pred_idx: Vec<usize> = f.pred_ids.iter().map(|id| pred_index[id]).collect();
        for &g in &gt_idx {
            gt_count[g] += 1.0;
        }
        for &p in &pred_idx {
            pred_count[p] += 1.0;
        }
        let iou = if gt_idx.is_empty() || pred_idx.is_empty() {
            Vec::new()
        } else {
            iou_matrix(f.gt_boxes, f.pred_boxes)
        };
        fds.push(FrameData {
            gt_idx,
            pred_idx,
            iou,
        });
    }
    result.num_gt = gt_count.iter().sum::<f64>() as usize;
    result.num_pred = pred_count.iter().sum::<f64>() as usize;

    // No possible true positives -> every gt is a miss or every pred is a false
    // positive (mirrors TrackEval's own early-return special case).
    if n_t == 0 {
        result.hota_fn_alphas = vec![result.num_gt as f64; n_a];
        result.loca_alphas = vec![1.0; n_a];
        return result;
    }
    if n_g == 0 {
        result.hota_fp_alphas = vec![result.num_pred as f64; n_a];
        result.loca_alphas = vec![1.0; n_a];
        return result;
    }

    // Phase 0: global alignment score from IoU-weighted co-occurrence.
    let mut pmc = vec![vec![0.0f64; n_t]; n_g]; // potential matches count
    for fd in &fds {
        if fd.iou.is_empty() {
            continue;
        }
        let (ng, np) = (fd.gt_idx.len(), fd.pred_idx.len());
        let mut row_sum = vec![0.0f64; ng];
        let mut col_sum = vec![0.0f64; np];
        for (i, iou_row) in fd.iou.iter().enumerate() {
            for (j, &v) in iou_row.iter().enumerate() {
                row_sum[i] += v;
                col_sum[j] += v;
            }
        }
        for i in 0..ng {
            for j in 0..np {
                let denom = row_sum[i] + col_sum[j] - fd.iou[i][j];
                if denom > 1e-10 {
                    pmc[fd.gt_idx[i]][fd.pred_idx[j]] += fd.iou[i][j] / denom;
                }
            }
        }
    }
    let mut gas = vec![vec![0.0f64; n_t]; n_g]; // global alignment score
    for g in 0..n_g {
        for t in 0..n_t {
            let denom = gt_count[g] + pred_count[t] - pmc[g][t];
            if denom > 1e-10 {
                gas[g][t] = pmc[g][t] / denom;
            }
        }
    }

    // Phase 1: per-frame optimal matching, then per-alpha accumulation.
    let mut tp = vec![0.0f64; n_a];
    let mut fn_ = vec![0.0f64; n_a];
    let mut fp = vec![0.0f64; n_a];
    let mut loca_sum = vec![0.0f64; n_a];
    let mut matches_counts = vec![vec![vec![0.0f64; n_t]; n_g]; n_a];

    for fd in &fds {
        let (ng, np) = (fd.gt_idx.len(), fd.pred_idx.len());
        if ng == 0 {
            for v in &mut fp {
                *v += np as f64;
            }
            continue;
        }
        if np == 0 {
            for v in &mut fn_ {
                *v += ng as f64;
            }
            continue;
        }

        let mut flat = vec![0.0f64; ng * np];
        for i in 0..ng {
            for j in 0..np {
                flat[i * np + j] = gas[fd.gt_idx[i]][fd.pred_idx[j]] * fd.iou[i][j];
            }
        }
        let (rows, cols) = lsap::solve(ng, np, &flat, true).expect("finite score matrix");

        for a in 0..n_a {
            let alpha = alphas[a];
            let mut num = 0.0;
            for (&i, &j) in rows.iter().zip(&cols) {
                if fd.iou[i][j] >= alpha - 1e-9 {
                    num += 1.0;
                    loca_sum[a] += fd.iou[i][j];
                    matches_counts[a][fd.gt_idx[i]][fd.pred_idx[j]] += 1.0;
                }
            }
            tp[a] += num;
            fn_[a] += ng as f64 - num;
            fp[a] += np as f64 - num;
        }
    }
    result.hota_tp_alphas = tp.clone();
    result.hota_fn_alphas = fn_.clone();
    result.hota_fp_alphas = fp.clone();

    // Phase 2: per-alpha DetA, AssA, AssRe, AssPr, LocA, HOTA.
    for a in 0..n_a {
        let deta = tp[a] / (tp[a] + fn_[a] + fp[a]).max(1.0);

        let mut ass_num = 0.0;
        let mut ass_re_num = 0.0;
        let mut ass_pr_num = 0.0;
        for g in 0..n_g {
            for t in 0..n_t {
                let c = matches_counts[a][g][t];
                if c > 0.0 {
                    ass_num += c * (c / (gt_count[g] + pred_count[t] - c).max(1.0));
                    ass_re_num += c * (c / gt_count[g].max(1.0));
                    ass_pr_num += c * (c / pred_count[t].max(1.0));
                }
            }
        }
        let assa = ass_num / tp[a].max(1.0);
        let loca = loca_sum[a].max(1e-10) / tp[a].max(1e-10);

        result.deta_alphas[a] = deta;
        result.assa_alphas[a] = assa;
        result.ass_re_alphas[a] = ass_re_num / tp[a].max(1.0);
        result.ass_pr_alphas[a] = ass_pr_num / tp[a].max(1.0);
        result.hota_alphas[a] = (deta * assa).sqrt();
        result.loca_alphas[a] = loca;
    }

    let n_a_f = n_a as f64;
    result.deta = result.deta_alphas.iter().sum::<f64>() / n_a_f;
    result.assa = result.assa_alphas.iter().sum::<f64>() / n_a_f;
    result.hota = result.hota_alphas.iter().sum::<f64>() / n_a_f;
    result.loca = result.loca_alphas.iter().sum::<f64>() / n_a_f;
    result
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
    fn perfect_tracking_scores_one() {
        let frames = [frame(&[1], &[A], &[1], &[A]), frame(&[1], &[A], &[1], &[A])];
        let m = compute_hota(&frames);
        assert!((m.hota - 1.0).abs() < 1e-9);
        assert!((m.deta - 1.0).abs() < 1e-9);
        assert!((m.assa - 1.0).abs() < 1e-9);
        assert!((m.loca - 1.0).abs() < 1e-9);
        assert_eq!(m.alphas.len(), 19);
        assert_eq!(m.hota_alphas.len(), 19);
        assert!(m.hota_tp_alphas.iter().all(|&v| v == 2.0));
        assert!(m.hota_fn_alphas.iter().all(|&v| v == 0.0));
        assert!(m.hota_fp_alphas.iter().all(|&v| v == 0.0));
        assert!(m.ass_re_alphas.iter().all(|&v| (v - 1.0).abs() < 1e-9));
        assert!(m.ass_pr_alphas.iter().all(|&v| (v - 1.0).abs() < 1e-9));
        assert!(m.loca_alphas.iter().all(|&v| (v - 1.0).abs() < 1e-9));
    }

    #[test]
    fn id_split_keeps_deta_but_halves_assa() {
        // One gt object perfectly localised, but covered by two predicted ids.
        // Detection is perfect (DetA=1) while association halves (AssA=0.5).
        let frames = [
            frame(&[1], &[A], &[10], &[A]),
            frame(&[1], &[A], &[10], &[A]),
            frame(&[1], &[A], &[20], &[A]),
            frame(&[1], &[A], &[20], &[A]),
        ];
        let m = compute_hota(&frames);
        assert!((m.deta - 1.0).abs() < 1e-9, "deta={}", m.deta);
        assert!((m.assa - 0.5).abs() < 1e-9, "assa={}", m.assa);
        assert!((m.hota - 0.5f64.sqrt()).abs() < 1e-9, "hota={}", m.hota);
        assert!((m.loca - 1.0).abs() < 1e-9);
        // 4 detections total, split 2/2 across two predicted ids.
        assert!(m.hota_tp_alphas.iter().all(|&v| v == 4.0));
        assert!(m.ass_re_alphas.iter().all(|&v| (v - 0.5).abs() < 1e-9));
        assert!(m.ass_pr_alphas.iter().all(|&v| (v - 1.0).abs() < 1e-9));
    }

    #[test]
    fn all_false_positives() {
        let frames = [frame(&[], &[], &[9], &[A])];
        let m = compute_hota(&frames);
        assert_eq!(m.hota, 0.0);
        assert_eq!(m.num_pred, 1);
        assert_eq!(m.num_gt, 0);
        assert!(m.hota_fp_alphas.iter().all(|&v| v == 1.0));
        assert!(m.loca_alphas.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn all_misses() {
        let frames = [frame(&[1], &[A], &[], &[])];
        let m = compute_hota(&frames);
        assert_eq!(m.hota, 0.0);
        assert_eq!(m.num_gt, 1);
        assert!(m.hota_fn_alphas.iter().all(|&v| v == 1.0));
        assert!(m.loca_alphas.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn empty_sequence() {
        let m = compute_hota(&[]);
        assert_eq!(m.num_frames, 0);
        assert_eq!(m.hota, 0.0);
        assert_eq!(m.alphas.len(), 19);
    }
}
