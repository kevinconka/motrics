//! Assignment primitives for matching predicted boxes to ground-truth boxes.
//!
//! Given an IoU matrix, [`match_boxes`] pairs rows (set A) to columns (set B)
//! subject to an IoU threshold. Two strategies are provided:
//!
//! * [`Method::Hungarian`] — optimal assignment maximising total IoU. The heavy
//!   lifting is delegated to the [`lsap`] crate, a Rust port of SciPy's
//!   `linear_sum_assignment` (Jonker–Volgenant). This is the same solver used by
//!   TrackEval / py-motmetrics, which keeps our numbers aligned with theirs.
//! * [`Method::Greedy`] — assign the highest-IoU pairs first; fast, but not
//!   guaranteed optimal.

/// Which assignment strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Hungarian,
    Greedy,
}

/// The result of matching set A (rows) against set B (columns).
#[derive(Debug, Default, Clone)]
pub struct MatchResult {
    /// Matched `(a_index, b_index)` pairs, ordered by `a_index`.
    pub matches: Vec<(usize, usize)>,
    /// IoU score for each matched pair, parallel to `matches`.
    pub scores: Vec<f64>,
    /// Indices of set A with no match.
    pub unmatched_a: Vec<usize>,
    /// Indices of set B with no match.
    pub unmatched_b: Vec<usize>,
}

/// Match set A (rows) to set B (columns) with an optimal linear-sum assignment.
///
/// Feeds the IoU matrix to [`lsap::solve`] in maximise mode, then discards pairs
/// with IoU below `threshold` (the standard MOT matching recipe). The solver
/// handles rectangular matrices directly, so no padding is required.
fn match_hungarian(iou: &[Vec<f64>], n_a: usize, n_b: usize, threshold: f64) -> MatchResult {
    // Flatten to the row-major layout lsap expects: cost[i * n_b + j].
    let mut flat = Vec::with_capacity(n_a * n_b);
    for row in iou {
        flat.extend_from_slice(row);
    }

    // IoU values are always finite in [0, 1], so lsap::solve cannot fail here.
    let (rows, cols) =
        lsap::solve(n_a, n_b, &flat, true).expect("finite IoU matrix is always solvable");

    let mut pairs: Vec<(usize, usize)> = rows.into_iter().zip(cols).collect();
    pairs.sort_by_key(|&(i, _)| i);

    let mut result = MatchResult::default();
    let mut matched_a = vec![false; n_a];
    let mut matched_b = vec![false; n_b];
    for (i, j) in pairs {
        if iou[i][j] >= threshold {
            result.matches.push((i, j));
            result.scores.push(iou[i][j]);
            matched_a[i] = true;
            matched_b[j] = true;
        }
    }
    for (i, &used) in matched_a.iter().enumerate() {
        if !used {
            result.unmatched_a.push(i);
        }
    }
    for (j, &used) in matched_b.iter().enumerate() {
        if !used {
            result.unmatched_b.push(j);
        }
    }
    result
}

/// Match set A (rows) to set B (columns) greedily by descending IoU.
///
/// Candidate pairs with IoU at or above `threshold` are considered in order of
/// decreasing IoU (ties broken by `(a_index, b_index)` for determinism); the
/// first free pair wins.
fn match_greedy(iou: &[Vec<f64>], n_a: usize, n_b: usize, threshold: f64) -> MatchResult {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (i, row) in iou.iter().enumerate() {
        for (j, &score) in row.iter().enumerate() {
            if score >= threshold {
                pairs.push((score, i, j));
            }
        }
    }
    // Sort by IoU descending, then by indices ascending for deterministic ties.
    pairs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    let mut used_a = vec![false; n_a];
    let mut used_b = vec![false; n_b];
    let mut result = MatchResult::default();
    for (score, i, j) in pairs {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            result.matches.push((i, j));
            result.scores.push(score);
        }
    }
    // matches came out in descending-IoU order; sort by a_index for a stable API.
    let mut order: Vec<usize> = (0..result.matches.len()).collect();
    order.sort_by_key(|&k| result.matches[k].0);
    result.matches = order.iter().map(|&k| result.matches[k]).collect();
    result.scores = order.iter().map(|&k| result.scores[k]).collect();

    for (i, &used) in used_a.iter().enumerate() {
        if !used {
            result.unmatched_a.push(i);
        }
    }
    for (j, &used) in used_b.iter().enumerate() {
        if !used {
            result.unmatched_b.push(j);
        }
    }
    result
}

/// Match two sets of boxes given their IoU matrix.
///
/// `iou` must be `n_a` rows by `n_b` columns. Empty inputs yield a result with
/// everything unmatched.
pub fn match_boxes(
    iou: &[Vec<f64>],
    n_a: usize,
    n_b: usize,
    threshold: f64,
    method: Method,
) -> MatchResult {
    if n_a == 0 || n_b == 0 {
        return MatchResult {
            unmatched_a: (0..n_a).collect(),
            unmatched_b: (0..n_b).collect(),
            ..Default::default()
        };
    }
    match method {
        Method::Hungarian => match_hungarian(iou, n_a, n_b, threshold),
        Method::Greedy => match_greedy(iou, n_a, n_b, threshold),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_finds_optimal_permutation() {
        let iou = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = match_boxes(&iou, 2, 2, 0.5, Method::Hungarian);
        assert_eq!(r.matches, vec![(0, 0), (1, 1)]);
        assert!(r.unmatched_a.is_empty() && r.unmatched_b.is_empty());
    }

    #[test]
    fn hungarian_beats_greedy_on_conflict() {
        // Greedy grabs (0,0)=0.9 first, stranding row 1 (only 0.0 left) and
        // forcing col 1 (0.8) unused. Optimal takes (0,1)+(1,0) = 0.8+0.6.
        let iou = vec![vec![0.9, 0.8], vec![0.6, 0.0]];

        let greedy = match_boxes(&iou, 2, 2, 0.5, Method::Greedy);
        assert_eq!(greedy.matches, vec![(0, 0)]);

        let hungarian = match_boxes(&iou, 2, 2, 0.5, Method::Hungarian);
        assert_eq!(hungarian.matches, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn threshold_filters_weak_matches() {
        let iou = vec![vec![0.4, 0.0], vec![0.0, 0.9]];
        let r = match_boxes(&iou, 2, 2, 0.5, Method::Hungarian);
        assert_eq!(r.matches, vec![(1, 1)]);
        assert_eq!(r.unmatched_a, vec![0]);
        assert_eq!(r.unmatched_b, vec![0]);
    }

    #[test]
    fn rectangular_more_rows_than_cols() {
        let iou = vec![vec![0.9], vec![0.7]];
        let r = match_boxes(&iou, 2, 1, 0.5, Method::Hungarian);
        assert_eq!(r.matches, vec![(0, 0)]);
        assert_eq!(r.unmatched_a, vec![1]);
        assert!(r.unmatched_b.is_empty());
    }

    #[test]
    fn empty_inputs() {
        let iou: Vec<Vec<f64>> = Vec::new();
        let r = match_boxes(&iou, 0, 3, 0.5, Method::Hungarian);
        assert!(r.matches.is_empty());
        assert_eq!(r.unmatched_b, vec![0, 1, 2]);
    }
}
