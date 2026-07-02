//! Assignment primitives for matching predicted boxes to ground-truth boxes.
//!
//! Given an IoU matrix, [`match_boxes`] pairs rows (set A) to columns (set B)
//! subject to an IoU threshold. Two strategies are provided:
//!
//! * [`Method::Hungarian`] — optimal assignment maximising total IoU
//!   (Kuhn–Munkres). This mirrors the `linear_sum_assignment` approach used by
//!   TrackEval / py-motmetrics.
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

/// Solve the assignment problem on a square cost matrix, minimising total cost.
///
/// Returns `assignment` where `assignment[row] = col`. Uses the O(n^3)
/// Hungarian algorithm with potentials (the e-maxx formulation adapted to
/// `f64`). The matrix must be square and non-empty.
fn hungarian_square(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    let inf = f64::INFINITY;

    // All vectors are 1-indexed; index 0 is a sentinel row/column.
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row assigned to column j (0 = none)
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;

            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        // Augment along the alternating path.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![usize::MAX; n];
    for j in 1..=n {
        if p[j] != 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    assignment
}

/// Match set A (rows) to set B (columns) using the Hungarian algorithm.
///
/// The IoU matrix is padded to a square cost matrix (`cost = -iou`, padding
/// cells `0.0`). After the optimal assignment is found, pairs with IoU below
/// `threshold` are discarded, mirroring the standard MOT matching recipe.
fn match_hungarian(iou: &[Vec<f64>], n_a: usize, n_b: usize, threshold: f64) -> MatchResult {
    let n = n_a.max(n_b);
    let mut cost = vec![vec![0.0f64; n]; n];
    for (i, row) in iou.iter().enumerate() {
        for (j, &score) in row.iter().enumerate() {
            cost[i][j] = -score;
        }
    }

    let assignment = hungarian_square(&cost);

    let mut result = MatchResult::default();
    let mut matched_b = vec![false; n_b];
    for i in 0..n_a {
        let j = assignment[i];
        if j < n_b && iou[i][j] >= threshold {
            result.matches.push((i, j));
            result.scores.push(iou[i][j]);
            matched_b[j] = true;
        } else {
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
        // Identity is clearly optimal here.
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
