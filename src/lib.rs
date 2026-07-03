//! `motrics` — an extremely fast MOT and HOTA metrics library.
//!
//! This crate is compiled into the `motrics._motrics` extension module. The
//! public, ergonomic API lives in the `motrics` Python package (see
//! `python/motrics/`), which re-exports the pieces below.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod assignment;
mod clear;
mod hota;
mod identity;
mod iou;

use assignment::Method;
use iou::Bbox;

/// Validate frame-aligned inputs and borrow them as a slice of [`clear::Frame`].
///
/// Shared by the sequence metrics (`compute_clear`, `compute_identity`).
fn build_frames<'a>(
    gt_ids: &'a [Vec<i64>],
    gt_boxes: &'a [Vec<Bbox>],
    pred_ids: &'a [Vec<i64>],
    pred_boxes: &'a [Vec<Bbox>],
) -> PyResult<Vec<clear::Frame<'a>>> {
    if gt_ids.len() != pred_ids.len() {
        return Err(PyValueError::new_err(format!(
            "gt and pred must have the same number of frames, got {} and {}",
            gt_ids.len(),
            pred_ids.len()
        )));
    }
    if gt_ids.len() != gt_boxes.len() || pred_ids.len() != pred_boxes.len() {
        return Err(PyValueError::new_err(
            "ids and boxes must have the same number of frames",
        ));
    }

    let mut frames = Vec::with_capacity(gt_ids.len());
    for i in 0..gt_ids.len() {
        if gt_ids[i].len() != gt_boxes[i].len() {
            return Err(PyValueError::new_err(format!(
                "frame {i}: gt_ids and gt_boxes length mismatch ({} vs {})",
                gt_ids[i].len(),
                gt_boxes[i].len()
            )));
        }
        if pred_ids[i].len() != pred_boxes[i].len() {
            return Err(PyValueError::new_err(format!(
                "frame {i}: pred_ids and pred_boxes length mismatch ({} vs {})",
                pred_ids[i].len(),
                pred_boxes[i].len()
            )));
        }
        frames.push(clear::Frame {
            gt_ids: &gt_ids[i],
            gt_boxes: &gt_boxes[i],
            pred_ids: &pred_ids[i],
            pred_boxes: &pred_boxes[i],
        });
    }
    Ok(frames)
}

/// Validate precomputed per-frame similarity matrices and borrow them as a
/// slice of [`clear::SimFrame`]. Shared by the similarity-matrix entry points
/// (`compute_clear_from_similarity`, `compute_identity_from_similarity`).
fn build_sim_frames<'a>(
    gt_ids: &'a [Vec<i64>],
    pred_ids: &'a [Vec<i64>],
    similarity: &'a [Vec<Vec<f64>>],
) -> PyResult<Vec<clear::SimFrame<'a>>> {
    if gt_ids.len() != pred_ids.len() || gt_ids.len() != similarity.len() {
        return Err(PyValueError::new_err(format!(
            "gt_ids, pred_ids, and similarity must have the same number of frames, got {}, {}, {}",
            gt_ids.len(),
            pred_ids.len(),
            similarity.len()
        )));
    }

    let mut frames = Vec::with_capacity(gt_ids.len());
    for (i, ((g, p), s)) in gt_ids.iter().zip(pred_ids).zip(similarity).enumerate() {
        if s.len() != g.len() {
            return Err(PyValueError::new_err(format!(
                "frame {i}: similarity has {} rows, expected {} (len(gt_ids[{i}]))",
                s.len(),
                g.len()
            )));
        }
        for (r, row) in s.iter().enumerate() {
            if row.len() != p.len() {
                return Err(PyValueError::new_err(format!(
                    "frame {i}: similarity row {r} has {} columns, expected {} (len(pred_ids[{i}]))",
                    row.len(),
                    p.len()
                )));
            }
        }
        frames.push(clear::SimFrame {
            gt_ids: g,
            pred_ids: p,
            similarity: s,
        });
    }
    Ok(frames)
}

/// Return the version of the compiled Rust core.
///
/// Wired through Cargo at build time so the native and Python versions can
/// never drift.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Whether the extension was compiled with debug assertions (used to warn
/// before reporting benchmark timings — a debug build is far slower).
#[pyfunction]
fn is_debug_build() -> bool {
    cfg!(debug_assertions)
}

/// Intersection-over-union of two `xyxy` boxes `(x1, y1, x2, y2)`.
#[pyfunction]
#[pyo3(name = "iou")]
fn iou_py(box_a: Bbox, box_b: Bbox) -> f64 {
    iou::iou(&box_a, &box_b)
}

/// Pairwise IoU matrix between two sets of `xyxy` boxes.
///
/// Returns a list of `len(boxes_a)` rows, each with `len(boxes_b)` IoU values.
#[pyfunction]
fn iou_matrix(boxes_a: Vec<Bbox>, boxes_b: Vec<Bbox>) -> Vec<Vec<f64>> {
    iou::iou_matrix(&boxes_a, &boxes_b)
}

/// The result of matching two sets of boxes.
#[pyclass(frozen)]
struct Matching {
    /// Matched `(a_index, b_index)` pairs, ordered by `a_index`.
    #[pyo3(get)]
    matches: Vec<(usize, usize)>,
    /// IoU score for each matched pair, parallel to `matches`.
    #[pyo3(get)]
    scores: Vec<f64>,
    /// Indices of `boxes_a` that were not matched.
    #[pyo3(get)]
    unmatched_a: Vec<usize>,
    /// Indices of `boxes_b` that were not matched.
    #[pyo3(get)]
    unmatched_b: Vec<usize>,
}

#[pymethods]
impl Matching {
    fn __repr__(&self) -> String {
        format!(
            "Matching(matches={} pairs, unmatched_a={}, unmatched_b={})",
            self.matches.len(),
            self.unmatched_a.len(),
            self.unmatched_b.len(),
        )
    }
}

/// Match two sets of `xyxy` boxes.
///
/// `method` is either `"hungarian"` (optimal, maximises total IoU) or
/// `"greedy"` (assign highest-IoU pairs first). Only pairs with IoU at or above
/// `iou_threshold` are kept.
#[pyfunction]
#[pyo3(signature = (boxes_a, boxes_b, iou_threshold=0.5, method="hungarian"))]
fn match_boxes(
    boxes_a: Vec<Bbox>,
    boxes_b: Vec<Bbox>,
    iou_threshold: f64,
    method: &str,
) -> PyResult<Matching> {
    let method = match method {
        "hungarian" => Method::Hungarian,
        "greedy" => Method::Greedy,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}, expected \"hungarian\" or \"greedy\""
            )))
        }
    };

    let n_a = boxes_a.len();
    let n_b = boxes_b.len();
    let matrix = iou::iou_matrix(&boxes_a, &boxes_b);
    let result = assignment::match_boxes(&matrix, n_a, n_b, iou_threshold, method);

    Ok(Matching {
        matches: result.matches,
        scores: result.scores,
        unmatched_a: result.unmatched_a,
        unmatched_b: result.unmatched_b,
    })
}

/// Accumulated CLEAR MOT metrics over a sequence.
#[pyclass(frozen)]
struct ClearMetrics {
    /// Multiple Object Tracking Accuracy: `1 - (FN + FP + IDSW) / num_gt`.
    #[pyo3(get)]
    mota: f64,
    /// Multiple Object Tracking Precision: mean IoU over matched pairs.
    #[pyo3(get)]
    motp: f64,
    /// Number of frames processed.
    #[pyo3(get)]
    num_frames: usize,
    /// Total ground-truth detections across all frames.
    #[pyo3(get)]
    num_gt: usize,
    /// True positives: matched (gt, pred) pairs.
    #[pyo3(get)]
    num_matches: usize,
    /// False positives: tracker detections with no match.
    #[pyo3(get)]
    num_false_positives: usize,
    /// Misses: ground-truth detections with no match.
    #[pyo3(get)]
    num_misses: usize,
    /// Identity switches.
    #[pyo3(get)]
    num_switches: usize,
}

#[pymethods]
impl ClearMetrics {
    fn __repr__(&self) -> String {
        format!(
            "ClearMetrics(mota={:.4}, motp={:.4}, tp={}, fp={}, fn={}, idsw={})",
            self.mota,
            self.motp,
            self.num_matches,
            self.num_false_positives,
            self.num_misses,
            self.num_switches,
        )
    }
}

impl From<clear::ClearMetrics> for ClearMetrics {
    fn from(m: clear::ClearMetrics) -> Self {
        ClearMetrics {
            mota: m.mota,
            motp: m.motp,
            num_frames: m.num_frames,
            num_gt: m.num_gt,
            num_matches: m.num_matches,
            num_false_positives: m.num_false_positives,
            num_misses: m.num_misses,
            num_switches: m.num_switches,
        }
    }
}

/// Compute CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches) for a sequence.
///
/// Inputs are frame-aligned: `gt_ids[t]` / `gt_boxes[t]` describe ground-truth
/// objects in frame `t` (and likewise `pred_ids` / `pred_boxes` for the
/// tracker). `gt_ids` and `pred_ids` must have the same number of frames, and
/// within each frame the id and box lists must have equal length.
#[pyfunction]
#[pyo3(signature = (gt_ids, gt_boxes, pred_ids, pred_boxes, iou_threshold=0.5))]
fn compute_clear(
    gt_ids: Vec<Vec<i64>>,
    gt_boxes: Vec<Vec<Bbox>>,
    pred_ids: Vec<Vec<i64>>,
    pred_boxes: Vec<Vec<Bbox>>,
    iou_threshold: f64,
) -> PyResult<ClearMetrics> {
    let frames = build_frames(&gt_ids, &gt_boxes, &pred_ids, &pred_boxes)?;
    Ok(clear::compute_clear(&frames, iou_threshold).into())
}

/// Compute CLEAR MOT metrics from precomputed per-frame similarity matrices
/// instead of boxes.
///
/// For callers that already hold pairwise scores (e.g. a `motmetrics`-style
/// distance matrix converted to similarity). `similarity[t][i][j]` scores
/// `gt_ids[t][i]` against `pred_ids[t][j]`; higher is better, the same
/// convention as IoU, and pairs below `threshold` are never matched.
#[pyfunction]
#[pyo3(signature = (gt_ids, pred_ids, similarity, threshold=0.5))]
fn compute_clear_from_similarity(
    gt_ids: Vec<Vec<i64>>,
    pred_ids: Vec<Vec<i64>>,
    similarity: Vec<Vec<Vec<f64>>>,
    threshold: f64,
) -> PyResult<ClearMetrics> {
    let frames = build_sim_frames(&gt_ids, &pred_ids, &similarity)?;
    Ok(clear::compute_clear_from_similarity(&frames, threshold).into())
}

/// Accumulated Identity metrics (IDF1/IDP/IDR) over a sequence.
#[pyclass(frozen)]
struct IdentityMetrics {
    /// Identity F1: `IDTP / (IDTP + 0.5 IDFP + 0.5 IDFN)`.
    #[pyo3(get)]
    idf1: f64,
    /// Identity precision: `IDTP / (IDTP + IDFP)`.
    #[pyo3(get)]
    idp: f64,
    /// Identity recall: `IDTP / (IDTP + IDFN)`.
    #[pyo3(get)]
    idr: f64,
    /// Identity true positives.
    #[pyo3(get)]
    idtp: usize,
    /// Identity false positives.
    #[pyo3(get)]
    idfp: usize,
    /// Identity false negatives.
    #[pyo3(get)]
    idfn: usize,
    /// Number of frames processed.
    #[pyo3(get)]
    num_frames: usize,
    /// Total ground-truth detections across all frames.
    #[pyo3(get)]
    num_gt: usize,
    /// Total predicted detections across all frames.
    #[pyo3(get)]
    num_pred: usize,
}

#[pymethods]
impl IdentityMetrics {
    fn __repr__(&self) -> String {
        format!(
            "IdentityMetrics(idf1={:.4}, idp={:.4}, idr={:.4}, idtp={}, idfp={}, idfn={})",
            self.idf1, self.idp, self.idr, self.idtp, self.idfp, self.idfn,
        )
    }
}

impl From<identity::IdentityMetrics> for IdentityMetrics {
    fn from(m: identity::IdentityMetrics) -> Self {
        IdentityMetrics {
            idf1: m.idf1,
            idp: m.idp,
            idr: m.idr,
            idtp: m.idtp,
            idfp: m.idfp,
            idfn: m.idfn,
            num_frames: m.num_frames,
            num_gt: m.num_gt,
            num_pred: m.num_pred,
        }
    }
}

/// Compute Identity metrics (IDF1, IDP, IDR) for a sequence.
///
/// Inputs are frame-aligned exactly like [`compute_clear`]. Identity metrics use
/// a single global bipartite matching between whole ground-truth and predicted
/// trajectories, so id consistency over time is rewarded.
#[pyfunction]
#[pyo3(signature = (gt_ids, gt_boxes, pred_ids, pred_boxes, iou_threshold=0.5))]
fn compute_identity(
    gt_ids: Vec<Vec<i64>>,
    gt_boxes: Vec<Vec<Bbox>>,
    pred_ids: Vec<Vec<i64>>,
    pred_boxes: Vec<Vec<Bbox>>,
    iou_threshold: f64,
) -> PyResult<IdentityMetrics> {
    let frames = build_frames(&gt_ids, &gt_boxes, &pred_ids, &pred_boxes)?;
    Ok(identity::compute_identity(&frames, iou_threshold).into())
}

/// Compute Identity metrics from precomputed per-frame similarity matrices
/// instead of boxes. Same similarity convention as
/// [`compute_clear_from_similarity`].
#[pyfunction]
#[pyo3(signature = (gt_ids, pred_ids, similarity, threshold=0.5))]
fn compute_identity_from_similarity(
    gt_ids: Vec<Vec<i64>>,
    pred_ids: Vec<Vec<i64>>,
    similarity: Vec<Vec<Vec<f64>>>,
    threshold: f64,
) -> PyResult<IdentityMetrics> {
    let frames = build_sim_frames(&gt_ids, &pred_ids, &similarity)?;
    Ok(identity::compute_identity_from_similarity(&frames, threshold).into())
}

/// HOTA metrics over a sequence (summarised, with per-alpha curves retained).
#[pyclass(frozen)]
struct HotaMetrics {
    /// HOTA score: mean over alpha of `sqrt(DetA * AssA)`.
    #[pyo3(get)]
    hota: f64,
    /// Detection accuracy: mean over alpha.
    #[pyo3(get)]
    deta: f64,
    /// Association accuracy: mean over alpha.
    #[pyo3(get)]
    assa: f64,
    /// Localization accuracy: mean over alpha.
    #[pyo3(get)]
    loca: f64,
    /// The alpha (localization) thresholds swept.
    #[pyo3(get)]
    alphas: Vec<f64>,
    /// Per-alpha HOTA scores, parallel to `alphas`.
    #[pyo3(get)]
    hota_alphas: Vec<f64>,
    /// Per-alpha DetA scores, parallel to `alphas`.
    #[pyo3(get)]
    deta_alphas: Vec<f64>,
    /// Per-alpha AssA scores, parallel to `alphas`.
    #[pyo3(get)]
    assa_alphas: Vec<f64>,
    /// Per-alpha LocA scores, parallel to `alphas`.
    #[pyo3(get)]
    loca_alphas: Vec<f64>,
    /// Per-alpha true positive counts, parallel to `alphas`.
    #[pyo3(get)]
    hota_tp_alphas: Vec<f64>,
    /// Per-alpha false negative counts, parallel to `alphas`.
    #[pyo3(get)]
    hota_fn_alphas: Vec<f64>,
    /// Per-alpha false positive counts, parallel to `alphas`.
    #[pyo3(get)]
    hota_fp_alphas: Vec<f64>,
    /// Per-alpha association recall, parallel to `alphas`.
    #[pyo3(get)]
    ass_re_alphas: Vec<f64>,
    /// Per-alpha association precision, parallel to `alphas`.
    #[pyo3(get)]
    ass_pr_alphas: Vec<f64>,
    /// Number of frames processed.
    #[pyo3(get)]
    num_frames: usize,
    /// Total ground-truth detections across all frames.
    #[pyo3(get)]
    num_gt: usize,
    /// Total predicted detections across all frames.
    #[pyo3(get)]
    num_pred: usize,
}

#[pymethods]
impl HotaMetrics {
    fn __repr__(&self) -> String {
        format!(
            "HotaMetrics(hota={:.4}, deta={:.4}, assa={:.4}, loca={:.4})",
            self.hota, self.deta, self.assa, self.loca,
        )
    }
}

/// Compute HOTA metrics (DetA, AssA, LocA, plus per-alpha curves) for a sequence.
///
/// Inputs are frame-aligned exactly like [`compute_clear`]. HOTA sweeps a set of
/// localization thresholds internally, so unlike the other metrics it takes no
/// single `iou_threshold`.
#[pyfunction]
fn compute_hota(
    gt_ids: Vec<Vec<i64>>,
    gt_boxes: Vec<Vec<Bbox>>,
    pred_ids: Vec<Vec<i64>>,
    pred_boxes: Vec<Vec<Bbox>>,
) -> PyResult<HotaMetrics> {
    let frames = build_frames(&gt_ids, &gt_boxes, &pred_ids, &pred_boxes)?;
    let m = hota::compute_hota(&frames);
    Ok(HotaMetrics {
        hota: m.hota,
        deta: m.deta,
        assa: m.assa,
        loca: m.loca,
        alphas: m.alphas,
        hota_alphas: m.hota_alphas,
        deta_alphas: m.deta_alphas,
        assa_alphas: m.assa_alphas,
        loca_alphas: m.loca_alphas,
        hota_tp_alphas: m.hota_tp_alphas,
        hota_fn_alphas: m.hota_fn_alphas,
        hota_fp_alphas: m.hota_fp_alphas,
        ass_re_alphas: m.ass_re_alphas,
        ass_pr_alphas: m.ass_pr_alphas,
        num_frames: m.num_frames,
        num_gt: m.num_gt,
        num_pred: m.num_pred,
    })
}

/// The `motrics._motrics` extension module.
#[pymodule]
fn _motrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_debug_build, m)?)?;
    m.add_function(wrap_pyfunction!(iou_py, m)?)?;
    m.add_function(wrap_pyfunction!(iou_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(match_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(compute_clear, m)?)?;
    m.add_function(wrap_pyfunction!(compute_clear_from_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_identity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_identity_from_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hota, m)?)?;
    m.add_class::<Matching>()?;
    m.add_class::<ClearMetrics>()?;
    m.add_class::<IdentityMetrics>()?;
    m.add_class::<HotaMetrics>()?;
    Ok(())
}
