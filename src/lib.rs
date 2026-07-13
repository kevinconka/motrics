//! `motrics` — an extremely fast MOT and HOTA metrics library.
//!
//! This crate is compiled into the `motrics._motrics` extension module. The
//! public, ergonomic API lives in the `motrics` Python package (see
//! `python/motrics/`), which re-exports the pieces below.

use std::borrow::Cow;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod accumulator;
mod assignment;
mod boxes;
mod clear;
mod hota;
mod identity;
mod iou;
mod iou3d;
mod mask;

use assignment::Method;
use boxes::{to_xyxy, BoxFormat, PyBoxes, PyBoxes3d};
use iou::Bbox;
use iou3d::Box3d;

/// Convert one `box_format` argument's worth of per-frame [`PyBoxes`] into
/// `xyxy`, borrowing (zero-copy) wherever the format and layout allow it.
fn to_xyxy_frames<'a>(
    boxes: &'a [PyBoxes<'a>],
    format: BoxFormat,
) -> PyResult<Vec<Cow<'a, [Bbox]>>> {
    boxes.iter().map(|b| b.as_boxes(format)).collect()
}

/// Validate frame-aligned inputs and borrow them as a slice of [`clear::Frame`].
///
/// Shared by the sequence metrics (`compute_clear`, `compute_identity`).
fn build_frames<'a>(
    gt_ids: &'a [Vec<i64>],
    gt_boxes: &'a [Cow<'a, [Bbox]>],
    pred_ids: &'a [Vec<i64>],
    pred_boxes: &'a [Cow<'a, [Bbox]>],
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
            gt_boxes: gt_boxes[i].as_ref(),
            pred_ids: &pred_ids[i],
            pred_boxes: pred_boxes[i].as_ref(),
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

/// Intersection-over-union of two boxes.
///
/// `box_format` is `"xyxy"` (`x1, y1, x2, y2`, the default) or `"xywh"`
/// (`x, y, width, height`).
#[pyfunction]
#[pyo3(name = "iou", signature = (box_a, box_b, box_format="xyxy"))]
fn iou_py(box_a: Bbox, box_b: Bbox, box_format: &str) -> PyResult<f64> {
    let format = BoxFormat::parse(box_format)?;
    Ok(iou::iou(&to_xyxy(box_a, format), &to_xyxy(box_b, format)))
}

/// Pairwise IoU matrix between two sets of boxes.
///
/// Each of `boxes_a`/`boxes_b` is a Python sequence of 4-tuples or a `(N, 4)`
/// float64 NumPy array (zero-copy for a contiguous `xyxy` array). `box_format`
/// is `"xyxy"` (default) or `"xywh"`.
///
/// Returns a list of `len(boxes_a)` rows, each with `len(boxes_b)` IoU values.
#[pyfunction]
#[pyo3(signature = (boxes_a, boxes_b, box_format="xyxy"))]
fn iou_matrix(boxes_a: PyBoxes, boxes_b: PyBoxes, box_format: &str) -> PyResult<Vec<Vec<f64>>> {
    let format = BoxFormat::parse(box_format)?;
    let a = boxes_a.as_boxes(format)?;
    let b = boxes_b.as_boxes(format)?;
    Ok(iou::iou_matrix(&a, &b))
}

/// Volumetric intersection-over-union of two oriented 3D boxes.
///
/// Each box is `[x, y, z, l, w, h, yaw]`: centre, full extents, and heading
/// (radians) about the vertical `y` axis in the `x`–`z` ground plane, the
/// KITTI / AB3DMOT convention. At `yaw = 0`, `l` runs along `x` and `w` along
/// `z`.
#[pyfunction]
fn iou_3d(box_a: Box3d, box_b: Box3d) -> f64 {
    iou3d::iou_3d(&box_a, &box_b)
}

/// Pairwise 3D IoU matrix between two sets of oriented 3D boxes.
///
/// Each of `boxes_a`/`boxes_b` is a Python sequence of 7-tuples or a `(N, 7)`
/// float64 NumPy array (zero-copy for a contiguous array); see [`iou_3d`] for
/// the box convention.
///
/// Returns a list of `len(boxes_a)` rows, each with `len(boxes_b)` IoU values.
#[pyfunction]
fn iou_3d_matrix(boxes_a: PyBoxes3d, boxes_b: PyBoxes3d) -> PyResult<Vec<Vec<f64>>> {
    let a = boxes_a.as_boxes()?;
    let b = boxes_b.as_boxes()?;
    Ok(iou3d::iou_3d_matrix(&a, &b))
}

/// A run-length-encoded binary mask, COCO/pycocotools convention: alternating
/// background/foreground run lengths walked column-major over an `(h, w)`
/// image, always starting with a (possibly zero-length) background run.
#[pyclass(frozen)]
struct Mask {
    rle: mask::Rle,
}

#[pymethods]
impl Mask {
    /// `size`: `(height, width)`. `counts`: alternating background/foreground
    /// run lengths (the *decoded* numeric form, not pycocotools' compressed
    /// string — use [`Mask.from_coco`] for that).
    #[new]
    #[pyo3(signature = (size, counts))]
    fn new(size: [usize; 2], counts: Vec<u32>) -> PyResult<Self> {
        Ok(Mask {
            rle: mask::Rle::new(size[0], size[1], counts).map_err(PyValueError::new_err)?,
        })
    }

    /// Decode pycocotools' compressed-string RLE form (`str` or `bytes`).
    #[staticmethod]
    fn from_coco(size: [usize; 2], counts: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Mask {
            rle: rle_from_coco_counts(size[0], size[1], counts)?,
        })
    }

    #[getter]
    fn size(&self) -> (usize, usize) {
        (self.rle.h(), self.rle.w())
    }

    #[getter]
    fn counts(&self) -> Vec<u32> {
        self.rle.counts().to_vec()
    }

    /// Foreground pixel count.
    fn area(&self) -> u64 {
        self.rle.area()
    }

    /// pycocotools' compressed-string RLE form.
    fn to_coco(&self) -> String {
        self.rle.to_compressed()
    }

    fn __repr__(&self) -> String {
        format!(
            "Mask(size=({}, {}), area={})",
            self.rle.h(),
            self.rle.w(),
            self.rle.area()
        )
    }
}

/// Decode a pycocotools-style `counts` value (`str`, `bytes`, or an
/// already-decoded list of run lengths) into an [`mask::Rle`].
fn rle_from_coco_counts(h: usize, w: usize, counts: &Bound<'_, PyAny>) -> PyResult<mask::Rle> {
    if let Ok(bytes) = counts.cast::<PyBytes>() {
        let s = std::str::from_utf8(bytes.as_bytes())
            .map_err(|_| PyValueError::new_err("RLE counts bytes must be valid ASCII"))?;
        return mask::Rle::from_compressed(h, w, s).map_err(PyValueError::new_err);
    }
    if let Ok(s) = counts.extract::<String>() {
        return mask::Rle::from_compressed(h, w, &s).map_err(PyValueError::new_err);
    }
    let counts: Vec<u32> = counts.extract()?;
    mask::Rle::new(h, w, counts).map_err(PyValueError::new_err)
}

/// Accept a [`Mask`] instance or a pycocotools-style dict
/// (`{"size": (h, w), "counts": ...}`, `counts` compressed or already decoded).
fn extract_rle(obj: &Bound<'_, PyAny>) -> PyResult<mask::Rle> {
    if let Ok(m) = obj.extract::<PyRef<Mask>>() {
        return Ok(m.rle.clone());
    }
    let [h, w]: [usize; 2] = obj.get_item("size")?.extract()?;
    rle_from_coco_counts(h, w, &obj.get_item("counts")?)
}

/// Foreground pixel count of a mask (a [`Mask`] or a pycocotools-style dict).
#[pyfunction]
fn mask_area(mask: &Bound<'_, PyAny>) -> PyResult<u64> {
    Ok(extract_rle(mask)?.area())
}

/// Intersection-over-union of two masks (each a [`Mask`] or a
/// pycocotools-style dict), of the same `(h, w)`.
///
/// If `is_crowd` is set, `mask_b` is a crowd/ignore region: the score is
/// intersection over `mask_a`'s own area rather than the union, matching
/// pycocotools' `iscrowd` semantics (spelled `is_crowd` here for consistency
/// with the rest of this library's snake_case parameters).
#[pyfunction]
#[pyo3(signature = (mask_a, mask_b, is_crowd=false))]
fn mask_iou(mask_a: &Bound<'_, PyAny>, mask_b: &Bound<'_, PyAny>, is_crowd: bool) -> PyResult<f64> {
    mask::iou(&extract_rle(mask_a)?, &extract_rle(mask_b)?, is_crowd).map_err(PyValueError::new_err)
}

/// Pairwise IoU matrix between two sets of masks.
///
/// `is_crowd`, if given, must have one entry per mask in `masks_b`; a `true`
/// entry makes that column an IoA-against-`masks_a`-only crowd region,
/// matching pycocotools' `iou(dt, gt, iscrowd)`.
#[pyfunction]
#[pyo3(signature = (masks_a, masks_b, is_crowd=None))]
fn mask_iou_matrix(
    masks_a: Vec<Bound<'_, PyAny>>,
    masks_b: Vec<Bound<'_, PyAny>>,
    is_crowd: Option<Vec<bool>>,
) -> PyResult<Vec<Vec<f64>>> {
    let a: Vec<mask::Rle> = masks_a.iter().map(extract_rle).collect::<PyResult<_>>()?;
    let b: Vec<mask::Rle> = masks_b.iter().map(extract_rle).collect::<PyResult<_>>()?;
    mask::iou_matrix(&a, &b, is_crowd.as_deref()).map_err(PyValueError::new_err)
}

/// Merge a list of same-size masks into their union (`intersect=False`, the
/// default) or intersection (`intersect=True`), matching pycocotools'
/// `merge(rles, intersect)`. An empty list yields an empty `Mask((0, 0), [])`.
/// Unlike pycocotools (which silently returns an empty mask on a size
/// mismatch between inputs), a genuine mismatch raises `ValueError`.
#[pyfunction]
#[pyo3(signature = (masks, intersect=false))]
fn mask_merge(masks: Vec<Bound<'_, PyAny>>, intersect: bool) -> PyResult<Mask> {
    let rles: Vec<mask::Rle> = masks.iter().map(extract_rle).collect::<PyResult<_>>()?;
    Ok(Mask {
        rle: mask::merge(&rles, intersect).map_err(PyValueError::new_err)?,
    })
}

/// Bounding box of a mask's foreground pixels, or `(0, 0, 0, 0)` if it has
/// none.
///
/// `box_format` is `"xyxy"` (default, matching every other box primitive in
/// this library — `iou`, `match_boxes`, `compute_clear`, ...) or `"xywh"`
/// (pycocotools' own `toBbox` convention).
#[pyfunction]
#[pyo3(signature = (mask, box_format="xyxy"))]
fn mask_to_bbox(mask: &Bound<'_, PyAny>, box_format: &str) -> PyResult<(f64, f64, f64, f64)> {
    let (x1, y1, x2, y2) = extract_rle(mask)?.bbox();
    match box_format {
        "xyxy" => Ok((x1, y1, x2, y2)),
        "xywh" => Ok((x1, y1, x2 - x1, y2 - y1)),
        other => Err(PyValueError::new_err(format!(
            "unknown box_format {other:?}, expected \"xyxy\" or \"xywh\""
        ))),
    }
}

/// Decode a mask to a dense `(h, w)` nested list of `0`/`1` values.
///
/// Returns `u32`, not `u8`: PyO3 converts `Vec<u8>` to a Python `bytes`
/// object rather than `list[int]`, which isn't what a "nested list" caller
/// expects here.
#[pyfunction]
fn mask_decode(mask: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u32>>> {
    let rle = extract_rle(mask)?;
    let dense = rle.to_dense();
    Ok((0..rle.h())
        .map(|r| {
            (0..rle.w())
                .map(|c| u32::from(dense[c * rle.h() + r]))
                .collect()
        })
        .collect())
}

/// Encode a dense `(h, w)` `uint8` NumPy array (any nonzero value is
/// foreground) into a [`Mask`].
#[pyfunction]
fn mask_encode(bitmap: PyReadonlyArray2<u8>) -> PyResult<Mask> {
    let view = bitmap.as_array();
    let (h, w) = (view.shape()[0], view.shape()[1]);
    let mut bits = Vec::with_capacity(h * w);
    for c in 0..w {
        for r in 0..h {
            bits.push(view[[r, c]]);
        }
    }
    Ok(Mask {
        rle: mask::Rle::from_dense(h, w, &bits).map_err(PyValueError::new_err)?,
    })
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

impl From<assignment::MatchResult> for Matching {
    fn from(result: assignment::MatchResult) -> Self {
        Matching {
            matches: result.matches,
            scores: result.scores,
            unmatched_a: result.unmatched_a,
            unmatched_b: result.unmatched_b,
        }
    }
}

/// Parse a `method` argument shared by `match_boxes` and `match_masks`.
fn parse_method(method: &str) -> PyResult<Method> {
    match method {
        "hungarian" => Ok(Method::Hungarian),
        "greedy" => Ok(Method::Greedy),
        other => Err(PyValueError::new_err(format!(
            "unknown method {other:?}, expected \"hungarian\" or \"greedy\""
        ))),
    }
}

/// Match two sets of boxes.
///
/// `method` is either `"hungarian"` (optimal, maximises total IoU) or
/// `"greedy"` (assign highest-IoU pairs first). Only pairs with IoU at or above
/// `iou_threshold` are kept. `box_format` is `"xyxy"` (default) or `"xywh"`.
#[pyfunction]
#[pyo3(signature = (boxes_a, boxes_b, iou_threshold=0.5, method="hungarian", box_format="xyxy"))]
fn match_boxes(
    boxes_a: PyBoxes,
    boxes_b: PyBoxes,
    iou_threshold: f64,
    method: &str,
    box_format: &str,
) -> PyResult<Matching> {
    let method = parse_method(method)?;
    let format = BoxFormat::parse(box_format)?;
    let boxes_a = boxes_a.as_boxes(format)?;
    let boxes_b = boxes_b.as_boxes(format)?;

    let n_a = boxes_a.len();
    let n_b = boxes_b.len();
    let matrix = iou::iou_matrix(&boxes_a, &boxes_b);
    let result = assignment::match_boxes(&matrix, n_a, n_b, iou_threshold, method);

    Ok(result.into())
}

/// Match two sets of masks, mirroring [`match_boxes`] for segmentation masks.
///
/// `method` is either `"hungarian"` (optimal, maximises total IoU) or
/// `"greedy"` (assign highest-IoU pairs first). Only pairs with IoU at or
/// above `iou_threshold` are kept.
#[pyfunction]
#[pyo3(signature = (masks_a, masks_b, iou_threshold=0.5, method="hungarian"))]
fn match_masks(
    masks_a: Vec<Bound<'_, PyAny>>,
    masks_b: Vec<Bound<'_, PyAny>>,
    iou_threshold: f64,
    method: &str,
) -> PyResult<Matching> {
    let method = parse_method(method)?;
    let a: Vec<mask::Rle> = masks_a.iter().map(extract_rle).collect::<PyResult<_>>()?;
    let b: Vec<mask::Rle> = masks_b.iter().map(extract_rle).collect::<PyResult<_>>()?;

    let n_a = a.len();
    let n_b = b.len();
    let matrix = mask::iou_matrix(&a, &b, None).map_err(PyValueError::new_err)?;
    let result = assignment::match_boxes(&matrix, n_a, n_b, iou_threshold, method);

    Ok(result.into())
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
    /// Mostly tracked: gt trajectories matched in more than 80% of their frames.
    #[pyo3(get)]
    mt: usize,
    /// Partially tracked: gt trajectories matched in 20–80% of their frames.
    #[pyo3(get)]
    pt: usize,
    /// Mostly lost: gt trajectories matched in less than 20% of their frames.
    #[pyo3(get)]
    ml: usize,
    /// Fragmentations: times a gt trajectory resumes after an interruption.
    #[pyo3(get)]
    frag: usize,
    /// Multiple Object Detection Accuracy: `(TP - FP) / max(1, TP + FN)`.
    #[pyo3(get)]
    moda: f64,
    /// MOTA over overlap: `(MOTP_sum - FP - IDSW) / max(1, TP + FN)`.
    #[pyo3(get)]
    smota: f64,
    /// MOTA with a log-scaled ID-switch penalty.
    #[pyo3(get)]
    motal: f64,
    /// CLEAR recall: `TP / max(1, TP + FN)`.
    #[pyo3(get)]
    clr_re: f64,
    /// CLEAR precision: `TP / max(1, TP + FP)`.
    #[pyo3(get)]
    clr_pr: f64,
    /// Per gt trajectory, the fraction of its frames that were matched (id
    /// switches count as matched), sorted descending. The basis for MT/PT/ML.
    #[pyo3(get)]
    track_ratios: Vec<f64>,
}

#[pymethods]
impl ClearMetrics {
    fn __repr__(&self) -> String {
        format!(
            "ClearMetrics(mota={:.4}, motp={:.4}, tp={}, fp={}, fn={}, idsw={}, mt={}, pt={}, ml={}, frag={})",
            self.mota,
            self.motp,
            self.num_matches,
            self.num_false_positives,
            self.num_misses,
            self.num_switches,
            self.mt,
            self.pt,
            self.ml,
            self.frag,
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
            mt: m.mt,
            pt: m.pt,
            ml: m.ml,
            frag: m.frag,
            moda: m.moda,
            smota: m.smota,
            motal: m.motal,
            clr_re: m.clr_re,
            clr_pr: m.clr_pr,
            track_ratios: m.track_ratios,
        }
    }
}

/// Compute CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches) for a sequence.
///
/// Inputs are frame-aligned: `gt_ids[t]` / `gt_boxes[t]` describe ground-truth
/// objects in frame `t` (and likewise `pred_ids` / `pred_boxes` for the
/// tracker). `gt_ids` and `pred_ids` must have the same number of frames, and
/// within each frame the id and box lists must have equal length. Each
/// frame's boxes may be a sequence of 4-tuples or a `(N, 4)` float64 NumPy
/// array (zero-copy for a contiguous `xyxy` array); `box_format` is `"xyxy"`
/// (default) or `"xywh"`.
#[pyfunction]
#[pyo3(signature = (gt_ids, gt_boxes, pred_ids, pred_boxes, iou_threshold=0.5, box_format="xyxy"))]
fn compute_clear(
    gt_ids: Vec<Vec<i64>>,
    gt_boxes: Vec<PyBoxes>,
    pred_ids: Vec<Vec<i64>>,
    pred_boxes: Vec<PyBoxes>,
    iou_threshold: f64,
    box_format: &str,
) -> PyResult<ClearMetrics> {
    let format = BoxFormat::parse(box_format)?;
    let gt_boxes = to_xyxy_frames(&gt_boxes, format)?;
    let pred_boxes = to_xyxy_frames(&pred_boxes, format)?;
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
#[pyo3(signature = (gt_ids, gt_boxes, pred_ids, pred_boxes, iou_threshold=0.5, box_format="xyxy"))]
fn compute_identity(
    gt_ids: Vec<Vec<i64>>,
    gt_boxes: Vec<PyBoxes>,
    pred_ids: Vec<Vec<i64>>,
    pred_boxes: Vec<PyBoxes>,
    iou_threshold: f64,
    box_format: &str,
) -> PyResult<IdentityMetrics> {
    let format = BoxFormat::parse(box_format)?;
    let gt_boxes = to_xyxy_frames(&gt_boxes, format)?;
    let pred_boxes = to_xyxy_frames(&pred_boxes, format)?;
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

impl From<hota::HotaMetrics> for HotaMetrics {
    fn from(m: hota::HotaMetrics) -> Self {
        HotaMetrics {
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
        }
    }
}

/// Compute HOTA metrics (DetA, AssA, LocA, plus per-alpha curves) for a sequence.
///
/// Inputs are frame-aligned exactly like [`compute_clear`]. HOTA sweeps a set of
/// localization thresholds internally, so unlike the other metrics it takes no
/// single `iou_threshold`.
#[pyfunction]
#[pyo3(signature = (gt_ids, gt_boxes, pred_ids, pred_boxes, box_format="xyxy"))]
fn compute_hota(
    gt_ids: Vec<Vec<i64>>,
    gt_boxes: Vec<PyBoxes>,
    pred_ids: Vec<Vec<i64>>,
    pred_boxes: Vec<PyBoxes>,
    box_format: &str,
) -> PyResult<HotaMetrics> {
    let format = BoxFormat::parse(box_format)?;
    let gt_boxes = to_xyxy_frames(&gt_boxes, format)?;
    let pred_boxes = to_xyxy_frames(&pred_boxes, format)?;
    let frames = build_frames(&gt_ids, &gt_boxes, &pred_ids, &pred_boxes)?;
    Ok(hota::compute_hota(&frames).into())
}

/// Compute HOTA metrics from precomputed per-frame similarity matrices instead
/// of boxes. Same similarity convention as [`compute_clear_from_similarity`].
#[pyfunction]
#[pyo3(signature = (gt_ids, pred_ids, similarity))]
fn compute_hota_from_similarity(
    gt_ids: Vec<Vec<i64>>,
    pred_ids: Vec<Vec<i64>>,
    similarity: Vec<Vec<Vec<f64>>>,
) -> PyResult<HotaMetrics> {
    let frames = build_sim_frames(&gt_ids, &pred_ids, &similarity)?;
    Ok(hota::compute_hota_from_similarity(&frames).into())
}

/// The result of [`evaluate`]: CLEAR, Identity, and HOTA computed together
/// from one shared similarity matrix.
#[pyclass(frozen)]
struct EvaluationResult {
    /// CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches).
    #[pyo3(get)]
    clear: Py<ClearMetrics>,
    /// Identity metrics (IDF1, IDP, IDR).
    #[pyo3(get)]
    identity: Py<IdentityMetrics>,
    /// HOTA metrics (DetA, AssA, LocA, plus per-alpha curves).
    #[pyo3(get)]
    hota: Py<HotaMetrics>,
}

#[pymethods]
impl EvaluationResult {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "EvaluationResult(clear={}, identity={}, hota={})",
            self.clear.borrow(py).__repr__(),
            self.identity.borrow(py).__repr__(),
            self.hota.borrow(py).__repr__(),
        )
    }
}

/// Compute CLEAR, Identity, and HOTA together for a sequence.
///
/// Inputs are frame-aligned exactly like [`compute_clear`]. Builds the
/// gt/pred similarity matrix once and reuses it for all three metrics,
/// instead of recomputing it once per metric (what calling `compute_clear`,
/// `compute_identity`, and `compute_hota` separately would do).
#[pyfunction]
#[pyo3(signature = (gt_ids, gt_boxes, pred_ids, pred_boxes, iou_threshold=0.5, box_format="xyxy"))]
fn evaluate(
    py: Python<'_>,
    gt_ids: Vec<Vec<i64>>,
    gt_boxes: Vec<PyBoxes>,
    pred_ids: Vec<Vec<i64>>,
    pred_boxes: Vec<PyBoxes>,
    iou_threshold: f64,
    box_format: &str,
) -> PyResult<EvaluationResult> {
    let format = BoxFormat::parse(box_format)?;
    let gt_boxes = to_xyxy_frames(&gt_boxes, format)?;
    let pred_boxes = to_xyxy_frames(&pred_boxes, format)?;
    let frames = build_frames(&gt_ids, &gt_boxes, &pred_ids, &pred_boxes)?;
    let similarity: Vec<Vec<Vec<f64>>> = frames
        .iter()
        .map(|f| iou::iou_matrix(f.gt_boxes, f.pred_boxes))
        .collect();
    let sim_frames: Vec<clear::SimFrame> = frames
        .iter()
        .zip(&similarity)
        .map(|(f, s)| clear::SimFrame {
            gt_ids: f.gt_ids,
            pred_ids: f.pred_ids,
            similarity: s,
        })
        .collect();
    Ok(EvaluationResult {
        clear: Py::new(
            py,
            ClearMetrics::from(clear::compute_clear_from_similarity(
                &sim_frames,
                iou_threshold,
            )),
        )?,
        identity: Py::new(
            py,
            IdentityMetrics::from(identity::compute_identity_from_similarity(
                &sim_frames,
                iou_threshold,
            )),
        )?,
        hota: Py::new(
            py,
            HotaMetrics::from(hota::compute_hota_from_similarity(&sim_frames)),
        )?,
    })
}

/// CLEAR and Identity read from a streaming [`Accumulator`] via
/// [`Accumulator::compute`].
#[pyclass(frozen)]
struct AccumulatorResult {
    /// CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches).
    #[pyo3(get)]
    clear: Py<ClearMetrics>,
    /// Identity metrics (IDF1, IDP, IDR).
    #[pyo3(get)]
    identity: Py<IdentityMetrics>,
}

#[pymethods]
impl AccumulatorResult {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "AccumulatorResult(clear={}, identity={})",
            self.clear.borrow(py).__repr__(),
            self.identity.borrow(py).__repr__(),
        )
    }
}

/// A streaming CLEAR + Identity accumulator.
///
/// Fold in one frame at a time with [`Accumulator::update`] (boxes) or
/// [`Accumulator::update_from_similarity`] (precomputed scores), then read the
/// metrics with [`Accumulator::compute`] — the online/large-sequence shape,
/// where the whole sequence is never held in memory. HOTA is not offered here:
/// its alpha sweep is inherently a whole-sequence computation, so use
/// [`evaluate`] or [`compute_hota`] for it.
#[pyclass]
struct Accumulator {
    inner: accumulator::Accumulator,
    format: BoxFormat,
}

#[pymethods]
impl Accumulator {
    /// `iou_threshold` is the match cutoff shared by both metrics; `box_format`
    /// is `"xyxy"` (default) or `"xywh"`, applied to every `update` call.
    #[new]
    #[pyo3(signature = (iou_threshold=0.5, box_format="xyxy"))]
    fn new(iou_threshold: f64, box_format: &str) -> PyResult<Self> {
        Ok(Accumulator {
            inner: accumulator::Accumulator::new(iou_threshold),
            format: BoxFormat::parse(box_format)?,
        })
    }

    /// Number of frames folded in so far.
    #[getter]
    fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    /// Fold one frame in. `gt_ids`/`gt_boxes` (and `pred_ids`/`pred_boxes`)
    /// must have equal length; each boxes argument is a sequence of 4-tuples or
    /// a contiguous `(N, 4)` float64 NumPy array, in this accumulator's
    /// `box_format`.
    #[pyo3(signature = (gt_ids, gt_boxes, pred_ids, pred_boxes))]
    fn update(
        &mut self,
        gt_ids: Vec<i64>,
        gt_boxes: PyBoxes,
        pred_ids: Vec<i64>,
        pred_boxes: PyBoxes,
    ) -> PyResult<()> {
        let gt_boxes = gt_boxes.as_boxes(self.format)?;
        let pred_boxes = pred_boxes.as_boxes(self.format)?;
        if gt_ids.len() != gt_boxes.len() {
            return Err(PyValueError::new_err(format!(
                "gt_ids and gt_boxes length mismatch ({} vs {})",
                gt_ids.len(),
                gt_boxes.len()
            )));
        }
        if pred_ids.len() != pred_boxes.len() {
            return Err(PyValueError::new_err(format!(
                "pred_ids and pred_boxes length mismatch ({} vs {})",
                pred_ids.len(),
                pred_boxes.len()
            )));
        }
        self.inner
            .update(&gt_ids, gt_boxes.as_ref(), &pred_ids, pred_boxes.as_ref());
        Ok(())
    }

    /// Fold one frame in from a precomputed similarity matrix instead of boxes.
    /// `similarity[i][j]` scores `gt_ids[i]` against `pred_ids[j]`; it must be
    /// `len(gt_ids)` rows by `len(pred_ids)` columns.
    #[pyo3(signature = (gt_ids, pred_ids, similarity))]
    fn update_from_similarity(
        &mut self,
        gt_ids: Vec<i64>,
        pred_ids: Vec<i64>,
        similarity: Vec<Vec<f64>>,
    ) -> PyResult<()> {
        if similarity.len() != gt_ids.len() {
            return Err(PyValueError::new_err(format!(
                "similarity has {} rows, expected {} (len(gt_ids))",
                similarity.len(),
                gt_ids.len()
            )));
        }
        for (r, row) in similarity.iter().enumerate() {
            if row.len() != pred_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "similarity row {r} has {} columns, expected {} (len(pred_ids))",
                    row.len(),
                    pred_ids.len()
                )));
            }
        }
        self.inner
            .update_from_similarity(&gt_ids, &pred_ids, &similarity);
        Ok(())
    }

    /// Finalize CLEAR and Identity from everything folded in so far. May be
    /// called at any point and does not consume the accumulator.
    fn compute(&self, py: Python<'_>) -> PyResult<AccumulatorResult> {
        let (clear, identity) = self.inner.compute();
        Ok(AccumulatorResult {
            clear: Py::new(py, ClearMetrics::from(clear))?,
            identity: Py::new(py, IdentityMetrics::from(identity))?,
        })
    }

    fn __repr__(&self) -> String {
        format!("Accumulator(num_frames={})", self.inner.num_frames())
    }
}

/// The `motrics._motrics` extension module.
#[pymodule]
fn _motrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_debug_build, m)?)?;
    m.add_function(wrap_pyfunction!(iou_py, m)?)?;
    m.add_function(wrap_pyfunction!(iou_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(iou_3d, m)?)?;
    m.add_function(wrap_pyfunction!(iou_3d_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(match_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(match_masks, m)?)?;
    m.add_function(wrap_pyfunction!(mask_area, m)?)?;
    m.add_function(wrap_pyfunction!(mask_iou, m)?)?;
    m.add_function(wrap_pyfunction!(mask_iou_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(mask_decode, m)?)?;
    m.add_function(wrap_pyfunction!(mask_encode, m)?)?;
    m.add_function(wrap_pyfunction!(mask_merge, m)?)?;
    m.add_function(wrap_pyfunction!(mask_to_bbox, m)?)?;
    m.add_class::<Mask>()?;
    m.add_function(wrap_pyfunction!(compute_clear, m)?)?;
    m.add_function(wrap_pyfunction!(compute_clear_from_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_identity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_identity_from_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hota, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hota_from_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_class::<Accumulator>()?;
    m.add_class::<AccumulatorResult>()?;
    m.add_class::<Matching>()?;
    m.add_class::<ClearMetrics>()?;
    m.add_class::<IdentityMetrics>()?;
    m.add_class::<HotaMetrics>()?;
    m.add_class::<EvaluationResult>()?;
    Ok(())
}
