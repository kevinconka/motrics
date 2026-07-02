//! `motrics` — an extremely fast MOT and HOTA metrics library.
//!
//! This crate is compiled into the `motrics._motrics` extension module. The
//! public, ergonomic API lives in the `motrics` Python package (see
//! `python/motrics/`), which re-exports the pieces below.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod assignment;
mod iou;

use assignment::Method;
use iou::Bbox;

/// Return the version of the compiled Rust core.
///
/// Wired through Cargo at build time so the native and Python versions can
/// never drift.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
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

/// The `motrics._motrics` extension module.
#[pymodule]
fn _motrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(iou_py, m)?)?;
    m.add_function(wrap_pyfunction!(iou_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(match_boxes, m)?)?;
    m.add_class::<Matching>()?;
    Ok(())
}
