//! Box format conversion and zero-copy NumPy box input.
//!
//! Boxes can be passed either as a Python sequence of `(a, b, c, d)`-style
//! tuples (copied into a `Vec`) or as a `(N, 4)` float64 NumPy array — a
//! contiguous, standard-layout array is reinterpreted in place as `&[Bbox]`
//! with no per-element conversion.

use std::borrow::Cow;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::iou::Bbox;
use crate::iou3d::Box3d;

/// A box coordinate convention: `xyxy` (`x1, y1, x2, y2`) or `xywh`
/// (`x, y, width, height`). Every core function accepts either, converting
/// `xywh` to `xyxy` once at the boundary.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BoxFormat {
    Xyxy,
    Xywh,
}

impl BoxFormat {
    pub fn parse(s: &str) -> PyResult<Self> {
        match s {
            "xyxy" => Ok(BoxFormat::Xyxy),
            "xywh" => Ok(BoxFormat::Xywh),
            other => Err(PyValueError::new_err(format!(
                "unknown box_format {other:?}, expected \"xyxy\" or \"xywh\""
            ))),
        }
    }
}

#[inline]
fn xywh_to_xyxy(b: Bbox) -> Bbox {
    [b[0], b[1], b[0] + b[2], b[1] + b[3]]
}

/// Convert a single box to `xyxy`, a no-op if it already is one.
#[inline]
pub fn to_xyxy(b: Bbox, format: BoxFormat) -> Bbox {
    match format {
        BoxFormat::Xyxy => b,
        BoxFormat::Xywh => xywh_to_xyxy(b),
    }
}

/// Boxes for one frame, accepted as either a Python sequence of 4-tuples or a
/// `(N, 4)` float64 NumPy array. `Array` is tried first so a NumPy array
/// always takes the zero-copy path rather than falling through to a
/// slower element-by-element sequence conversion.
#[derive(FromPyObject)]
pub enum PyBoxes<'py> {
    Array(PyReadonlyArray2<'py, f64>),
    List(Vec<Bbox>),
}

impl PyBoxes<'_> {
    /// Borrow (or, if a format conversion or a non-contiguous array layout
    /// requires it, copy) the boxes as `xyxy`.
    pub fn as_boxes(&self, format: BoxFormat) -> PyResult<Cow<'_, [Bbox]>> {
        match self {
            PyBoxes::List(v) => match format {
                BoxFormat::Xyxy => Ok(Cow::Borrowed(v)),
                BoxFormat::Xywh => Ok(Cow::Owned(v.iter().map(|&b| to_xyxy(b, format)).collect())),
            },
            PyBoxes::Array(arr) => {
                let view = arr.as_array();
                if view.ncols() != 4 {
                    return Err(PyValueError::new_err(format!(
                        "boxes array must have shape (N, 4), got {:?}",
                        view.shape()
                    )));
                }
                // `to_slice` (not `as_slice`) reuses the view's own borrow
                // lifetime instead of shortening it to this local variable,
                // so the zero-copy `Cow::Borrowed` below actually compiles.
                match view.to_slice() {
                    Some(flat) => {
                        let boxes: &[Bbox] = bytemuck::cast_slice(flat);
                        match format {
                            BoxFormat::Xyxy => Ok(Cow::Borrowed(boxes)),
                            BoxFormat::Xywh => Ok(Cow::Owned(
                                boxes.iter().map(|&b| to_xyxy(b, format)).collect(),
                            )),
                        }
                    }
                    // Non-contiguous (e.g. a transposed or strided view): copy row by row.
                    None => Ok(Cow::Owned(
                        view.rows()
                            .into_iter()
                            .map(|r| to_xyxy([r[0], r[1], r[2], r[3]], format))
                            .collect(),
                    )),
                }
            }
        }
    }
}

/// Oriented 3D boxes for one set, accepted as either a Python sequence of
/// 7-tuples (`[x, y, z, l, w, h, yaw]`) or a `(N, 7)` float64 NumPy array
/// (zero-copy for a contiguous array). `Array` is tried first so a NumPy array
/// always takes the zero-copy path.
#[derive(FromPyObject)]
pub enum PyBoxes3d<'py> {
    Array(PyReadonlyArray2<'py, f64>),
    List(Vec<Box3d>),
}

impl PyBoxes3d<'_> {
    /// Borrow (or, for a non-contiguous array layout, copy) the boxes.
    pub fn as_boxes(&self) -> PyResult<Cow<'_, [Box3d]>> {
        match self {
            PyBoxes3d::List(v) => Ok(Cow::Borrowed(v)),
            PyBoxes3d::Array(arr) => {
                let view = arr.as_array();
                if view.ncols() != 7 {
                    return Err(PyValueError::new_err(format!(
                        "3D boxes array must have shape (N, 7), got {:?}",
                        view.shape()
                    )));
                }
                match view.to_slice() {
                    Some(flat) => Ok(Cow::Borrowed(bytemuck::cast_slice(flat))),
                    // Non-contiguous (e.g. a strided view): copy row by row.
                    None => Ok(Cow::Owned(
                        view.rows()
                            .into_iter()
                            .map(|r| [r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
                            .collect(),
                    )),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xywh_converts_to_xyxy() {
        assert_eq!(xywh_to_xyxy([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 4.0, 6.0]);
    }
}
