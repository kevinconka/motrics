//! Bounding-box intersection-over-union (IoU) primitives.
//!
//! Boxes use the `xyxy` convention: `[x1, y1, x2, y2]` with `x2 >= x1` and
//! `y2 >= y1`. Degenerate boxes (zero or negative area) are handled gracefully
//! and simply contribute zero area.

/// A bounding box in `xyxy` format: `[x1, y1, x2, y2]`.
pub type Bbox = [f64; 4];

/// Area of a single `xyxy` box, clamped to be non-negative.
#[inline]
fn area(b: &Bbox) -> f64 {
    let w = (b[2] - b[0]).max(0.0);
    let h = (b[3] - b[1]).max(0.0);
    w * h
}

/// Intersection-over-union of two `xyxy` boxes.
///
/// Returns a value in `[0, 1]`. Returns `0.0` when the union area is zero
/// (e.g. both boxes are degenerate).
pub fn iou(a: &Bbox, b: &Bbox) -> f64 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);

    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;

    let union = area(a) + area(b) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Pairwise IoU matrix between two sets of boxes.
///
/// The result is `boxes_a.len()` rows by `boxes_b.len()` columns, where
/// `result[i][j] == iou(&boxes_a[i], &boxes_b[j])`.
pub fn iou_matrix(boxes_a: &[Bbox], boxes_b: &[Bbox]) -> Vec<Vec<f64>> {
    boxes_a
        .iter()
        .map(|a| boxes_b.iter().map(|b| iou(a, b)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn identical_boxes_have_iou_one() {
        approx(iou(&[0.0, 0.0, 10.0, 10.0], &[0.0, 0.0, 10.0, 10.0]), 1.0);
    }

    #[test]
    fn disjoint_boxes_have_iou_zero() {
        approx(iou(&[0.0, 0.0, 10.0, 10.0], &[20.0, 20.0, 30.0, 30.0]), 0.0);
    }

    #[test]
    fn half_overlap() {
        // Two 10x10 boxes overlapping in a 5x10 strip.
        // inter = 50, union = 100 + 100 - 50 = 150 -> 1/3.
        approx(
            iou(&[0.0, 0.0, 10.0, 10.0], &[5.0, 0.0, 15.0, 10.0]),
            1.0 / 3.0,
        );
    }

    #[test]
    fn degenerate_box_is_zero() {
        approx(iou(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 10.0, 10.0]), 0.0);
    }

    #[test]
    fn matrix_shape_and_values() {
        let a = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]];
        let b = [[0.0, 0.0, 10.0, 10.0]];
        let m = iou_matrix(&a, &b);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 1);
        approx(m[0][0], 1.0);
        approx(m[1][0], 0.0);
    }
}
