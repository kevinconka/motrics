//! Oriented 3D bounding-box intersection-over-union (KITTI / AB3DMOT
//! convention).
//!
//! A 3D box is seven numbers, `[x, y, z, l, w, h, yaw]`:
//! - `(x, y, z)` is the box **centre**;
//! - `l`, `w`, `h` are the full extents along the box's own length, width and
//!   height axes;
//! - `yaw` (radians) is the heading, a rotation about the vertical `y` axis in
//!   the `x`–`z` ground plane. At `yaw = 0`, `l` runs along `x` and `w` along
//!   `z` (KITTI's `roty` convention); `h` always runs along `y`.
//!
//! IoU is volumetric: the bird's-eye-view (BEV) footprints — rotated
//! rectangles in the `x`–`z` plane — are intersected exactly with a
//! Sutherland–Hodgman convex clip, and that area is multiplied by the overlap
//! of the two `[y - h/2, y + h/2]` height intervals to give the intersection
//! volume. This is the 3D IoU used by AB3DMOT's KITTI-3D MOT evaluation.

/// A 3D box: `[x, y, z, l, w, h, yaw]` (centre, extents, heading in radians).
pub type Box3d = [f64; 7];

const X: usize = 0;
const Y: usize = 1;
const Z: usize = 2;
const L: usize = 3;
const W: usize = 4;
const H: usize = 5;
const YAW: usize = 6;

/// A point in the `x`–`z` ground plane.
type Pt = (f64, f64);

/// The four BEV corners of a box in the `x`–`z` plane, in order.
fn bev_corners(b: &Box3d) -> [Pt; 4] {
    let (c, s) = (b[YAW].cos(), b[YAW].sin());
    let (hl, hw) = (b[L] / 2.0, b[W] / 2.0);
    // Local corners as (along-length, along-width) offsets, walked in order.
    let local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)];
    local.map(|(dx, dz)| (b[X] + dx * c + dz * s, b[Z] - dx * s + dz * c))
}

/// Signed area of a polygon (positive when counter-clockwise) via the
/// shoelace formula.
fn signed_area(poly: &[Pt]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..n {
        let (x1, z1) = poly[i];
        let (x2, z2) = poly[(i + 1) % n];
        a += x1 * z2 - x2 * z1;
    }
    a / 2.0
}

/// Whether `p` is on the inside (left) of the directed edge `a -> b`, for a
/// counter-clockwise clip polygon.
fn inside(p: Pt, a: Pt, b: Pt) -> bool {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0) >= 0.0
}

/// Intersection of the segment `s -> e` with the infinite line through
/// `a -> b`. The caller guarantees the two are not parallel (they straddle a
/// clip edge).
fn line_intersect(s: Pt, e: Pt, a: Pt, b: Pt) -> Pt {
    let (a1, a2) = (b.1 - a.1, a.0 - b.0);
    let c1 = a1 * a.0 + a2 * a.1;
    let (b1, b2) = (e.1 - s.1, s.0 - e.0);
    let c2 = b1 * s.0 + b2 * s.1;
    let denom = a1 * b2 - b1 * a2;
    ((b2 * c1 - a2 * c2) / denom, (a1 * c2 - b1 * c1) / denom)
}

/// Clip the convex polygon `subject` against the convex polygon `clip`
/// (Sutherland–Hodgman). Both must be counter-clockwise.
fn clip_polygon(subject: &[Pt], clip: &[Pt]) -> Vec<Pt> {
    let mut output = subject.to_vec();
    for i in 0..clip.len() {
        if output.is_empty() {
            break;
        }
        let a = clip[i];
        let b = clip[(i + 1) % clip.len()];
        let input = std::mem::take(&mut output);
        for j in 0..input.len() {
            let cur = input[j];
            let prev = input[(j + input.len() - 1) % input.len()];
            let cur_in = inside(cur, a, b);
            let prev_in = inside(prev, a, b);
            if cur_in {
                if !prev_in {
                    output.push(line_intersect(prev, cur, a, b));
                }
                output.push(cur);
            } else if prev_in {
                output.push(line_intersect(prev, cur, a, b));
            }
        }
    }
    output
}

/// Area of the intersection of two BEV rectangles (given as corner lists).
fn bev_intersection_area(a: &[Pt], b: &[Pt]) -> f64 {
    // Normalise both to counter-clockwise so the inside test is consistent.
    let to_ccw = |mut p: Vec<Pt>| {
        if signed_area(&p) < 0.0 {
            p.reverse();
        }
        p
    };
    let clipped = clip_polygon(&to_ccw(a.to_vec()), &to_ccw(b.to_vec()));
    signed_area(&clipped).abs()
}

/// Volumetric IoU of two oriented 3D boxes.
///
/// Returns a value in `[0, 1]`, or `0.0` when the union volume is zero (e.g.
/// a degenerate box).
pub fn iou_3d(a: &Box3d, b: &Box3d) -> f64 {
    let bev = bev_intersection_area(&bev_corners(a), &bev_corners(b));
    let h_overlap = ((a[Y] + a[H] / 2.0).min(b[Y] + b[H] / 2.0)
        - (a[Y] - a[H] / 2.0).max(b[Y] - b[H] / 2.0))
    .max(0.0);
    let inter = bev * h_overlap;
    let union = a[L] * a[W] * a[H] + b[L] * b[W] * b[H] - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Pairwise 3D IoU matrix, `boxes_a.len()` rows by `boxes_b.len()` columns.
pub fn iou_3d_matrix(boxes_a: &[Box3d], boxes_b: &[Box3d]) -> Vec<Vec<f64>> {
    boxes_a
        .iter()
        .map(|a| boxes_b.iter().map(|b| iou_3d(a, b)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn identical_boxes_have_iou_one() {
        let b: Box3d = [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3];
        approx(iou_3d(&b, &b), 1.0);
    }

    #[test]
    fn disjoint_boxes_have_iou_zero() {
        let a: Box3d = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0];
        let b: Box3d = [100.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0];
        approx(iou_3d(&a, &b), 0.0);
    }

    #[test]
    fn axis_aligned_half_overlap_in_x() {
        // Two 2x2x2 cubes offset by 1 along x: BEV overlap 1x2, full height.
        // inter = 1*2*2 = 4, union = 8 + 8 - 4 = 12 -> 1/3.
        let a: Box3d = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0];
        let b: Box3d = [1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0];
        approx(iou_3d(&a, &b), 1.0 / 3.0);
    }

    #[test]
    fn no_height_overlap_is_zero() {
        // Same footprint, stacked apart vertically.
        let a: Box3d = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0];
        let b: Box3d = [0.0, 5.0, 0.0, 2.0, 2.0, 2.0, 0.0];
        approx(iou_3d(&a, &b), 0.0);
    }

    #[test]
    fn yaw_symmetry_of_square_footprint() {
        // A square-footprint box rotated 90 degrees is identical to itself.
        let a: Box3d = [0.0, 0.0, 0.0, 2.0, 2.0, 3.0, 0.0];
        let b: Box3d = [0.0, 0.0, 0.0, 2.0, 2.0, 3.0, PI / 2.0];
        approx(iou_3d(&a, &b), 1.0);
    }

    #[test]
    fn matrix_shape_and_values() {
        let a: [Box3d; 2] = [
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            [100.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        ];
        let b: [Box3d; 1] = [[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]];
        let m = iou_3d_matrix(&a, &b);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 1);
        approx(m[0][0], 1.0);
        approx(m[1][0], 0.0);
    }
}
