//! COCO run-length-encoded (RLE) binary mask primitives.
//!
//! RLE stores a binary mask as alternating run lengths of background/
//! foreground pixels, walked in column-major order (down each column, then
//! on to the next) — the convention pycocotools and every TrackEval mask
//! dataset (KITTI-MOTS, BDD-MOTS, DAVIS) use. The *compressed* string form
//! further packs each run length as a delta-from-two-runs-back, variable-length
//! (LEB128-like) integer, matching pycocotools' `maskApi.c` byte-for-byte —
//! this module's codec was validated against a real `pycocotools` build
//! before being ported here (see `tests/test_mask.py`).
//!
//! Area, union and intersection are computed by walking both run-length
//! sequences in lockstep (`rle_area`/`iou`), never materialising a dense
//! pixel array — the whole point of RLE. `decode`/`Rle::from_dense` exist for
//! interop with dense NumPy masks, not for the similarity computation itself.

/// A run-length-encoded binary mask: `counts` are alternating background/
/// foreground run lengths (always starting with a background run, which may
/// be `0`), covering `h * w` pixels in column-major order.
///
/// Fields are private and every public constructor validates that `counts`
/// covers exactly `h * w` pixels: [`iou`]'s lockstep walk assumes this and
/// never terminates otherwise (see [`Rle::new`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rle {
    h: usize,
    w: usize,
    counts: Vec<u32>,
}

impl Rle {
    /// Build an [`Rle`] from already-decoded run lengths, checking that they
    /// cover exactly `h * w` pixels.
    pub fn new(h: usize, w: usize, counts: Vec<u32>) -> Result<Rle, String> {
        let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
        let expected = (h as u64) * (w as u64);
        if total != expected {
            return Err(format!(
                "RLE counts cover {total} pixels, expected {expected} ({h} * {w})"
            ));
        }
        Ok(Rle { h, w, counts })
    }

    pub fn h(&self) -> usize {
        self.h
    }

    pub fn w(&self) -> usize {
        self.w
    }

    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// Total foreground pixel count: the sum of the odd-indexed (1st, 3rd, ...)
    /// runs, since runs always alternate starting with background.
    pub fn area(&self) -> u64 {
        self.counts
            .iter()
            .skip(1)
            .step_by(2)
            .map(|&c| c as u64)
            .sum()
    }

    /// Bounding box of the foreground pixels as `(x1, y1, x2, y2)` — this
    /// crate's `xyxy` convention (`x2`/`y2` exclusive) — or `(0, 0, 0, 0)`
    /// for an empty mask. Computed directly from the run lengths in
    /// `O(len(counts))`, without decoding to a dense mask: each foreground
    /// run's column-major span converts to a `(row, col)` range in one step
    /// (`divmod` by `h`), since a run confined to one column only touches
    /// that column's rows, while a run spanning multiple columns always
    /// touches every row (column-major order guarantees the run covers each
    /// of those columns fully except possibly the first and last).
    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        let h = self.h as u64;
        let mut bounds: Option<(u64, u64, u64, u64)> = None; // (min_row, max_row, min_col, max_col)
        let mut offset: u64 = 0;
        for (i, &run) in self.counts.iter().enumerate() {
            if i % 2 == 1 && run > 0 {
                let start = offset;
                let end = offset + u64::from(run) - 1;
                let (start_col, start_row) = (start / h, start % h);
                let (end_col, end_row) = (end / h, end % h);
                let (run_min_row, run_max_row) = if start_col == end_col {
                    (start_row, end_row)
                } else {
                    (0, h - 1)
                };
                bounds = Some(match bounds {
                    None => (run_min_row, run_max_row, start_col, end_col),
                    Some((min_row, max_row, min_col, max_col)) => (
                        min_row.min(run_min_row),
                        max_row.max(run_max_row),
                        min_col.min(start_col),
                        max_col.max(end_col),
                    ),
                });
            }
            offset += u64::from(run);
        }
        match bounds {
            None => (0.0, 0.0, 0.0, 0.0),
            Some((min_row, max_row, min_col, max_col)) => (
                min_col as f64,
                min_row as f64,
                (max_col + 1) as f64,
                (max_row + 1) as f64,
            ),
        }
    }

    /// Decode to a dense column-major `0`/`1` byte mask of length `h * w`.
    pub fn to_dense(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.h * self.w);
        let mut value = 0u8;
        for &run in &self.counts {
            out.resize(out.len() + run as usize, value);
            value ^= 1;
        }
        out
    }

    /// Build an [`Rle`] from a dense column-major `0`/`1` (or any-nonzero-is-
    /// foreground) byte mask of length `h * w`.
    pub fn from_dense(h: usize, w: usize, bits: &[u8]) -> Rle {
        let mut counts = Vec::new();
        let mut value = 0u8;
        let mut run = 0u32;
        for &b in bits {
            let v = u8::from(b != 0);
            if v == value {
                run += 1;
            } else {
                counts.push(run);
                value = v;
                run = 1;
            }
        }
        counts.push(run);
        Rle { h, w, counts }
    }

    /// Decode pycocotools' compressed-string RLE form: each run length is a
    /// delta from the run two positions back (0 for the first two runs),
    /// packed 5 bits at a time with a continuation bit, ASCII-shifted by 48.
    ///
    /// Rejects anything a valid pycocotools encoder could never produce
    /// (a byte outside the packed-value alphabet, a group long enough to
    /// overflow the `i64` accumulator's shift, or a decoded run length
    /// outside `u32`) with a clean error rather than panicking or silently
    /// truncating — `s` is untrusted external input (a Python `str`/`bytes`).
    pub fn from_compressed(h: usize, w: usize, s: &str) -> Result<Rle, String> {
        // 13 groups of 5 bits comfortably covers any real u32 delta (up to
        // ~8 groups) while keeping every shift amount (5 * k) well under
        // i64's 64-bit width.
        const MAX_GROUPS_PER_RUN: u32 = 13;

        let bytes = s.as_bytes();
        let mut counts: Vec<u32> = Vec::new();
        let mut p = 0usize;
        while p < bytes.len() {
            let mut x: i64 = 0;
            let mut k: u32 = 0;
            loop {
                if p >= bytes.len() {
                    return Err("truncated RLE counts string".to_string());
                }
                if k >= MAX_GROUPS_PER_RUN {
                    return Err("RLE counts string has an oversized run-length group".to_string());
                }
                let byte = bytes[p];
                if !(48..=111).contains(&byte) {
                    return Err(format!("invalid byte {byte:#04x} in RLE counts string"));
                }
                let c = i64::from(byte) - 48;
                x |= (c & 0x1f) << (5 * k);
                let more = c & 0x20 != 0;
                p += 1;
                k += 1;
                if !more {
                    if c & 0x10 != 0 {
                        x |= -1i64 << (5 * k);
                    }
                    break;
                }
            }
            if counts.len() > 2 {
                x += i64::from(counts[counts.len() - 2]);
            }
            if !(0..=i64::from(u32::MAX)).contains(&x) {
                return Err(format!(
                    "RLE counts string decodes to an out-of-range run length ({x})"
                ));
            }
            counts.push(x as u32);
        }
        Rle::new(h, w, counts)
    }

    /// Encode to pycocotools' compressed-string RLE form (the inverse of
    /// [`Rle::from_compressed`]).
    pub fn to_compressed(&self) -> String {
        let mut out = Vec::new();
        for (i, &count) in self.counts.iter().enumerate() {
            let mut x: i64 = i64::from(count);
            if i > 2 {
                x -= i64::from(self.counts[i - 2]);
            }
            loop {
                let mut c = x & 0x1f;
                x >>= 5;
                let more = if c & 0x10 != 0 { x != -1 } else { x != 0 };
                if more {
                    c |= 0x20;
                }
                out.push((c + 48) as u8);
                if !more {
                    break;
                }
            }
        }
        // SAFETY: every byte pushed is in 48..=97, valid ASCII.
        String::from_utf8(out).expect("RLE compressed string is always ASCII")
    }
}

/// Sum the intersection and union pixel counts of two same-size masks by
/// walking both run-length sequences in lockstep — `O(len(a) + len(b))`,
/// no dense decode.
fn intersection_and_union(a: &Rle, b: &Rle) -> (u64, u64) {
    let (mut ca, mut cb) = (a.counts[0], b.counts[0]);
    let (mut ka, mut kb) = (1usize, 1usize);
    let (mut va, mut vb) = (false, false);
    let (mut inter, mut union) = (0u64, 0u64);
    loop {
        let c = ca.min(cb);
        if va || vb {
            union += u64::from(c);
            if va && vb {
                inter += u64::from(c);
            }
        }
        ca -= c;
        if ca == 0 && ka < a.counts.len() {
            ca = a.counts[ka];
            ka += 1;
            va = !va;
        }
        cb -= c;
        if cb == 0 && kb < b.counts.len() {
            cb = b.counts[kb];
            kb += 1;
            vb = !vb;
        }
        if ca == 0 && cb == 0 {
            break;
        }
    }
    (inter, union)
}

/// Intersection-over-union of two RLE masks of the same `(h, w)`.
///
/// If `is_crowd` is set, `b` is treated as a crowd/ignore region: the score
/// is intersection over `a`'s own area instead of the union (so `a` scores
/// 1.0 whenever it lies entirely inside `b`), matching pycocotools' `iscrowd`
/// semantics and TrackEval's mask-dataset ignore-region handling.
pub fn iou(a: &Rle, b: &Rle, is_crowd: bool) -> Result<f64, String> {
    if a.h != b.h || a.w != b.w {
        return Err(format!(
            "mask size mismatch: {:?} vs {:?}",
            (a.h, a.w),
            (b.h, b.w)
        ));
    }
    if a.counts.is_empty() || b.counts.is_empty() {
        return Ok(0.0);
    }
    let (inter, union) = intersection_and_union(a, b);
    if inter == 0 {
        return Ok(0.0);
    }
    let denom = if is_crowd { a.area() } else { union };
    if denom == 0 {
        Ok(0.0)
    } else {
        Ok(inter as f64 / denom as f64)
    }
}

/// Pairwise IoU matrix between two sets of RLE masks.
///
/// `is_crowd`, if given, must have one entry per `b` mask (a column); a
/// `true` entry makes that column an IoA-against-`a`-only crowd region
/// rather than a normal IoU pair, matching pycocotools' `iou(dt, gt,
/// iscrowd)` semantics (spelled `is_crowd` here for consistency with
/// [`iou`]'s parameter of the same meaning).
pub fn iou_matrix(
    a: &[Rle],
    b: &[Rle],
    is_crowd: Option<&[bool]>,
) -> Result<Vec<Vec<f64>>, String> {
    if let Some(flags) = is_crowd {
        if flags.len() != b.len() {
            return Err(format!(
                "is_crowd must have one entry per mask in the second set, got {} for {}",
                flags.len(),
                b.len()
            ));
        }
    }
    a.iter()
        .map(|ma| {
            b.iter()
                .enumerate()
                .map(|(j, mb)| iou(ma, mb, is_crowd.map(|f| f[j]).unwrap_or(false)))
                .collect()
        })
        .collect()
}

/// Merge two same-size masks into one, coalescing adjacent same-value runs,
/// computing their union or intersection. The lockstep walk is the same one
/// [`intersection_and_union`] uses; unlike that function this one emits the
/// merged run-length sequence itself rather than just area sums.
///
/// The result's `counts` always sum to `h * w` (the walk consumes exactly
/// that much from each input, which itself already satisfies the
/// invariant), so this builds the output `Rle` directly rather than paying
/// for a redundant [`Rle::new`] coverage check.
fn pairwise_merge(a: &Rle, b: &Rle, intersect: bool) -> Rle {
    let (mut ca, mut cb) = (a.counts[0], b.counts[0]);
    let (mut ka, mut kb) = (1usize, 1usize);
    let (mut va, mut vb) = (false, false);
    let mut value = false;
    let mut counts = Vec::new();
    let mut run = 0u32;
    loop {
        let c = ca.min(cb);
        run += c;
        ca -= c;
        if ca == 0 && ka < a.counts.len() {
            ca = a.counts[ka];
            ka += 1;
            va = !va;
        }
        cb -= c;
        if cb == 0 && kb < b.counts.len() {
            cb = b.counts[kb];
            kb += 1;
            vb = !vb;
        }
        let previous = value;
        value = if intersect { va && vb } else { va || vb };
        let done = ca == 0 && cb == 0;
        if value != previous || done {
            counts.push(run);
            run = 0;
        }
        if done {
            break;
        }
    }
    Rle {
        h: a.h,
        w: a.w,
        counts,
    }
}

/// Merge a list of same-size masks into their union (`intersect=false`, the
/// default) or intersection (`intersect=true`), matching pycocotools'
/// `merge(rles, intersect)`. An empty list yields an empty `0x0` mask
/// (matching pycocotools). Unlike pycocotools, which silently returns an
/// empty mask on a size mismatch between inputs, this raises a clear error.
pub fn merge(masks: &[Rle], intersect: bool) -> Result<Rle, String> {
    let Some(first) = masks.first() else {
        return Rle::new(0, 0, Vec::new());
    };
    for other in &masks[1..] {
        if other.h != first.h || other.w != first.w {
            return Err(format!(
                "mask size mismatch: {:?} vs {:?}",
                (first.h, first.w),
                (other.h, other.w)
            ));
        }
    }
    let mut acc = first.clone();
    for other in &masks[1..] {
        acc = pairwise_merge(&acc, other, intersect);
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    // Golden vectors validated against a real pycocotools build (docstring
    // example from pycocotools/mask.py: M=[0 0 1 1 1 0 1] -> counts=[2,3,1,1]).
    #[test]
    fn compressed_roundtrip_matches_pycocotools_docstring_example() {
        let rle = Rle {
            h: 7,
            w: 1,
            counts: vec![2, 3, 1, 1],
        };
        assert_eq!(rle.to_compressed(), "231N");
        assert_eq!(Rle::from_compressed(7, 1, "231N").unwrap(), rle);
    }

    #[test]
    fn compressed_roundtrip_all_zeros_and_ones() {
        let zeros = Rle {
            h: 4,
            w: 4,
            counts: vec![16],
        };
        assert_eq!(zeros.to_compressed(), "`0");
        let ones = Rle {
            h: 4,
            w: 4,
            counts: vec![0, 16],
        };
        assert_eq!(ones.to_compressed(), "0`0");
        assert_eq!(Rle::from_compressed(4, 4, "`0").unwrap(), zeros);
        assert_eq!(Rle::from_compressed(4, 4, "0`0").unwrap(), ones);
    }

    #[test]
    fn compressed_roundtrip_single_pixel() {
        // 5x5, single foreground pixel at column-major offset 17 (row 2, col 3).
        let rle = Rle {
            h: 5,
            w: 5,
            counts: vec![17, 1, 7],
        };
        assert_eq!(rle.to_compressed(), "a017");
        assert_eq!(Rle::from_compressed(5, 5, "a017").unwrap(), rle);
    }

    #[test]
    fn dense_roundtrip() {
        let bits = [0u8, 0, 1, 1, 1, 0, 1];
        let rle = Rle::from_dense(7, 1, &bits);
        assert_eq!(rle.counts, vec![2, 3, 1, 1]);
        assert_eq!(rle.to_dense(), bits);
    }

    #[test]
    fn area_counts_foreground_only() {
        let rle = Rle {
            h: 7,
            w: 1,
            counts: vec![2, 3, 1, 1],
        };
        assert_eq!(rle.area(), 4);
    }

    #[test]
    fn identical_masks_have_iou_one() {
        let a = Rle::from_dense(3, 3, &[1, 1, 0, 0, 1, 0, 0, 0, 1]);
        let b = a.clone();
        approx(iou(&a, &b, false).unwrap(), 1.0);
    }

    #[test]
    fn disjoint_masks_have_iou_zero() {
        let a = Rle::from_dense(2, 2, &[1, 1, 0, 0]);
        let b = Rle::from_dense(2, 2, &[0, 0, 1, 1]);
        approx(iou(&a, &b, false).unwrap(), 0.0);
    }

    #[test]
    fn half_overlap() {
        // a = {0,1}, b = {1,2} out of 4 pixels -> inter=1, union=3.
        let a = Rle::from_dense(2, 2, &[1, 1, 0, 0]);
        let b = Rle::from_dense(2, 2, &[0, 1, 1, 0]);
        approx(iou(&a, &b, false).unwrap(), 1.0 / 3.0);
    }

    #[test]
    fn crowd_scores_intersection_over_a_area() {
        // a is fully inside the crowd region b, but b is much larger.
        let a = Rle::from_dense(4, 1, &[1, 1, 0, 0]);
        let b = Rle::from_dense(4, 1, &[1, 1, 1, 1]);
        approx(iou(&a, &b, true).unwrap(), 1.0); // inter=2, area(a)=2
        approx(iou(&a, &b, false).unwrap(), 0.5); // inter=2, union=4
    }

    #[test]
    fn mismatched_size_errors() {
        let a = Rle {
            h: 2,
            w: 2,
            counts: vec![0, 4],
        };
        let b = Rle {
            h: 3,
            w: 3,
            counts: vec![0, 9],
        };
        assert!(iou(&a, &b, false).is_err());
    }

    #[test]
    fn iou_matrix_shape_and_crowd_column() {
        let a = vec![Rle::from_dense(2, 2, &[1, 1, 0, 0])];
        let b = vec![
            Rle::from_dense(2, 2, &[1, 1, 0, 0]),
            Rle::from_dense(2, 2, &[1, 1, 1, 1]),
        ];
        let m = iou_matrix(&a, &b, Some(&[false, true])).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].len(), 2);
        approx(m[0][0], 1.0);
        approx(m[0][1], 1.0); // crowd column: inter=2 / area(a)=2
    }

    // Regression: counts that don't cover exactly h * w used to reach `iou`
    // unchecked, and its lockstep walk never terminates once the shorter
    // side is exhausted while the other still has pixels left (see the PR
    // discussion). `Rle::new` — the only public way to build an `Rle` from
    // raw counts — now rejects this before it can reach `iou` at all.
    #[test]
    fn new_rejects_undersized_counts() {
        assert!(Rle::new(10, 1, vec![5]).is_err());
    }

    #[test]
    fn new_rejects_oversized_counts() {
        assert!(Rle::new(2, 2, vec![0, 100]).is_err());
    }

    #[test]
    fn new_accepts_exact_coverage() {
        assert!(Rle::new(2, 2, vec![0, 4]).is_ok());
    }

    #[test]
    fn from_compressed_rejects_invalid_byte() {
        // '!' (0x21) is outside the valid 48..=111 packed-value alphabet.
        assert!(Rle::from_compressed(2, 2, "!").is_err());
    }

    #[test]
    fn from_compressed_rejects_oversized_run_length_group() {
        // A run of 20 continuation-flagged bytes never terminates within
        // MAX_GROUPS_PER_RUN and must error, not shift-overflow-panic.
        let malformed = "o".repeat(20); // 'o' - 48 = 0x2f, continuation bit set
        assert!(Rle::from_compressed(2, 2, &malformed).is_err());
    }

    #[test]
    fn from_compressed_rejects_truncated_coverage() {
        // A syntactically valid but short compressed string (decodes fine,
        // just doesn't cover h * w) must be rejected too.
        assert!(Rle::from_compressed(10, 1, "5").is_err());
    }

    #[test]
    fn bbox_of_empty_mask_is_zero() {
        let empty = Rle::from_dense(4, 4, &[0u8; 16]);
        assert_eq!(empty.bbox(), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn bbox_matches_pycocotools_example() {
        // Validated against real pycocotools: bits (row-major)
        // [[0,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,0,0]] -> toBbox (xywh) = [1,1,3,2].
        #[rustfmt::skip]
        let column_major = [
            0, 0, 0, 0, // column 0
            0, 1, 0, 0, // column 1
            0, 0, 0, 0, // column 2
            0, 0, 1, 0, // column 3
        ];
        let rle = Rle::from_dense(4, 4, &column_major);
        assert_eq!(rle.bbox(), (1.0, 1.0, 4.0, 3.0)); // xyxy = xywh (1,1,3,2) + 1 on the max end
    }

    #[test]
    fn bbox_full_mask_covers_everything() {
        let rle = Rle::from_dense(3, 5, &[1u8; 15]);
        assert_eq!(rle.bbox(), (0.0, 0.0, 5.0, 3.0));
    }

    #[test]
    fn merge_union_of_two_masks() {
        // a = {0,1}, b = {1,2} (row-major, 1x4) -> union = {0,1,2}.
        let a = Rle::from_dense(4, 1, &[1, 1, 0, 0]);
        let b = Rle::from_dense(4, 1, &[0, 1, 1, 0]);
        let union = merge(&[a, b], false).unwrap();
        assert_eq!(union.to_dense(), vec![1, 1, 1, 0]);
    }

    #[test]
    fn merge_intersection_of_two_masks() {
        let a = Rle::from_dense(4, 1, &[1, 1, 0, 0]);
        let b = Rle::from_dense(4, 1, &[0, 1, 1, 0]);
        let inter = merge(&[a, b], true).unwrap();
        assert_eq!(inter.to_dense(), vec![0, 1, 0, 0]);
    }

    #[test]
    fn merge_of_three_masks() {
        let a = Rle::from_dense(3, 1, &[1, 0, 0]);
        let b = Rle::from_dense(3, 1, &[0, 1, 0]);
        let c = Rle::from_dense(3, 1, &[0, 0, 1]);
        let union = merge(&[a, b, c], false).unwrap();
        assert_eq!(union.to_dense(), vec![1, 1, 1]);
    }

    #[test]
    fn merge_of_empty_list_is_empty_mask() {
        let empty = merge(&[], false).unwrap();
        assert_eq!(empty.h(), 0);
        assert_eq!(empty.w(), 0);
        assert_eq!(empty.area(), 0);
    }

    #[test]
    fn merge_of_one_mask_is_itself() {
        let a = Rle::from_dense(2, 2, &[1, 0, 0, 1]);
        let merged = merge(std::slice::from_ref(&a), false).unwrap();
        assert_eq!(merged, a);
    }

    #[test]
    fn merge_size_mismatch_errors() {
        let a = Rle::from_dense(2, 2, &[1, 0, 0, 1]);
        let b = Rle::from_dense(3, 3, &[0u8; 9]);
        assert!(merge(&[a, b], false).is_err());
    }
}
