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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rle {
    pub h: usize,
    pub w: usize,
    pub counts: Vec<u32>,
}

impl Rle {
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
    pub fn from_compressed(h: usize, w: usize, s: &str) -> Result<Rle, String> {
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
                let c = i64::from(bytes[p]) - 48;
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
            if x < 0 {
                return Err("RLE counts string decodes to a negative run length".to_string());
            }
            counts.push(x as u32);
        }
        Ok(Rle { h, w, counts })
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
/// `iscrowd`, if given, must have one entry per `b` mask (a column); a `true`
/// entry makes that column an IoA-against-`a`-only crowd region rather than a
/// normal IoU pair, matching pycocotools' `iou(dt, gt, iscrowd)`.
pub fn iou_matrix(a: &[Rle], b: &[Rle], iscrowd: Option<&[bool]>) -> Result<Vec<Vec<f64>>, String> {
    if let Some(flags) = iscrowd {
        if flags.len() != b.len() {
            return Err(format!(
                "iscrowd must have one entry per mask in the second set, got {} for {}",
                flags.len(),
                b.len()
            ));
        }
    }
    a.iter()
        .map(|ma| {
            b.iter()
                .enumerate()
                .map(|(j, mb)| iou(ma, mb, iscrowd.map(|f| f[j]).unwrap_or(false)))
                .collect()
        })
        .collect()
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
}
