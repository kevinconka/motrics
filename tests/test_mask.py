"""Tests for the mask-IoU similarity kernel (`Mask`, `mask_*`).

Unit tests exercise the Python binding layer directly (argument flexibility,
error paths); the RLE codec itself is validated byte-for-byte against real
`pycocotools` output both here (`test_mask_matches_pycocotools`) and in the
Rust unit tests in `src/mask.rs` (golden vectors taken from the same
validation this file runs).
"""

from __future__ import annotations

import motrics
import pytest

pycocotools_mask = pytest.importorskip("pycocotools.mask")
np = pytest.importorskip("numpy")


def test_mask_encode_decode_round_trip() -> None:
    bits = [[0, 1, 0], [0, 1, 1], [1, 0, 0]]
    bitmap = np.array(bits, dtype=np.uint8)
    mask = motrics.mask_encode(bitmap)
    assert mask.size == (3, 3)
    assert mask.area() == sum(sum(row) for row in bits)
    assert motrics.mask_decode(mask) == bits


def test_mask_new_accepts_decoded_counts() -> None:
    # frame from the pycocotools docstring: M=[0 0 1 1 1 0 1] -> counts=[2,3,1,1].
    mask = motrics.Mask((7, 1), [2, 3, 1, 1])
    assert mask.area() == 4
    assert mask.to_coco() == "231N"


def test_mask_from_coco_decodes_compressed_string() -> None:
    mask = motrics.Mask.from_coco((7, 1), "231N")
    assert mask.counts == [2, 3, 1, 1]
    assert mask.area() == 4


def test_mask_iou_identical_masks() -> None:
    a = motrics.Mask((2, 2), [0, 4])  # all foreground
    b = motrics.Mask((2, 2), [0, 4])
    assert motrics.mask_iou(a, b) == pytest.approx(1.0)


def test_mask_iou_disjoint_masks() -> None:
    a = motrics.Mask((4, 1), [0, 2, 2])  # foreground rows 0-1
    b = motrics.Mask((4, 1), [2, 2, 0])  # foreground rows 2-3
    assert motrics.mask_iou(a, b) == pytest.approx(0.0)


def test_mask_iou_crowd_scores_intersection_over_a() -> None:
    a = motrics.Mask((4, 1), [0, 2, 2])  # 2 fg pixels
    b = motrics.Mask((4, 1), [0, 4])  # all 4 fg pixels, the "crowd" region
    assert motrics.mask_iou(a, b, is_crowd=True) == pytest.approx(1.0)
    assert motrics.mask_iou(a, b, is_crowd=False) == pytest.approx(0.5)


def test_mask_iou_size_mismatch_raises() -> None:
    a = motrics.Mask((2, 2), [0, 4])
    b = motrics.Mask((3, 3), [0, 9])
    with pytest.raises(ValueError, match="mask size mismatch"):
        motrics.mask_iou(a, b)


def test_mask_iou_accepts_pycocotools_style_dict() -> None:
    rle_dict: motrics._motrics.RleDict = {"size": [7, 1], "counts": "231N"}
    mask = motrics.Mask((7, 1), [2, 3, 1, 1])
    assert motrics.mask_iou(rle_dict, mask) == pytest.approx(1.0)


def test_mask_iou_matrix_shape_and_crowd_column() -> None:
    a = [motrics.Mask((2, 2), [0, 4])]
    b = [motrics.Mask((2, 2), [0, 4]), motrics.Mask((2, 2), [2, 2])]
    matrix = motrics.mask_iou_matrix(a, b, iscrowd=[False, True])
    assert len(matrix) == 1
    assert len(matrix[0]) == 2
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(0.5)  # crowd column: inter=2 / area(a[0])=4


def test_mask_iou_matrix_iscrowd_length_mismatch_raises() -> None:
    a = [motrics.Mask((2, 2), [0, 4])]
    b = [motrics.Mask((2, 2), [0, 4]), motrics.Mask((2, 2), [2, 2])]
    with pytest.raises(ValueError, match="iscrowd"):
        motrics.mask_iou_matrix(a, b, iscrowd=[False])


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (17, 9), (64, 48), (100, 137)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mask_matches_pycocotools(shape: tuple[int, int], seed: int) -> None:
    """Round-trip encode/decode/area/iou against real (unmodified) pycocotools
    on random masks, including degenerate (all-zero/all-one) edge cases."""
    rng = np.random.RandomState(seed)
    a_bits = (rng.rand(*shape) > 0.5).astype(np.uint8)
    b_bits = (rng.rand(*shape) > 0.5).astype(np.uint8)

    rle_a = pycocotools_mask.encode(np.asfortranarray(a_bits))
    rle_b = pycocotools_mask.encode(np.asfortranarray(b_bits))

    m_a = motrics.mask_encode(a_bits)
    m_b = motrics.mask_encode(b_bits)

    assert m_a.to_coco().encode() == rle_a["counts"]
    assert m_a.area() == pycocotools_mask.area([rle_a])[0]
    assert motrics.mask_decode(m_a) == a_bits.tolist()

    ref_iou = pycocotools_mask.iou([rle_a], [rle_b], [0])[0][0]
    assert motrics.mask_iou(m_a, m_b) == pytest.approx(ref_iou, abs=1e-9)

    ref_ioa = pycocotools_mask.iou([rle_a], [rle_b], [1])[0][0]
    assert motrics.mask_iou(m_a, m_b, is_crowd=True) == pytest.approx(ref_ioa, abs=1e-9)

    # Also directly through pycocotools' own RLE dicts (compressed bytes counts).
    assert motrics.mask_iou(rle_a, rle_b) == pytest.approx(ref_iou, abs=1e-9)


def test_mask_iou_matrix_matches_pycocotools() -> None:
    rng = np.random.RandomState(3)
    a_bits = [(rng.rand(30, 20) > 0.5).astype(np.uint8) for _ in range(3)]
    b_bits = [(rng.rand(30, 20) > 0.5).astype(np.uint8) for _ in range(4)]
    rles_a = [pycocotools_mask.encode(np.asfortranarray(b)) for b in a_bits]
    rles_b = [pycocotools_mask.encode(np.asfortranarray(b)) for b in b_bits]
    iscrowd = [False, True, False, True]

    ref = pycocotools_mask.iou(rles_a, rles_b, iscrowd)
    got = motrics.mask_iou_matrix(rles_a, rles_b, iscrowd=iscrowd)
    assert np.allclose(got, ref, atol=1e-9)
