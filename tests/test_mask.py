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
    matrix = motrics.mask_iou_matrix(a, b, is_crowd=[False, True])
    assert len(matrix) == 1
    assert len(matrix[0]) == 2
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(0.5)  # crowd column: inter=2 / area(a[0])=4


def test_mask_iou_matrix_is_crowd_length_mismatch_raises() -> None:
    a = [motrics.Mask((2, 2), [0, 4])]
    b = [motrics.Mask((2, 2), [0, 4]), motrics.Mask((2, 2), [2, 2])]
    with pytest.raises(ValueError, match="is_crowd"):
        motrics.mask_iou_matrix(a, b, is_crowd=[False])


def test_mask_new_rejects_counts_not_covering_the_mask_area() -> None:
    # Regression: counts that don't sum to h * w used to reach `mask_iou`
    # unvalidated and hang forever in its lockstep intersection walk.
    with pytest.raises(ValueError, match="RLE counts cover"):
        motrics.Mask((10, 1), [5])


def test_mask_iou_rejects_malformed_dict_instead_of_hanging() -> None:
    malformed: motrics._motrics.RleDict = {"size": [10, 1], "counts": [5]}
    valid: motrics._motrics.RleDict = {"size": [10, 1], "counts": [0, 10]}
    with pytest.raises(ValueError, match="RLE counts cover"):
        motrics.mask_iou(malformed, valid)


@pytest.mark.parametrize(
    ("counts", "match"),
    [
        ("!", "invalid byte"),  # outside the valid packed-value alphabet
        ("o" * 20, "oversized run-length group"),  # never sets the stop bit
        ("5", "RLE counts cover"),  # decodes fine but under-covers h * w
        # 12 continuation bytes + 1 terminating byte with the sign bit set
        # pushes the post-increment group count to 13, overflowing the
        # sign-extension shift (5 * 13 = 65) if not rejected first.
        ("P" * 12 + "@", "oversized signed run-length group"),
    ],
)
def test_mask_from_coco_rejects_malformed_strings(counts: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        motrics.Mask.from_coco((100, 1), counts)


def test_mask_new_rejects_dimensions_that_overflow() -> None:
    with pytest.raises(ValueError, match="overflows"):
        motrics.Mask((2**32, 2**32), [])


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
    got = motrics.mask_iou_matrix(rles_a, rles_b, is_crowd=iscrowd)
    assert np.allclose(got, ref, atol=1e-9)


def test_mask_merge_union_and_intersection() -> None:
    a = motrics.Mask((4, 1), [0, 2, 2])  # foreground rows 0-1
    b = motrics.Mask((4, 1), [1, 2, 1])  # foreground rows 1-2
    union = motrics.mask_merge([a, b])
    inter = motrics.mask_merge([a, b], intersect=True)
    assert motrics.mask_decode(union) == [[1], [1], [1], [0]]
    assert motrics.mask_decode(inter) == [[0], [1], [0], [0]]


def test_mask_merge_of_empty_list_is_empty_mask() -> None:
    empty = motrics.mask_merge([])
    assert empty.size == (0, 0)
    assert empty.area() == 0


def test_mask_merge_of_multiple_empty_masks_is_empty_mask() -> None:
    # Regression: merge([]) can return a 0x0 mask with empty counts, which
    # the pairwise merge walk can't index into. Merging that result with
    # itself must not crash.
    empty = motrics.mask_merge([])
    merged = motrics.mask_merge([empty, motrics.mask_merge([])])
    assert merged.size == (0, 0)
    assert merged.area() == 0


def test_mask_merge_size_mismatch_raises() -> None:
    a = motrics.Mask((2, 2), [0, 4])
    b = motrics.Mask((3, 3), [0, 9])
    with pytest.raises(ValueError, match="mask size mismatch"):
        motrics.mask_merge([a, b])


def test_mask_merge_matches_pycocotools() -> None:
    rng = np.random.RandomState(4)
    for intersect in (False, True):
        bits = [(rng.rand(15, 12) > 0.5).astype(np.uint8) for _ in range(4)]
        rles = [pycocotools_mask.encode(np.asfortranarray(b)) for b in bits]
        ref = pycocotools_mask.merge(rles, intersect=intersect)
        got = motrics.mask_merge(rles, intersect=intersect)
        assert got.to_coco().encode() == ref["counts"]


def test_mask_to_bbox_default_is_xyxy() -> None:
    # 4x4, single foreground pixel at row=1, col=1 -> xywh (1,1,1,1).
    mask = motrics.Mask((4, 4), [5, 1, 10])
    assert motrics.mask_to_bbox(mask) == (1.0, 1.0, 2.0, 2.0)
    assert motrics.mask_to_bbox(mask, box_format="xywh") == (1.0, 1.0, 1.0, 1.0)


def test_mask_to_bbox_empty_mask_is_zero() -> None:
    empty = motrics.Mask((4, 4), [16])
    assert motrics.mask_to_bbox(empty) == (0.0, 0.0, 0.0, 0.0)


def test_mask_to_bbox_unknown_box_format_raises() -> None:
    mask = motrics.Mask((2, 2), [0, 4])
    with pytest.raises(ValueError, match="unknown box_format"):
        motrics.mask_to_bbox(mask, box_format="nope")  # ty: ignore[invalid-argument-type]


def test_mask_to_bbox_matches_pycocotools() -> None:
    rng = np.random.RandomState(5)
    for _ in range(20):
        shape = (rng.randint(1, 25), rng.randint(1, 25))
        bits = (rng.rand(*shape) > 0.5).astype(np.uint8)
        rle = pycocotools_mask.encode(np.asfortranarray(bits))
        ref_xywh = tuple(pycocotools_mask.toBbox([rle])[0].tolist())
        got_xywh = motrics.mask_to_bbox(rle, box_format="xywh")
        assert got_xywh == pytest.approx(ref_xywh, abs=1e-9)
