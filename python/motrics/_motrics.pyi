"""Type stubs for the compiled ``motrics._motrics`` extension module."""

from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt

__version__: str

# A bounding box: 4 floats, in whichever convention `box_format` specifies.
Bbox = Sequence[float]
# One frame's boxes: a sequence of 4-tuples, or a zero-copy `(N, 4)` float64
# NumPy array (zero-copy only for a contiguous array in `"xyxy"` format).
Boxes = Sequence[Bbox] | npt.NDArray[np.float64]
# `"xyxy"` is `(x1, y1, x2, y2)`; `"xywh"` is `(x, y, width, height)`.
BoxFormat = Literal["xyxy", "xywh"]

def version() -> str:
    """Return the version of the compiled Rust core."""
    ...

def is_debug_build() -> bool:
    """Whether the extension was compiled with debug assertions.

    Debug builds are ~10x slower than release builds; use this to warn before
    reporting any performance measurement.
    """
    ...

def iou(box_a: Bbox, box_b: Bbox, box_format: BoxFormat = "xyxy") -> float:
    """Intersection-over-union of two boxes."""
    ...

def iou_matrix(
    boxes_a: Boxes, boxes_b: Boxes, box_format: BoxFormat = "xyxy"
) -> list[list[float]]:
    """Pairwise IoU matrix, ``len(boxes_a)`` rows by ``len(boxes_b)`` columns."""
    ...

class RleDict(TypedDict):
    """A pycocotools-style RLE mask dict, as read straight from a COCO/KITTI-MOTS/
    BDD-MOTS/DAVIS annotation file or returned by ``pycocotools.mask.encode``."""

    size: Sequence[int]  # (h, w); pycocotools itself uses a `[h, w]` list
    counts: str | bytes | Sequence[int]

class Mask:
    """A run-length-encoded binary mask (COCO/pycocotools convention):
    alternating background/foreground run lengths, walked column-major over an
    ``(h, w)`` image."""

    def __init__(self, size: Sequence[int], counts: Sequence[int]) -> None: ...
    @staticmethod
    def from_coco(size: Sequence[int], counts: str | bytes) -> Mask:
        """Decode pycocotools' compressed-string RLE form."""
        ...

    @property
    def size(self) -> tuple[int, int]: ...
    @property
    def counts(self) -> list[int]:
        """Alternating background/foreground run lengths (decoded form)."""

    def area(self) -> int:
        """Foreground pixel count."""
        ...

    def to_coco(self) -> str:
        """pycocotools' compressed-string RLE form."""
        ...

    def __repr__(self) -> str: ...

# Anything accepted where a mask is expected: a `Mask` instance, or a
# pycocotools-style RLE dict (`counts` compressed `str`/`bytes`, or an
# already-decoded list of run lengths).
MaskLike = Mask | RleDict

def mask_area(mask: MaskLike) -> int:
    """Foreground pixel count of a mask."""
    ...

def mask_iou(mask_a: MaskLike, mask_b: MaskLike, is_crowd: bool = False) -> float:
    """Intersection-over-union of two masks of the same ``(h, w)``.

    If ``is_crowd`` is set, ``mask_b`` is a crowd/ignore region: the score is
    intersection over ``mask_a``'s own area rather than the union, matching
    pycocotools' ``iscrowd`` semantics.
    """
    ...

def mask_iou_matrix(
    masks_a: Sequence[MaskLike],
    masks_b: Sequence[MaskLike],
    is_crowd: Sequence[bool] | None = None,
) -> list[list[float]]:
    """Pairwise IoU matrix between two sets of masks.

    ``is_crowd``, if given, must have one entry per mask in ``masks_b``; a
    ``True`` entry makes that column an IoA-against-``masks_a``-only crowd
    region, matching pycocotools' ``iou(dt, gt, iscrowd)``.
    """
    ...

def mask_decode(mask: MaskLike) -> list[list[int]]:
    """Decode a mask to a dense ``(h, w)`` nested list of ``0``/``1`` values."""
    ...

def mask_encode(bitmap: npt.NDArray[np.uint8]) -> Mask:
    """Encode a dense ``(h, w)`` ``uint8`` NumPy array (any nonzero value is
    foreground) into a :class:`Mask`."""
    ...

def mask_merge(masks: Sequence[MaskLike], intersect: bool = False) -> Mask:
    """Merge masks into their union (default) or intersection.

    Matches pycocotools' ``merge(rles, intersect)``. An empty ``masks``
    yields an empty ``Mask((0, 0), [])``. Unlike pycocotools, which silently
    returns an empty mask on a size mismatch between inputs, a genuine
    mismatch raises ``ValueError``.
    """
    ...

def mask_to_bbox(
    mask: MaskLike, box_format: BoxFormat = "xyxy"
) -> tuple[float, float, float, float]:
    """Bounding box of a mask's foreground pixels, or ``(0, 0, 0, 0)`` if it
    has none.

    ``box_format`` is ``"xyxy"`` (default, matching every other box
    primitive in this library) or ``"xywh"`` (pycocotools' own ``toBbox``
    convention).
    """
    ...

class Matching:
    """The result of matching two sets of boxes or masks."""

    @property
    def matches(self) -> list[tuple[int, int]]:
        """Matched ``(a_index, b_index)`` pairs, ordered by ``a_index``."""

    @property
    def scores(self) -> list[float]:
        """IoU score for each matched pair, parallel to ``matches``."""

    @property
    def unmatched_a(self) -> list[int]:
        """Indices of set A that were not matched."""

    @property
    def unmatched_b(self) -> list[int]:
        """Indices of set B that were not matched."""

    def __repr__(self) -> str: ...

def match_boxes(
    boxes_a: Boxes,
    boxes_b: Boxes,
    iou_threshold: float = 0.5,
    method: str = "hungarian",
    box_format: BoxFormat = "xyxy",
) -> Matching:
    """Match two sets of boxes.

    ``method`` is ``"hungarian"`` (optimal, maximises total IoU) or ``"greedy"``
    (assign highest-IoU pairs first). Only pairs with IoU at or above
    ``iou_threshold`` are kept.
    """
    ...

def match_masks(
    masks_a: Sequence[MaskLike],
    masks_b: Sequence[MaskLike],
    iou_threshold: float = 0.5,
    method: str = "hungarian",
) -> Matching:
    """Match two sets of masks, mirroring :func:`match_boxes` for segmentation
    masks.
    """
    ...

class ClearMetrics:
    """Accumulated CLEAR MOT metrics over a sequence."""

    @property
    def mota(self) -> float:
        """Multiple Object Tracking Accuracy: ``1 - (FN + FP + IDSW) / num_gt``."""

    @property
    def motp(self) -> float:
        """Multiple Object Tracking Precision: mean IoU over matched pairs."""

    @property
    def num_frames(self) -> int:
        """Number of frames processed."""

    @property
    def num_gt(self) -> int:
        """Total ground-truth detections across all frames."""

    @property
    def num_matches(self) -> int:
        """True positives: matched ``(gt, pred)`` pairs."""

    @property
    def num_false_positives(self) -> int:
        """False positives: tracker detections with no match."""

    @property
    def num_misses(self) -> int:
        """Misses: ground-truth detections with no match."""

    @property
    def num_switches(self) -> int:
        """Identity switches."""

    def __repr__(self) -> str: ...

def compute_clear(
    gt_ids: Sequence[Sequence[int]],
    gt_boxes: Sequence[Boxes],
    pred_ids: Sequence[Sequence[int]],
    pred_boxes: Sequence[Boxes],
    iou_threshold: float = 0.5,
    box_format: BoxFormat = "xyxy",
) -> ClearMetrics:
    """Compute CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches).

    Inputs are frame-aligned: ``gt_ids[t]`` / ``gt_boxes[t]`` describe the
    ground-truth objects in frame ``t``, and ``pred_ids`` / ``pred_boxes`` the
    tracker's. ``gt_ids`` and ``pred_ids`` must have the same number of frames,
    and within each frame the id and box lists must have equal length. Each
    frame's boxes may be a contiguous ``(N, 4)`` float64 NumPy array instead of
    a list of tuples, for a zero-copy read in ``"xyxy"`` format.
    """
    ...

def compute_clear_from_similarity(
    gt_ids: Sequence[Sequence[int]],
    pred_ids: Sequence[Sequence[int]],
    similarity: Sequence[Sequence[Sequence[float]]],
    threshold: float = 0.5,
) -> ClearMetrics:
    """Compute CLEAR MOT metrics from precomputed per-frame similarity matrices.

    For callers that already hold pairwise scores (e.g. a ``motmetrics``-style
    distance matrix converted to similarity) instead of boxes.
    ``similarity[t][i][j]`` scores ``gt_ids[t][i]`` against ``pred_ids[t][j]``;
    higher is better, the same convention as IoU, and pairs below
    ``threshold`` are never matched.
    """
    ...

class IdentityMetrics:
    """Accumulated Identity metrics (IDF1/IDP/IDR) over a sequence."""

    @property
    def idf1(self) -> float:
        """Identity F1: ``IDTP / (IDTP + 0.5 IDFP + 0.5 IDFN)``."""

    @property
    def idp(self) -> float:
        """Identity precision: ``IDTP / (IDTP + IDFP)``."""

    @property
    def idr(self) -> float:
        """Identity recall: ``IDTP / (IDTP + IDFN)``."""

    @property
    def idtp(self) -> int:
        """Identity true positives."""

    @property
    def idfp(self) -> int:
        """Identity false positives."""

    @property
    def idfn(self) -> int:
        """Identity false negatives."""

    @property
    def num_frames(self) -> int:
        """Number of frames processed."""

    @property
    def num_gt(self) -> int:
        """Total ground-truth detections across all frames."""

    @property
    def num_pred(self) -> int:
        """Total predicted detections across all frames."""

    def __repr__(self) -> str: ...

def compute_identity(
    gt_ids: Sequence[Sequence[int]],
    gt_boxes: Sequence[Boxes],
    pred_ids: Sequence[Sequence[int]],
    pred_boxes: Sequence[Boxes],
    iou_threshold: float = 0.5,
    box_format: BoxFormat = "xyxy",
) -> IdentityMetrics:
    """Compute Identity metrics (IDF1, IDP, IDR) for a sequence.

    Inputs are frame-aligned exactly like :func:`compute_clear`. Identity metrics
    use a single global bipartite matching between whole ground-truth and
    predicted trajectories, rewarding id consistency over time.
    """
    ...

def compute_identity_from_similarity(
    gt_ids: Sequence[Sequence[int]],
    pred_ids: Sequence[Sequence[int]],
    similarity: Sequence[Sequence[Sequence[float]]],
    threshold: float = 0.5,
) -> IdentityMetrics:
    """Compute Identity metrics from precomputed per-frame similarity matrices.

    Same similarity convention as :func:`compute_clear_from_similarity`.
    """
    ...

class HotaMetrics:
    """HOTA metrics over a sequence (summarised, with per-alpha curves)."""

    @property
    def hota(self) -> float:
        """HOTA score: mean over alpha of ``sqrt(DetA * AssA)``."""

    @property
    def deta(self) -> float:
        """Detection accuracy: mean over alpha."""

    @property
    def assa(self) -> float:
        """Association accuracy: mean over alpha."""

    @property
    def loca(self) -> float:
        """Localization accuracy: mean over alpha."""

    @property
    def alphas(self) -> list[float]:
        """The alpha (localization) thresholds swept."""

    @property
    def hota_alphas(self) -> list[float]:
        """Per-alpha HOTA scores, parallel to ``alphas``."""

    @property
    def deta_alphas(self) -> list[float]:
        """Per-alpha DetA scores, parallel to ``alphas``."""

    @property
    def assa_alphas(self) -> list[float]:
        """Per-alpha AssA scores, parallel to ``alphas``."""

    @property
    def loca_alphas(self) -> list[float]:
        """Per-alpha LocA scores, parallel to ``alphas``."""

    @property
    def hota_tp_alphas(self) -> list[float]:
        """Per-alpha true positive counts, parallel to ``alphas``."""

    @property
    def hota_fn_alphas(self) -> list[float]:
        """Per-alpha false negative counts, parallel to ``alphas``."""

    @property
    def hota_fp_alphas(self) -> list[float]:
        """Per-alpha false positive counts, parallel to ``alphas``."""

    @property
    def ass_re_alphas(self) -> list[float]:
        """Per-alpha association recall, parallel to ``alphas``."""

    @property
    def ass_pr_alphas(self) -> list[float]:
        """Per-alpha association precision, parallel to ``alphas``."""

    @property
    def num_frames(self) -> int:
        """Number of frames processed."""

    @property
    def num_gt(self) -> int:
        """Total ground-truth detections across all frames."""

    @property
    def num_pred(self) -> int:
        """Total predicted detections across all frames."""

    def __repr__(self) -> str: ...

def compute_hota(
    gt_ids: Sequence[Sequence[int]],
    gt_boxes: Sequence[Boxes],
    pred_ids: Sequence[Sequence[int]],
    pred_boxes: Sequence[Boxes],
    box_format: BoxFormat = "xyxy",
) -> HotaMetrics:
    """Compute HOTA metrics (DetA, AssA, LocA, plus per-alpha curves).

    Inputs are frame-aligned exactly like :func:`compute_clear`. HOTA sweeps a
    set of localization thresholds internally, so unlike the other metrics it
    takes no single ``iou_threshold``.
    """
    ...

def compute_hota_from_similarity(
    gt_ids: Sequence[Sequence[int]],
    pred_ids: Sequence[Sequence[int]],
    similarity: Sequence[Sequence[Sequence[float]]],
) -> HotaMetrics:
    """Compute HOTA metrics from precomputed per-frame similarity matrices.

    Same similarity convention as :func:`compute_clear_from_similarity`.
    """
    ...

class EvaluationResult:
    """The result of :func:`evaluate`: CLEAR, Identity, and HOTA computed
    together from one shared similarity matrix."""

    @property
    def clear(self) -> ClearMetrics:
        """CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches)."""

    @property
    def identity(self) -> IdentityMetrics:
        """Identity metrics (IDF1, IDP, IDR)."""

    @property
    def hota(self) -> HotaMetrics:
        """HOTA metrics (DetA, AssA, LocA, plus per-alpha curves)."""

    def __repr__(self) -> str: ...

def evaluate(
    gt_ids: Sequence[Sequence[int]],
    gt_boxes: Sequence[Boxes],
    pred_ids: Sequence[Sequence[int]],
    pred_boxes: Sequence[Boxes],
    iou_threshold: float = 0.5,
    box_format: BoxFormat = "xyxy",
) -> EvaluationResult:
    """Compute CLEAR, Identity, and HOTA together for a sequence.

    Inputs are frame-aligned exactly like :func:`compute_clear`. Builds the
    gt/pred similarity matrix once and reuses it for all three metrics,
    instead of recomputing it once per metric (what calling ``compute_clear``,
    ``compute_identity``, and ``compute_hota`` separately would do).
    """
    ...

class AccumulatorResult:
    """CLEAR and Identity read from a streaming :class:`Accumulator`."""

    @property
    def clear(self) -> ClearMetrics:
        """CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches)."""

    @property
    def identity(self) -> IdentityMetrics:
        """Identity metrics (IDF1, IDP, IDR)."""

    def __repr__(self) -> str: ...

class Accumulator:
    """A streaming CLEAR + Identity accumulator.

    Fold in one frame at a time with :meth:`update` (boxes) or
    :meth:`update_from_similarity` (precomputed scores), then read the metrics
    with :meth:`compute` — the online/large-sequence shape, where the whole
    sequence is never held in memory. HOTA is not offered here: its alpha
    sweep is inherently a whole-sequence computation, so use :func:`evaluate`
    or :func:`compute_hota` for it.
    """

    def __init__(
        self, iou_threshold: float = 0.5, box_format: BoxFormat = "xyxy"
    ) -> None: ...
    @property
    def num_frames(self) -> int:
        """Number of frames folded in so far."""

    def update(
        self,
        gt_ids: Sequence[int],
        gt_boxes: Boxes,
        pred_ids: Sequence[int],
        pred_boxes: Boxes,
    ) -> None:
        """Fold one frame in.

        ``gt_ids``/``gt_boxes`` (and ``pred_ids``/``pred_boxes``) must have
        equal length; each boxes argument is a sequence of 4-tuples or a
        contiguous ``(N, 4)`` float64 NumPy array, in this accumulator's
        ``box_format``.
        """
        ...

    def update_from_similarity(
        self,
        gt_ids: Sequence[int],
        pred_ids: Sequence[int],
        similarity: Sequence[Sequence[float]],
    ) -> None:
        """Fold one frame in from a precomputed similarity matrix.

        ``similarity[i][j]`` scores ``gt_ids[i]`` against ``pred_ids[j]``; it
        must be ``len(gt_ids)`` rows by ``len(pred_ids)`` columns.
        """
        ...

    def compute(self) -> AccumulatorResult:
        """Finalize CLEAR and Identity from everything folded in so far.

        May be called at any point and does not consume the accumulator.
        """
        ...

    def __repr__(self) -> str: ...
