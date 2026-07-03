"""Type stubs for the compiled ``motrics._motrics`` extension module."""

from collections.abc import Sequence

__version__: str

# A bounding box in xyxy format: (x1, y1, x2, y2).
Bbox = Sequence[float]

def version() -> str:
    """Return the version of the compiled Rust core."""
    ...

def is_debug_build() -> bool:
    """Whether the extension was compiled with debug assertions.

    Debug builds are ~10x slower than release builds; use this to warn before
    reporting any performance measurement.
    """
    ...

def iou(box_a: Bbox, box_b: Bbox) -> float:
    """Intersection-over-union of two ``xyxy`` boxes ``(x1, y1, x2, y2)``."""
    ...

def iou_matrix(boxes_a: Sequence[Bbox], boxes_b: Sequence[Bbox]) -> list[list[float]]:
    """Pairwise IoU matrix, ``len(boxes_a)`` rows by ``len(boxes_b)`` columns."""
    ...

class Matching:
    """The result of matching two sets of boxes."""

    @property
    def matches(self) -> list[tuple[int, int]]:
        """Matched ``(a_index, b_index)`` pairs, ordered by ``a_index``."""

    @property
    def scores(self) -> list[float]:
        """IoU score for each matched pair, parallel to ``matches``."""

    @property
    def unmatched_a(self) -> list[int]:
        """Indices of ``boxes_a`` that were not matched."""

    @property
    def unmatched_b(self) -> list[int]:
        """Indices of ``boxes_b`` that were not matched."""

    def __repr__(self) -> str: ...

def match_boxes(
    boxes_a: Sequence[Bbox],
    boxes_b: Sequence[Bbox],
    iou_threshold: float = 0.5,
    method: str = "hungarian",
) -> Matching:
    """Match two sets of ``xyxy`` boxes.

    ``method`` is ``"hungarian"`` (optimal, maximises total IoU) or ``"greedy"``
    (assign highest-IoU pairs first). Only pairs with IoU at or above
    ``iou_threshold`` are kept.
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
    gt_boxes: Sequence[Sequence[Bbox]],
    pred_ids: Sequence[Sequence[int]],
    pred_boxes: Sequence[Sequence[Bbox]],
    iou_threshold: float = 0.5,
) -> ClearMetrics:
    """Compute CLEAR MOT metrics (MOTA, MOTP, FP, FN, ID switches).

    Inputs are frame-aligned: ``gt_ids[t]`` / ``gt_boxes[t]`` describe the
    ground-truth objects in frame ``t``, and ``pred_ids`` / ``pred_boxes`` the
    tracker's. ``gt_ids`` and ``pred_ids`` must have the same number of frames,
    and within each frame the id and box lists must have equal length.
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
    gt_boxes: Sequence[Sequence[Bbox]],
    pred_ids: Sequence[Sequence[int]],
    pred_boxes: Sequence[Sequence[Bbox]],
    iou_threshold: float = 0.5,
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
    gt_boxes: Sequence[Sequence[Bbox]],
    pred_ids: Sequence[Sequence[int]],
    pred_boxes: Sequence[Sequence[Bbox]],
) -> HotaMetrics:
    """Compute HOTA metrics (DetA, AssA, LocA, plus per-alpha curves).

    Inputs are frame-aligned exactly like :func:`compute_clear`. HOTA sweeps a
    set of localization thresholds internally, so unlike the other metrics it
    takes no single ``iou_threshold``.
    """
    ...
