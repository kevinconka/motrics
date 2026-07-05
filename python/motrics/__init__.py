"""motrics — an extremely fast MOT and HOTA metrics library, written in Rust.

The heavy lifting lives in the compiled ``motrics._motrics`` extension module.
This package re-exports the public API so users only ever import from
``motrics``.
"""

from motrics._motrics import (
    ClearMetrics,
    EvaluationResult,
    HotaMetrics,
    IdentityMetrics,
    Mask,
    Matching,
    __version__,
    compute_clear,
    compute_clear_from_similarity,
    compute_hota,
    compute_hota_from_similarity,
    compute_identity,
    compute_identity_from_similarity,
    iou,
    iou_matrix,
    is_debug_build,
    mask_area,
    mask_decode,
    mask_encode,
    mask_iou,
    mask_iou_matrix,
    mask_merge,
    mask_to_bbox,
    match_boxes,
    version,
)
from motrics.frames import Frames, evaluate
from motrics.kitti import load_kitti, load_kitti_gt, preprocess_kitti
from motrics.motchallenge import (
    align_frames,
    load_motchallenge,
    load_motchallenge_gt,
    preprocess_motchallenge,
)

__all__ = [
    "ClearMetrics",
    "EvaluationResult",
    "Frames",
    "HotaMetrics",
    "IdentityMetrics",
    "Mask",
    "Matching",
    "__version__",
    "align_frames",
    "compute_clear",
    "compute_clear_from_similarity",
    "compute_hota",
    "compute_hota_from_similarity",
    "compute_identity",
    "compute_identity_from_similarity",
    "evaluate",
    "iou",
    "iou_matrix",
    "is_debug_build",
    "load_kitti",
    "load_kitti_gt",
    "load_motchallenge",
    "load_motchallenge_gt",
    "mask_area",
    "mask_decode",
    "mask_encode",
    "mask_iou",
    "mask_iou_matrix",
    "mask_merge",
    "mask_to_bbox",
    "match_boxes",
    "preprocess_kitti",
    "preprocess_motchallenge",
    "version",
]
