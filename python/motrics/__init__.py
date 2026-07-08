"""motrics — an extremely fast MOT and HOTA metrics library, written in Rust.

The heavy lifting lives in the compiled ``motrics._motrics`` extension module.
This package re-exports the public API so users only ever import from
``motrics``.
"""

from motrics._motrics import (
    Accumulator,
    AccumulatorResult,
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
    match_masks,
    version,
)
from motrics.bdd100k import load_bdd100k, load_bdd100k_gt, preprocess_bdd100k
from motrics.davis import load_davis, preprocess_davis
from motrics.frames import Frames, evaluate
from motrics.kitti import load_kitti, load_kitti_gt, preprocess_kitti
from motrics.kitti_mots import (
    load_kitti_mots,
    load_kitti_mots_gt,
    preprocess_kitti_mots,
)
from motrics.motchallenge import (
    align_frames,
    load_motchallenge,
    load_motchallenge_gt,
    preprocess_motchallenge,
)

__all__ = [
    "Accumulator",
    "AccumulatorResult",
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
    "load_bdd100k",
    "load_bdd100k_gt",
    "load_davis",
    "load_kitti",
    "load_kitti_gt",
    "load_kitti_mots",
    "load_kitti_mots_gt",
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
    "match_masks",
    "preprocess_bdd100k",
    "preprocess_davis",
    "preprocess_kitti",
    "preprocess_kitti_mots",
    "preprocess_motchallenge",
    "version",
]
