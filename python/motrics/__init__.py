"""motrics — an extremely fast MOT and HOTA metrics library, written in Rust.

The heavy lifting lives in the compiled ``motrics._motrics`` extension module.
This package re-exports the public API so users only ever import from
``motrics``.
"""

from motrics._motrics import (
    ClearMetrics,
    HotaMetrics,
    IdentityMetrics,
    Matching,
    __version__,
    compute_clear,
    compute_hota,
    compute_identity,
    iou,
    iou_matrix,
    is_debug_build,
    match_boxes,
    version,
)
from motrics.motchallenge import align_frames, load_motchallenge

__all__ = [
    "ClearMetrics",
    "HotaMetrics",
    "IdentityMetrics",
    "Matching",
    "__version__",
    "align_frames",
    "compute_clear",
    "compute_hota",
    "compute_identity",
    "iou",
    "iou_matrix",
    "is_debug_build",
    "load_motchallenge",
    "match_boxes",
    "version",
]
