"""motrics — an extremely fast MOT and HOTA metrics library, written in Rust.

The heavy lifting lives in the compiled ``motrics._motrics`` extension module.
This package re-exports the public API so users only ever import from
``motrics``.
"""

from motrics._motrics import (
    ClearMetrics,
    IdentityMetrics,
    Matching,
    __version__,
    compute_clear,
    compute_identity,
    iou,
    iou_matrix,
    match_boxes,
    version,
)

__all__ = [
    "ClearMetrics",
    "IdentityMetrics",
    "Matching",
    "__version__",
    "compute_clear",
    "compute_identity",
    "iou",
    "iou_matrix",
    "match_boxes",
    "version",
]
