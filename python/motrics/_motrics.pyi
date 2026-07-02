"""Type stubs for the compiled ``motrics._motrics`` extension module."""

from collections.abc import Sequence

__version__: str

# A bounding box in xyxy format: (x1, y1, x2, y2).
Bbox = Sequence[float]

def version() -> str:
    """Return the version of the compiled Rust core."""
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
