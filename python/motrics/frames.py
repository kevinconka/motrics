"""A container bundling one side (ground truth or predictions) of an
evaluation, plus the ``evaluate()`` entry point built on top of it.

``compute_clear``/``compute_identity``/``compute_hota`` each take the same
four parallel ``(gt_ids, gt_boxes, pred_ids, pred_boxes)`` arguments — fine
for one metric, repetitive for the common case of wanting all three.
``Frames`` bundles each side into one object; ``evaluate()`` takes two of
them and returns all three metrics computed from a single shared similarity
matrix, rather than recomputing it once per metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from motrics._motrics import EvaluationResult
from motrics._motrics import evaluate as _evaluate

# A bounding box: 4 floats, in whichever convention `box_format` specifies.
Bbox = Sequence[float]
# One frame's boxes: a sequence of 4-tuples, or a zero-copy `(N, 4)` float64
# NumPy array (zero-copy only for a contiguous array in `"xyxy"` format).
Boxes = Sequence[Bbox] | npt.NDArray[np.float64]
# `"xyxy"` is `(x1, y1, x2, y2)`; `"xywh"` is `(x, y, width, height)`.
BoxFormat = Literal["xyxy", "xywh"]


@dataclass(frozen=True, slots=True)
class Frames:
    """Frame-aligned object ids and boxes for one side of an evaluation.

    ``ids[t]``/``boxes[t]`` describe the objects detected in frame ``t``;
    within a frame the two lists must have equal length (checked by whichever
    metric function consumes this). Construct one for ground truth and one
    for predictions, then pass both to :func:`evaluate` (or unpack
    ``.ids``/``.boxes`` into :func:`motrics.compute_clear` and friends).
    """

    ids: Sequence[Sequence[int]]
    boxes: Sequence[Boxes]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.boxes):
            raise ValueError(
                "ids and boxes must have the same number of frames, got "
                f"{len(self.ids)} and {len(self.boxes)}"
            )

    @property
    def num_frames(self) -> int:
        """Number of frames."""
        return len(self.ids)

    @property
    def num_dets(self) -> int:
        """Total detections across all frames."""
        return sum(len(f) for f in self.ids)


def evaluate(
    gt: Frames,
    pred: Frames,
    *,
    iou_threshold: float = 0.5,
    box_format: BoxFormat = "xyxy",
) -> EvaluationResult:
    """Compute CLEAR, Identity, and HOTA together for a sequence.

    Builds the gt/pred similarity matrix once and reuses it for all three
    metrics, instead of recomputing it once per metric (what calling
    :func:`motrics.compute_clear`, :func:`motrics.compute_identity`, and
    :func:`motrics.compute_hota` separately would do).
    """
    return _evaluate(gt.ids, gt.boxes, pred.ids, pred.boxes, iou_threshold, box_format)
