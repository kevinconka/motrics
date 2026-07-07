"""DAVIS (unsupervised) tracking ingest helpers.

Parse the DAVIS annotation format: one indexed-palette PNG per frame in a
per-sequence directory, sorted by filename. A pixel's value is the object id
it belongs to; ``0`` is background and ``255`` is a "void" region (unlabelled/
ambiguous pixels). Ground truth and tracker results use the identical format,
so a single :func:`load_davis` reads both.

Two functions, mirroring the other dataset adapters:

- ``load_davis`` — parse a sequence directory into per-frame ids/masks.
- ``preprocess_davis`` — pair a ground-truth and a tracker sequence into the
  ``(gt_ids, pred_ids, similarity)`` triple the ``compute_*_from_similarity``
  functions take, matching TrackEval's ``DAVIS`` numbers.

**On void regions.** TrackEval's ``DAVIS`` *appears* to drop void pixels, but
does not affect its scores by doing so: it computes the gt/tracker similarity
first, and only then zeroes void pixels out of the tracker masks — which the
metrics never re-read (they consume the already-computed similarity). Void is
also a reserved pixel value (``255``), never an object id, so it never enters
an object mask to begin with. The metric numbers are therefore identical
whether or not void is subtracted, and this adapter — faithful to TrackEval —
scores on the raw per-object masks.

Reading the indexed PNGs needs a PNG decoder; like TrackEval, this imports
Pillow lazily, so ``motrics``' core stays dependency-free for callers that do
not use this adapter.
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

import numpy as np

from motrics._motrics import Mask, mask_encode, mask_iou_matrix

if TYPE_CHECKING:
    import numpy.typing as npt

# Per-frame detections, ordered by frame: each entry is ``(ids, masks)``.
_ByFrame = list[tuple[list[int], list[Mask]]]

#: Reserved pixel value marking a "void"/ignore region rather than an object.
_VOID_ID = 255


def _read_indexed_pngs(seq_dir: str | PathLike[str]) -> list[npt.NDArray[np.integer]]:
    """Read a sequence directory's frames as ``(h, w)`` index arrays, sorted by
    filename."""
    from os import listdir
    from os.path import join

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - trivial guard
        raise ModuleNotFoundError(
            "reading DAVIS PNG masks requires Pillow; install it with "
            "`pip install pillow`"
        ) from exc

    seq_dir = str(seq_dir)
    names = sorted(n for n in listdir(seq_dir) if n.lower().endswith(".png"))
    frames = []
    for name in names:
        frame = np.array(Image.open(join(seq_dir, name)))
        if frame.ndim != 2:
            raise ValueError(
                f"{name}: expected an indexed (single-channel) PNG mask, got an "
                f"array with shape {frame.shape}; a pixel value is the object id"
            )
        frames.append(frame)
    return frames


def _split_frame(frame: npt.NDArray[np.integer]) -> tuple[list[int], list[Mask]]:
    """Split one index array into per-object ``(ids, masks)``, background and
    void excluded."""
    ids = [int(v) for v in np.unique(frame) if v not in (0, _VOID_ID)]
    masks = [mask_encode((frame == i).astype(np.uint8)) for i in ids]
    return ids, masks


def load_davis(seq_dir: str | PathLike[str]) -> _ByFrame:
    """Parse a DAVIS sequence directory (ground truth or tracker result) into
    per-frame detections.

    Args:
        seq_dir: Directory of indexed-PNG masks, one per frame; frames are
            ordered by filename.

    Returns:
        A list ordered by frame; each entry is ``(ids, masks)`` for that frame.
    """
    return [_split_frame(frame) for frame in _read_indexed_pngs(seq_dir)]


def preprocess_davis(
    gt: _ByFrame,
    pred: _ByFrame,
) -> tuple[list[list[int]], list[list[int]], list[list[list[float]]]]:
    """Pair a ground-truth and a tracker sequence exactly as TrackEval's
    ``DAVIS`` does before scoring.

    DAVIS has a single "general" class and removes no detections, so this is
    purely the per-frame mask IoU between ground truth and tracker (see the
    module docstring on why void regions do not change the numbers).

    Args:
        gt: Ground truth from :func:`load_davis`.
        pred: Tracker result from :func:`load_davis`. Must have the same number
            of frames as ``gt``.

    Returns:
        ``(gt_ids, pred_ids, similarity)`` ready for
        ``compute_clear_from_similarity`` / ``compute_identity_from_similarity``
        / ``compute_hota_from_similarity``: mask similarity has no core
        ``compute_*`` overload of its own, only the precomputed-similarity one.
        ``similarity[t]`` is the mask IoU matrix between ``gt_ids[t]`` and
        ``pred_ids[t]``.
    """
    if len(gt) != len(pred):
        raise ValueError(
            f"frame count mismatch: gt has {len(gt)}, pred has {len(pred)}"
        )

    gt_ids: list[list[int]] = []
    pred_ids: list[list[int]] = []
    similarity: list[list[list[float]]] = []
    for (g_ids, g_masks), (p_ids, p_masks) in zip(gt, pred, strict=True):
        gt_ids.append(list(g_ids))
        pred_ids.append(list(p_ids))
        similarity.append(mask_iou_matrix(g_masks, p_masks))
    return gt_ids, pred_ids, similarity
