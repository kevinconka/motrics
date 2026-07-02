"""Shared dataset discovery and loading for the benchmark suite.

A *dataset* is a directory of sequence sub-directories, each laid out in the
MOTChallenge convention::

    <sequence>/gt/gt.txt     # ground truth
    <sequence>/pred.txt      # tracker results

Both the synthetic fixtures (``benchmarks/data``) and real MOTChallenge
sequences fetched by ``download.py`` (``benchmarks/data/real``) use this layout,
so the benchmark loads either through one code path — the public
``motrics.load_motchallenge`` / ``motrics.align_frames`` readers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import motrics

DATA_DIR = Path(__file__).parent / "data"
REAL_DIR = DATA_DIR / "real"

Bbox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Sequence:
    """One frame-aligned sequence, ready for the metric functions."""

    name: str
    gt_ids: list[list[int]]
    gt_boxes: list[list[Bbox]]
    pred_ids: list[list[int]]
    pred_boxes: list[list[Bbox]]

    @property
    def num_frames(self) -> int:
        return len(self.gt_ids)

    @property
    def num_gt_dets(self) -> int:
        return sum(len(f) for f in self.gt_ids)

    @property
    def num_pred_dets(self) -> int:
        return sum(len(f) for f in self.pred_ids)


def load_sequence(seq_dir: Path, *, min_confidence: float | None = None) -> Sequence:
    """Load one ``<sequence>/{gt/gt.txt,pred.txt}`` directory into a `Sequence`."""
    gt = motrics.load_motchallenge(seq_dir / "gt" / "gt.txt")
    pred = motrics.load_motchallenge(
        seq_dir / "pred.txt", min_confidence=min_confidence
    )
    gt_ids, gt_boxes, pred_ids, pred_boxes = motrics.align_frames(gt, pred)
    return Sequence(seq_dir.name, gt_ids, gt_boxes, pred_ids, pred_boxes)


def discover_sequences(root: Path) -> list[Path]:
    """Return sequence directories under ``root`` (those with ``gt/gt.txt``)."""
    if not root.exists():
        return []
    seqs = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir()
        and child.name != "real"
        and (child / "gt" / "gt.txt").is_file()
        and (child / "pred.txt").is_file()
    ]
    return seqs


def default_root() -> tuple[Path, str]:
    """Pick the dataset to benchmark: real sequences if present, else synthetic.

    Returns ``(root, kind)`` where ``kind`` is ``"real"`` or ``"synthetic"``.
    """
    if discover_sequences(REAL_DIR):
        return REAL_DIR, "real"
    return DATA_DIR, "synthetic"


def ensure_fixtures() -> None:
    """Generate the synthetic fixtures if absent (they are generated, not committed)."""
    if not discover_sequences(DATA_DIR):
        import generate_fixtures

        generate_fixtures.main()


def load_dataset(
    root: Path | None = None, *, min_confidence: float | None = None
) -> tuple[list[Sequence], str]:
    """Load every sequence under ``root`` (or the auto-selected default)."""
    if root is None:
        root, kind = default_root()
        if kind == "synthetic":
            ensure_fixtures()
    else:
        kind = "real" if root.resolve() == REAL_DIR.resolve() else "synthetic"
    sequences = [
        load_sequence(seq_dir, min_confidence=min_confidence)
        for seq_dir in discover_sequences(root)
    ]
    return sequences, kind
