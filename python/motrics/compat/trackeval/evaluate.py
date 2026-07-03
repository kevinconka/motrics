"""Implementation of :func:`evaluate_mot_challenge`."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

from motrics import (
    ClearMetrics,
    HotaMetrics,
    IdentityMetrics,
    compute_clear,
    compute_hota,
    compute_identity,
    load_motchallenge,
    load_motchallenge_gt,
    preprocess_motchallenge,
)

PathType = str | PathLike[str]
MetricDict = dict[str, Any]
SeqResult = dict[str, MetricDict]


def evaluate_mot_challenge(
    sequences: Mapping[str, tuple[PathType, PathType]],
    *,
    benchmark: str = "MOT17",
    iou_threshold: float = 0.5,
) -> dict[str, SeqResult]:
    """Evaluate CLEAR, Identity, and HOTA over a MOTChallenge benchmark.

    Args:
        sequences: Mapping of sequence name to ``(gt_path, pred_path)``,
            ``gt_path`` a MOTChallenge ``gt.txt`` and ``pred_path`` a tracker
            results file in the same format.
        benchmark: ``"MOT20"`` additionally treats ``non_mot_vehicle`` as a
            distractor class; every other benchmark uses the same set.
        iou_threshold: Threshold for both the distractor-match preprocessing
            step and the CLEAR/Identity assignment.

    Returns:
        Mapping of sequence name to ``{"clear": {...}, "identity": {...},
        "hota": {...}}`` (field names match :class:`motrics.ClearMetrics` /
        :class:`motrics.IdentityMetrics` / :class:`motrics.HotaMetrics`),
        plus a ``"COMBINED_SEQ"`` entry aggregated across all sequences (see
        module docstring for how HOTA combination differs from TrackEval's
        exact one).
    """
    results: dict[str, SeqResult] = {}
    for name, (gt_path, pred_path) in sequences.items():
        gt = load_motchallenge_gt(gt_path)
        pred = load_motchallenge(pred_path)
        args = preprocess_motchallenge(
            gt, pred, iou_threshold=iou_threshold, benchmark=benchmark
        )
        results[name] = {
            "clear": _clear_dict(compute_clear(*args, iou_threshold)),
            "identity": _identity_dict(compute_identity(*args, iou_threshold)),
            "hota": _hota_dict(compute_hota(*args)),
        }
    results["COMBINED_SEQ"] = _combine(results)
    return results


def _clear_dict(c: ClearMetrics) -> MetricDict:
    return {
        "mota": c.mota,
        "motp": c.motp,
        "num_frames": c.num_frames,
        "num_gt": c.num_gt,
        "num_matches": c.num_matches,
        "num_false_positives": c.num_false_positives,
        "num_misses": c.num_misses,
        "num_switches": c.num_switches,
    }


def _identity_dict(i: IdentityMetrics) -> MetricDict:
    return {
        "idf1": i.idf1,
        "idp": i.idp,
        "idr": i.idr,
        "idtp": i.idtp,
        "idfp": i.idfp,
        "idfn": i.idfn,
    }


def _hota_dict(h: HotaMetrics) -> MetricDict:
    return {"hota": h.hota, "deta": h.deta, "assa": h.assa, "loca": h.loca}


def _combine(per_seq: dict[str, SeqResult]) -> SeqResult:
    clears = [r["clear"] for r in per_seq.values()]
    idents = [r["identity"] for r in per_seq.values()]
    hotas = [r["hota"] for r in per_seq.values()]

    num_gt = sum(c["num_gt"] for c in clears)
    num_fp = sum(c["num_false_positives"] for c in clears)
    num_fn = sum(c["num_misses"] for c in clears)
    num_sw = sum(c["num_switches"] for c in clears)
    num_matches = sum(c["num_matches"] for c in clears)
    combined_clear = {
        "mota": 1.0 - (num_fn + num_fp + num_sw) / num_gt if num_gt else float("nan"),
        "motp": (
            sum(c["motp"] * c["num_matches"] for c in clears) / num_matches
            if num_matches
            else float("nan")
        ),
        "num_gt": num_gt,
        "num_matches": num_matches,
        "num_false_positives": num_fp,
        "num_misses": num_fn,
        "num_switches": num_sw,
    }

    idtp = sum(i["idtp"] for i in idents)
    idfp = sum(i["idfp"] for i in idents)
    idfn = sum(i["idfn"] for i in idents)
    combined_identity = {
        "idf1": _safe_div(idtp, idtp + 0.5 * idfp + 0.5 * idfn),
        "idp": _safe_div(idtp, idtp + idfp),
        "idr": _safe_div(idtp, idtp + idfn),
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
    }

    weights = [c["num_gt"] for c in clears]
    total_weight = sum(weights)

    def _wavg(values: list[float]) -> float:
        if not total_weight:
            return float("nan")
        return sum(v * w for v, w in zip(values, weights, strict=True)) / total_weight

    combined_hota = {
        "hota": _wavg([h["hota"] for h in hotas]),
        "deta": _wavg([h["deta"] for h in hotas]),
        "assa": _wavg([h["assa"] for h in hotas]),
        "loca": _wavg([h["loca"] for h in hotas]),
    }

    return {
        "clear": combined_clear,
        "identity": combined_identity,
        "hota": combined_hota,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")
