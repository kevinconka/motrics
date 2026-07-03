"""Evaluate a MOTChallenge benchmark the way `TrackEval
<https://github.com/JonathonLuiten/TrackEval>`_'s ``Evaluator`` +
``MotChallenge2DBox`` does, without installing TrackEval::

    import motrics.compat.trackeval as trackeval

    results = trackeval.evaluate_mot_challenge(
        {
            "MOT17-02-FRCNN": ("data/MOT17-02/gt/gt.txt", "results/MOT17-02.txt"),
            "MOT17-04-FRCNN": ("data/MOT17-04/gt/gt.txt", "results/MOT17-04.txt"),
        }
    )
    print(results["COMBINED_SEQ"]["clear"]["mota"])
    print(results["MOT17-02-FRCNN"]["hota"]["hota"])

This is a small, functional subset of TrackEval's ``Evaluator``: one
function, MOTChallenge only, no config dict, dataset-class plugin system, or
file/plot output. Per-sequence ground truth is filtered through
:func:`motrics.preprocess_motchallenge`, so per-sequence CLEAR/Identity/HOTA
match TrackEval's own numbers. ``COMBINED_SEQ`` CLEAR and Identity are exact
re-aggregations from summed counts (as TrackEval's own ``combine_sequences``
does); ``COMBINED_SEQ`` HOTA is a detection-count-weighted average of the
per-sequence scores, since :class:`motrics.HotaMetrics` doesn't expose the
raw per-alpha counts TrackEval's exact combination needs — close enough for
tracking benchmark comparisons, but not bit-exact.
"""

from __future__ import annotations

from motrics.compat.trackeval.evaluate import evaluate_mot_challenge

__all__ = ["evaluate_mot_challenge"]
