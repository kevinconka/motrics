"""Drop-in for `TrackEval <https://github.com/JonathonLuiten/TrackEval>`_'s
MOTChallenge evaluation API, backed by motrics' Rust core::

    import motrics.compat.trackeval as trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    evaluator = trackeval.Evaluator(eval_config)

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = "data/gt/mot_challenge/"
    dataset_config["TRACKERS_FOLDER"] = "data/trackers/mot_challenge/"
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    results, messages = evaluator.evaluate(dataset_list, metrics_list)
    print(
        results["MotChallenge2DBox"]["my_tracker"]["COMBINED_SEQ"]["pedestrian"][
            "CLEAR"
        ]["MOTA"]
    )

Same class names, config keys, directory/seqmap conventions, and result shape
(``results[dataset][tracker][seq_or_"COMBINED_SEQ"][cls][metric]["FIELD"]``) as
real TrackEval, no ``trackeval``/``scipy`` install required — but only for the
subset described below; an evaluation script using unsupported config or
metric fields will raise rather than silently return a wrong number.

What's NOT implemented (see each module's docstring for specifics):

- ``Evaluator``: parallel evaluation, error-handling config
  (``BREAK_ON_ERROR``/etc.), printing/file output/plotting. Always serial,
  always raises immediately, never writes files.
- ``datasets.MotChallenge2DBox``: zipped input (``INPUT_AS_ZIP``), classes
  other than ``pedestrian`` (TrackEval's own MOT Challenge adapter is
  pedestrian-only too, so this isn't a gap versus TrackEval itself),
  ``DO_PREPROC=False``, and ``BENCHMARK="MOT15"`` (raises at construction).
- ``metrics.CLEAR``: only ``MOTA``/``MOTP`` — not ``MT``/``PT``/``ML``/``Frag``/
  ``MODA``/``sMOTA``/etc., which need mostly-tracked/lost and fragmentation
  bookkeeping the Rust core doesn't compute yet.
- ``metrics.{IDEucl,JAndF,TrackMAP,VACE}``: not implemented at all.
"""

from __future__ import annotations

from motrics.compat.trackeval import datasets, metrics
from motrics.compat.trackeval._utils import TrackEvalException
from motrics.compat.trackeval.eval import Evaluator

__all__ = ["Evaluator", "TrackEvalException", "datasets", "metrics"]
