"""``Evaluator``, mirroring ``trackeval.eval.Evaluator``.

Only the serial evaluation path is implemented: no ``USE_PARALLEL``
(multiprocessing), no ``BREAK_ON_ERROR``/``RETURN_ON_ERROR``/``LOG_ON_ERROR``
(errors always propagate immediately, i.e. ``BREAK_ON_ERROR=True``), no
``show_progressbar``, and no printing/file output/plotting
(``PRINT_RESULTS``/``OUTPUT_SUMMARY``/``OUTPUT_DETAILED``/``PLOT_CURVES``) —
those config keys are accepted (so existing config dicts don't raise
``KeyError``) but silently have no effect. Class-combination
(``should_classes_combine``/``use_super_categories``) isn't implemented either
since :class:`~motrics.compat.trackeval.datasets.MotChallenge2DBox` never sets
them ``True`` (matching TrackEval's own pedestrian-only MOT Challenge
behaviour).
"""

from __future__ import annotations

from typing import Any

from motrics.compat.trackeval._utils import init_config, validate_metrics_list
from motrics.compat.trackeval.metrics.count import Count


class Evaluator:
    @staticmethod
    def get_default_eval_config() -> dict[str, Any]:
        return {
            "USE_PARALLEL": False,
            "NUM_PARALLEL_CORES": 8,
            "BREAK_ON_ERROR": True,
            "RETURN_ON_ERROR": False,
            "LOG_ON_ERROR": None,
            "PRINT_RESULTS": True,
            "PRINT_ONLY_COMBINED": False,
            "PRINT_CONFIG": True,
            "TIME_PROGRESS": True,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_EMPTY_CLASSES": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": True,
        }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = init_config(config, self.get_default_eval_config(), "Eval")

    def evaluate(
        self,
        dataset_list: list[Any],
        metrics_list: list[Any],
        show_progressbar: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluate every tracker in every dataset against every metric.

        Returns ``(output_res, output_msg)`` — ``output_res[dataset_name][tracker]``
        is ``{seq: {cls: {metric_name: {field: value}}}, "COMBINED_SEQ": {...}}``,
        and ``output_msg[dataset_name][tracker]`` is ``"Success"`` (errors
        propagate as exceptions rather than being recorded here).
        """
        metrics_list = [*metrics_list, Count()]
        metric_names = validate_metrics_list(metrics_list)
        output_res: dict[str, Any] = {}
        output_msg: dict[str, Any] = {}

        for dataset in dataset_list:
            dataset_name = dataset.get_name()
            output_res[dataset_name] = {}
            output_msg[dataset_name] = {}
            tracker_list, seq_list, class_list = dataset.get_eval_info()

            for tracker in tracker_list:
                res = {
                    seq: _eval_sequence(
                        seq, dataset, tracker, class_list, metrics_list, metric_names
                    )
                    for seq in sorted(seq_list)
                }
                res["COMBINED_SEQ"] = {
                    cls: {
                        metric_name: metric.combine_sequences(
                            {
                                seq: seq_res[cls][metric_name]
                                for seq, seq_res in res.items()
                            }
                        )
                        for metric, metric_name in zip(
                            metrics_list, metric_names, strict=True
                        )
                    }
                    for cls in class_list
                }
                output_res[dataset_name][tracker] = res
                output_msg[dataset_name][tracker] = "Success"

        return output_res, output_msg


def _eval_sequence(
    seq: str,
    dataset: Any,
    tracker: str,
    class_list: list[str],
    metrics_list: list[Any],
    metric_names: list[str],
) -> dict[str, Any]:
    raw_data = dataset.get_raw_seq_data(tracker, seq)
    seq_res = {}
    for cls in class_list:
        data = dataset.get_preprocessed_seq_data(raw_data, cls)
        seq_res[cls] = {
            metric_name: metric.eval_sequence(data)
            for metric, metric_name in zip(metrics_list, metric_names, strict=True)
        }
    return seq_res
