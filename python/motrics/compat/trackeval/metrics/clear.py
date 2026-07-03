"""``CLEAR`` metric (MOTA/MOTP), mirroring ``trackeval.metrics.CLEAR``.

Only the MOTA/MOTP subset of TrackEval's real field set is implemented —
``MT``/``PT``/``ML``/``Frag``/``MODA``/``sMOTA``/``CLR_Re``/``CLR_Pr``/``MOTAL``/
``CLR_F1``/``FP_per_frame``/``MTR``/``PTR``/``MLR`` need mostly-tracked/lost and
fragmentation bookkeeping the Rust core doesn't compute yet. ``eval_sequence``
delegates to :func:`motrics.compute_clear` (independently parity-tested
against TrackEval); this class only translates field names and combination.
"""

from __future__ import annotations

from typing import Any

from motrics import compute_clear
from motrics.compat.trackeval._utils import init_config
from motrics.compat.trackeval.metrics._base_metric import _BaseMetric


class CLEAR(_BaseMetric):
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        return {"THRESHOLD": 0.5, "PRINT_CONFIG": True}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.integer_fields = ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "CLR_Frames"]
        self.float_fields = ["MOTA", "MOTP"]
        self.fields = self.float_fields + self.integer_fields
        self.summed_fields = [*self.integer_fields, "MOTP_sum"]
        self.summary_fields = self.fields
        self.config = init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config["THRESHOLD"])

    def eval_sequence(self, data: dict[str, Any]) -> dict[str, Any]:
        m = compute_clear(
            data["gt_ids"],
            data["gt_dets"],
            data["tracker_ids"],
            data["tracker_dets"],
            self.threshold,
        )
        res = {
            "CLR_TP": m.num_matches,
            "CLR_FN": m.num_misses,
            "CLR_FP": m.num_false_positives,
            "IDSW": m.num_switches,
            "CLR_Frames": m.num_frames,
            "MOTP_sum": m.motp * m.num_matches,
        }
        return self._compute_final_fields(res)

    def combine_sequences(self, all_res: dict[str, dict[str, Any]]) -> dict[str, Any]:
        res = {field: self._combine_sum(all_res, field) for field in self.summed_fields}
        return self._compute_final_fields(res)

    @staticmethod
    def _compute_final_fields(res: dict[str, Any]) -> dict[str, Any]:
        res["MOTA"] = (res["CLR_TP"] - res["CLR_FP"] - res["IDSW"]) / max(
            1.0, res["CLR_TP"] + res["CLR_FN"]
        )
        res["MOTP"] = res["MOTP_sum"] / max(1.0, res["CLR_TP"])
        return res
