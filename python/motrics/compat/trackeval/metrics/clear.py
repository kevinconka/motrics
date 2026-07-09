"""``CLEAR`` metric (MOTA/MOTP + MT/PT/ML/Frag), mirroring
``trackeval.metrics.CLEAR``.

``eval_sequence`` delegates the per-frame matching to
:func:`motrics.compute_clear` (which now also returns MT/PT/ML/Frag and the
derived MODA/sMOTA/MOTAL scores); this class only translates field names and
reproduces TrackEval's empty-sequence short-circuits and final-field formulas.
"""

from __future__ import annotations

import math
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
        main_integer_fields = [
            "CLR_TP",
            "CLR_FN",
            "CLR_FP",
            "IDSW",
            "MT",
            "PT",
            "ML",
            "Frag",
        ]
        main_float_fields = [
            "MOTA",
            "MOTP",
            "MODA",
            "CLR_Re",
            "CLR_Pr",
            "MTR",
            "PTR",
            "MLR",
            "sMOTA",
        ]
        self.integer_fields = [*main_integer_fields, "CLR_Frames"]
        self.float_fields = [
            *main_float_fields,
            "CLR_F1",
            "FP_per_frame",
            "MOTAL",
            "MOTP_sum",
        ]
        self.fields = self.float_fields + self.integer_fields
        self.summed_fields = [*self.integer_fields, "MOTP_sum"]
        self.summary_fields = main_float_fields + main_integer_fields
        self.config = init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config["THRESHOLD"])

    def eval_sequence(self, data: dict[str, Any]) -> dict[str, Any]:
        res: dict[str, Any] = dict.fromkeys(self.fields, 0)

        # Short-circuits matching TrackEval, which return before computing the
        # derived fields (so they stay 0, MLR aside).
        if data["num_tracker_dets"] == 0:
            res["CLR_FN"] = data["num_gt_dets"]
            res["ML"] = data["num_gt_ids"]
            res["MLR"] = 1.0
            return res
        if data["num_gt_dets"] == 0:
            res["CLR_FP"] = data["num_tracker_dets"]
            res["MLR"] = 1.0
            return res

        m = compute_clear(
            data["gt_ids"],
            data["gt_dets"],
            data["tracker_ids"],
            data["tracker_dets"],
            self.threshold,
        )
        res.update(
            CLR_TP=m.num_matches,
            CLR_FN=m.num_misses,
            CLR_FP=m.num_false_positives,
            IDSW=m.num_switches,
            MT=m.mt,
            PT=m.pt,
            ML=m.ml,
            Frag=m.frag,
            CLR_Frames=m.num_frames,
            MOTP_sum=m.motp * m.num_matches,
        )
        return self._compute_final_fields(res)

    def combine_sequences(self, all_res: dict[str, dict[str, Any]]) -> dict[str, Any]:
        res = {field: self._combine_sum(all_res, field) for field in self.summed_fields}
        return self._compute_final_fields(res)

    @staticmethod
    def _compute_final_fields(res: dict[str, Any]) -> dict[str, Any]:
        num_gt_ids = res["MT"] + res["ML"] + res["PT"]
        res["MTR"] = res["MT"] / max(1.0, num_gt_ids)
        res["MLR"] = res["ML"] / max(1.0, num_gt_ids)
        res["PTR"] = res["PT"] / max(1.0, num_gt_ids)
        res["CLR_Re"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + res["CLR_FN"])
        res["CLR_Pr"] = res["CLR_TP"] / max(1.0, res["CLR_TP"] + res["CLR_FP"])
        res["MODA"] = (res["CLR_TP"] - res["CLR_FP"]) / max(
            1.0, res["CLR_TP"] + res["CLR_FN"]
        )
        res["MOTA"] = (res["CLR_TP"] - res["CLR_FP"] - res["IDSW"]) / max(
            1.0, res["CLR_TP"] + res["CLR_FN"]
        )
        res["MOTP"] = res["MOTP_sum"] / max(1.0, res["CLR_TP"])
        res["sMOTA"] = (res["MOTP_sum"] - res["CLR_FP"] - res["IDSW"]) / max(
            1.0, res["CLR_TP"] + res["CLR_FN"]
        )
        res["CLR_F1"] = res["CLR_TP"] / max(
            1.0, res["CLR_TP"] + 0.5 * res["CLR_FN"] + 0.5 * res["CLR_FP"]
        )
        res["FP_per_frame"] = res["CLR_FP"] / max(1.0, res["CLR_Frames"])
        safe_log_idsw = math.log10(res["IDSW"]) if res["IDSW"] > 0 else res["IDSW"]
        res["MOTAL"] = (res["CLR_TP"] - res["CLR_FP"] - safe_log_idsw) / max(
            1.0, res["CLR_TP"] + res["CLR_FN"]
        )
        return res
