"""``HOTA`` metric, mirroring ``trackeval.metrics.HOTA``.

``eval_sequence`` delegates to :func:`motrics.compute_hota`, whose per-alpha
raw counts (``hota_tp_alphas``/``hota_fn_alphas``/``hota_fp_alphas``,
``ass_re_alphas``/``ass_pr_alphas``, ``loca_alphas``) are verified bit-exact
against TrackEval's own ``HOTA.eval_sequence``; this class only translates
field names and combination.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from motrics import compute_hota
from motrics.compat.trackeval.metrics._base_metric import _BaseMetric


class HOTA(_BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_array_fields = ["HOTA_TP", "HOTA_FN", "HOTA_FP"]
        self.float_array_fields = [
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
            "OWTA",
        ]
        self.float_fields = ["HOTA(0)", "LocA(0)", "HOTALocA(0)"]
        self.fields = (
            self.float_array_fields + self.integer_array_fields + self.float_fields
        )
        self.summary_fields = self.float_array_fields + self.float_fields

    def eval_sequence(self, data: dict[str, Any]) -> dict[str, Any]:
        m = compute_hota(
            data["gt_ids"], data["gt_dets"], data["tracker_ids"], data["tracker_dets"]
        )
        res = {
            "HOTA_TP": np.array(m.hota_tp_alphas),
            "HOTA_FN": np.array(m.hota_fn_alphas),
            "HOTA_FP": np.array(m.hota_fp_alphas),
            "AssA": np.array(m.assa_alphas),
            "AssRe": np.array(m.ass_re_alphas),
            "AssPr": np.array(m.ass_pr_alphas),
            "LocA": np.array(m.loca_alphas),
        }
        return self._compute_final_fields(res)

    def combine_sequences(self, all_res: dict[str, dict[str, Any]]) -> dict[str, Any]:
        res = {
            field: self._combine_sum(all_res, field)
            for field in self.integer_array_fields
        }
        for field in ["AssRe", "AssPr", "AssA"]:
            res[field] = self._combine_weighted_av(
                all_res, field, res, weight_field="HOTA_TP"
            )
        loca_weighted_sum = sum(
            all_res[k]["LocA"] * all_res[k]["HOTA_TP"] for k in all_res
        )
        res["LocA"] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(
            1e-10, res["HOTA_TP"]
        )
        return self._compute_final_fields(res)

    @staticmethod
    def _compute_final_fields(res: dict[str, Any]) -> dict[str, Any]:
        res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
        res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
        res["DetA"] = res["HOTA_TP"] / np.maximum(
            1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"]
        )
        res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
        res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
        res["HOTA(0)"] = res["HOTA"][0]
        res["LocA(0)"] = res["LocA"][0]
        res["HOTALocA(0)"] = res["HOTA(0)"] * res["LocA(0)"]
        return res
