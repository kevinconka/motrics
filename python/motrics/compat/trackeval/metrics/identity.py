"""``Identity`` metric (IDF1/IDP/IDR), mirroring ``trackeval.metrics.Identity``.

``eval_sequence`` delegates to :func:`motrics.compute_identity`; this class
only translates field names and combination.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from motrics import compute_identity
from motrics.compat.trackeval._utils import init_config
from motrics.compat.trackeval.metrics._base_metric import _BaseMetric


class Identity(_BaseMetric):
    @staticmethod
    def get_default_config() -> dict[str, Any]:
        return {"THRESHOLD": 0.5, "PRINT_CONFIG": True}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.integer_fields = ["IDTP", "IDFN", "IDFP"]
        self.float_fields = ["IDF1", "IDR", "IDP"]
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.fields
        self.config = init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config["THRESHOLD"])

    def eval_sequence(self, data: dict[str, Any]) -> dict[str, Any]:
        m = compute_identity(
            data["gt_ids"],
            data["gt_dets"],
            data["tracker_ids"],
            data["tracker_dets"],
            self.threshold,
        )
        return self._compute_final_fields(
            {"IDTP": m.idtp, "IDFN": m.idfn, "IDFP": m.idfp}
        )

    def combine_sequences(self, all_res: dict[str, dict[str, Any]]) -> dict[str, Any]:
        res = {
            field: self._combine_sum(all_res, field) for field in self.integer_fields
        }
        return self._compute_final_fields(res)

    @staticmethod
    def _compute_final_fields(res: dict[str, Any]) -> dict[str, Any]:
        res["IDR"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + res["IDFN"])
        res["IDP"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + res["IDFP"])
        res["IDF1"] = res["IDTP"] / np.maximum(
            1.0, res["IDTP"] + 0.5 * res["IDFP"] + 0.5 * res["IDFN"]
        )
        return res
