"""``Count`` metric: raw detection/id counts, mirroring ``trackeval.metrics.Count``."""

from __future__ import annotations

from typing import Any

from motrics.compat.trackeval.metrics._base_metric import _BaseMetric


class Count(_BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self.integer_fields = ["Dets", "GT_Dets", "IDs", "GT_IDs"]
        self.fields = self.integer_fields
        self.summary_fields = self.fields

    def eval_sequence(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "Dets": data["num_tracker_dets"],
            "GT_Dets": data["num_gt_dets"],
            "IDs": data["num_tracker_ids"],
            "GT_IDs": data["num_gt_ids"],
            "Frames": data["num_timesteps"],
        }

    def combine_sequences(self, all_res: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            field: self._combine_sum(all_res, field) for field in self.integer_fields
        }
