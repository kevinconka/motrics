"""Shared base for metric classes, mirroring ``trackeval.metrics._base_metric``.

Table printing, file output, and plotting (``print_table``, ``summary_results``,
``detailed_results``, ``plot_single_tracker_results``) are out of scope — they're
CLI-script side effects that don't affect the values :class:`Evaluator.evaluate`
returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class _BaseMetric(ABC):
    def __init__(self) -> None:
        self.plottable = False
        self.integer_fields: list[str] = []
        self.float_fields: list[str] = []
        self.array_labels: np.ndarray = np.array([])
        self.integer_array_fields: list[str] = []
        self.float_array_fields: list[str] = []
        self.fields: list[str] = []
        self.summary_fields: list[str] = []

    @abstractmethod
    def eval_sequence(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute this metric's fields for one sequence."""

    @abstractmethod
    def combine_sequences(self, all_res: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Combine per-sequence results into one (the ``COMBINED_SEQ`` entry)."""

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    @staticmethod
    def _combine_sum(all_res: dict[str, dict[str, Any]], field: str) -> Any:
        return sum(res[field] for res in all_res.values())

    @staticmethod
    def _combine_weighted_av(
        all_res: dict[str, dict[str, Any]],
        field: str,
        comb_res: dict[str, Any],
        weight_field: str,
    ) -> Any:
        numerator = sum(res[field] * res[weight_field] for res in all_res.values())
        return numerator / np.maximum(1.0, comb_res[weight_field])
