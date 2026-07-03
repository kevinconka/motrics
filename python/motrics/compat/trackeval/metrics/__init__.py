"""Metric classes mirroring ``trackeval.metrics`` — ``Count``, ``Identity``,
``CLEAR``, ``HOTA``. See each class's docstring for what's supported."""

from __future__ import annotations

from motrics.compat.trackeval.metrics.clear import CLEAR
from motrics.compat.trackeval.metrics.count import Count
from motrics.compat.trackeval.metrics.hota import HOTA
from motrics.compat.trackeval.metrics.identity import Identity

__all__ = ["CLEAR", "HOTA", "Count", "Identity"]
