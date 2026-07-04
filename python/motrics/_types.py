"""Shared type aliases used across motrics' Python ingest modules."""

from __future__ import annotations

#: A bounding box in xyxy format: (x1, y1, x2, y2).
Bbox = tuple[float, float, float, float]
