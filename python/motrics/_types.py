"""Shared type aliases used across motrics' Python ingest modules."""

from __future__ import annotations

#: A bounding box in xyxy format: (x1, y1, x2, y2).
Bbox = tuple[float, float, float, float]

#: An oriented 3D box: (x, y, z, l, w, h, yaw) -- centre, full extents, and
#: heading in radians about the vertical y axis (KITTI/AB3DMOT convention).
Box3d = tuple[float, float, float, float, float, float, float]
