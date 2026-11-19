"""Closed road geometry from public approximate position samples.

The ribbon is an approximate corridor around the reference trajectory. It is
not a survey of the actual circuit boundaries or exact F1 lateral placement.
Coordinates are never translated or scaled by this module.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


def closed_geometry(xy, spacing=3.0, half_width=10.0):
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 12 or not np.isfinite(points).all():
        raise ValueError("Expected at least 12 finite XY reference positions")
    if spacing <= 0 or half_width < 3:
        raise ValueError("spacing must be positive and road half-width at least 3 m")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > .05]
    points = points[keep]
    if np.linalg.norm(points[-1] - points[0]) < .05:
        points = points[:-1]
    gap = float(np.linalg.norm(points[-1] - points[0]))
    segments = np.linalg.norm(np.diff(np.vstack([points, points[0]]), axis=0), axis=1)
    if gap > max(50., 4 * float(np.median(segments))):
        raise ValueError(f"Reference is not a complete closed lap: endpoint gap {gap:.1f} m")
    distance = np.r_[0., np.cumsum(segments)]
    length = float(distance[-1])
    s = np.linspace(0, length, max(24, int(np.ceil(length / spacing))), endpoint=False)
    curve = CubicSpline(distance, np.vstack([points, points[0]]), axis=0, bc_type="periodic")
    reference = curve(s)
    # Light periodic smoothing removes broadcast position quantization. No claim
    # is made that this reconstructs the real surveyed road center or edges.
    step = length / len(s)
    center = gaussian_filter1d(reference, sigma=max(.5, 4. / step), axis=0, mode="wrap")
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    curvature = (np.roll(np.unwrap(heading), -1) - np.roll(np.unwrap(heading), 1)) / (2 * step)
    # Use wrapped heading differences at the closure seam.
    delta = np.roll(heading, -1) - np.roll(heading, 1)
    curvature = np.arctan2(np.sin(delta), np.cos(delta)) / (2 * step)
    widths = np.maximum(half_width, np.linalg.norm(reference - center, axis=1) + 2.5)
    return dict(s_m=s, reference_xy=reference, center_xy=center, normal_xy=normal,
                heading_rad=heading, curvature_per_m=curvature, half_width_m=widths,
                length_m=length, source_endpoint_gap_m=gap)


def load_geometry(path, spacing=3.0, half_width=10.0):
    with np.load(path, allow_pickle=False) as reference:
        return closed_geometry(np.column_stack([reference["x_m"], reference["y_m"]]), spacing, half_width)


def signed_face_areas(left, right):
    """XY signed doubled areas for upward-facing road ribbon triangles."""
    left, right = np.asarray(left), np.asarray(right)
    a = right - left
    b = np.roll(right, -1, axis=0) - left
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
