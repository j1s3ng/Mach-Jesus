"""Deterministic geometric candidate detector, with NO semantic classification.

This baseline removes a flat ground estimate, voxelizes, finds connected
components, and fits PCA-oriented boxes. Scores are an uncalibrated heuristic
of point support and shape; they are not probabilities. Flat-ground assumptions,
occlusion and closely spaced objects can cause severe errors.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .io import validate_points


def detect(points: np.ndarray) -> list[dict]:
    """Detect vehicle-sized geometric candidates from Nx4 sensor points only."""
    cloud = validate_points(points)
    if len(cloud) < 7:
        return []
    xyz = cloud[:, :3].astype(np.float64)
    roi = ((xyz[:, 0] > 0.5) & (xyz[:, 0] < 90) &
           (np.abs(xyz[:, 1]) < 35) & (xyz[:, 2] > -5) & (xyz[:, 2] < 5))
    xyz = xyz[roi]
    if len(xyz) < 7:
        return []
    # The low elevation quantile needs road returns. With no ground points,
    # this estimate can cut off object bottoms: an explicit baseline limitation.
    ground_z = float(np.percentile(xyz[:, 2], 10))
    xyz = xyz[xyz[:, 2] > ground_z + 0.22]
    if len(xyz) < 7:
        return []
    _, indices, counts = np.unique(np.floor(xyz / 0.18).astype(np.int64), axis=0,
                                   return_index=True, return_counts=True)
    xyz = xyz[indices]
    # Cap candidate workload on unexpectedly dense recordings, deterministically.
    if len(xyz) > 60000:
        stride = int(np.ceil(len(xyz) / 60000))
        xyz, counts = xyz[::stride], counts[::stride]
    tree = cKDTree(xyz)
    visited = np.zeros(len(xyz), dtype=bool)
    boxes = []
    for start in range(len(xyz)):
        if visited[start]:
            continue
        component, pending = [], [start]
        visited[start] = True
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in tree.query_ball_point(xyz[current], r=0.85):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    pending.append(neighbor)
        if len(component) < 7:
            continue
        cluster = xyz[component]
        center_xy = cluster[:, :2].mean(axis=0)
        covariance = np.cov(cluster[:, :2] - center_xy, rowvar=False)
        _, axes = np.linalg.eigh(covariance)
        major = axes[:, -1]
        yaw = float(np.arctan2(major[1], major[0]))
        # Canonical orientation modulo pi avoids nondeterministic eigenvector sign.
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        local = (cluster[:, :2] - center_xy) @ rotation
        lo, hi = local.min(axis=0), local.max(axis=0)
        size_xy = hi - lo + 0.22
        height = float(np.ptp(cluster[:, 2]) + 0.15)
        # A single visible rear/side face is almost planar. Retain these thin
        # partial boxes instead of inventing an unseen vehicle length.
        if not (0.65 <= size_xy[0] <= 9 and 0.25 <= size_xy[1] <= 4.5 and 0.35 <= height <= 4):
            continue
        center = np.r_[center_xy + ((lo + hi) / 2) @ rotation.T,
                       (cluster[:, 2].min() + cluster[:, 2].max()) / 2]
        support = int(counts[component].sum())
        score = float(np.clip(0.25 + 0.55 * (1 - np.exp(-support / 45.0)) +
                              0.15 * min(size_xy[0] / 3.5, 1), 0, 0.95))
        boxes.append(dict(center=center.tolist(), size=[*size_xy.tolist(), height], yaw=yaw,
                          score=score, score_type="uncalibrated_geometric_heuristic",
                          label="vehicle_sized_candidate", num_points=support))
    return sorted(boxes, key=lambda box: (-box["score"], box["center"][0], box["center"][1]))
