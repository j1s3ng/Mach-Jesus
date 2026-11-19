"""Unclassified emergency obstacles from measured LiDAR geometry only.

Points must already be level ego x-forward/y-left/z-up coordinates, with z
expressed above the world's ground datum, as produced by the racer controller.
This channel complements learned detection. It does not identify cars or infer
that space hidden behind a return is empty, and its support strength is not a
learned or calibrated object probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .hotlap_driver import project_near


@dataclass
class GeometryTrack:
    identifier: int
    center: np.ndarray
    surface: np.ndarray
    velocity: np.ndarray
    timestamp: float
    lateral: float
    support: int
    voxel_count: int
    height_min: float
    height_max: float
    lateral_min: float
    lateral_max: float
    longitudinal_span: float
    strength: float
    hits: int = 1
    velocity_hits: int = 0
    speed_uncertainty: float = 45.


class LidarHazardChannel:
    def __init__(self, reference, *, max_range_m=100., min_height_m=.30,
                 max_height_m=3., corridor_half_width_m=6., rear_range_m=15.,
                 voxel_size_m=.25, cluster_radius_m=1.2, minimum_points=3,
                 max_age_s=.6, ground_height_m=0.):
        values = [max_range_m, min_height_m, max_height_m, corridor_half_width_m,
                  rear_range_m, voxel_size_m, cluster_radius_m, max_age_s, ground_height_m]
        if (not np.isfinite(values).all() or min(values[:-1]) <= 0.
                or max_height_m <= min_height_m or minimum_points < 3
                or int(minimum_points) != minimum_points):
            raise ValueError("Invalid raw LiDAR geometry parameters")
        self.reference = reference
        self.max_range_m = float(max_range_m)
        self.min_height_m, self.max_height_m = float(min_height_m), float(max_height_m)
        self.corridor_half_width_m, self.rear_range_m = float(corridor_half_width_m), float(rear_range_m)
        self.voxel_size_m, self.cluster_radius_m = float(voxel_size_m), float(cluster_radius_m)
        self.minimum_points, self.max_age_s = int(minimum_points), float(max_age_s)
        self.ground_height_m = float(ground_height_m)
        self.tracks = []
        self.next_identifier = 1
        self.last_timestamp = None
        self.last_scan = {}

    def _project(self, xy, ego_progress):
        """Exact projection onto the nearest few local ordered map segments."""
        reference = self.reference
        count = len(reference.xy)
        base = int(np.searchsorted(reference.s, ego_progress % reference.length, side="right") - 1)
        base += math.floor(ego_progress / reference.length) * count
        radius = int(math.ceil((self.max_range_m + 15.) / np.min(reference.ds))) + 3
        extended = np.arange(base - radius, base + radius + 1)
        indices = extended % count
        centers = reference.xy[indices] + .5 * reference.segment[indices]
        nearest = cKDTree(centers).query(xy, k=min(4, len(centers)))[1]
        nearest = np.asarray(nearest).reshape(len(xy), -1)
        segment_indices = indices[nearest]
        delta = xy[:, None, :] - reference.xy[segment_indices]
        segments = reference.segment[segment_indices]
        fraction = np.clip(np.sum(delta * segments, axis=2) / reference.ds[segment_indices] ** 2, 0., 1.)
        projected = reference.xy[segment_indices] + fraction[:, :, None] * segments
        choice = np.argmin(np.sum((xy[:, None, :] - projected) ** 2, axis=2), axis=1)
        row = np.arange(len(xy))
        selected = segment_indices[row, choice]
        normals = np.column_stack((-reference.segment[selected, 1], reference.segment[selected, 0])) / reference.ds[selected, None]
        lateral = np.sum((xy - projected[row, choice]) * normals, axis=1)
        progress = (extended[nearest[row, choice]] // count) * reference.length + reference.s[selected]
        progress += fraction[row, choice] * reference.ds[selected]
        return progress, lateral

    def _ground(self, points):
        # The current course is flat. Estimate a small datum correction only
        # from nearby low returns; distant depth-layer bias must not raise curbs
        # into obstacles or make a hidden road plane look measured.
        candidate = points[(np.linalg.norm(points[:, :2], axis=1) < 20.)
                           & (np.abs(points[:, 1]) < 6.)
                           & (points[:, 2] > self.ground_height_m - .5)
                           & (points[:, 2] < self.ground_height_m + .25), 2]
        if len(candidate) < 20:
            return self.ground_height_m
        correction = float(np.median(candidate)) - self.ground_height_m
        return self.ground_height_m + float(np.clip(correction, -.10, .10))

    def _clusters(self, points, ego_position, ego_yaw, ego_progress):
        xyz = np.asarray(points, float)[:, :3]
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        finite_count = len(xyz)
        ranges = np.linalg.norm(xyz[:, :2] - [1.6, 0.], axis=1)
        within = ranges <= self.max_range_m
        xyz, ranges = xyz[within], ranges[within]
        ground = self._ground(xyz) if len(xyz) else self.ground_height_m
        self_mask = ((xyz[:, 0] >= -1.2) & (xyz[:, 0] <= 4.7) & (np.abs(xyz[:, 1]) <= 1.1))
        height = xyz[:, 2] - ground
        keep = (~self_mask) & (height >= self.min_height_m) & (height <= self.max_height_m)
        selected = xyz[keep]
        diagnostics = {"finite_points": int(finite_count), "in_range_points": int(len(ranges)), "above_ground_nonself_points": int(len(selected)),
                       "ground_datum_m": ground, "maximum_return_range_m": float(ranges.max()) if len(ranges) else None}
        if len(selected) < self.minimum_points:
            return [], diagnostics
        c, s = math.cos(ego_yaw), math.sin(ego_yaw)
        rotation = np.array([[c, -s], [s, c]])
        world = selected.copy()
        world[:, :2] = selected[:, :2] @ rotation.T + np.asarray(ego_position)[:2]
        keys = np.floor(world / self.voxel_size_m).astype(np.int64)
        _, inverse, support = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        voxels = np.column_stack([np.bincount(inverse, weights=world[:, axis]) / support for axis in range(3)])
        progress, lateral = self._project(voxels[:, :2], ego_progress)
        accepted = ((progress >= ego_progress - self.rear_range_m) & (progress <= ego_progress + self.max_range_m)
                    & (np.abs(lateral) <= self.corridor_half_width_m))
        voxels, support, progress, lateral = (value[accepted] for value in (voxels, support, progress, lateral))
        diagnostics["corridor_voxels"] = int(len(voxels))
        diagnostics["corridor_point_support"] = int(support.sum())
        if len(voxels) < 2:
            return [], diagnostics
        parent = np.arange(len(voxels))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for first, second in cKDTree(voxels).query_pairs(self.cluster_radius_m):
            root_a, root_b = find(first), find(second)
            if root_a != root_b:
                parent[root_b] = root_a
        groups = {}
        for index in range(len(voxels)):
            groups.setdefault(find(index), []).append(index)
        clusters = []
        for members in groups.values():
            members = np.asarray(members)
            cluster = voxels[members]
            point_support = int(support[members].sum())
            horizontal_span = max(float(np.ptp(progress[members])), float(np.ptp(lateral[members])))
            vertical_span = float(np.ptp(cluster[:, 2]))
            if point_support < self.minimum_points or len(members) < 2 or max(horizontal_span, vertical_span) < .4:
                continue
            lower, upper = np.percentile(progress[members], [5., 95.])
            lat_lower, lat_upper = np.percentile(lateral[members], [5., 95.])
            representative = members[int(np.argmin(np.abs(progress[members] - lower)))]
            center = np.average(cluster[:, :2], axis=0, weights=support[members])
            strength = min(.95, .20 + .35 * min(1., point_support / 12.) + .30 * min(1., len(members) / 8.)
                           + .10 * min(1., max(horizontal_span, vertical_span)))
            clusters.append({"center": center, "surface": voxels[representative, :2].copy(),
                "lateral": float((lat_lower + lat_upper) / 2.), "support": point_support,
                "voxel_count": int(len(members)), "height_min": float(cluster[:, 2].min() - ground),
                "height_max": float(cluster[:, 2].max() - ground), "lateral_min": float(lat_lower),
                "lateral_max": float(lat_upper), "longitudinal_span": float(upper - lower), "strength": strength})
        return clusters, diagnostics

    def update(self, points, ego_position, ego_yaw, ego_progress, timestamp):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("Expected measured LiDAR points shaped [N,4]")
        timestamp = float(timestamp)
        if not np.isfinite(np.r_[ego_position, ego_yaw, ego_progress, timestamp]).all():
            raise ValueError("Ego state and timestamp must be finite")
        if self.last_timestamp is not None:
            if timestamp < self.last_timestamp - 1e-9:
                raise ValueError("LiDAR timestamps must be monotonic")
            if timestamp <= self.last_timestamp + 1e-9:
                return self.observation(ego_progress, timestamp)
        self.last_timestamp = timestamp
        clusters, self.last_scan = self._clusters(points, ego_position, ego_yaw, ego_progress)
        self.last_scan["input_points"] = int(len(points))
        self.tracks = [track for track in self.tracks if timestamp - track.timestamp <= self.max_age_s]
        used = set()
        if self.tracks and clusters:
            costs = np.full((len(self.tracks), len(clusters) + len(self.tracks)), 12.)
            for i, track in enumerate(self.tracks):
                dt = timestamp - track.timestamp
                predicted = track.center + track.velocity * dt
                projection = project_near(self.reference, predicted, ego_progress, window=self.max_range_m + 15.)
                tangent = np.array([math.cos(projection["heading_rad"]), math.sin(projection["heading_rad"])])
                normal = np.array([-tangent[1], tangent[0]])
                longitudinal_gate = 3. + (85. if track.hits < 3 else 8. + track.speed_uncertainty) * dt
                for j, cluster in enumerate(clusters):
                    delta = cluster["center"] - predicted
                    along, across = float(np.dot(delta, tangent)), float(np.dot(delta, normal))
                    if abs(along) <= longitudinal_gate and abs(across) <= 2. + 8. * dt:
                        costs[i, j] = (along / max(1., longitudinal_gate / 3.)) ** 2 + (across / 1.) ** 2
            rows, columns = linear_sum_assignment(costs)
            for i, j in zip(rows, columns):
                if j >= len(clusters) or costs[i, j] >= 12.:
                    continue
                track, cluster = self.tracks[i], clusters[j]
                dt = timestamp - track.timestamp
                measured_velocity = (cluster["center"] - track.center) / dt
                norm = np.linalg.norm(measured_velocity)
                residual = float(np.linalg.norm(measured_velocity - track.velocity))
                consistent = track.hits == 1 or residual < 40.
                track.velocity_hits = track.velocity_hits + 1 if norm <= 85. and consistent and dt <= .45 else 0
                if norm > 85.:
                    measured_velocity *= 85. / norm
                track.velocity = measured_velocity if track.hits == 1 else .5 * track.velocity + .5 * measured_velocity
                track.speed_uncertainty = max(2., .5 * track.speed_uncertainty + .5 * residual)
                for name, value in cluster.items():
                    setattr(track, name, value)
                track.timestamp = timestamp
                track.hits += 1
                used.add(int(j))
        for index, cluster in enumerate(clusters):
            if index not in used:
                self.tracks.append(GeometryTrack(self.next_identifier, velocity=np.zeros(2), timestamp=timestamp, **cluster))
                self.next_identifier += 1
        return self.observation(ego_progress, timestamp)

    def observation(self, ego_progress, timestamp):
        if not np.isfinite([ego_progress, timestamp]).all():
            raise ValueError("Observation progress and timestamp must be finite")
        if self.last_timestamp is not None and timestamp < self.last_timestamp - 1e-9:
            raise ValueError("Cannot observe before the latest LiDAR scan")
        rows = []
        for track in self.tracks:
            age = max(0., timestamp - track.timestamp)
            if age > self.max_age_s:
                continue
            surface = track.surface + track.velocity * age
            projection = project_near(self.reference, surface, ego_progress, window=self.max_range_m + 15.)
            tangent = np.array([math.cos(projection["heading_rad"]), math.sin(projection["heading_rad"])])
            speed = float(np.dot(track.velocity, tangent))
            rows.append({"track_id": f"geometry-{track.identifier}", "source": "raw_lidar_geometry",
                "object_class": "unclassified_obstacle", "relative_distance_m": projection["progress_m"] - ego_progress,
                "lateral_m": track.lateral, "position_m": surface.tolist(), "speed_mps": speed,
                "velocity_valid": track.velocity_hits >= 2, "speed_std_mps": track.speed_uncertainty,
                "position_std_m": max(.25, .5 * self.voxel_size_m + age * max(2., track.speed_uncertainty)),
                "geometry_confidence": track.strength, "confidence_kind": "uncalibrated_geometry_support",
                "measurement_age_s": age, "age_s": age, "measurement_hits": track.hits,
                "persistent": track.hits >= 2, "point_support": track.support, "voxel_count": track.voxel_count,
                "height_min_m": track.height_min, "height_max_m": track.height_max,
                "lateral_min_m": track.lateral_min, "lateral_max_m": track.lateral_max,
                "observed_longitudinal_span_m": track.longitudinal_span,
                "distance_definition": "nearest supported observed surface; not a predicted vehicle center"})
        rows.sort(key=lambda row: abs(row["relative_distance_m"]))
        return {"geometry_hazards": rows, "hazard_tracks": rows, "diagnostics": dict(self.last_scan),
                "range_limit_m": self.max_range_m, "free_space_verified": False, "occluded_space": "unknown",
                "visibility_note": "No return or a clear scan cannot establish that fully occluded space is empty."}
