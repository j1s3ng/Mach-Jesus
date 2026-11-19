"""Occlusion-aware ray/box synthetic LiDAR demo, not a learned sensor model.

Nearest ray intersections occlude farther boxes and road. This is deliberately
simple geometry, not a racing simulator, photorealistic data, or KITTI data.
Rays originate 1.4 m above a flat road. Saved points and boxes are translated
into vehicle-ground ego coordinates: road z=0, x-forward, y-left, z-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import COORDINATE_FRAME, write_frame


def _scan(boxes: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    azimuth, elevation = np.meshgrid(np.linspace(-np.pi, np.pi, 720, endpoint=False),
                                     np.deg2rad(np.linspace(-22, 8, 24)))
    directions = np.stack([np.cos(elevation) * np.cos(azimuth),
                           np.cos(elevation) * np.sin(azimuth), np.sin(elevation)], axis=-1).reshape(-1, 3)
    distance = np.full(len(directions), np.inf)
    downward = directions[:, 2] < -1e-8
    distance[downward] = -1.4 / directions[downward, 2]
    owner = np.full(len(directions), -1, dtype=int)
    for index, box in enumerate(boxes):
        yaw = box["yaw"]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        origin = -np.asarray(box["center"]) @ rotation
        ray = directions @ rotation
        half = np.asarray(box["size"]) / 2
        safe_ray = np.where(np.abs(ray) < 1e-10, 1e-10, ray)
        ta, tb = (-half - origin) / safe_ray, (half - origin) / safe_ray
        entry, leave = np.minimum(ta, tb).max(axis=1), np.maximum(ta, tb).min(axis=1)
        hit = (leave >= np.maximum(entry, 0)) & (entry > 0) & (entry < distance)
        distance[hit], owner[hit] = entry[hit], index
    keep = (distance > 0.5) & (distance < 85) & (rng.random(len(distance)) > 0.025)
    xyz = directions[keep] * (distance[keep] + rng.normal(0, 0.02, keep.sum()))[:, None]
    intensity = np.where(owner[keep] >= 0, 0.8, 0.22) + rng.normal(0, 0.025, keep.sum())
    points = np.column_stack([xyz, np.clip(intensity, 0, 1)]).astype(np.float32)
    return points, [int(np.count_nonzero(owner[keep] == index)) for index in range(len(boxes))]


def generate_demo(output: Path, frames: int = 120, seed: int = 7) -> Path:
    """Write deterministic 10 Hz frames; refuse to append to an existing manifest."""
    output = Path(output)
    if frames <= 0:
        raise ValueError("frames must be positive")
    if (output / "manifest.jsonl").exists():
        raise FileExistsError(f"Recording already exists: {output / 'manifest.jsonl'}")
    rng = np.random.default_rng(seed)
    for index in range(frames):
        phase = (index % 40) / 40
        boxes = [
            dict(center=[15 + 4 * np.sin(phase * 2 * np.pi), 1.8 + .6 * np.sin(phase * 2 * np.pi), -.65], size=[4.3, 1.8, 1.5], yaw=.07, label="car", track_id="car_front"),
            dict(center=[28 - 7 * phase, -4.2, -.6], size=[4.6, 1.9, 1.6], yaw=-.08, label="car", track_id="car_left"),
            dict(center=[35 + 3 * phase, 6, -.25], size=[6, 2.5, 2.3], yaw=.15, label="truck", track_id="truck_far"),
            dict(center=[35, 3.0, -.65], size=[4.1, 1.8, 1.5], yaw=.02, label="car", track_id="car_occluded"),
        ]
        points, visibility = _scan(boxes, rng)
        # Ray calculations use the sensor origin, but recordings share the
        # vehicle-ground origin used by Webots and the Vmax transfer workflow.
        points[:, 2] += 1.4
        # Evaluate only objects with enough returns for this toy scan. This is
        # a synthetic visibility convention, not an official benchmark filter.
        gt = [dict(box, center=[*box["center"][:2], box["center"][2] + 1.4],
                   visible_lidar_points=count)
              for box, count in zip(boxes, visibility) if count >= 8]
        write_frame(output, f"{index:06d}", points, index * 100_000_000,
                    f"synthetic_{index // 40:03d}", gt_boxes=gt,
                    source="synthetic_ray_box_demo_not_real_data")
    metadata = {
        "source": "synthetic_ray_box_demo_not_real_data",
        "coordinate_frame": COORDINATE_FRAME,
        "coordinate_origin": "ego_vehicle_reference_at_ground",
        "lidar_to_ego_translation_m": [0.0, 0.0, 1.4],
        "road_plane_z_m": 0.0,
        "sensor_period_ms": 100,
        "frames": frames,
        "seed": seed,
        "intensity": "synthetic class-based reflectance heuristic, not physical reflectance",
    }
    (output / "sensor_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output / "manifest.jsonl"
