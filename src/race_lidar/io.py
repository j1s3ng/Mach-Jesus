"""Portable frame format, with SI units and an explicit right-handed ego frame.

manifest.jsonl has one JSON object per frame: frame_id, scene_id,
timestamp_ns, points_file (relative .npy path), coordinate_frame, source,
optional image_file (relative RGB .png path), and optional gt_boxes.
Points have shape N x 4: x-forward, y-left, z-up in metres, intensity in
the source's documented scale (no implicit intensity normalization).
Boxes use center [x,y,z], size [length,width,height] in metres, and yaw in
radians counterclockwise about +z. Origin must be documented by each source;
Webots uses the vehicle ground origin after the sensor extrinsic transform.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

COORDINATE_FRAME = "ego_x_forward_y_left_z_up"


def validate_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("LiDAR points must have shape (N, 4): x, y, z, intensity")
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError("LiDAR points must contain finite numeric values")
    return array.astype(np.float32, copy=False)


def validate_box(box: dict) -> dict:
    for key in ("center", "size"):
        value = np.asarray(box.get(key, []), dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Box {key} must contain three finite values")
    if np.any(np.asarray(box["size"]) <= 0):
        raise ValueError("Box dimensions must be positive")
    if not np.isfinite(float(box.get("yaw", float("nan")))):
        raise ValueError("Box yaw must be finite")
    if "score" in box and not np.isfinite(float(box["score"])):
        raise ValueError("Box confidence must be finite")
    return box


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Frame paths must stay inside the recording directory")
    return path


def write_frame(
    root: Path,
    frame_id: str,
    points: np.ndarray,
    timestamp_ns: int,
    scene_id: str,
    gt_boxes: list[dict] | None = None,
    image: np.ndarray | None = None,
    *,
    source: str = "local_recording",
) -> Path:
    """Append one frame; duplicate scene/frame IDs are rejected, never overwritten."""
    root = Path(root)
    cloud = validate_points(points)
    for name, value in (("frame_id", frame_id), ("scene_id", scene_id)):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"{name} must be a nonempty simple identifier")
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns < 0:
        raise ValueError("timestamp_ns must be a nonnegative integer")
    if gt_boxes is not None:
        for box in gt_boxes:
            validate_box(box)
    rgb = None
    if image is not None:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Camera image must be a uint8 RGB array of shape (H,W,3)")
    manifest = root / "manifest.jsonl"
    if manifest.exists():
        for entry in read_manifest(root):
            if (entry["scene_id"], entry["frame_id"]) == (scene_id, frame_id):
                raise ValueError(f"Duplicate frame {scene_id}/{frame_id}")
    stem = f"{scene_id}__{frame_id}"
    relative = f"points/{stem}.npy"
    (root / "points").mkdir(parents=True, exist_ok=True)
    np.save(root / relative, cloud, allow_pickle=False)
    entry = dict(frame_id=frame_id, scene_id=scene_id, timestamp_ns=int(timestamp_ns),
                 points_file=relative, coordinate_frame=COORDINATE_FRAME, source=source)
    if gt_boxes is not None:
        entry["gt_boxes"] = gt_boxes
    if rgb is not None:
        (root / "images").mkdir(exist_ok=True)
        entry["image_file"] = f"images/{stem}.png"
        Image.fromarray(rgb).save(root / entry["image_file"])
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, allow_nan=False) + "\n")
    return manifest


def read_manifest(root: Path) -> list[dict]:
    """Read a directory or explicit manifest.jsonl path, rejecting duplicate IDs."""
    path = Path(root)
    path = path / "manifest.jsonl" if path.is_dir() else path
    entries, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                for key in ("frame_id", "scene_id", "timestamp_ns", "points_file", "coordinate_frame"):
                    if key not in entry:
                        raise ValueError(f"Missing {key}")
                if entry["coordinate_frame"] != COORDINATE_FRAME:
                    raise ValueError("Unsupported coordinate frame; convert to ego coordinates first")
                key = (entry["scene_id"], entry["frame_id"])
                if key in seen:
                    raise ValueError(f"Duplicate scene/frame ID: {key}")
                seen.add(key)
                entries.append(entry)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid manifest line {line_number}: {exc}") from exc
    return entries


def read_frame(manifest_entry: dict, root: Path) -> dict:
    """Load a frame; ground-truth annotations remain separate from sensor points."""
    entry = dict(manifest_entry)
    if entry.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError("Unsupported coordinate frame")
    entry["points"] = validate_points(np.load(_safe_path(Path(root), entry["points_file"]), allow_pickle=False))
    if "gt_boxes" in entry:
        for box in entry["gt_boxes"]:
            validate_box(box)
    if "image_file" in entry:
        with Image.open(_safe_path(Path(root), entry["image_file"])) as image:
            entry["image"] = np.asarray(image.convert("RGB"))
    return entry
