"""Offline inference + annotated BEV diagnostics from manifest recordings.

predictions.jsonl repeats frame_id, scene_id, timestamp_ns, coordinate_frame,
source, backend and inference_ms, with boxes containing center/size/yaw/score.
metrics.json separates measured inference latency from full replay wall time.
Cluster inference runs locally; the remote backend can call a CUDA model over
an SSH tunnel. Ground truth is passed only to metrics and visualization.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .baseline import detect
from .io import COORDINATE_FRAME, read_frame, read_manifest
from .metrics import evaluate_frames


def render_bev(points: np.ndarray, boxes: list[dict], gt_boxes: list[dict] | None = None,
               title: str = "", size: tuple[int, int] = (960, 720),
               footer="Geometric baseline | confidence is uncalibrated | not a trained detector") -> Image.Image:
    """Forward is up, ego-left is screen-left; cyan predictions, amber labels."""
    width, height = size
    canvas = Image.new("RGB", size, (12, 19, 28))
    draw = ImageDraw.Draw(canvas)
    scale = min((width - 60) / 60, (height - 100) / 65)
    def pixel(x, y):
        return (width / 2 - y * scale, height - 45 - x * scale)
    for x in range(0, 66, 10):
        draw.line([pixel(x, -30), pixel(x, 30)], fill=(37, 49, 62))
        draw.text(pixel(x, -29), f"{x}m", fill=(130, 145, 160))
    for y in range(-30, 31, 10):
        draw.line([pixel(0, y), pixel(65, y)], fill=(37, 49, 62))
    for x, y, z, intensity in points:
        if 0 <= x <= 65 and abs(y) <= 30:
            px, py = pixel(float(x), float(y))
            color = (70, 90, 105) if z < 0.2 else (155, 190, 175)
            draw.point((round(px), round(py)), fill=color)
    for candidates, color in ((gt_boxes or [], (255, 190, 70)), (boxes, (70, 220, 255))):
        for box in candidates:
            x, y, _ = box["center"]
            length, box_width, _ = box["size"]
            yaw = box["yaw"]
            corners = np.array([[length / 2, box_width / 2], [length / 2, -box_width / 2],
                                [-length / 2, -box_width / 2], [-length / 2, box_width / 2]])
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            corners = corners @ rotation.T + [x, y]
            poly = [pixel(*p) for p in corners]
            draw.line(poly + [poly[0]], fill=color, width=2)
            if "score" in box:
                draw.text(pixel(x + length / 2 + .7, y), f"{box['score']:.2f}", fill=color)
    draw.ellipse([width / 2 - 5, height - 50, width / 2 + 5, height - 40], fill=(255, 255, 255))
    draw.text((16, 12), title[:120], fill=(235, 240, 245))
    draw.text((16, 32), "BEV | forward ^ | ego-left < | cyan: candidates | amber: ground truth", fill=(185, 200, 215))
    draw.text((16, height - 22), footer, fill=(185, 200, 215))
    return canvas


def replay(input_dir: Path, output_dir: Path, backend: str = "cluster", max_frames: int | None = None,
           make_video: bool = True, distance_threshold: float = 2.0,
           detector_url="http://127.0.0.1:8765") -> dict:
    """Run point-only inference, write predictions, metrics and diagnostics."""
    if backend == "cluster":
        predictor = detect
    elif backend == "remote":
        from .remote import remote_detect
        predictor = lambda points: remote_detect(points,detector_url)
    else:
        raise ValueError("Expected backend='cluster' or 'remote'")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    entries = read_manifest(input_dir)
    root = input_dir if input_dir.is_dir() else input_dir.parent
    entries = entries[:max_frames]
    if not entries:
        raise ValueError("The recording has no frames")
    if root.resolve() == output_dir.resolve():
        raise ValueError("Replay output must be separate from the input recording")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Replay output already exists; choose a fresh directory to preserve results")
    output_dir.mkdir(parents=True, exist_ok=True)
    bev_dir = output_dir / "bev"
    bev_dir.mkdir(exist_ok=True)
    durations, predictions, labels, thumbnails = [], [], [], []
    selected = set(np.linspace(0, len(entries) - 1, min(12, len(entries)), dtype=int).tolist())
    writer, video_path = None, output_dir / "replay.mp4"
    intervals = [(b["timestamp_ns"] - a["timestamp_ns"]) / 1e9
                 for a, b in zip(entries, entries[1:])
                 if a["scene_id"] == b["scene_id"] and b["timestamp_ns"] > a["timestamp_ns"]]
    fps = float(np.clip(1 / np.median(intervals), 1, 120)) if intervals else 10.0
    start_wall = time.perf_counter()
    try:
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for index, entry in enumerate(entries):
                frame = read_frame(entry, root)
                start = time.perf_counter_ns()
                boxes = predictor(frame["points"])
                elapsed = (time.perf_counter_ns() - start) / 1e6
                durations.append(elapsed)
                prediction = dict(frame_id=entry["frame_id"], scene_id=entry["scene_id"],
                                  timestamp_ns=entry["timestamp_ns"], coordinate_frame=COORDINATE_FRAME,
                                  source=entry.get("source", "unspecified"), backend=backend,
                                  inference_ms=elapsed, boxes=boxes)
                predictions.append(prediction)
                labels.append(entry)
                handle.write(json.dumps(prediction, allow_nan=False) + "\n")
                image = render_bev(frame["points"], boxes, frame.get("gt_boxes"),
                                   f"{entry['scene_id']} / {entry['frame_id']} | {elapsed:.1f} ms | {entry.get('source', 'unspecified')}",
                                   footer="Remote model | latency includes network transfer" if backend == "remote" else
                                   "Geometric baseline | confidence is uncalibrated | not a trained detector")
                image.save(bev_dir / f"{index:06d}.png")
                if index in selected:
                    thumb = image.copy()
                    thumb.thumbnail((480, 360))
                    thumbnails.append(thumb)
                if make_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, image.size)
                        if not writer.isOpened():
                            writer.release()
                            writer = None
                            make_video = False
                    if writer is not None:
                        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    cols = min(3, len(thumbnails))
    contact = Image.new("RGB", (480 * cols, 360 * math.ceil(len(thumbnails) / cols)), (12, 19, 28))
    for index, thumb in enumerate(thumbnails):
        contact.paste(thumb, ((index % cols) * 480, (index // cols) * 360))
    contact.save(output_dir / "contact_sheet.jpg", quality=90)
    timings = np.asarray(durations)
    # The pretrained model returns cars only within 50 m. Evaluate every backend
    # over this same supported region; keep full detections/labels in diagnostics.
    metric_predictions = [{**frame, "boxes": [box for box in frame["boxes"]
                           if np.linalg.norm(box["center"][:2]) <= 50.0]} for frame in predictions]
    metric_labels = [{**frame, "gt_boxes": [box for box in frame["gt_boxes"]
                      if np.linalg.norm(box["center"][:2]) <= 50.0]} if "gt_boxes" in frame else frame
                     for frame in labels]
    evaluation = evaluate_frames(metric_predictions, metric_labels, distance_threshold)
    evaluation["evaluation_radius_m"] = 50.0
    summary = dict(
        backend="deterministic_geometric_clustering_cpu" if backend == "cluster" else "remote_detector",
        frames=len(entries),
        data_sources=sorted({entry.get("source", "unspecified") for entry in entries}),
        confidence="uncalibrated heuristic, not a probability" if backend == "cluster" else "model scores; calibration not established",
        warmup="none; first frame included",
        timing_scope="point-only inference call; I/O/rendering/video excluded; remote includes transfer/round trip",
        inference_ms_mean=float(timings.mean()), inference_ms_p50=float(np.percentile(timings, 50)),
        inference_ms_p95=float(np.percentile(timings, 95)),
        inference_fps=len(entries) / (timings.sum() / 1000),
        replay_wall_seconds=time.perf_counter() - start_wall, playback_fps=fps,
        video=str(video_path) if make_video and video_path.exists() else None,
        evaluation=evaluation,
    )
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
