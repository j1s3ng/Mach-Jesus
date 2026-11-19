"""Calibrated pinhole overlay for the included Webots front camera.

No recognition or ground-truth labels are used. Camera axes follow ego axes:
x forward, y left, z up; image pixels increase right and down.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .io import read_frame, read_manifest
from .baseline import detect


def project_points(points, width=640, height=360, fov=1.4, translation=(1.6, 0, 1.2)):
    xyz = np.asarray(points)[:, :3] - translation
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 0] > .1)
    focal = width / (2 * math.tan(fov / 2))
    uv = np.full((len(xyz), 2), np.nan)
    uv[valid, 0] = (width - 1) / 2 - focal * xyz[valid, 1] / xyz[valid, 0]
    uv[valid, 1] = (height - 1) / 2 - focal * xyz[valid, 2] / xyz[valid, 0]
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, valid


def overlay_frame(frame, boxes, caption="RGB + projected LiDAR | cyan: geometric candidates"):
    image = Image.fromarray(frame["image"]).copy()
    draw = ImageDraw.Draw(image)
    uv, valid = project_points(frame["points"], *image.size)
    valid &= frame["points"][:, 2] > .2
    for pixel, point in zip(uv[valid][::2], frame["points"][valid][::2]):
        distance = min(np.linalg.norm(point[:2]) / 80, 1)
        color = (int(255 * (1-distance)), int(220 * distance), 210)
        draw.ellipse((pixel[0]-1, pixel[1]-1, pixel[0]+1, pixel[1]+1), fill=color)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for box in boxes:
        length, width, height = box["size"]
        corners = np.array([[x*length/2,y*width/2,z*height/2] for z in [-1,1]
                            for x,y in [(1,1),(1,-1),(-1,-1),(-1,1)]])
        a = box["yaw"]
        rotation = np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
        corners = corners @ rotation.T + box["center"]
        pixels, visible = project_points(corners, *image.size)
        for first, second in edges:
            if visible[first] and visible[second]:
                draw.line([tuple(pixels[first]),tuple(pixels[second])], fill=(40,245,255), width=2)
    draw.rectangle((0,0,image.width,32), fill=(9,18,29))
    draw.text((10,10), caption, fill=(235,240,245))
    return image


def export_overlay(recording, output, count=6):
    recording, output = Path(recording), Path(output)
    entries = read_manifest(recording)
    if not (recording / "sensor_metadata.json").exists():
        raise ValueError("Overlay requires the included Webots calibration metadata")
    output.mkdir(parents=True, exist_ok=True)
    images = []
    for index in np.linspace(0, len(entries)-1, min(count,len(entries)), dtype=int):
        frame = read_frame(entries[index], recording)
        images.append(overlay_frame(frame, detect(frame["points"])))
        images[-1].save(output / f"camera_lidar_{index:06d}.png")
    contact = Image.new("RGB", (640*2, 360*math.ceil(len(images)/2)))
    for i, image in enumerate(images): contact.paste(image, ((i%2)*640,(i//2)*360))
    contact.save(output / "camera_lidar_contact.jpg", quality=92)
    return output / "camera_lidar_contact.jpg"
