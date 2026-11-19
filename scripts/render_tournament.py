#!/usr/bin/env python3
"""Compose synchronized physical race footage and observed live standings.

Usage: python scripts/render_tournament.py HEAT_DIR --output HEAT_DIR/race_mosaic.mp4
Output uses mp4v at the source 5 Hz; convert it to H.264 for browser playback.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from race_lidar.race_results import evaluate_heat

cv2.setNumThreads(1)

WIDTH, HEIGHT = 1280, 720
PANEL = (13, 20, 31)
MUTED = (151, 165, 185)
WHITE = (239, 244, 250)
GOLD = (255, 211, 105)


def font(size, bold=False):
    candidates = [
        Path("/System/Library/Fonts/Supplemental") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
        Path("/System/Library/Fonts/Menlo.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONTS = {size: font(size) for size in (11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 28, 32, 36, 48)}
BOLD = {size: font(size, True) for size in (13, 14, 16, 18, 20, 22, 24, 28, 32, 36, 48)}


def color(entry):
    value = entry.get("color", "0.55 0.65 0.8")
    values = value.replace(",", " ").split() if isinstance(value, str) else value
    return tuple(int(np.clip(float(component), 0, 1) * 255) for component in values[:3])


def text_color(entry):
    # Preserve each car's hue while keeping dark blue/purple labels legible.
    return tuple(round(0.72 * component + 0.28 * 255) for component in color(entry))


def clock(seconds):
    seconds = max(0, float(seconds or 0))
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def policy(entry):
    value = entry.get("policy", "")
    return {"reference": "Baseline", "neural_v1": "NN V1", "neural_v2": "NN V2", "neural_v3": "NN V3",
            "neural_rich_seed7": "Rich NN 7", "neural_rich_seed17": "Rich NN 17"}.get(value, entry.get("label", value))


def fit_text(draw, text, xy, max_width, size=16, fill=WHITE, bold=False):
    selected = BOLD.get(size, FONTS[size]) if bold else FONTS[size]
    while text and draw.textlength(text, font=selected) > max_width:
        text = text[:-2].rstrip("…") + "…"
    draw.text(xy, text, fill=fill, font=selected)


def finished(entry):
    claim = entry.get("valid_finish", entry.get("finished", entry.get("completed", False)))
    return bool(claim) and entry.get("valid_lap") is not False and entry.get("finish_time_s") is not None


def failed(entry):
    status = entry.get("result_status", entry.get("status"))
    return bool(entry.get("failed")) or bool(entry.get("terminal") and not finished(entry)) or status in {
        "failed", "completed_invalid", "invalid_finish", "dnf", "timeout", "stopped",
    } or bool(entry.get("completed") and not finished(entry))


def standings(rows):
    return sorted(rows, key=lambda row: (
        0 if finished(row) else (2 if failed(row) else 1),
        float(row["finish_time_s"]) if finished(row) else -float(row.get("progress_m", 0)),
        int(row.get("grid_index", 0)),
    ))


def select_view(cars, simulation_time=None, first_finish_time=None, finish_hold_seconds=3.0):
    """Keep live footage on the race once valid finishers leave the contest."""
    active = [car for car in cars if not finished(car) and not failed(car)]
    if finished(cars[0]) and simulation_time is not None and first_finish_time is not None:
        if 0 <= simulation_time - first_finish_time < finish_hold_seconds:
            return cars[0], 1, "WINNER FINISHED"
    if finished(cars[0]) and active:
        selected = active[0]
        return selected, cars.index(selected) + 1, "LEADING ACTIVE RACER"
    role = "WINNER ONBOARD" if finished(cars[0]) else ("LAST RECORDED VIEW" if failed(cars[0]) else "RACE LEADER")
    return cars[0], 1, role


def read_json(path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


class Onboard:
    def __init__(self, path, fallback_fps):
        self.path = path
        self.capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]) if path.exists() else None
        opened = self.capture is not None and self.capture.isOpened()
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) if opened else fallback_fps
        if not self.fps or not math.isfinite(self.fps):
            self.fps = fallback_fps
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        self.index = -1
        self.last = Image.new("RGB", (640, 360), PANEL)
        self.ended = not opened

    def frame(self, index):
        while not self.ended and self.index < index:
            okay, frame = self.capture.read()
            if not okay:
                self.ended = True
                break
            self.index += 1
            self.last = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.ended:
            held = ImageEnhance.Brightness(self.last.convert("L").convert("RGB")).enhance(0.65)
            ImageDraw.Draw(held).text((16, 325), "FEED ENDED · LAST RECORDED FRAME", fill=WHITE, font=BOLD[14])
            return held
        return self.last

    def close(self):
        if self.capture is not None:
            self.capture.release()


class Renderer:
    def __init__(self, heat):
        self.heat = heat
        self.config = read_json(heat / "config.json")
        self.results = read_json(heat / "heat_results.json")
        self.entrants = self.config["entrants"]
        self.entries = {entry["car_id"]: entry for entry in self.entrants}
        self.capacity = self.config.get("mode") == "capacity"
        self.period = float(self.config.get("video_period_ms", 200)) / 1000
        self.fps = 1 / self.period
        # Cameras record every world-period from time zero, including the grid.
        # The common race-start delay must never be added to this frame origin.
        self.frame_origin = self.period
        self.start_time = float(self.config.get("start_time_s", 1.0))
        self.length = float(self.config.get("reference_length_m", 1)) * int(self.config.get("laps", 1))
        self.board = [json.loads(line) for line in (heat / "leaderboard.jsonl").read_text().splitlines() if line.strip()]
        if not self.board:
            raise ValueError("leaderboard.jsonl has no observations")
        self.board.sort(key=lambda row: row.get("simulation_time_s", row.get("time_s", 0) + self.start_time))
        self.times = [float(row.get("simulation_time_s", row.get("time_s", 0) + self.start_time)) for row in self.board]
        self.first_finish_time = next((when for when, row in zip(self.times, self.board)
                                       if any(finished(car) for car in row["cars"])), None)
        self.videos = {entry["car_id"]: Onboard(heat / "cars" / entry["car_id"] / "onboard.mp4", self.fps)
                       for entry in self.entrants}
        self.total_frames = max(max(video.count for video in self.videos.values()),
                                int(round((self.times[-1] - self.frame_origin) / self.period)) + 1)
        self.map_xy = None
        reference = Path(self.config.get("reference_path", ""))
        if reference.is_file():
            with np.load(reference, allow_pickle=False) as data:
                self.map_xy = np.column_stack((data["x_m"], data["y_m"]))

    def board_at(self, simulation_time):
        index = max(0, bisect.bisect_right(self.times, simulation_time + 1e-7) - 1)
        row = self.board[index]
        cars = standings([{**self.entries[car["car_id"]], **car} for car in row["cars"]])
        return row, cars

    def map_panel(self, canvas, cars, y, height):
        if self.map_xy is None or height < 70:
            return
        draw = ImageDraw.Draw(canvas)
        draw.text((978, y), "TRACK POSITION", fill=MUTED, font=FONTS[11])
        low, high = self.map_xy.min(axis=0), self.map_xy.max(axis=0)
        extent = np.maximum(high-low, 1)
        scale = min(280 / extent[0], (height - 27) / extent[1])
        center = (low + high) / 2
        def project(point):
            delta = np.asarray(point[:2]) - center
            return (1120 + float(delta[0]) * scale, y + 20 + (height - 27) / 2 - float(delta[1]) * scale)
        line = [project(point) for point in self.map_xy]
        draw.line(line + [line[0]], fill=(74, 90, 111), width=3)
        # Draw the leader last so its marker remains legible in tight packs.
        for car in reversed(cars):
            if "position_m" not in car:
                continue
            x, yy = project(car["position_m"])
            radius = 4 if car is cars[0] else 3
            draw.ellipse((x-radius, yy-radius, x+radius, yy+radius), fill=color(car), outline=WHITE if car is cars[0] else None)

    def leaderboard(self, canvas, cars, race_time, final=False):
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((960, 0, 1279, 719), fill=PANEL)
        draw.text((978, 13), "FINAL ORDER" if final else "LIVE STANDINGS", fill=WHITE, font=BOLD[18])
        draw.text((978, 37), f"{len(cars)} RACERS  ·  {clock(race_time)}", fill=MUTED, font=FONTS[13])
        row_height = min(39, max(24, 532 // max(len(cars), 1)))
        leader_progress = float(cars[0].get("progress_m", 0))
        for rank, car in enumerate(cars, 1):
            y = 65 + (rank - 1) * row_height
            tone = color(car)
            if rank % 2:
                draw.rectangle((968, y, 1271, y+row_height-1), fill=(20, 29, 42))
            draw.rectangle((970, y+3, 974, y+row_height-4), fill=tone)
            draw.text((980, y+3), f"{rank:02}", fill=GOLD if rank == 1 else WHITE, font=BOLD[14])
            title = f"{car['car_id']}  {policy(car)}"
            fit_text(draw, title, (1008, y+2), 147, size=14, fill=text_color(car), bold=True)
            # A future result file cannot reveal a finish in a live frame.
            if finished(car):
                status = clock(car["finish_time_s"])
            elif final and not self.capacity:
                status = "INVALID" if car.get("status") == "invalid_finish" else "DNF"
            elif failed(car) and not self.capacity:
                status = "OUT"
            elif self.videos[car["car_id"]].ended:
                status = "NO FEED"
            else:
                status = f"{max(0, min(100, float(car.get('progress_m', 0))/self.length*100)):.1f}%"
            draw.text((1167, y+3), status, fill=WHITE if finished(car) else MUTED, font=FONTS[13])
            bar_y = y + row_height - 5
            progress = np.clip(float(car.get("progress_m", 0)) / self.length, 0, 1)
            draw.line((1008, bar_y, 1256, bar_y), fill=(42, 55, 72), width=2)
            if progress:
                draw.line((1008, bar_y, 1008+248*progress, bar_y), fill=tone, width=2)
            if row_height >= 35 and not finished(car):
                gap = max(0, leader_progress - float(car.get("progress_m", 0)))
                draw.text((1008, y+18), "LEADER" if rank == 1 else f"{gap:.0f} m to leader", fill=MUTED, font=FONTS[11])
        bottom = 65 + len(cars) * row_height + 13
        self.map_panel(canvas, cars, bottom, HEIGHT - bottom - 13)

    def race_frame(self, index):
        simulation_time = self.frame_origin + index * self.period
        board, cars = self.board_at(simulation_time)
        race_time = max(0, simulation_time-self.start_time)
        frames = {car_id: video.frame(int(round(index*self.period*video.fps))) for car_id, video in self.videos.items()}
        leader, view_rank, view_role = select_view(cars, simulation_time, self.first_finish_time)
        leader_id = leader["car_id"]
        canvas = Image.new("RGB", (WIDTH, HEIGHT), PANEL)
        canvas.paste(frames[leader_id].resize((960, 540), Image.Resampling.BILINEAR), (0, 0))
        draw = ImageDraw.Draw(canvas)
        # Replace the source's existing text-only header; retain the whole camera image.
        draw.rectangle((0, 0, 959, 98), fill=(8, 14, 23))
        draw.rectangle((0, 0, 8, 98), fill=color(leader))
        title = f"{'CAPACITY TRIAL' if self.capacity else 'YAS MARINA AI RACE'}  /  {len(cars)} CARS"
        draw.text((22, 11), title, fill=MUTED, font=FONTS[14])
        fit_text(draw, f"P{view_rank}  {leader_id} · {leader.get('label', policy(leader))}", (22, 33), 710, size=28, bold=True)
        draw.text((22, 72), f"{view_role}  |  OWN LiDAR → VOXELNEXT → DRIVER", fill=(145, 201, 219), font=FONTS[13])
        draw.text((797, 18), clock(race_time), fill=WHITE, font=BOLD[28])
        leader_status = "WINNER FINISHED" if view_role == "WINNER FINISHED" else ("FINISHED" if finished(leader) else ("OUT" if failed(leader) else ("GRID" if simulation_time < self.start_time else "RACING")))
        draw.text((781 if view_role == "WINNER FINISHED" else 803, 57), leader_status, fill=GOLD if finished(leader) else MUTED, font=BOLD[14])
        contenders = sorted([car for car in cars if car["car_id"] != leader_id],
                            key=lambda car: (finished(car) or failed(car), cars.index(car)))[:4]
        for tile in range(4):
            x = tile * 240
            if tile < len(contenders):
                car = contenders[tile]
                canvas.paste(frames[car["car_id"]].resize((240, 135), Image.Resampling.BILINEAR), (x, 540))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((x, 675, x+239, 719), fill=(17, 26, 39))
                draw.line((x, 675, x+239, 675), fill=color(car), width=3)
                tile_rank = cars.index(car) + 1
                draw.text((x+10, 682), f"P{tile_rank}  {car['car_id']} · {policy(car)}", fill=text_color(car), font=BOLD[14])
                status = "FINISHED " + clock(car["finish_time_s"]) if finished(car) else f"{max(0, float(car.get('progress_m',0))):.0f} m / {self.length:.0f} m"
                draw.text((x+10, 702), status, fill=MUTED, font=FONTS[11])
            else:
                draw.rectangle((x, 540, x+239, 719), fill=(20, 29, 42))
                draw.text((x+20, 594), "INDEPENDENT AI", fill=WHITE, font=BOLD[18])
                draw.text((x+20, 624), "Shared rules · same car", fill=MUTED, font=FONTS[13])
            draw.line((x, 540, x, 719), fill=(47, 62, 82), width=1)
        self.leaderboard(canvas, cars, race_time)
        return canvas

    def final_rows(self):
        reported = {row["car_id"]: row for row in self.results.get("cars", [])}
        live = {row["car_id"]: row for row in self.board[-1]["cars"]}
        if self.capacity:
            return standings([{**entry, **reported.get(entry["car_id"], {}), **live.get(entry["car_id"], {})}
                              for entry in self.entrants])
        # Match analyze_directory exactly: the frozen terminal controller result
        # replaces the aggregate record. Shutdown summaries and post-finish
        # director progress describe cooldown movement, not classification.
        records = dict(reported)
        for entry in self.entrants:
            car_id = entry["car_id"]
            path = self.heat / "cars" / car_id / "result.json"
            if path.exists():
                records[car_id] = read_json(path)
        validated = evaluate_heat(self.config, records, self.results.get("collisions_by_car"))
        series = read_json(self.heat.parent / "results.json")
        authoritative = next((heat["standings"] for heat in series.get("heats", [])
                              if heat.get("heat_id") == self.heat.name), None)
        if authoritative is not None:
            keys = ("car_id", "position", "status", "valid_finish", "clean_finish", "finish_time_s", "progress_m")
            observed = [tuple(row.get(key) for key in keys) for row in validated]
            expected = [tuple(row.get(key) for key in keys) for row in authoritative]
            if observed != expected:
                raise ValueError("Final video classification differs from the authoritative series results")
        rows = []
        for audit in validated:
            car_id = audit["car_id"]
            row = {**self.entries[car_id], **records.get(car_id, {}), **audit, "finished": audit["valid_finish"]}
            if "position_m" in live.get(car_id, {}):
                row["position_m"] = live[car_id]["position_m"]  # map display only
            rows.append(row)
        return rows

    def final_card(self):
        rows = self.final_rows()
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (9, 15, 25))
        draw = ImageDraw.Draw(canvas)
        title = "CAPACITY TRIAL COMPLETE" if self.capacity else "FINAL RACE CLASSIFICATION"
        draw.text((42, 30), title, fill=MUTED, font=BOLD[22])
        if self.capacity:
            draw.text((42, 77), "ORDER AT TRIAL STOP", fill=WHITE, font=BOLD[32])
        elif finished(rows[0]):
            draw.text((42, 77), "WINNER", fill=GOLD, font=BOLD[28])
            fit_text(draw, f"{rows[0]['car_id']} · {rows[0].get('label', policy(rows[0]))}", (42, 119), 850, size=36, bold=True)
            draw.text((42, 166), clock(rows[0]["finish_time_s"]), fill=color(rows[0]), font=BOLD[36])
        else:
            draw.text((42, 77), "NO VALID FINISH RECORDED", fill=WHITE, font=BOLD[28])
        if self.capacity:
            rtf = self.results.get("real_time_factor")
            label = f"{len(rows)} cars  |  {self.results.get('simulation_seconds', 0):.1f} simulated seconds"
            if rtf is not None:
                label += f"  |  {rtf:.2f}× real time"
            draw.text((42, 128), label, fill=MUTED, font=FONTS[20])
        # Three podium cards show observed order; the sidebar retains every entrant.
        for index, car in enumerate(rows[:3]):
            x, y, w = 42 + index*292, 258, 270
            draw.rounded_rectangle((x, y, x+w, 500), radius=12, fill=(22, 32, 47))
            draw.rectangle((x, y, x+w, y+5), fill=color(car))
            draw.text((x+20, y+20), f"P{index+1}", fill=GOLD if index == 0 else WHITE, font=BOLD[36])
            draw.text((x+20, y+78), car["car_id"], fill=WHITE, font=BOLD[28])
            fit_text(draw, car.get("label", policy(car)), (x+20, y+119), w-40, size=20, fill=text_color(car), bold=True)
            if finished(car):
                status = clock(car["finish_time_s"])
            elif self.capacity:
                status = f"{float(car.get('progress_m',0)):.0f} m"
            elif car.get("status") == "invalid_finish":
                status = "INVALID FINISH"
            else:
                status = f"DNF · {float(car.get('progress_m',0))/self.length*100:.1f}%"
            draw.text((x+20, y+172), status, fill=MUTED, font=FONTS[20])
        learned = sorted({policy(entry) for entry in self.entrants if entry.get("policy") != "reference"})
        legend = "Baseline = reference controller    " + " / ".join(learned) + " = trained drivers"
        fit_text(draw, legend, (42, 565), 900, size=17, fill=MUTED)
        draw.text((42, 600), "All views: recorded Webots cars · independent LiDAR perception · shared vehicle rules", fill=MUTED, font=FONTS[15])
        draw.text((42, 666), "Measured race order and finish times", fill=(103, 125, 151), font=FONTS[14])
        # Board may show the final result only after all live footage has elapsed.
        self.leaderboard(canvas, rows, self.results.get("simulation_seconds", 0), final=True)
        return canvas

    def close(self):
        for video in self.videos.values():
            video.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heat_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final-seconds", type=float, default=5)
    args = parser.parse_args()
    output = args.output or args.heat_dir / "race_mosaic.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(args.heat_dir)
    renderer.final_rows()  # Validate every final rank before encoding the timeline.
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), renderer.fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"Could not open output video {output}")
    try:
        preview_index = min(renderer.total_frames-1, max(0, int(5 * renderer.fps)))
        for index in range(renderer.total_frames):
            frame = renderer.race_frame(index)
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            if index == preview_index:
                frame.save(output.with_name(output.stem + "_preview.jpg"), quality=92)
            if index % max(1, int(renderer.fps*20)) == 0:
                print(f"Rendered {index+1}/{renderer.total_frames} race frames", flush=True)
        final = renderer.final_card()
        final.save(output.with_name(output.stem + "_final.jpg"), quality=92)
        final_frames = max(0, round(args.final_seconds * renderer.fps))
        bgr = cv2.cvtColor(np.asarray(final), cv2.COLOR_RGB2BGR)
        for _ in range(final_frames):
            writer.write(bgr)
    finally:
        writer.release()
        renderer.close()
    metadata = {"output": str(output.resolve()), "width": WIDTH, "height": HEIGHT, "fps": renderer.fps,
                "codec": "mp4v", "race_frames": renderer.total_frames, "final_frames": final_frames,
                "duration_seconds": (renderer.total_frames+final_frames)/renderer.fps,
                "first_frame_simulation_time_s": renderer.frame_origin, "common_race_start_time_s": renderer.start_time,
                "alignment": "frame i is world video_period*(i+1); latest leaderboard observation at or before world time",
                "camera_selection": "hold first valid winner for 3 simulation seconds after first finished board observation, then highest-ranked active racer; explicit role and actual rank",
                "cars": {car_id: {"source_frames": video.count, "source_fps": video.fps} for car_id, video in renderer.videos.items()},
                "capacity_trial": renderer.capacity}
    metadata["final_classification"] = [{key: row.get(key) for key in
                                        ("car_id", "position", "status", "valid_finish", "finish_time_s", "progress_m")}
                                       for row in renderer.final_rows()]
    output.with_suffix(".render.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
