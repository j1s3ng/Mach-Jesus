"""CPU-only old/new tracker comparison using frozen boxes and separate poses.

Replay runs in separate processes with each source tree first on sys.path. All
sensor frames are replayed at their recorded cadence; --stride only subsamples
evaluation. Other-car poses are used exclusively after tracking, for evaluation.
Center matching and visibility-unknown potential-target recall are not official
detector AP or a claim that every simulator car is visible to the sensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (2., 4., 6.)


def read_rows(path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def at(row):
    return round(float(row["simulation_time_s"]), 6)


def heats(directory, selected=None):
    manifest = directory / "tournament.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text())["heats"]
        return [(entry["heat_id"], directory / entry.get("path", entry["heat_id"]))
                for entry in entries if not selected or entry["heat_id"] in selected]
    return [(directory.name, directory)]


def blank_counts():
    return {"frames": 0, "predictions": 0, "potential_targets": 0, "true_positives": 0,
            "false_positives": 0, "false_negatives": 0, "nearest_stopped_threat_frames": 0,
            "nearest_stopped_threat_hits": 0, "detected_nearest_stopped_frames": 0,
            "detected_nearest_stopped_hits": 0, "detected_potential_targets": 0,
            "detected_potential_target_hits": 0}


def summarize(counts):
    return {**counts,
        "center_match_precision": counts["true_positives"] / counts["predictions"] if counts["predictions"] else None,
        "visibility_unknown_potential_target_recall": counts["true_positives"] / counts["potential_targets"] if counts["potential_targets"] else None,
        "detector_conditioned_target_recall": counts["detected_potential_target_hits"] / counts["detected_potential_targets"] if counts["detected_potential_targets"] else None,
        "nearest_stopped_potential_threat_recall": counts["nearest_stopped_threat_hits"] / counts["nearest_stopped_threat_frames"] if counts["nearest_stopped_threat_frames"] else None,
        "detected_nearest_stopped_threat_recall": counts["detected_nearest_stopped_hits"] / counts["detected_nearest_stopped_frames"] if counts["detected_nearest_stopped_frames"] else None}


def worker(args):
    # This is intentionally before importing either source tree or NumPy.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    source = args.tracker_source.resolve()
    sys.path.insert(0, str(source / "src"))
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from race_lidar.hotlap_driver import RacingReference, project_near
    from race_lidar.race_policy import TournamentTraffic

    def match(predictions, targets, threshold):
        if not predictions or not targets:
            return set()
        predicted = np.asarray([row["position_m"][:2] for row in predictions])
        actual = np.asarray([row["center"] for row in targets])
        distances = np.linalg.norm(predicted[:, None, :] - actual[None, :, :], axis=2)
        costs = np.full((len(predictions), len(targets) + len(predictions)), 1000.)
        costs[:, :len(targets)] = np.where(distances <= threshold, distances, 1e6)
        indices, columns = linear_sum_assignment(costs)
        return {int(column) for index, column in zip(indices, columns)
                if column < len(targets) and distances[index, column] <= threshold}

    def in_window(row, limit):
        return (-45. <= row["relative_distance_m"] <= limit and abs(row["lateral_m"]) <= 7.)

    def raw_boxes(boxes, ego, reference):
        yaw = ego["yaw_rad"]
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        rows = []
        for box in boxes:
            center = np.asarray(box.get("center", []), float)
            size = np.asarray(box.get("size", []), float)
            score = float(box.get("score", 0.))
            if center.shape != (3,) or size.shape != (3,) or not np.isfinite(np.r_[center, size, score]).all():
                continue
            if score < .35 or not (3. <= size[0] <= 7.5 and 1.25 <= size[1] <= 2.9 and .55 <= size[2] <= 2.5):
                continue
            if np.linalg.norm(center[:2]) < 3.:
                continue
            point = rotation @ center[:2] + np.asarray(ego["position_m"])[:2]
            projection = project_near(reference, point, ego["progress_m"], window=110.)
            row = {"position_m": point.tolist(), "relative_distance_m": projection["progress_m"] - ego["progress_m"],
                   "lateral_m": projection["lateral_m"]}
            if in_window(row, 95.):
                rows.append(row)
        return rows

    def potential_targets(poses, car_id, timestamp, ego, reference):
        targets = []
        for other_id, samples in poses.items():
            if other_id == car_id or timestamp not in samples:
                continue
            other = samples[timestamp]
            center = np.asarray(other["position_m"])[:2] + 1.6 * np.array([math.cos(other["yaw_rad"]), math.sin(other["yaw_rad"])])
            distance = float(np.linalg.norm(center - np.asarray(ego["position_m"])[:2]))
            if not 3. <= distance <= 100.:
                continue
            projection = project_near(reference, center, ego["progress_m"], window=110.)
            target = {"car_id": other_id, "center": center.tolist(), "relative_distance_m": projection["progress_m"] - ego["progress_m"],
                      "lateral_m": projection["lateral_m"], "speed_mps": other["speed_mps"],
                      "gross_pose_tilt": abs(other.get("roll_rad", 0.)) > .6 or abs(other["position_m"][2]) > 1.}
            if in_window(target, 95.):
                targets.append(target)
        return targets

    started = time.perf_counter()
    totals = {channel: {str(int(limit)): {str(int(threshold)): blank_counts() for threshold in THRESHOLDS}
                       for limit in (50., 95.)} for channel in ("raw_detector", "confirmed_tracks", "control_hazards")}
    per_car = []
    frames_replayed = frames_evaluated = poses_missing = 0
    config_sums = {}
    event_path = args.result.with_suffix(".stopped_events.jsonl")
    with event_path.open("w") as events:
        for heat_id, heat in heats(args.directory, args.heats):
            config_path = heat / "config.json"
            config = json.loads(config_path.read_text())
            config_sums[heat_id] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            reference = RacingReference.from_npz(config["reference_path"])
            # Opponent traces stay in evaluation-only storage. They are never
            # passed to tracker.update, observation, or a driving policy.
            poses = {entry["car_id"]: {at(row): row for row in read_rows(heat / "cars" / entry["car_id"] / "control_trace.jsonl")}
                     for entry in config["entrants"]}
            for entry in config["entrants"]:
                car_id = entry["car_id"]
                if args.cars and car_id not in args.cars:
                    continue
                tracker_options = {}
                if args.variant == "candidate_v2" and args.rapid_hazard_confidence is not None:
                    tracker_options["rapid_hazard_confidence"] = args.rapid_hazard_confidence
                traffic = TournamentTraffic(reference, **tracker_options)
                car_frames = car_evaluated = car_missing = 0
                predictions = read_rows(heat / "cars" / car_id / "predictions.jsonl")
                for frame_index, prediction in enumerate(predictions):
                    timestamp = at(prediction)
                    if args.max_time is not None and timestamp - config.get("start_time_s", 1.) > args.max_time:
                        break
                    if args.max_frames is not None and frame_index >= args.max_frames:
                        break
                    ego = poses[car_id].get(timestamp)
                    if ego is None:
                        car_missing += 1
                        continue
                    traffic.update(prediction["boxes"], ego["position_m"], ego["yaw_rad"], ego["progress_m"],
                                   prediction["simulation_time_s"], ego_speed=ego["speed_mps"])
                    car_frames += 1
                    if frame_index % args.stride:
                        continue
                    observation = traffic.observation(ego["progress_m"], ego["speed_mps"], prediction["simulation_time_s"])
                    # Evaluation truth is accessed only after tracker output.
                    targets = potential_targets(poses, car_id, timestamp, ego, reference)
                    raw = raw_boxes(prediction["boxes"], ego, reference)
                    channels = {"raw_detector": raw, "confirmed_tracks": observation["confirmed_tracks"],
                                "control_hazards": observation.get("hazard_tracks", observation["confirmed_tracks"])}
                    car_evaluated += 1
                    for limit in (50., 95.):
                        candidates = [target for target in targets if in_window(target, limit)]
                        lane_ahead = [(i, target) for i, target in enumerate(candidates)
                                      if target["relative_distance_m"] > 0.
                                      and abs(target["lateral_m"] - ego["cross_track_error_m"]) < 2.3]
                        nearest = min(lane_ahead, key=lambda row: row[1]["relative_distance_m"]) if lane_ahead else None
                        stopped_index = nearest[0] if nearest and nearest[1]["relative_distance_m"] > 7. and nearest[1]["speed_mps"] < 1. else None
                        for threshold in THRESHOLDS:
                            raw_hits = match([row for row in raw if in_window(row, limit)], candidates, threshold)
                            event = {"heat": heat_id, "car_id": car_id, "time_s": timestamp - config.get("start_time_s", 1.),
                                     "limit_m": limit, "threshold_m": threshold,
                                     "target_car_id": nearest[1]["car_id"] if stopped_index is not None else None,
                                     "gap_m": nearest[1]["relative_distance_m"] if stopped_index is not None else None,
                                     "detected": stopped_index in raw_hits if stopped_index is not None else None}
                            for channel, rows in channels.items():
                                predicted = [row for row in rows if in_window(row, limit)]
                                hits = match(predicted, candidates, threshold)
                                count = totals[channel][str(int(limit))][str(int(threshold))]
                                count["frames"] += 1
                                count["predictions"] += len(predicted)
                                count["potential_targets"] += len(candidates)
                                count["true_positives"] += len(hits)
                                count["false_positives"] += len(predicted) - len(hits)
                                count["false_negatives"] += len(candidates) - len(hits)
                                count["detected_potential_targets"] += len(raw_hits)
                                count["detected_potential_target_hits"] += len(hits & raw_hits)
                                if stopped_index is not None:
                                    count["nearest_stopped_threat_frames"] += 1
                                    count["nearest_stopped_threat_hits"] += stopped_index in hits
                                    count["detected_nearest_stopped_frames"] += stopped_index in raw_hits
                                    count["detected_nearest_stopped_hits"] += stopped_index in raw_hits and stopped_index in hits
                                    event[channel] = stopped_index in hits
                            if stopped_index is not None:
                                events.write(json.dumps(event, allow_nan=False) + "\n")
                frames_replayed += car_frames
                frames_evaluated += car_evaluated
                poses_missing += car_missing
                per_car.append({"heat": heat_id, "car_id": car_id, "frames_replayed": car_frames,
                                "frames_evaluated": car_evaluated, "missing_ego_poses": car_missing})
                print(f"{args.variant}: {heat_id}/{car_id}: replayed {car_frames}, evaluated {car_evaluated}", flush=True)
    result = {"variant": args.variant, "source_root": str(source), "tracker_source_sha256": hashlib.sha256((source / "src/race_lidar/race_policy.py").read_bytes()).hexdigest(),
              "config_sha256": config_sums, "evaluation_stride": args.stride, "frames_replayed": frames_replayed,
              "frames_evaluated": frames_evaluated, "missing_ego_poses": poses_missing,
              "wall_seconds": time.perf_counter() - started, "gpu_used": False, "opponent_gt_passed_to_tracker": False,
              "tracker_configuration": {"rapid_hazard_confidence": getattr(traffic, "rapid_hazard_confidence", None),
                                        "hazard_hold_s": getattr(traffic, "hazard_hold_s", None)},
              "stopped_events_path": str(event_path), "cars": per_car,
              "metrics": {channel: {limit: {threshold: summarize(count) for threshold, count in by_threshold.items()}
                                    for limit, by_threshold in by_limit.items()} for channel, by_limit in totals.items()}}
    args.result.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--new-source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--cars", nargs="+")
    parser.add_argument("--heats", nargs="+")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--rapid-hazard-confidence", type=float,
                        help="Candidate-only first-scan hazard threshold override")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tracker-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    results = []
    for variant, source in (("frozen_v1", args.old_source), ("candidate_v2", args.new_source)):
        result = args.output / f"{variant}.json"
        command = [sys.executable, str(Path(__file__).resolve()), str(args.directory.resolve()),
                   "--worker", "--old-source", str(args.old_source.resolve()), "--tracker-source", str(source.resolve()),
                   "--output", str(args.output.resolve()), "--variant", variant, "--result", str(result.resolve()),
                   "--stride", str(args.stride)]
        for name, values in (("--cars", args.cars), ("--heats", args.heats)):
            if values:
                command.extend([name, *values])
        for name, value in (("--max-frames", args.max_frames), ("--max-time", args.max_time)):
            if value is not None:
                command.extend([name, str(value)])
        if args.rapid_hazard_confidence is not None:
            command.extend(["--rapid-hazard-confidence", str(args.rapid_hazard_confidence)])
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        results.append(json.loads(result.read_text()))
    old, new = results
    if old["frames_replayed"] != new["frames_replayed"] or old["frames_evaluated"] != new["frames_evaluated"]:
        raise RuntimeError("Old/new evaluation populations differ")
    report = {"schema": "tracker_center_replay_comparison_v1", "run_directory": str(args.directory.resolve()),
        "frames_replayed_per_variant": old["frames_replayed"], "frames_evaluated_per_variant": old["frames_evaluated"],
        "evaluation_stride": args.stride, "gpu_used": False, "production_sources_modified_by_comparison": False,
        "method": ["Separate subprocesses import frozen and candidate source trees to avoid Python module collisions.",
                   "Replay every saved measured detector frame at its original timestamps; stride affects evaluation only.",
                   "Tracker input contains its own ego pose and saved boxes only. Other-car physical poses are accessed after tracker output for evaluation.",
                   "One-to-one 2D vehicle-center matching at 2, 4 and 6 metres; potential targets within 100 m Euclidean range, -45 m rear to 50 m/95 m ahead, +/-7 m lateral.",
                   "Nearest in-lane stopped threat means the closest potential vehicle ahead, within 2.3 m of ego's track lateral position, with a precontact gap >7 m and body speed <1 m/s."],
        "limits": ["Potential-target recall has unknown visibility and occlusion. It is not detector AP and does not imply every simulator car should be detected.",
                   "Raw-detector-conditioned recall isolates some association losses but is still approximate center matching, not manual sensor annotation.",
                   "Ground-truth center uses rear-axle XY plus a nominal 1.6 m forward yaw offset. Full pitch was not logged; large-tilt vehicle centers have additional approximation error.",
                   "Completed and failed cars remain potential hazards while physically present. Frames can include post-finish cooldown evaluations.",
                   "This offline observation comparison cannot establish physical collision-rate or lap-time improvement; independent physical trials remain necessary."],
        "variants": results}
    path = args.output / "comparison.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"comparison": str(path), "frames_per_variant": old["frames_replayed"],
                      "evaluated_per_variant": old["frames_evaluated"]}, indent=2))


if __name__ == "__main__":
    main()
