"""Replay a tournament's measured perception and frozen driver policies on CPU.

Example: python scripts/audit_tournament_drivers.py runs/my-tournament --wait

The default requires finished heat reports and closed controller summaries. It
does not contact a simulator/GPU, change a policy, or read the collision audit as
a control input. ``--allow-incomplete`` explicitly audits a live snapshot.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from race_lidar.hotlap_driver import RacingReference, longitudinal_control
from race_lidar.learned_driver import LearnedDriver
from race_lidar.race_policy import RacePolicy, TournamentTraffic, policy_state

TOLERANCE = 1e-6
SOURCE_PATHS = (
    "webots/controllers/tournament_racer/tournament_racer.py",
    "src/race_lidar/race_policy.py",
    "src/race_lidar/learned_driver.py",
    "src/race_lidar/hotlap_driver.py",
)
TRAFFIC_KEYS = ("front_distance_m", "front_lateral_m", "front_speed_mps",
                "left_clearance_m", "right_clearance_m")
ACTION_KEYS = ("target_speed_mps", "target_offset_m")


def json_rows(path):
    """Only a truncated final live record may be skipped; other damage fails."""
    lines = path.read_text().splitlines()
    rows = []
    truncated = 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            truncated += 1
    return rows, truncated


def timestamp(row):
    return round(float(row["simulation_time_s"]), 6)


def maximum_difference(actual, expected, keys):
    return max((float(np.max(np.abs(np.asarray(actual[key]) - np.asarray(expected[key]))))
                for key in keys), default=0.)


def heat_paths(directory):
    manifest_path = directory / "tournament.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return [(str(row["heat_id"]), directory / row.get("path", row["heat_id"]))
                for row in manifest["heats"]]
    if (directory / "config.json").exists():
        return [(directory.name, directory)]
    raise ValueError("Expected tournament.json or a heat config.json")


def readiness(directory):
    missing = []
    for heat_id, heat in heat_paths(directory):
        if not (heat / "config.json").exists() or not (heat / "heat_results.json").exists():
            missing.append(f"{heat_id}: heat report")
            continue
        config = json.loads((heat / "config.json").read_text())
        for entry in config["entrants"]:
            if not (heat / "cars" / entry["car_id"] / "controller_summary.json").exists():
                missing.append(f"{heat_id}/{entry['car_id']}: controller closure")
    return missing


def reference_from_config(config):
    limits = config.get("limits", {})
    return RacingReference.from_npz(config["reference_path"],
        max_speed=float(limits.get("max_speed_mps", config.get("max_speed_mps", 60.))),
        lateral_accel=float(limits.get("lateral_accel_mps2", config.get("lateral_accel_mps2", 8.))),
        brake_decel=float(limits.get("brake_decel_mps2", config.get("brake_decel_mps2", 8.))),
        acceleration=float(limits.get("acceleration_mps2", config.get("acceleration_mps2", 5.))))


def audit_car(heat, config, entry, reference):
    car_id = entry["car_id"]
    folder = Path(entry.get("output_dir", heat / "cars" / car_id))
    traces, trace_truncated = json_rows(folder / "control_trace.jsonl")
    predictions, prediction_truncated = json_rows(folder / "predictions.jsonl")
    actions, action_truncated = json_rows(folder / "policy_actions.jsonl")
    result = json.loads((folder / "result.json").read_text())
    trace_by_time = {timestamp(row): row for row in traces}
    action_by_time = {timestamp(row): row for row in actions}
    traffic = TournamentTraffic(reference)
    spec = entry.get("policy", "reference")
    if isinstance(spec, dict):
        policy_name = str(spec.get("name", spec.get("type", "neural")))
        model_path = spec.get("model_path", entry.get("model_path"))
    else:
        policy_name, model_path = str(spec), entry.get("model_path")
    model = LearnedDriver.from_npz(model_path) if model_path else None
    policy = RacePolicy(reference, model=model, name=policy_name,
                        seed=int(entry.get("seed", config.get("seed", 0))))
    checked = missing = traffic_mismatches = state_mismatches = 0
    raw_mismatches = neutral_mismatches = causal_mismatches = 0
    maximum_traffic_error = maximum_state_error = 0.
    for prediction in predictions:
        at = timestamp(prediction)
        trace, action = trace_by_time.get(at), action_by_time.get(at)
        if trace is None or action is None:
            missing += 1
            continue
        traffic.update(prediction["boxes"], trace["position_m"], trace["yaw_rad"],
                       trace["progress_m"], prediction["simulation_time_s"],
                       ego_speed=trace["speed_mps"])
        observed = traffic.observation(trace["progress_m"], trace["speed_mps"],
                                       prediction["simulation_time_s"])
        state = policy_state(reference, trace["progress_m"], trace["cross_track_error_m"],
                             trace["yaw_rad"], trace["speed_mps"],
                             trace["target_lane_offset_m"], observed)
        traffic_error = maximum_difference(observed, action["inputs"], TRAFFIC_KEYS)
        own_keys = [key for key in state if key not in TRAFFIC_KEYS]
        state_error = maximum_difference(state, action["inputs"], own_keys)
        traffic_mismatches += int(traffic_error > TOLERANCE)
        state_mismatches += int(state_error > TOLERANCE)
        maximum_traffic_error = max(maximum_traffic_error, traffic_error)
        maximum_state_error = max(maximum_state_error, state_error)
        replay = policy.predict(state, observed, action["feasible_speed_cap_mps"])
        raw_mismatches += int(maximum_difference(replay["raw_action"], action["raw_action"], ACTION_KEYS) > TOLERANCE)
        neutral_mismatches += int(maximum_difference(replay["no_traffic_counterfactual"],
                                                    action["no_traffic_counterfactual"], ACTION_KEYS) > TOLERANCE)
        causal_mismatches += replay["vision_changed_action"] != action["vision_changed_action"]
        checked += 1
    slow = [row for row in traces if row["time_s"] > 15. and row["speed_mps"] < 1.]
    continued = sum(row["simulation_time_s"] > slow[0]["simulation_time_s"] for row in actions) if slow else None
    progress_delta = np.diff([row["progress_m"] for row in traces])
    latencies = [row["inference_ms"] for row in predictions]
    checksum = model.checkpoint_sha256 if model else None
    checkpoint_matches = result.get("model_sha256") == checksum
    truncated = trace_truncated + prediction_truncated + action_truncated
    errors = []
    for count, message in ((missing, "missing timestamp joins"), (traffic_mismatches, "traffic feature mismatches"),
                           (state_mismatches, "own-state feature mismatches"), (raw_mismatches, "raw action mismatches"),
                           (neutral_mismatches, "counterfactual mismatches"), (causal_mismatches, "causal flag mismatches")):
        if count:
            errors.append(f"{count} {message}")
    if not checkpoint_matches:
        errors.append("result checkpoint checksum differs from replay checkpoint")
    if len(predictions) != len(actions) and config.get("perception_enabled", True):
        errors.append("prediction and policy-decision counts differ")
    if len(trace_by_time) != len(traces) or len(action_by_time) != len(actions):
        errors.append("duplicate trace or decision timestamps")
    if np.any(progress_delta < -.01):
        errors.append("ordered progress reversed")
    return {"car_id": car_id, "policy": policy_name, "status": result["status"],
        "audit_passed": not errors, "errors": errors, "last_trace_race_time_s": traces[-1]["time_s"] if traces else None,
        "prediction_frames": len(predictions), "nonempty_detection_frames": sum(bool(row["boxes"]) for row in predictions),
        "policy_decisions": len(actions), "decisions_with_confirmed_tracks": sum(bool(row["confirmed_tracks"]) for row in actions),
        "vision_changed_raw_actions": sum(row["vision_changed_action"] for row in actions),
        "measurement_frames_replayed": checked, "missing_timestamp_joins": missing,
        "traffic_feature_mismatches": traffic_mismatches, "own_state_feature_mismatches": state_mismatches,
        "maximum_absolute_traffic_feature_error": maximum_traffic_error,
        "maximum_absolute_own_state_feature_error": maximum_state_error,
        "policy_actions_replayed": checked, "neural_actions_replayed": checked if model else 0,
        "raw_action_mismatches": raw_mismatches, "counterfactual_mismatches": neutral_mismatches,
        "causal_flag_mismatches": causal_mismatches, "model_sha256": checksum,
        "result_checkpoint_matches": checkpoint_matches, "truncated_final_lines": truncated,
        "inference_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "near_zero_samples_after_15s": len(slow), "decisions_after_first_near_zero_sample": continued}


def audit_directory(directory, *, allow_incomplete=False):
    directory = Path(directory).resolve()
    incomplete = readiness(directory)
    if incomplete and not allow_incomplete:
        raise ValueError("Tournament is not closed: " + "; ".join(incomplete))
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    fingerprints = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}
    recorded_hashes = provenance.get("sha256", {})
    source_matches = {path: fingerprints[path] == recorded_hashes[path]
                      for path in SOURCE_PATHS if path in recorded_hashes}
    heats = []
    for heat_id, heat in heat_paths(directory):
        if not (heat / "config.json").exists():
            continue
        config = json.loads((heat / "config.json").read_text())
        reference = reference_from_config(config)
        cars = [audit_car(heat, config, entry, reference) for entry in config["entrants"]]
        heats.append({"heat_id": heat_id, "physics_threads": config.get("physics_threads", config.get("rules", {}).get("physics_threads")),
                      "audit_passed": all(car["audit_passed"] for car in cars), "cars": cars})
    count_keys = ("prediction_frames", "policy_decisions", "measurement_frames_replayed", "policy_actions_replayed",
                  "neural_actions_replayed", "traffic_feature_mismatches", "own_state_feature_mismatches",
                  "raw_action_mismatches", "counterfactual_mismatches", "causal_flag_mismatches", "missing_timestamp_joins")
    totals = {key: sum(car[key] for heat in heats for car in heat["cars"]) for key in count_keys}
    return {"schema": "tournament_driver_audit_v2", "audited_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tournament_directory": str(directory), "final": not incomplete, "incomplete": incomplete,
        "audit_passed": not incomplete and all(heat["audit_passed"] for heat in heats) and all(source_matches.values()),
        "gpu_used_by_audit": False, "policy_or_model_changes_by_audit": False, "tolerance": TOLERANCE,
        "method": ["Join recorded detector predictions, own-ego telemetry and policy decisions by simulation timestamp rounded to one microsecond.",
          "Rebuild each car's temporal tracks using only its saved boxes and own pose/speed. Compare all five traffic features.",
          "Rebuild own-state features from own pose/progress and shared map. Replay each frozen neural checkpoint or conventional policy, including its no-traffic counterfactual and causal-action flag.",
          "Compare replay checkpoint hashes with official result hashes and source hashes with the tournament's pre-run provenance."],
        "source_sha256": fingerprints, "source_matches_prerun_provenance": source_matches,
        "source_review": {"opponent_ground_truth_used_for_control": False,
          "control_localization": "Only own getSelf pose/orientation/velocity and the shared map; no opponent node lookup.",
          "pose_resets_or_writes_in_racer": False, "racer_terminates_simulation": False,
          "opponent_state_path": "Own LiDAR -> detector HTTP boxes -> temporal tracks -> policy and identical guards."},
        "blocked_car_review": {"runtime": "Cars continue physical stepping and 5 Hz perception/decisions while blocked; failed cars brake and continue stepping.",
          "recovery": "No reverse/contact recovery. Physical pileups or occupied lanes can yield legitimate time-limit DNFs.",
          "shared_low_speed_creep": "Common longitudinal feedback retains about 0.12 throttle near zero target/body speed until speed error triggers braking.",
          "zero_target_zero_body_speed_throttle_brake": list(longitudinal_control(0., 0., 0., mass=880.)),
          "newly_seen_stopped_vehicle_limitation": "At 5 Hz, the same-direction initial velocity prior can prevent a newly seen stationary car from associating when ego speed exceeds about 30 m/s: predicted motion exceeds the 6 m gate before the track is confirmed. The unconfirmed track can also suppress a nearby replacement. This common frozen perception limitation can worsen approaches to new pileups.",
          "stopped_vehicle_cpu_reproduction": {"ego_speed_mps": 40., "stationary_vehicle_initial_gap_m": 50.,
            "scan_period_s": .2, "scans": 6, "final_gap_m": 10., "confirmed_tracks_after_six_scans": 0}},
        "limits": ["No raw cloud is persisted for every frame; this reconstructs the saved box-to-control path, not the external GPU model's internal inference.",
          "Detector GPU/checkpoint identity and throughput belong to the separate capacity audit. Collision cleanliness belongs to the separate supervisor audit.",
          "This audits the specified tournament only; engineering-pilot counts are not imported.",
          "Classical reference pace and neural predicted pace are different algorithms with identical downstream physical/traffic guards.",
          "The multi-threaded simulation is an observed two-heat comparison, not a guarantee of deterministic or universal policy superiority."],
        "totals": totals, "heats": heats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for all manifest heats and controller closures")
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args()
    if not 1. <= args.poll_seconds <= 60.:
        parser.error("poll-seconds must be between 1 and 60")
    if args.wait:
        while missing := readiness(args.directory):
            print("Waiting for completed tournament: " + "; ".join(missing[:3]), flush=True)
            time.sleep(args.poll_seconds)
    audit = audit_directory(args.directory, allow_incomplete=args.allow_incomplete)
    output = args.output or args.directory / "driver_audit.json"
    output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "final": audit["final"], "audit_passed": audit["audit_passed"],
                      "totals": audit["totals"]}, indent=2), flush=True)
    raise SystemExit(0 if audit["audit_passed"] else 1)


if __name__ == "__main__":
    main()
