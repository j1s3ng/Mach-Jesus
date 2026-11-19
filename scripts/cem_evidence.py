"""Expected-world and closed-evidence bindings for engineering reward search."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from race_lidar.hotlap_driver import RacingReference
from scripts.improvement_ledger import (OFFICIAL_DETECTOR_SHA256, canonical, digest,
                                        environment_rules_hash, scenario_identity)
from scripts.improvement_queue_edit import atomic_json

EVIDENCE_VERSION = "cem_closed_expected_world_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenario_from_job(job):
    flags = {"--scenario": "scenario", "--cars": "cars", "--rotation": "rotation", "--seed": "seed",
             "--spacing": "spacing", "--seconds": "seconds", "--obstacle-progress": "obstacle_progress",
             "--grid-jitter": "grid_jitter", "--ego-progress": "ego_progress", "--ego-lane": "ego_lane",
             "--blocked-side": "blocked_side"}
    args, values = job["args"], {}
    for index, flag in enumerate(args):
        if flag in flags:
            if index + 1 >= len(args):
                raise ValueError("Incomplete scenario flag in expected queue job")
            values[flags[flag]] = args[index + 1]
    return scenario_identity(values)


def expected_case(job, *, snapshot_path, snapshot_sha256, policy, model_sha256):
    """Derive immutable expected spawns/rules from the queued job, not its output."""
    snapshot = Path(snapshot_path).resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text())
    policy_spec = next((row for row in manifest["metadata"]["policies"] if row["policy"] == policy), None)
    if not policy_spec or not policy_spec.get("model_relpath"):
        raise ValueError("Expected learned policy is absent from frozen roster")
    reference_path, model_path = snapshot / "reference.npz", snapshot / policy_spec["model_relpath"]
    if sha(model_path) != model_sha256:
        raise ValueError("Expected checkpoint does not match the frozen model")
    args = job["args"]
    for flag, value in (("--source-root", snapshot), ("--reference", reference_path)):
        if flag not in args or Path(args[args.index(flag) + 1]).resolve() != value:
            raise ValueError("Expected job source/reference does not match selected foundation")
    if "--single-policy" not in args or args[args.index("--single-policy") + 1] != policy:
        raise ValueError("CEM evidence requires the explicitly selected homogeneous policy")
    scene = scenario_from_job(job)
    reference = RacingReference.from_npz(reference_path)
    rng = np.random.default_rng(scene["seed"])
    entrants = []
    for index in range(scene["cars"]):
        progress = -scene["spacing"] * index - float(rng.uniform(0., scene["grid_jitter"]))
        lane = (-2.5 if index % 2 == 0 else 2.5) + float(rng.uniform(-scene["grid_jitter"], scene["grid_jitter"]))
        static = scene["scenario"] == "stopped" and index > 0
        if scene["scenario"] == "stopped":
            progress = scene["ego_progress"] if index == 0 else scene["obstacle_progress"]
            lane = scene["ego_lane"] if index == 0 else 0. if index == 1 else -3.2 if scene["blocked_side"] == "right" else 3.2
        elif scene["scenario"] == "parking":
            progress = reference.length - 130. - scene["spacing"] * index
        point, yaw, _, _ = reference.sample(progress)
        point = point + lane * np.array([-math.sin(yaw), math.cos(yaw)])
        entrants.append({"car_id": f"R{index + 1:02d}", "grid_index": index, "seed": scene["seed"] + index,
                         "start_progress_m": progress, "start_lane_m": lane, "is_static": static,
                         "policy": "stationary_obstacle" if static else policy,
                         "model_path": None if static else str(model_path),
                         "expected_initial_xy_m": point.tolist(), "expected_initial_yaw_rad": yaw})
    rules = {"protocol": "race-improvement-v2", "reference_length_m": reference.length, "laps": 1,
             "sensor_period_ms": 200, "physics_step_ms": 20, "max_speed_mps": 60.,
             "lateral_accel_budget_mps2": 8., "friction_coefficient": 1.5,
             "vehicle_model": "generic Webots Car 880kg electric", "physics_threads": 8,
             "field_size": scene["cars"], "control_version": "predictive_v2", "visible_distance_m": 70.,
             "grid_spacing_m": scene["spacing"], "post_finish_mode": "physical_staggered_runoff_parking",
             "race_cutoff_seconds": scene["seconds"], "grid_jitter_m": scene["grid_jitter"],
             "lidar_max_range_m": 100., "brake_decel_mps2": 8., "acceleration_mps2": 5.,
             "track_error_limit_m": 6., "total_vehicle_mass_kg": 880.}
    return {"version": EVIDENCE_VERSION, "job_id": job["job_id"], "queue_job_sha256": digest(job),
            "output": str((Path(__file__).resolve().parents[1] / job["output"]).resolve()),
            "scenario": scene, "rules": rules, "rules_sha256": environment_rules_hash(rules),
            "snapshot_path": str(snapshot), "snapshot_sha256": snapshot_sha256,
            "reference_path": str(reference_path), "reference_sha256": sha(reference_path),
            "model_sha256": model_sha256, "policy": policy, "entrants": entrants}


def config_issues(config, expected):
    """Validate exact seeded case identity including every configured spawn."""
    issues = []
    scene = expected["scenario"]
    values = {"scenario": scene["scenario"], "seed": scene["seed"], "grid_rotation": scene["rotation"],
              "output_dir": expected["output"],
              "reference_length_m": expected["rules"]["reference_length_m"], "laps": 1,
              "start_time_s": 1., "max_sim_seconds": 1. + scene["seconds"],
              "source_root": expected["snapshot_path"], "reference_path": expected["reference_path"],
              "snapshot_sha256": expected["snapshot_sha256"], "policy_parameters": {"visible_distance_m": 70.},
              "mode": "capacity" if scene["scenario"] != "race" else "race"}
    for key, value in values.items():
        if config.get(key) != value:
            issues.append("expected_case_config_mismatch:" + key)
    if config.get("rules") != expected["rules"]:
        issues.append("expected_case_rules_mismatch")
    entries = config.get("entrants", [])
    if not isinstance(entries, list) or [row.get("car_id") for row in entries] != [row["car_id"] for row in expected["entrants"]]:
        issues.append("expected_case_entrant_ids_mismatch"); entries = []
    for actual, planned in zip(entries, expected["entrants"]):
        for key in ("grid_index", "seed", "start_progress_m", "start_lane_m", "policy", "model_path"):
            if actual.get(key) != planned[key]:
                issues.append(f"expected_case_spawn_or_policy_mismatch:{planned['car_id']}:{key}")
        if bool(actual.get("is_static", False)) != planned["is_static"]:
            issues.append("expected_case_static_role_mismatch:" + planned["car_id"])
    if config.get("cars") != {row["car_id"]: row for row in entries}:
        issues.append("config_cars_entrant_mapping_mismatch")
    if scene["scenario"] == "stopped":
        if config.get("hazard") != {"ego_car_id": "R01", "progress_m": scene["obstacle_progress"]}:
            issues.append("expected_hazard_identity_mismatch")
    return issues


def closed_issues(run, config, expected):
    """Bind scored/physical entrants and initial measured poses to the case.

    First Supervisor poses are offline *case identity evidence*, not a student
    input, trajectory label, lane oracle or counterfactual control observation.
    """
    run = Path(run)
    issues = []
    required = ("heat_results.json", "gt_poses.jsonl")
    if any(not (run / name).is_file() for name in required):
        return ["missing_closed_authoritative_heat_or_spawn_poses"]
    heat = json.loads((run / "heat_results.json").read_text())
    if heat.get("status") not in ("time_limit", "all_cars_reported") or heat.get("mode") != config["mode"]:
        issues.append("heat_is_not_a_closed_matching_mode")
    if heat.get("rules") != config["rules"] or heat.get("car_count") != len(expected["entrants"]):
        issues.append("closed_heat_rules_or_field_identity_mismatch")
    actual = heat.get("cars", [])
    if not isinstance(actual, list) or len(actual) != len(expected["entrants"]) or {row.get("car_id") for row in actual} != {row["car_id"] for row in expected["entrants"]}:
        issues.append("closed_heat_entrant_ids_mismatch"); actual = []
    by_id = {row["car_id"]: row for row in actual}
    if not isinstance(heat.get("collisions_by_car"), dict) or set(heat["collisions_by_car"]) != set(by_id):
        issues.append("missing_authoritative_per_car_collision_counts")
    try:
        with (run / "gt_poses.jsonl").open() as stream:
            first = json.loads(next(line for line in stream if line.strip()))
        poses = first["cars"]
        if isinstance(poses, dict):
            poses = [{"car_id": key, **value} for key, value in poses.items()]
        if len(poses) != len(expected["entrants"]):
            issues.append("initial_physical_pose_count_mismatch")
        poses = {row["car_id"]: row for row in poses}
        if first["timestamp_ns"] > 500_000_000 or set(poses) != set(by_id):
            issues.append("initial_physical_pose_clock_or_identity_mismatch")
        for entry in expected["entrants"]:
            pose = poses[entry["car_id"]]
            error = np.linalg.norm(np.asarray(pose["position_m"][:2]) - entry["expected_initial_xy_m"])
            rotation = np.asarray(pose["rotation"]).reshape(3, 3)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            yaw_error = math.atan2(math.sin(yaw - entry["expected_initial_yaw_rad"]), math.cos(yaw - entry["expected_initial_yaw_rad"]))
            if not np.isfinite(error) or error > .05 or abs(yaw_error) > .01:
                issues.append("initial_physical_spawn_mismatch:" + entry["car_id"])
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        issues.append("invalid_initial_physical_spawn_evidence")
    for entry in expected["entrants"]:
        folder = run / "cars" / entry["car_id"]
        if not (folder / "result.json").is_file():
            issues.append("missing_closed_car_result:" + entry["car_id"]); continue
        result = json.loads((folder / "result.json").read_text())
        if result.get("car_id") != entry["car_id"]:
            issues.append("closed_car_identity_mismatch:" + entry["car_id"])
        if not entry["is_static"]:
            if not (folder / "controller_summary.json").is_file():
                issues.append("controller_not_closed:" + entry["car_id"])
            for key in ("start_progress_m", "start_lane_m"):
                if result.get(key) != entry[key] or by_id.get(entry["car_id"], {}).get(key) != entry[key]:
                    issues.append(f"closed_result_spawn_mismatch:{entry['car_id']}:{key}")
    return issues


def fresh_audit(run, expected, cache_dir):
    """Re-audit each closed evidence fingerprint once, then verify cached files.

    Stat signatures avoid repeatedly hashing large closed logs. Any source
    size/mtime change, expected case change, evaluator change or frozen-source
    change invalidates the cache and triggers fresh file hashes plus CPU audit.
    """
    from scripts.evaluate_improvement_attempt import evaluate_attempt
    run, cache_dir = Path(run), Path(cache_dir)
    snapshot = Path(expected["snapshot_path"])
    manifest = json.loads((snapshot / "manifest.json").read_text())
    files = [run / name for name in ("config.json", "heat_results.json", "ledger_outcome.json", "gt_poses.jsonl")]
    for entry in expected["entrants"]:
        folder = run / "cars" / entry["car_id"]
        files.append(folder / "result.json")
        if not entry["is_static"]:
            files.extend(folder / name for name in ("controller_summary.json", "control_trace.jsonl", "policy_actions.jsonl", "predictions.jsonl"))
    files.append(snapshot / "manifest.json")
    files.extend(snapshot / row["path"] for row in manifest["files"])
    files.extend(Path(__file__).resolve().parent / name for name in ("evaluate_improvement_attempt.py", "improvement_ledger.py", "cem_evidence.py"))
    def file_signature(path):
        stat = path.stat()
        row = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        # Authoritative configs/results, source and models are small enough to
        # rehash every poll even if a copy tool preserves timestamps. Closed
        # large JSONL clouds/telemetry use stat invalidation plus their archived
        # full SHA256; this is not protection against adversarial timestamp forgery.
        if path.suffix in (".json", ".py", ".npz"):
            row["sha256"] = sha(path)
        return row
    signature = [file_signature(path) for path in files]
    identity = digest({"expected_case": expected, "stat_signature": signature, "version": EVIDENCE_VERSION})
    cache = cache_dir / (hashlib.sha256(str(run.resolve()).encode()).hexdigest() + ".json")
    if cache.is_file():
        saved = json.loads(cache.read_text())
        if saved.get("identity") == identity and saved.get("audit_sha256") == digest(saved.get("audit")):
            return saved["audit"], {"identity": identity, "cached": True, "cache_path": str(cache)}
    file_hashes = [{"path": str(path.resolve()), "sha256": sha(path)} for path in files]
    audited = evaluate_attempt(run, snapshot, expected["snapshot_sha256"],
        expected_rules_sha256=expected["rules_sha256"], expected_policy=expected["policy"],
        expected_model_sha256=expected["model_sha256"], expected_detector_checkpoint_sha256=OFFICIAL_DETECTOR_SHA256,
        expected_control_parameters={"visible_distance_m": 70.})
    after = [file_signature(path) for path in files]
    if signature != after:
        raise ValueError("Closed evidence changed during independent CPU audit")
    # The evaluator returns validated rows, but convenience aggregate flags are
    # normally added by the heat launcher, so reconstruct them independently.
    entries = audited.get("entrants", [])
    audited["all_clean_finishes"] = bool(entries) and all(row.get("clean_finish") is True for row in entries)
    audited["all_valid_finishes"] = bool(entries) and all(row.get("valid_finish") is True for row in entries)
    cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(cache, {"identity": identity, "file_sha256": file_hashes, "audit": audited, "audit_sha256": digest(audited)})
    return audited, {"identity": identity, "cached": False, "cache_path": str(cache)}
