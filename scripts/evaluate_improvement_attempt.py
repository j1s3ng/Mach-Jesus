#!/usr/bin/env python3
"""Convert closed simulator artifacts into a conservative improvement-ledger outcome."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from race_lidar.race_results import evaluate_heat
from scripts.improvement_ledger import (OFFICIAL_DETECTOR_SHA256, candidate_descriptor, canonical,
    control_parameters, digest, environment_rules, environment_rules_hash, frozen_policy_parameters, verify_snapshot)


def read_json(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def policy_key(spec):
    return str(spec.get("name", spec.get("type", "neural"))) if isinstance(spec, dict) else str(spec)


def encoded_policy_parameters(value):
    try:
        return canonical(control_parameters(value) if value is not None else None).decode()
    except (ValueError, TypeError):
        return canonical({"invalid_recorded_parameters": str(value)}).decode()


def rows(path, issues, label):
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        issues.append(f"{label}:missing_or_corrupt_log:{type(error).__name__}")
        return []


def no_ground_truth_review(snapshot, manifest):
    """Inspect frozen control code, excluding the explicitly privileged director.

    This checks the current project's direct Webots APIs. It complements recorded
    perception identities and is not a proof about arbitrary adversarial Python.
    """
    forbidden = {"getFromDef", "getFromId", "getRoot", "getFromDevice", "getSFNode", "getMFNode",
                 "setSFVec3f", "setSFRotation", "setVelocity", "resetPhysics", "simulationResetPhysics",
                 "simulationReset", "addForce", "addForceWithOffset", "addTorque"}
    checked, violations = [], []
    for item in manifest["files"]:
        name = item["path"]
        control = name.endswith("tournament_racer.py") or "/src/race_lidar/" in "/"+name
        if not control or not name.endswith(".py"):
            continue
        source = Path(snapshot) / name
        tree = ast.parse(source.read_text(), filename=name)
        checked.append(name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
                violations.append(f"{name}:{node.lineno}:{node.func.attr}")
    racer_present = any(name.endswith("tournament_racer.py") for name in checked)
    return {"verified": bool(racer_present and not violations), "files_checked": checked,
            "forbidden_control_calls": violations,
            "scope": "Frozen direct control API review plus per-car measured-perception identities; not formal verification of arbitrary Python."}


def imported_runtime_sources(controller_path):
    """Require new control modules only when this frozen racer imports them."""
    names = {"lidar_hazards", "rich_driver"}
    imported = set()
    for node in ast.walk(ast.parse(Path(controller_path).read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.module == "race_lidar":
                imported.update(alias.name for alias in node.names if alias.name in names)
            else:
                imported.update(name for name in names if node.module == "race_lidar."+name)
        elif isinstance(node, ast.Import):
            imported.update(name for name in names for alias in node.names if alias.name == "race_lidar."+name)
    return {f"src/race_lidar/{name}.py" for name in imported}


def evaluate_attempt(run_directory, snapshot_path=None, expected_snapshot_sha256=None,
                     hazard=None, expected_rules_sha256=None, expected_policy=None,
                     expected_model_sha256=None, expected_detector_checkpoint_sha256=None,
                     expected_control_parameters=None):
    run = Path(run_directory).resolve()
    config = read_json(run / "config.json")
    aggregate = read_json(run / "heat_results.json")
    issues, records, audits = [], {}, []
    snapshot_path = snapshot_path or config.get("source_snapshot_path") or config.get("snapshot_path")
    manifest, snapshot_verified, source_review = None, False, {"verified": False, "files_checked": []}
    try:
        if snapshot_path is None:
            raise ValueError("missing_snapshot")
        manifest = verify_snapshot(snapshot_path)
        snapshot_verified = (expected_snapshot_sha256 is None
                             or manifest["snapshot_sha256"] == expected_snapshot_sha256)
        if not snapshot_verified:
            issues.append("snapshot_identity_mismatch")
        source_review = no_ground_truth_review(snapshot_path, manifest)
        if not source_review["verified"]:
            issues.append("frozen_control_source_review_failed")
    except (OSError, ValueError, SyntaxError, KeyError) as error:
        issues.append(f"snapshot_verification_failed:{type(error).__name__}")
    raw_rules_hash = digest(config.get("rules") or {})
    rules_hash = environment_rules_hash(config.get("rules") or {})
    if expected_rules_sha256 is not None and rules_hash != expected_rules_sha256:
        issues.append("common_rules_identity_mismatch")
    collisions = aggregate.get("collisions_by_car")
    if not isinstance(collisions, dict):
        issues.append("missing_authoritative_collision_audit")
        collisions = {}
    director_records = aggregate.get("cars", [])
    if isinstance(director_records, list):
        director_records = {str(row["car_id"]): row for row in director_records}
    else:
        director_records = {str(key): value for key, value in director_records.items()}
    hazard = hazard if hazard is not None else config.get("hazard")
    if isinstance(hazard, dict):
        hazard = dict(hazard)
    elif hazard is not None:
        raise ValueError("hazard must be a dictionary")
    entrants = [entry for entry in config["entrants"] if not entry.get("is_static", False)]
    evaluated_ids = config.get("evaluated_car_ids")
    if hazard is not None:
        ego = str(hazard.get("ego_car_id", config.get("ego_car_id", entrants[0]["car_id"])))
        evaluated_ids = [ego]
    elif evaluated_ids is None:
        evaluated_ids = [str(entry["car_id"]) for entry in entrants]
    evaluated_ids = [str(cid) for cid in evaluated_ids]
    snapshot_hashes = {item["sha256"] for item in (manifest or {}).get("files", [])}
    manifest_files = {item["path"]: item["sha256"] for item in (manifest or {}).get("files", [])}
    controller_name = next((name for name in manifest_files
                            if name.endswith("webots/controllers/tournament_racer/tournament_racer.py")), None)
    source_prefix = Path(*Path(controller_name).parts[:-4]) if controller_name else Path()
    required_sources = {"webots/controllers/tournament_racer/tournament_racer.py",
                        "src/race_lidar/race_policy.py", "src/race_lidar/race_control.py",
                        "src/race_lidar/hotlap_driver.py", "src/race_lidar/learned_driver.py"}
    if snapshot_path and controller_name:
        try:
            required_sources.update(imported_runtime_sources(Path(snapshot_path)/controller_name))
        except (OSError, SyntaxError):
            issues.append("cannot_inspect_frozen_racer_imports")
    for entry in entrants:
        cid = str(entry["car_id"])
        folder = run / "cars" / cid
        own = []
        try:
            result = read_json(folder / "result.json")
        except (OSError, ValueError) as error:
            issues.append(f"{cid}:missing_frozen_result:{type(error).__name__}")
            continue
        records[cid] = result
        if result.get("runtime_policy_parameters") is not None:
            try:
                control_parameters(result["runtime_policy_parameters"])
            except (ValueError, TypeError):
                own.append("invalid_runtime_policy_parameters")
        if result.get("policy") != policy_key(entry.get("policy", "reference")):
            own.append("runtime_policy_identity_mismatch")
        runtime_hashes = result.get("runtime_source_sha256") or {}
        if not isinstance(runtime_hashes, dict):
            own.append("invalid_runtime_source_hash_map")
            runtime_hashes = {}
        expected_root = (Path(snapshot_path)/source_prefix).resolve() if snapshot_path else None
        reported_root = result.get("runtime_source_root")
        if not reported_root or expected_root is None or Path(reported_root).resolve() != expected_root:
            own.append("runtime_source_root_not_frozen_snapshot")
        for name in sorted(required_sources):
            expected_hash = manifest_files.get(str(source_prefix/name))
            if expected_hash is None:
                own.append(f"runtime_source_missing_from_snapshot:{name}")
            elif runtime_hashes.get(name) != expected_hash:
                own.append(f"runtime_source_checksum_mismatch:{name}")
        for name, recorded_hash in runtime_hashes.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                own.append(f"unsafe_runtime_source_path:{name}")
                continue
            expected_hash = manifest_files.get(str(source_prefix/relative))
            if expected_hash is None:
                own.append(f"reported_runtime_source_not_in_snapshot:{name}")
            elif recorded_hash != expected_hash:
                own.append(f"runtime_source_checksum_mismatch:{name}")
        if result.get("car_id") != cid:
            own.append("result_identity_mismatch")
        if not (folder / "controller_summary.json").exists():
            own.append("controller_not_closed")
        overlap = collisions.get(cid)
        if not finite(overlap) or overlap < 0 or int(overlap) != overlap:
            own.append("missing_or_invalid_collision_count")
        for field in ("track_violation_steps", "position_resets_after_spawn"):
            value = result.get(field)
            if not finite(value) or value < 0 or int(value) != value:
                own.append(f"missing_or_invalid_{field}")
        if result.get("position_resets_after_spawn") not in (None, 0):
            own.append("runtime_pose_reset")
        if result.get("opponent_ground_truth_used_for_control") is not False:
            own.append("no_opponent_ground_truth_assertion_missing")
        if result.get("perception_enabled") is not True:
            own.append("measured_perception_disabled")
        for field in ("sensor_period_ms", "physics_step_ms"):
            expected = config.get("rules", {}).get(field, config.get(field))
            if expected is None or result.get(field) != expected:
                own.append(f"runtime_{field}_mismatch")
        aliases = {"max_speed_mps": ("max_speed_mps",),
                   "lateral_accel_mps2": ("lateral_accel_budget_mps2", "lateral_accel_mps2"),
                   "brake_decel_mps2": ("brake_decel_mps2",),
                   "acceleration_mps2": ("acceleration_mps2",)}
        limits = result.get("physical_limits", {})
        for actual, names in aliases.items():
            expected = next((config.get("rules", {}).get(name, config.get(name)) for name in names
                             if config.get("rules", {}).get(name, config.get(name)) is not None), None)
            if expected is None or limits.get(actual) != expected:
                own.append(f"physical_limit_mismatch:{actual}")
        model = entry.get("model_path")
        if isinstance(entry.get("policy"), dict):
            model = entry["policy"].get("model_path", model)
        if model:
            try:
                model_hash = checksum(model)
                if model_hash != result.get("model_sha256"):
                    own.append("runtime_model_checksum_mismatch")
                if model_hash not in snapshot_hashes:
                    own.append("runtime_model_not_in_snapshot")
            except OSError:
                own.append("runtime_model_unavailable")
        elif result.get("neural_driver_used"):
            own.append("neural_model_path_missing")
        try:
            reference_hash = checksum(config["reference_path"])
            if reference_hash != result.get("reference_sha256"):
                own.append("runtime_reference_checksum_mismatch")
            if reference_hash not in snapshot_hashes:
                own.append("runtime_reference_not_in_snapshot")
        except (OSError, KeyError):
            own.append("runtime_reference_unavailable")
        traces = rows(folder / "control_trace.jsonl", own, "trace")
        predictions = rows(folder / "predictions.jsonl", own, "predictions")
        actions = rows(folder / "policy_actions.jsonl", own, "actions")
        if not traces:
            own.append("no_physical_telemetry")
        if any(len(row.get("position_m", [])) != 3 or not all(finite(value) for value in row.get("position_m", []))
               for row in traces):
            own.append("missing_or_invalid_own_pose_telemetry")
        for previous, current in zip(traces, traces[1:]):
            if current["simulation_time_s"] <= previous["simulation_time_s"]:
                own.append("nonmonotonic_telemetry_clock")
            if current["progress_m"] < previous["progress_m"] - .001:
                own.append("nonmonotonic_ordered_progress")
        frames = {row.get("frame_id"): row for row in predictions}
        if len(frames) != len(predictions):
            own.append("duplicate_perception_frame_ids")
        if any(row.get("car_id") != cid for row in predictions + actions):
            own.append("perception_or_action_car_identity_mismatch")
        for action in actions:
            prediction = frames.get(action.get("vision_frame_id"))
            if prediction is None or prediction["simulation_time_s"] > action["simulation_time_s"] + 1e-7:
                own.append("action_has_missing_or_future_vision_frame")
        terminal_time = result.get("simulation_time_s")
        if not finite(terminal_time):
            own.append("missing_official_terminal_clock")
        else:
            for log, field in ((predictions, "sensor_frames"), (actions, "policy_decisions")):
                count = sum(row["simulation_time_s"] < terminal_time-1e-7 for row in log)
                if count != result.get(field):
                    own.append(f"official_{field}_count_mismatch")
        if result.get("completed") and result.get("valid_lap"):
            target = config.get("reference_length_m", config.get("rules", {}).get("reference_length_m"))
            target = target * config.get("laps", config.get("rules", {}).get("laps", 1)) if target else None
            brackets = [(a, b) for a, b in zip(traces, traces[1:])
                        if target is not None and a["progress_m"] < target <= b["progress_m"]]
            if len(brackets) != 1:
                own.append("full_finish_crossing_not_observed")
            else:
                a, b = brackets[0]
                fraction = (target-a["progress_m"]) / (b["progress_m"]-a["progress_m"])
                observed = a["simulation_time_s"] + fraction*(b["simulation_time_s"]-a["simulation_time_s"])
                observed -= config.get("start_time_s", 1.)
                reported = result.get("finish_time_s")
                if not finite(reported) or abs(observed-reported) > .15:
                    own.append("finish_clock_not_supported_by_physical_trace")
            independent = director_records.get(cid, {}).get("progress_m")
            if not finite(independent) or target is None or independent < target-.001:
                own.append("director_progress_does_not_support_finish")
        own = sorted(set(own))
        audits.append({"car_id": cid, "issues": own, "trace_rows": len(traces),
                       "prediction_rows": len(predictions), "action_rows": len(actions),
                       "runtime_sources_checked": len(runtime_hashes),
                       "required_runtime_sources": sorted(required_sources)})
        issues.extend(f"{cid}:{issue}" for issue in own)
    scoring_config = dict(config, entrants=entrants)
    standings = evaluate_heat(scoring_config, records, collisions)
    selected = [dict(row, policy_key=records.get(row["car_id"], {}).get("policy"),
                     model_sha256=records.get(row["car_id"], {}).get("model_sha256"),
                     runtime_detector_checkpoint_sha256=records.get(row["car_id"], {}).get("runtime_detector_checkpoint_sha256"),
                     runtime_policy_parameters=records.get(row["car_id"], {}).get("runtime_policy_parameters"))
                for row in standings if row["car_id"] in evaluated_ids]
    if len(selected) != len(evaluated_ids) or len(set(evaluated_ids)) != len(evaluated_ids):
        issues.append("missing_or_duplicate_evaluated_entrants")
    expected_descriptor = None
    parameters = expected_control_parameters if expected_control_parameters is not None else config.get("policy_parameters")
    frozen_parameters = None
    if manifest is not None:
        try:
            source_policy = expected_policy or (selected[0]["policy_key"] if selected else "reference")
            frozen_parameters = frozen_policy_parameters(manifest, source_policy)
        except ValueError:
            issues.append("invalid_frozen_policy_parameters")
    if parameters is None:
        parameters = frozen_parameters
    if parameters is not None:
        try:
            parameters = control_parameters(parameters)
            if frozen_parameters != parameters:
                issues.append("control_parameters_not_frozen_in_snapshot")
        except ValueError:
            issues.append("invalid_expected_control_parameters")
    expected_detector = (expected_detector_checkpoint_sha256
                         if expected_detector_checkpoint_sha256 is not None
                         else config.get("expected_detector_checkpoint_sha256"))
    if expected_policy is not None:
        try:
            descriptor_input = {
                "policy_key": expected_policy, "model_sha256": expected_model_sha256,
                "detector_sha256": expected_detector or OFFICIAL_DETECTOR_SHA256}
            if parameters is not None:
                descriptor_input["control_parameters"] = parameters
            expected_descriptor = candidate_descriptor({"descriptor": descriptor_input})
            expected_detector = expected_descriptor["detector_sha256"]
        except ValueError:
            issues.append("invalid_expected_candidate_descriptor")
    if expected_detector is not None:
        try:
            candidate_descriptor({"descriptor": {"policy_key": "reference", "model_sha256": None,
                                                  "detector_sha256": expected_detector}})
        except ValueError:
            issues.append("invalid_expected_detector_checkpoint")
        if expected_detector != OFFICIAL_DETECTOR_SHA256 and expected_detector not in snapshot_hashes:
            issues.append("adapted_detector_not_in_frozen_snapshot")
    for row in selected:
        if expected_policy is not None and row["policy_key"] != expected_policy:
            issues.append(f"{row['car_id']}:policy_not_expected_candidate")
        if (expected_policy is not None or expected_model_sha256 is not None) and row["model_sha256"] != expected_model_sha256:
            issues.append(f"{row['car_id']}:checkpoint_not_expected_candidate")
        if expected_detector is not None and row["runtime_detector_checkpoint_sha256"] != expected_detector:
            issues.append(f"{row['car_id']}:runtime_detector_not_expected_checkpoint")
        if parameters is not None:
            try:
                if control_parameters(row["runtime_policy_parameters"]) != parameters:
                    issues.append(f"{row['car_id']}:runtime_control_parameters_mismatch")
            except ValueError:
                issues.append(f"{row['car_id']}:runtime_control_parameters_missing_or_invalid")
    observed = {(record.get("policy"), record.get("model_sha256"), record.get("runtime_detector_checkpoint_sha256"),
                 encoded_policy_parameters(record.get("runtime_policy_parameters")))
                for record in records.values()}
    homogeneous = len(observed) == 1 and len(records) == len(entrants)
    observed_descriptors = []
    for key, model, detector, encoded_parameters in sorted(observed, key=lambda item: tuple(map(str, item))):
        descriptor = {"policy_key": key, "model_sha256": model, "detector_sha256": detector}
        actual_parameters = json.loads(encoded_parameters)
        if actual_parameters is not None:
            descriptor["control_parameters"] = actual_parameters
        observed_descriptors.append(descriptor)
    hazard_pass, hazard_evidence = None, None
    if hazard is not None:
        obstacle = hazard.get("progress_m")
        if obstacle is None and finite(hazard.get("progress_fraction")):
            length = config.get("reference_length_m", config.get("rules", {}).get("reference_length_m"))
            obstacle = hazard["progress_fraction"] * length if length else None
        if not finite(obstacle):
            issues.append("missing_static_obstacle_progress")
        record = records.get(ego, {})
        ego_progress = record.get("progress_m", record.get("ordered_progress_m"))
        director_progress = director_records.get(ego, {}).get("progress_m")
        passed = bool(finite(obstacle) and finite(ego_progress) and ego_progress >= obstacle+20.)
        if passed and (not finite(director_progress) or director_progress < obstacle+20.-6.):
            issues.append("director_progress_does_not_support_hazard_pass")
        zero_incidents = (collisions.get(ego) == 0 and record.get("track_violation_steps") == 0
                          and record.get("position_resets_after_spawn") == 0)
        measured_vision = record.get("sensor_frames", 0) > 0
        hazard_pass = bool(passed and zero_incidents and measured_vision)
        hazard_evidence = {"ego_car_id": ego, "obstacle_progress_m": obstacle,
                           "required_ego_progress_m": obstacle+20. if finite(obstacle) else None,
                           "ego_progress_m": ego_progress, "director_progress_m": director_progress,
                           "zero_incidents": zero_incidents, "measured_vision_frames": record.get("sensor_frames")}
    return {"schema_version": 1, "audit_verified": not issues,
            "evaluation_source_sha256": checksum(__file__),
            "snapshot_verified": snapshot_verified,
            "snapshot_sha256": (manifest or {}).get("snapshot_sha256"), "rules_sha256": rules_hash,
            "raw_rules_sha256": raw_rules_hash, "environment_rules": environment_rules(config.get("rules") or {}),
            "expected_control_parameters": parameters,
            "expected_candidate_descriptor": expected_descriptor,
            "expected_detector_checkpoint_sha256": expected_detector,
            "observed_candidate_descriptors": observed_descriptors, "homogeneous_field": homogeneous,
            "entrants": selected, "hazard_pass": hazard_pass, "hazard_evidence": hazard_evidence,
            "audit_issues": sorted(set(issues)), "per_car_audits": audits,
            "control_source_review": source_review, "run_directory": str(run),
            "method": "Frozen files and runtime hashes; authoritative supervisor overlap counts; own-perception identity/clock/counter checks; ordered physical finish traces.",
            "limits": ["Supervisor overlaps are conservative envelope samples, not proven physical-contact events.",
                       "Direct control API review is not formal verification; raw GPU inference internals are not replayed here."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-sha256")
    parser.add_argument("--rules-sha256")
    parser.add_argument("--expected-policy")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-detector-sha256")
    parser.add_argument("--expected-control-parameters", type=Path)
    parser.add_argument("--ego")
    parser.add_argument("--hazard-progress", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    hazard = ({"ego_car_id": args.ego, "progress_m": args.hazard_progress}
              if args.hazard_progress is not None else None)
    if hazard is not None and hazard["ego_car_id"] is None:
        hazard.pop("ego_car_id")
    outcome = evaluate_attempt(args.run_directory, args.snapshot, args.snapshot_sha256,
                               hazard, args.rules_sha256, args.expected_policy, args.expected_model_sha256,
                               args.expected_detector_sha256,
                               read_json(args.expected_control_parameters) if args.expected_control_parameters else None)
    output = args.output or args.run_directory / "ledger_outcome.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(outcome, indent=2, allow_nan=False)+"\n"); temporary.replace(output)
    print(json.dumps({key: outcome[key] for key in ("audit_verified", "snapshot_verified", "hazard_pass", "audit_issues")}, indent=2))
    return 0 if outcome["audit_verified"] and outcome["snapshot_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
