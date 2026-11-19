"""Matched engineering detector/driver failure proof; no control modifications."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from race_lidar.rich_driver import RichDriver
from race_lidar.race_control import combine_observations
from scripts.analyze_rich_measured import guarded
from scripts.improvement_ledger import verify_snapshot
from scripts.training_sources import validate_training_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sanitized_state(row):
    return {**row["inputs"], "hazard_tracks": row["hazard_tracks"]}


def summarize_action(row, model, teacher):
    state = sanitized_state(row)
    label = teacher.intent_teacher(state)
    return {"simulation_time_s": row["simulation_time_s"], "measured_inputs": state,
            "recorded_raw_action": row["raw_action"], "replayed_action": model.predict(state),
            "frozen005_teacher_lane_m": label["target_offset_m"],
            "frozen005_teacher_raw_lane_costs": label["raw_lane_costs"].tolist(),
            "frozen005_teacher_regularized_lane_costs": label["lane_costs"].tolist(),
            "one_step_fixed_guard": guarded(row["raw_action"], state, row["feasible_speed_cap_mps"], 70.)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=ROOT / "runs/improvement-12h/engineering/rich005-s17-left")
    parser.add_argument("--new", type=Path, default=ROOT / "runs/improvement-12h/engineering/cv425-nn00517-left")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher_path = ROOT / "runs/improvement-12h/rich-driver-005/source/scripts/train_rich_driver.py"
    spec = importlib.util.spec_from_file_location("frozen005_trajectory_teacher", teacher_path)
    teacher = importlib.util.module_from_spec(spec); spec.loader.exec_module(teacher)
    loaded, hashes = [], {}
    for run in (args.old, args.new):
        validate_training_source(run, require_config=True)
        config = json.loads((run / "config.json").read_text())
        manifest = verify_snapshot(config["source_root"])
        result = json.loads((run / "cars/R01/result.json").read_text())
        audit = json.loads((run / "ledger_outcome.json").read_text())
        action = rows(run / "cars/R01/policy_actions.jsonl")
        trace = rows(run / "cars/R01/control_trace.jsonl")
        with (run / "gt_poses.jsonl").open() as stream:
            spawn = json.loads(next(line for line in stream if line.strip()))
        for name in ("config.json", "ledger_outcome.json", "heat_results.json", "cars/R01/result.json",
                     "cars/R01/policy_actions.jsonl", "cars/R01/predictions.jsonl", "cars/R01/control_trace.jsonl"):
            hashes[str((run / name).resolve())] = sha(run / name)
        loaded.append({"run": run, "config": config, "manifest": manifest, "result": result,
                       "audit": audit, "actions": action, "trace": trace, "spawn": spawn})
    old, new = loaded
    spawn_keys = ("car_id", "grid_index", "start_progress_m", "start_lane_m", "seed", "policy", "is_static")
    spawns = [{key: entry.get(key) for key in spawn_keys} for entry in old["config"]["entrants"]]
    spawns_new = [{key: entry.get(key) for key in spawn_keys} for entry in new["config"]["entrants"]]
    pose_error = max(float(np.max(np.abs(np.asarray(a["position_m"]) - np.asarray(b["position_m"]))))
                     for a, b in zip(old["spawn"]["cars"], new["spawn"]["cars"]))
    model_paths = [Path(item["config"]["entrants"][0]["model_path"]) for item in loaded]
    checks = {"rules_identical": old["config"]["rules"] == new["config"]["rules"],
              "seed_identical": old["config"]["seed"] == new["config"]["seed"],
              "configured_all_spawns_identical": spawns == spawns_new,
              "first_physical_pose_all_ids_identical": [r["car_id"] for r in old["spawn"]["cars"]] == [r["car_id"] for r in new["spawn"]["cars"]],
              "first_physical_pose_max_abs_error_m": pose_error,
              "initial_own_trace_position_identical": old["trace"][0]["position_m"] == new["trace"][0]["position_m"],
              "all8_runtime_source_hashes_identical": old["result"]["runtime_source_sha256"] == new["result"]["runtime_source_sha256"] and len(old["result"]["runtime_source_sha256"]) == 8,
              "driver_checkpoint_identical": sha(model_paths[0]) == sha(model_paths[1]) == old["result"]["model_sha256"] == new["result"]["model_sha256"],
              "reference_identical": sha(old["config"]["reference_path"]) == sha(new["config"]["reference_path"]),
              "both_runtime_assert_no_opponent_ground_truth_control": all(item["result"].get("opponent_ground_truth_used_for_control") is False for item in loaded),
              "both_audits_verified": all(item["audit"]["audit_verified"] is True for item in loaded)}
    if any(value is not True for key, value in checks.items() if key != "first_physical_pose_max_abs_error_m") or pose_error > 1e-8:
        raise ValueError("Matched-case identity proof failed: " + json.dumps(checks))
    model = RichDriver.from_npz(model_paths[0])
    old_by_time = {row["timestamp_ns"]: row for row in old["actions"]}
    differences, wrong_switches = [], []
    for row in new["actions"]:
        previous = old_by_time.get(row["timestamp_ns"])
        if previous is None:
            continue
        a, b = previous["raw_action"], row["raw_action"]
        if abs(a["target_speed_mps"] - b["target_speed_mps"]) > 1e-6 or a["target_offset_m"] != b["target_offset_m"]:
            differences.append({"simulation_time_s": row["simulation_time_s"], "old_action": a, "new_action": b,
                "ego_body_speed_difference_mps": row["inputs"]["body_speed_mps"] - previous["inputs"]["body_speed_mps"],
                "ego_lateral_difference_m": row["inputs"]["cross_track_m"] - previous["inputs"]["cross_track_m"]})
        if row["simulation_time_s"] <= 15. and b["target_offset_m"] > 1.6 and row["inputs"]["last_target_offset_m"] < -.2:
            wrong_switches.append(summarize_action(row, model, teacher))
    critical = min(new["actions"], key=lambda row: abs(row["simulation_time_s"] - 7.2))
    critical_old = old_by_time[critical["timestamp_ns"]]
    state = sanitized_state(critical)
    ablations = {}
    for name, hazards in (("all_measured_rows", state["hazard_tracks"]),
        ("geometry_only", [row for row in state["hazard_tracks"] if row.get("source") == "raw_lidar_geometry"]),
        ("without_aged_high_covariance_fragments", [row for row in state["hazard_tracks"] if not
             (row.get("source") == "raw_lidar_geometry" and row.get("position_std_m", 0.) >= 8. and not row.get("velocity_valid", True))])):
        observed = {**state, "hazard_tracks": hazards}
        ablations[name] = {"same_recorded_ego": True, "prediction": model.predict(observed),
                           "frozen005_teacher_lane_m": teacher.intent_teacher(observed)["target_offset_m"]}
    outcomes = {}
    for name, item in zip(("original_detector", "cv425"), loaded):
        row = item["audit"]["entrants"][0]
        outcomes[name] = {"run": str(item["run"].resolve()), "snapshot_sha256": item["manifest"]["snapshot_sha256"],
            "detector_sha256": row["runtime_detector_checkpoint_sha256"], "hazard_pass": item["audit"]["hazard_pass"],
            "hazard_evidence": item["audit"]["hazard_evidence"],
            **{key: row[key] for key in ("progress_m", "collision_samples", "track_violation_steps", "position_resets_after_spawn", "incident_audit_complete")}}
    report = {"scope": "Post-12:21-cutoff matched engineering negative case, n=1 pair; no new combined-model lap claim.",
        "comparison_is_randomized_trial": False, "checks": checks, "configured_spawns": spawns,
        "driver_sha256": model.checkpoint_sha256, "runtime_source_sha256": old["result"]["runtime_source_sha256"],
        "frozen005_teacher_sha256": sha(teacher_path), "file_sha256": hashes, "outcomes": outcomes,
        "first_raw_policy_divergence": differences[0] if differences else None,
        "critical_7p2s_original": summarize_action(critical_old, model, teacher),
        "critical_7p2s_cv425": summarize_action(critical, model, teacher),
        "wrong_side_switches_before15s": wrong_switches, "measured_input_ablations": ablations,
        "cause_status": "Inferred observation-shift generalization failure: a blocked-side NN reversal activates the unchanged guard, loses accumulated lateral command, and precedes a collision-free stop. Closed-loop causal repair is not yet established.",
        "ground_truth_scope": "Only first Supervisor poses checked offline for spawn identity. No opponent truth enters model features, teacher labels, action replay or fitting.",
        "limitations": ["Two physical trajectories diverge after the first changed perception/action; later time-aligned states are not identical counterfactual worlds.",
                        "Ablations are CPU sensitivity probes on one measured ego state, not demonstrated physical interventions.",
                        "More accurate detections can shift the downstream measured hazard distribution without being false positives."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks, "first_divergence": report["first_raw_policy_divergence"],
                      "outcomes": outcomes, "critical_ablation": ablations}, indent=2))


if __name__ == "__main__":
    main()
