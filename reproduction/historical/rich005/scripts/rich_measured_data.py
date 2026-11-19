"""Strict engineering-only observed-state loader and row-local expert labels."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from race_lidar.race_policy import RacePolicy
from scripts.training_sources import validate_training_source

# "holdout" here means exclusion from gradients/checkpoint selection only.
# The20-car engineering run was already seen in pre-fit design diagnostics.
ALLOWED_RUNS = {"rich001-field4-r0": "actual_fit", "rich001-field20-r0": "actual_run_holdout"}
STATE_KEYS = ("progress_fraction", "body_speed_mps", "heading_error_rad", "cross_track_m", "curvature_ahead",
              "last_target_offset_m", "front_distance_m", "front_lateral_m", "front_speed_mps",
              "left_clearance_m", "right_clearance_m")
HAZARD_KEYS = ("relative_distance_m", "lateral_m", "speed_mps", "position_std_m", "position_uncertainty_m",
               "speed_std_mps", "speed_uncertainty_mps", "lateral_min_m", "lateral_max_m",
               "velocity_valid", "age_s", "source")


def validate_unsealed_source(path):
    validate_training_source(path)
    return Path(path).resolve()


def validate_run(path):
    # Keep the original lexical path until the shared guard has checked aliases.
    validate_training_source(path)
    path = Path(path).resolve()
    if path.name not in ALLOWED_RUNS or path.parent.name != "engineering":
        raise ValueError("Only the two explicitly authorized engineering runs may supply measured training states")
    validate_training_source(path, require_config=True)
    return path


def load_run(path, *, stride=1):
    path = validate_run(path)
    if stride < 1:
        raise ValueError("Measured-state stride must be positive")
    samples, manifest = [], []
    for source in sorted(path.glob("cars/*/policy_actions.jsonl")):
        eligibility = validate_training_source(source, require_config=True)
        included = 0
        for index, line in enumerate(source.open()):
            if index % stride:
                continue
            row = json.loads(line)
            if row.get("phase", "racing") != "racing":
                continue
            # No recorded actions, GT fields, leaderboard, outcomes or feasible
            # cap are copied into the student's feature state.
            state = {key: row["inputs"][key] for key in STATE_KEYS}
            hazards = row.get("hazard_tracks", row["inputs"].get("hazard_tracks", []))
            state["hazard_tracks"] = [{key: hazard[key] for key in HAZARD_KEYS if key in hazard}
                                     for hazard in hazards if isinstance(hazard, dict)]
            cap = float(row["feasible_speed_cap_mps"])
            if not math.isfinite(cap) or cap < 0.:
                raise ValueError("Recorded feasible label cap must be finite and nonnegative")
            samples.append({"state": state, "label_feasible_speed_mps": min(60., cap),
                            "run_id": path.name, "split_role": ALLOWED_RUNS[path.name],
                            "car_id": source.parent.name, "row_index": index,
                            "simulation_time_s": float(row["simulation_time_s"])})
            included += 1
        manifest.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": included,
                         "training_eligibility": eligibility})
    if not samples:
        raise ValueError(f"No authorized measured policy states in {path}")
    return samples, manifest


def load_engineering_run(path, *, stride=1):
    """Load an explicitly named future development run, preserving row whitelist.

    Unlike the frozen004 two-source loader, this accepts new engineering runs
    only when their own config explicitly declares development/training use.
    No recursive run discovery and no reserved evaluation data are permitted.
    """
    original = Path(path)
    validate_training_source(original, require_config=True)
    path = original.resolve()
    if path.parent.name != "engineering":
        raise ValueError("New measured training inputs must be an explicitly named engineering run")
    config = json.loads((path / "config.json").read_text())
    if config.get("dataset_split") not in ("engineering", "development", "train", "training"):
        raise ValueError("New engineering run needs an explicit development/training dataset_split")
    if config.get("excluded_from_training") is not False:
        raise ValueError("New engineering run must explicitly set excluded_from_training=false")
    if stride < 1:
        raise ValueError("Measured-state stride must be positive")
    samples, manifest = [], []
    for source in sorted(path.glob("cars/*/policy_actions.jsonl")):
        eligibility = validate_training_source(source, require_config=True)
        included = 0
        for index, line in enumerate(source.open()):
            if index % stride or not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase", "racing") != "racing":
                continue
            state = {key: row["inputs"][key] for key in STATE_KEYS}
            hazards = row.get("hazard_tracks", row["inputs"].get("hazard_tracks", []))
            state["hazard_tracks"] = [{key: hazard[key] for key in HAZARD_KEYS if key in hazard}
                                     for hazard in hazards if isinstance(hazard, dict)]
            cap = float(row["feasible_speed_cap_mps"])
            if not math.isfinite(cap) or cap < 0.:
                raise ValueError("Recorded feasible label cap must be finite and nonnegative")
            samples.append({"state": state, "label_feasible_speed_mps": min(60., cap),
                            "run_id": path.name, "split_role": "engineering_development",
                            "car_id": source.parent.name, "row_index": index,
                            "simulation_time_s": float(row["simulation_time_s"])})
            included += 1
        manifest.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "rows": included, "training_eligibility": eligibility})
    if not samples:
        raise ValueError(f"No eligible engineering measured states in {path}")
    return samples, manifest


def reference_counterfactual(state, feasible_speed):
    """Current full-row reference policy, initialized from this measured state."""
    expert = RacePolicy(None, model=None, name="FullRowsReference", seed=0)
    expert.classic_lane = float(state["last_target_offset_m"])
    actual = float(state["cross_track_m"])
    expert.preferred_side = 1. if abs(3.2 - actual) <= abs(-3.2 - actual) else -1.
    observation = {key: state[key] for key in ("front_distance_m", "front_lateral_m", "front_speed_mps",
                                              "left_clearance_m", "right_clearance_m")}
    observation["hazard_tracks"] = state["hazard_tracks"]
    return expert.predict(state, observation, feasible_speed)["raw_action"]
