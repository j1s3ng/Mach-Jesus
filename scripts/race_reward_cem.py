"""Bounded final-layer CEM using actual on-policy engineering race rewards.

Preparation/collection is CPU-only. Physical execution is exclusively through
the existing locked queue consumer. Enqueuing requires explicit --enqueue and
an eligible foundation that passed its named hazard and4/8/20 engineering runs.
No formal outcome is a fitting, ranking, or calibration input.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
from race_lidar.rich_driver import RichDriver, save_rich_model
from scripts.improvement_ledger import Ledger, OFFICIAL_DETECTOR_SHA256, freeze_snapshot, verify_snapshot
from scripts.improvement_queue_edit import append_jobs, atomic_json
from scripts.training_sources import validate_training_source
from scripts.cem_evidence import (closed_issues, config_issues, expected_case as build_expected_case, fresh_audit)
from scripts.improvement_ledger import environment_rules_hash

PARAMETER_NAMES = ("speed_gain", "speed_offset_mps", "right_logit_bias", "center_logit_bias", "left_logit_bias")
IDENTITY = np.array([1., 0., 0., 0., 0.])
INITIAL_STD = np.array([.035, .5, .10, .10, .10])
MINIMUM_STD = np.array([.008, .10, .025, .025, .025])
FIXED_RULES = {"max_speed_mps": 60., "lateral_accel_budget_mps2": 8., "brake_decel_mps2": 8.,
               "acceleration_mps2": 5., "total_vehicle_mass_kg": 880., "friction_coefficient": 1.5,
               "physics_step_ms": 20, "sensor_period_ms": 200}
HAZARDS = ("visibility-entry", "right", "left", "flying-straight")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def project_parameters(values):
    """Project logits to the intersection of zero-sum and the ±0.2 box."""
    result = np.asarray(values, float).copy()
    if result.shape != (5,) or not np.isfinite(result).all():
        raise ValueError("Calibration needs five finite parameters")
    result[0] = np.clip(result[0], .92, 1.08); result[1] = np.clip(result[1], -1., 1.)
    if np.max(np.abs(result[2:])) <= .2 and abs(result[2:].sum()) < 1e-14:
        return result
    low, high = result[2:].min() - .2, result[2:].max() + .2
    for _ in range(64):
        middle = (low + high) / 2.
        if np.clip(result[2:] - middle, -.2, .2).sum() > 0.:
            low = middle
        else:
            high = middle
    result[2:] = np.clip(result[2:] - (low + high) / 2., -.2, .2)
    return result


def export_calibrated(foundation, destination, parameters, *, generation, reward_history_sha256=None):
    validate_training_source(foundation)
    model = RichDriver.from_npz(foundation)
    parameters = project_parameters(parameters)
    weights, biases = [value.copy() for value in model.weights], [value.copy() for value in model.biases]
    weights[-1][:3] *= np.float32(parameters[0])
    biases[-1][:3] *= np.float32(parameters[0]); biases[-1][:3] += np.float32(parameters[1] / 60.)
    biases[-1][3:] += parameters[2:].astype(np.float32)
    metadata = {**model.metadata, "method": "bounded_cem_direct_policy_search_from_physical_engineering_rewards",
                "reinforcement_learning": True, "reinforcement_learning_method": "black_box_parameter_search_not_policy_gradient",
                "calibration_search_status": "immutable_unscored_proposal", "calibration_generation": generation,
                "calibration_foundation_sha256": sha(foundation), "calibration_parameter_names": list(PARAMETER_NAMES),
                "calibration_script_sha256": sha(__file__),
                "calibration_parameters": parameters.tolist(), "calibration_reward_history_sha256": reward_history_sha256,
                "calibration_bounds": {"speed_gain": [.92, 1.08], "speed_offset_mps": [-1., 1.],
                                       "each_logit_bias": [-.2, .2], "logit_bias_sum": 0.},
                "runtime_contract_changed": False, "physical_limits_changed": False}
    return save_rich_model(destination, weights, biases, model.mean, model.scale, metadata)


def sample_generation(mean, std, generation, seed, *, count=6):
    if count != 6 or generation not in (0, 1):
        raise ValueError("Initial CEM campaign is limited to two generations of six candidates")
    rng = np.random.default_rng(seed + generation)
    result = [project_parameters(mean + rng.normal(size=5) * std) for _ in range(count)]
    result[0] = IDENTITY.copy() if generation == 0 else project_parameters(mean)
    return result


def update_distribution(candidates, old_mean, old_std):
    feasible = sorted([row for row in candidates if row["reward"]["feasible"]],
                      key=lambda row: row["reward"]["reward"], reverse=True)
    if not feasible:
        raise ValueError("CEM cannot learn elites from rejected/unsafe candidates")
    elites = feasible[:2]
    values = np.stack([row["parameters"] for row in elites])
    mean = project_parameters(.5 * np.asarray(old_mean) + .5 * values.mean(0))
    std = np.maximum(MINIMUM_STD, .5 * np.asarray(old_std) + .5 * values.std(0))
    return mean, std, [row["candidate_id"] for row in elites]


def outcome(run, *, expected_case, cache_dir, model_sha256, policy, kind, snapshot_sha256=None, field_size=4):
    """Read closed eligible engineering evidence; missing evidence remains pending."""
    run = Path(run).resolve()
    if run.parent.name != "engineering":
        raise ValueError("Physical reward sources must be explicitly named engineering runs")
    if not (run / "config.json").exists():
        return None
    eligibility = validate_training_source(run, require_config=True)
    config = json.loads((run / "config.json").read_text())
    if config.get("excluded_from_training") is not False or config.get("dataset_split") not in ("development", "engineering", "train", "training"):
        raise ValueError("CEM rewards require an explicitly training-eligible development split")
    if not (run / "ledger_outcome.json").exists():
        return None
    audit = json.loads((run / "ledger_outcome.json").read_text())
    reasons = config_issues(config, expected_case)
    if expected_case["job_id"] != run.name or expected_case["output"] != str(run):
        reasons.append("expected_case_output_identity_mismatch")
    if audit.get("rules_sha256") != environment_rules_hash(config.get("rules", {})):
        reasons.append("saved_audit_current_rules_mismatch")
    if not reasons:
        try:
            reasons.extend(closed_issues(run, config, expected_case))
        except (OSError, ValueError, KeyError, TypeError):
            reasons.append("corrupt_closed_authoritative_evidence")
    evidence_binding = {"independent_current_audit": False}
    if not reasons:
        try:
            current, evidence_binding = fresh_audit(run, expected_case, cache_dir)
            for key in ("audit_verified", "snapshot_verified", "rules_sha256", "raw_rules_sha256", "snapshot_sha256", "entrants", "hazard_pass", "hazard_evidence"):
                if current.get(key) != audit.get(key):
                    reasons.append("saved_audit_disagrees_with_current_evidence:" + key)
            audit = current
            evidence_binding["independent_current_audit"] = True
        except (OSError, ValueError, KeyError, TypeError):
            reasons.append("independent_closed_evidence_audit_failed")
    if audit.get("audit_verified") is not True or audit.get("snapshot_verified") is not True:
        reasons.append("unverified_physical_evidence")
    if snapshot_sha256 and audit.get("snapshot_sha256") != snapshot_sha256:
        reasons.append("snapshot_identity_mismatch")
    if config.get("visible_distance_m") != 70.:
        reasons.append("shared_guard_visibility_changed")
    for key, expected in FIXED_RULES.items():
        if config.get("rules", {}).get(key) != expected:
            reasons.append("fixed_physical_rule_changed:" + key)
    if kind == "field" and config.get("rules", {}).get("field_size") != field_size:
        reasons.append("field_size_changed")
    entrants = audit.get("entrants", [])
    expected_count = 1 if kind == "hazard" else field_size
    if not isinstance(entrants, list) or len(entrants) != expected_count:
        reasons.append("missing_or_extra_scored_entrants"); entrants = entrants if isinstance(entrants, list) else []
    incident = False
    for row in entrants:
        if row.get("model_sha256") != model_sha256 or row.get("policy_key") != policy:
            reasons.append("runtime_model_or_policy_mismatch")
        if row.get("runtime_detector_checkpoint_sha256") != OFFICIAL_DETECTOR_SHA256:
            reasons.append("runtime_detector_changed")
        if row.get("runtime_policy_parameters") != {"visible_distance_m": 70.}:
            reasons.append("runtime_guard_parameters_changed")
        if row.get("incident_audit_complete") is not True:
            reasons.append("incomplete_incident_audit")
        for key in ("collision_samples", "track_violation_steps", "position_resets_after_spawn"):
            value = row.get(key)
            if not finite_number(value) or value < 0.:
                reasons.append("missing_incident_count:" + key)
            elif value > 0.:
                incident = True
    evidence = audit.get("hazard_evidence") or {}
    if kind == "hazard":
        progress, target = evidence.get("ego_progress_m"), evidence.get("required_ego_progress_m")
        valid_numbers = all(finite_number(value) for value in (progress, target)) and target > 0.
        if not valid_numbers:
            reasons.append("missing_hazard_progress")
        passed = (audit.get("hazard_pass") is True and evidence.get("zero_incidents") is True
                  and valid_numbers and progress >= target)
        if evidence.get("zero_incidents") is False:
            incident = True
        elif evidence.get("zero_incidents") is not True:
            reasons.append("missing_hazard_incident_audit")
        mean_finish = None
        progress_fraction = min(1., max(0., progress / target)) if valid_numbers else 0.
    elif kind == "field":
        passed = audit.get("all_clean_finishes") is True
        finishes = []
        for row in entrants:
            finish = row.get("finish_time_s")
            progress, target = row.get("progress_m"), row.get("target_progress_m")
            valid_numbers = all(finite_number(value) for value in (finish, progress, target))
            passed &= (row.get("valid_finish") is True and row.get("clean_finish") is True and valid_numbers
                       and finish > 0. and target > 0. and progress >= target - 1e-3)
            if finite_number(finish) and finish > 0.:
                finishes.append(finish)
        mean_finish = float(np.mean(finishes)) if len(finishes) == expected_count else None
        fractions = []
        for row in entrants:
            progress, target = row.get("progress_m"), row.get("target_progress_m")
            if finite_number(progress) and finite_number(target) and target > 0.:
                fractions.append(min(1., max(0., progress / target)))
            else:
                reasons.append("missing_field_progress"); fractions.append(0.)
        progress_fraction = float(np.mean(fractions)) if fractions else 0.
    else:
        raise ValueError("Outcome kind must be hazard or field")
    usable = not reasons
    return {"run": str(run), "kind": kind, "eligible_source": eligibility, "outcome_sha256": sha(run / "ledger_outcome.json"),
            "expected_case": expected_case, "current_evidence_binding": evidence_binding,
            "usable": usable, "passed": bool(usable and passed and not incident), "incident": incident,
            "rejection_reasons": reasons, "mean_finish_time_s": mean_finish, "progress_fraction": progress_fraction,
            "audit_verified": audit.get("audit_verified"), "snapshot_sha256": audit.get("snapshot_sha256")}


def race_reward(cases, *, field_cutoff_s=480.):
    """Safety/evidence rejection precedes actual on-policy pace reward."""
    if any(row is None for row in cases):
        return None
    if any(not row["usable"] for row in cases):
        return {"feasible": False, "reward": -2000., "reason": "invalid_evidence", "cases": cases}
    if any(row["incident"] for row in cases):
        return {"feasible": False, "reward": -1000., "reason": "physical_incident", "cases": cases}
    fields = [row for row in cases if row["kind"] == "field"]
    if not all(row["passed"] for row in cases) or len(fields) != 1 or fields[0]["mean_finish_time_s"] is None:
        progress = float(np.mean([row["progress_fraction"] for row in cases]))
        return {"feasible": False, "reward": -100. + 40. * progress, "reason": "incomplete_safety_or_finish_gate", "cases": cases}
    mean_finish = fields[0]["mean_finish_time_s"]
    return {"feasible": True, "reward": float(np.clip(100. + 20. * (1. - mean_finish / field_cutoff_s), 100., 120.)),
            "mean_finish_time_s": mean_finish, "reason": "all_required_cases_clean", "cases": cases}


def job_batch(candidate, *, root=ROOT, seed=101, rotation=0, validation=False):
    snapshot = Path(candidate["snapshot_path"])
    prefix = candidate["candidate_id"] + ("-validation" if validation else "")
    common = ["--source-root", str(snapshot), "--reference", str(snapshot / "reference.npz"),
              "--candidate-id", candidate["candidate_id"], "--single-policy", candidate["policy"],
              "--no-video", "--pointcloud-stride", "0", "--stage", "engineering", "--split", "development",
              "--visibility", "70", "--seed", str(seed)]
    jobs = []
    if not validation:
        for case in HAZARDS:
            args = common + ["--scenario", "stopped", "--seconds", "120"]
            if case in ("left", "right"):
                args += ["--cars", "3", "--obstacle-progress", "1000", "--ego-progress", "900", "--blocked-side", case]
            else:
                args += ["--cars", "2", "--obstacle-progress", "300" if case == "visibility-entry" else "1000", "--ego-progress", "0"]
            identity = prefix + "-" + case
            jobs.append({"job_id": identity, "output": f"runs/improvement-12h/engineering/{identity}",
                         "estimated_wall_seconds": 120, "max_wall_seconds": 300, "args": args})
    identity = prefix + "-field4"
    jobs.append({"job_id": identity, "output": f"runs/improvement-12h/engineering/{identity}",
                 "estimated_wall_seconds": 300, "max_wall_seconds": 600,
                 "args": common + ["--scenario", "race", "--seconds", "480", "--cars", "4", "--spacing", "14",
                                   "--grid-jitter", ".3", "--rotation", str(rotation)],
                 "requires": [{"job_id": job["job_id"], "true_fields": ["audit_verified", "hazard_pass"]} for job in jobs]})
    return jobs


def bind_jobs(candidate, jobs):
    return {job["job_id"]: build_expected_case(job, snapshot_path=candidate["snapshot_path"],
        snapshot_sha256=candidate["snapshot_sha256"], policy=candidate["policy"],
        model_sha256=candidate["model_sha256"]) for job in jobs}


def freeze_candidate(state, parameters, generation, index):
    campaign, folder = Path(state["campaign"]), Path(state["output"])
    identity = f"{state['search_id']}-g{generation}-c{index}"
    policy = "neural_" + identity.replace("-", "_")
    destination = folder / identity
    if (destination / "candidate.json").exists():
        recovered = json.loads((destination / "candidate.json").read_text())
        if recovered["parameters"] != parameters.tolist() or sha(recovered["model_path"]) != recovered["model_sha256"]:
            raise ValueError("Existing immutable CEM candidate conflicts with resumed plan")
        verify_snapshot(recovered["snapshot_path"])
        return recovered
    destination.mkdir(exist_ok=True)
    if sha(state["foundation_model"]) != state["foundation_sha256"]:
        raise ValueError("Foundation model changed after CEM initialization")
    history_hash = state["preparing_generation"]["reward_history_sha256"]
    model = destination / "model.npz"
    if not model.exists():
        export_calibrated(state["foundation_model"], model, parameters, generation=generation, reward_history_sha256=history_hash)
    else:
        existing = RichDriver.from_npz(model)
        if existing.metadata.get("calibration_parameters") != parameters.tolist() or existing.metadata.get("calibration_reward_history_sha256") != history_hash:
            raise ValueError("Incomplete CEM model artifact conflicts with its saved plan")
    foundation = Path(state["foundation_snapshot"])
    manifest = verify_snapshot(foundation)
    files = {row["path"]: foundation / row["path"] for row in manifest["files"] if not row["path"].startswith("models/")}
    key = f"models/{policy}.npz"; files[key] = model
    snapshot = campaign / "snapshots" / identity
    if snapshot.exists():
        frozen = verify_snapshot(snapshot)
        if {row["path"]: row["sha256"] for row in frozen["files"]}.get(key) != sha(model):
            raise ValueError("Incomplete CEM snapshot conflicts with its model")
    else:
        frozen = freeze_snapshot(snapshot, files, metadata={"candidate_id": identity,
            "policies": [{"policy": "reference", "label": "Reference"}, {"policy": policy, "label": identity, "model_relpath": key}],
            "policy_parameters": {"visible_distance_m": 70.}, "protocol": "race-improvement-v2",
            "training": "bounded CEM direct policy search from actual engineering race rewards",
            "foundation_snapshot_sha256": manifest["snapshot_sha256"]})
    ledger = Ledger(campaign / "ledger.jsonl")
    registered = [row for row in ledger.records() if row["event_type"] == "candidate_frozen" and row["payload"]["candidate_id"] == identity]
    if registered:
        if registered[-1]["payload"].get("snapshot_sha256") != frozen["snapshot_sha256"]:
            raise ValueError("CEM registration conflicts with resumed snapshot")
    else:
        ledger.register_candidate(identity, snapshot,
            {"policy_key": policy, "model_sha256": sha(model), "detector_sha256": OFFICIAL_DETECTOR_SHA256,
             "control_parameters": {"visible_distance_m": 70.}}, search_id=state["search_id"], generation=generation)
    candidate = {"candidate_id": identity, "policy": policy, "parameters": parameters.tolist(), "model_path": str(model),
                 "model_sha256": sha(model), "snapshot_path": str(snapshot), "snapshot_sha256": frozen["snapshot_sha256"],
                 "generation": generation, "index": index}
    candidate["expected_cases"] = bind_jobs(candidate, job_batch(candidate))
    atomic_json(destination / "candidate.json", candidate)
    return candidate


def foundation_ready(state):
    results, field_sizes, hazards = [], set(), set()
    for path in state["foundation_runs"]:
        run = Path(path)
        if not (run / "config.json").exists():
            return False, "foundation_run_pending", results
        validate_training_source(run, require_config=True)
        config = json.loads((run / "config.json").read_text())
        hazard = config.get("scenario") == "stopped"
        size = int(config["rules"]["field_size"])
        result = outcome(run, expected_case=state["foundation_expected_cases"][str(run.resolve())],
                         cache_dir=Path(state["output"]) / "evidence_cache",
                         model_sha256=state["foundation_sha256"], policy=state["foundation_policy"],
                         kind="hazard" if hazard else "field", snapshot_sha256=state["foundation_snapshot_sha256"], field_size=size)
        if result is None:
            return False, "foundation_outcome_pending", results
        results.append(result)
        if not result["passed"]:
            return False, "foundation_physical_gate_failed", results
        if hazard:
            moving = [entry for entry in config["entrants"] if not entry.get("is_static")]
            static = [entry for entry in config["entrants"] if entry.get("is_static")]
            hazards.add(json.dumps({"ego": moving[0]["start_progress_m"],
                                    "obstacles": [(entry["start_progress_m"], entry["start_lane_m"]) for entry in static]}, sort_keys=True))
        else:
            field_sizes.add(size)
    if not {4, 8, 20}.issubset(field_sizes) or len(hazards) < 4:
        return False, "foundation_requires_four_distinct_hazards_and_clean_fields4_8_20", results
    return True, "foundation_engineering_gates_verified", results


def candidate_reward(candidate, campaign, cache_dir):
    root = Path(campaign) / "engineering"
    hazard_cases = [outcome(root / (candidate["candidate_id"] + "-" + case),
                           expected_case=candidate["expected_cases"][candidate["candidate_id"] + "-" + case], cache_dir=cache_dir,
                           model_sha256=candidate["model_sha256"],
                           policy=candidate["policy"], kind="hazard", snapshot_sha256=candidate["snapshot_sha256"]) for case in HAZARDS]
    if any(row is None for row in hazard_cases):
        return None
    if any(not row["passed"] for row in hazard_cases):
        # Field4 is intentionally skipped by queue dependencies; don't wait for
        # a nonexistent field outcome or invent a finish reward.
        return race_reward(hazard_cases)
    field = outcome(root / (candidate["candidate_id"] + "-field4"),
                    expected_case=candidate["expected_cases"][candidate["candidate_id"] + "-field4"], cache_dir=cache_dir,
                    model_sha256=candidate["model_sha256"],
                    policy=candidate["policy"], kind="field", snapshot_sha256=candidate["snapshot_sha256"])
    return race_reward(hazard_cases + [field])


def queue_budget(state, jobs):
    campaign = Path(state["campaign"])
    deadline = dt.datetime.fromisoformat(json.loads((campaign / "campaign.json").read_text())["deadline_utc"]).timestamp()
    queue = json.loads((campaign / "queue.json").read_text()) if (campaign / "queue.json").exists() else []
    # Consumer progress is scheduling-only; no formal outcomes/rewards are read.
    status = json.loads((campaign / "queue_status.json").read_text()) if (campaign / "queue_status.json").exists() else {}
    completed = set(status.get("completed", {}))
    backlog = sum(job.get("estimated_wall_seconds", 900) for job in queue if job["job_id"] not in completed)
    requested = sum(job["estimated_wall_seconds"] for job in jobs)
    return time.time() + backlog + requested + 300. < deadline, {"pending_queue_estimate_s": backlog,
              "requested_estimate_s": requested, "deadline_epoch_s": deadline, "checked_epoch_s": time.time()}


def step(state, *, enqueue=False):
    folder, campaign = Path(state["output"]), Path(state["campaign"])
    if state["status"] in ("complete", "stopped"):
        return state
    for name, expected in state.get("source_sha256", {}).items():
        if sha(ROOT / name) != expected or sha(folder / "source" / name) != expected:
            raise ValueError("CEM source changed after its immutable initialization: " + name)
    deadline = dt.datetime.fromisoformat(json.loads((campaign / "campaign.json").read_text())["deadline_utc"]).timestamp()
    if time.time() >= deadline - 300.:
        state.update(status="stopped", reason="campaign_deadline_reserve"); return state
    ready, reason, foundation = foundation_ready(state)
    state["foundation_gate"] = {"ready": ready, "reason": reason, "cases": foundation}
    if not ready:
        state["status"] = "waiting_for_foundation"; return state
    pending = next((generation for generation in state["generations"] if not generation.get("completed")), None)
    if pending is not None:
        for candidate in pending["candidates"]:
            if "reward" not in candidate:
                reward = candidate_reward(candidate, campaign, folder / "evidence_cache")
                if reward is not None:
                    candidate["reward"] = reward
                    atomic_json(folder / candidate["candidate_id"] / "physical_reward.json", reward)
        if not all("reward" in candidate for candidate in pending["candidates"]):
            state["status"] = "waiting_for_physical_rewards"; return state
        pending["completed"] = True
        feasible = [row for row in pending["candidates"] if row["reward"]["feasible"]]
        if not feasible:
            state.update(status="stopped", reason="no_feasible_generation_elites"); return state
        mean, std, elite_ids = update_distribution(pending["candidates"], state["mean"], state["std"])
        state.update(mean=mean.tolist(), std=std.tolist()); pending["elite_ids"] = elite_ids
    if len(state["generations"]) < 2:
        if not enqueue:
            state["status"] = "ready_for_generation_explicit_enqueue"; return state
        generation = len(state["generations"])
        # Check capacity before freezing/registering an entire generation.
        if "preparing_generation" not in state:
            prototype = {"candidate_id": "estimate", "policy": "neural_estimate", "snapshot_path": "/estimate"}
            jobs_estimate = job_batch(prototype) * 6
            fits, budget = queue_budget(state, jobs_estimate); state["last_budget_check"] = budget
            if not fits:
                state.update(status="stopped", reason="deadline_reserve_prevents_complete_generation"); return state
            values = sample_generation(np.asarray(state["mean"]), np.asarray(state["std"]), generation, state["seed"])
            history = {"method": "actual_on_policy_engineering_reward_history",
                       "completed_generations": [row for row in state["generations"] if row.get("completed")]}
            history_path = folder / f"generation-{generation}-reward-history.json"
            atomic_json(history_path, history)
            state["preparing_generation"] = {"generation": generation, "parameters": [row.tolist() for row in values],
                                             "reward_history_sha256": sha(history_path), "reward_history_path": str(history_path)}
            atomic_json(folder / "state.json", state)
        plan = state["preparing_generation"]
        if plan["generation"] != generation:
            raise ValueError("Resumed CEM generation index changed")
        parameters = [np.asarray(row) for row in plan["parameters"]]
        candidates = [freeze_candidate(state, values, generation, index) for index, values in enumerate(parameters)]
        jobs = [job for candidate in candidates for job in job_batch(candidate)]
        append_jobs(campaign / "queue.json", jobs)
        state["generations"].append({"index": generation, "completed": False, "candidates": candidates})
        del state["preparing_generation"]
        state["status"] = "waiting_for_physical_rewards"; return state
    if "validation" not in state:
        candidates = [row for generation in state["generations"] for row in generation["candidates"] if row["reward"]["feasible"]]
        winner = max(candidates, key=lambda row: row["reward"]["reward"])
        baseline = state["generations"][0]["candidates"][0]
        if np.allclose(winner["parameters"], IDENTITY, atol=1e-10):
            state.update(status="complete", reason="identity_foundation_remains_best", winner=winner); return state
        if not enqueue:
            state["status"] = "ready_for_whole_run_validation_explicit_enqueue"; return state
        jobs = job_batch(baseline, seed=211, rotation=1, validation=True) + job_batch(winner, seed=211, rotation=1, validation=True)
        fits, budget = queue_budget(state, jobs); state["last_budget_check"] = budget
        if not fits:
            state.update(status="stopped", reason="deadline_reserve_prevents_validation", winner=winner); return state
        append_jobs(campaign / "queue.json", jobs)
        validation_expected = {}
        for candidate in (baseline, winner):
            validation_expected.update(bind_jobs(candidate, job_batch(candidate, seed=211, rotation=1, validation=True)))
        state["validation"] = {"baseline": baseline, "winner": winner, "jobs": [job["job_id"] for job in jobs],
                               "expected_cases": validation_expected,
                               "seed": 211, "rotation": 1, "used_for_cem_updates": False}
    validation = state["validation"]
    results = {}
    for role in ("baseline", "winner"):
        candidate = validation[role]
        results[role] = outcome(campaign / "engineering" / (candidate["candidate_id"] + "-validation-field4"),
             expected_case=validation["expected_cases"][candidate["candidate_id"] + "-validation-field4"], cache_dir=folder / "evidence_cache",
             model_sha256=candidate["model_sha256"], policy=candidate["policy"], kind="field", snapshot_sha256=candidate["snapshot_sha256"])
    validation["results"] = results
    if any(value is None for value in results.values()):
        state["status"] = "waiting_for_whole_run_validation"; return state
    improved = (all(value["passed"] for value in results.values()) and
                results["winner"]["mean_finish_time_s"] <= results["baseline"]["mean_finish_time_s"] - 1.)
    validation["development_improvement"] = improved
    state.update(status="complete", reason="development_gain_verified" if improved else "no_verified_development_gain",
                 winner=validation["winner"], formal_promotion=False)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--foundation-model", type=Path, required=True); init.add_argument("--foundation-snapshot", type=Path, required=True)
    init.add_argument("--foundation-policy", required=True); init.add_argument("--foundation-runs", type=Path, nargs="+", required=True)
    init.add_argument("--output", type=Path, required=True); init.add_argument("--search-id", required=True)
    init.add_argument("--campaign", type=Path, default=ROOT / "runs/improvement-12h"); init.add_argument("--seed", type=int, default=20261114)
    for name in ("step", "watch"):
        command = sub.add_parser(name); command.add_argument("--output", type=Path, required=True)
        command.add_argument("--enqueue", action="store_true"); command.add_argument("--poll-seconds", type=float, default=20.)
    args = parser.parse_args()
    if args.command == "init":
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Choose a fresh CEM output directory")
        if not args.search_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in args.search_id):
            parser.error("Search ID must contain lowercase letters, digits, hyphen or underscore")
        validate_training_source(args.foundation_model)
        model = RichDriver.from_npz(args.foundation_model); manifest = verify_snapshot(args.foundation_snapshot)
        if sha(args.foundation_model) not in {row["sha256"] for row in manifest["files"]}:
            parser.error("Foundation checkpoint is not present in its verified frozen source bundle")
        if manifest.get("metadata", {}).get("policy_parameters") != {"visible_distance_m": 70.}:
            parser.error("Foundation must use the fixed70m shared guard")
        args.output.mkdir(parents=True)
        source_names = ("scripts/race_reward_cem.py", "scripts/cem_evidence.py", "scripts/improvement_queue_edit.py",
                        "scripts/training_sources.py", "scripts/evaluate_improvement_attempt.py", "scripts/improvement_ledger.py")
        for name in source_names:
            destination = args.output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, destination)
        state = {"schema_version": 1, "status": "waiting_for_foundation", "search_id": args.search_id,
                 "output": str(args.output.resolve()), "campaign": str(args.campaign.resolve()), "seed": args.seed,
                 "foundation_model": str(args.foundation_model.resolve()), "foundation_sha256": model.checkpoint_sha256,
                 "foundation_snapshot": str(args.foundation_snapshot.resolve()), "foundation_snapshot_sha256": manifest["snapshot_sha256"],
                 "foundation_policy": args.foundation_policy, "foundation_runs": [str(path.resolve()) for path in args.foundation_runs],
                 "mean": IDENTITY.tolist(), "std": INITIAL_STD.tolist(), "generations": [],
                 "generation_limit": 2, "candidates_per_generation": 6, "formal_sources_for_learning": [],
                 "source_sha256": {name: sha(args.output / "source" / name) for name in source_names},
                 "method": "bounded_direct_policy_search_from_actual_on_policy_engineering_rewards"}
        queue = json.loads((args.campaign / "queue.json").read_text())
        expected = {}
        for run in args.foundation_runs:
            matches = [job for job in queue if (ROOT / job["output"]).resolve() == run.resolve()]
            if len(matches) != 1:
                parser.error("Every foundation run must match one authoritative queued engineering job")
            expected[str(run.resolve())] = build_expected_case(matches[0], snapshot_path=args.foundation_snapshot,
                snapshot_sha256=manifest["snapshot_sha256"], policy=args.foundation_policy, model_sha256=model.checkpoint_sha256)
        state["foundation_expected_cases"] = expected
        atomic_json(args.output / "state.json", state); print(json.dumps({"status": state["status"], "no_jobs_enqueued": True})); return
    with (args.output / "cem.lock").open("a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state = json.loads((args.output / "state.json").read_text())
            state = step(state, enqueue=args.enqueue); atomic_json(args.output / "state.json", state)
            print(json.dumps({"status": state["status"], "reason": state.get("reason"),
                              "generations": len(state["generations"])}), flush=True)
            if args.command == "step" or state["status"] in ("complete", "stopped"):
                break
            time.sleep(max(5., min(60., args.poll_seconds)))


if __name__ == "__main__":
    main()
