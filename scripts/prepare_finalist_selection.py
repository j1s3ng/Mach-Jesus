#!/usr/bin/env python3
"""Prepare prospective engineering-only finalist rules; no queues or ledger writes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from race_lidar.rich_driver import RichDriver
from scripts.improvement_ledger import Ledger, OFFICIAL_DETECTOR_SHA256, canonical, digest, scenario_identity

IDENTITY = np.array([1., 0., 0., 0., 0.])
MODEL_SHA = {"neural_rich005_seed17": "d9837d76d29f80368dae8ec8f611aa67b4c3e7885ed5bab51876fbff34afd9da",
             "neural_rich006_seed17": "730802bc23036458e76007f0f17b8f13a3894457e14d82b642f8f571f84d83c6",
             "neural_rich005_seed7": "89bafe77e99d160aa97aab5510f587112a6c461e2cd7383a381a4af429c94989"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_model_identity(path):
    """Hash actual float32 inference arrays and the feature/output encoding.

    Training metadata, model names, paths and seeds do not create distinct racers.
    This excludes numerical clones, not every possible algebraic NN symmetry.
    """
    model = RichDriver.from_npz(path)
    contract = {key: model.metadata[key] for key in ("format", "feature_names", "output_names")}
    contract["lane_summary_mode"] = model.metadata.get("lane_summary_mode", "nearest")
    h = hashlib.sha256(canonical(contract))
    arrays = [("feature_mean", model.mean), ("feature_scale", model.scale)]
    arrays.extend((f"weight_{i}", value) for i, value in enumerate(model.weights))
    arrays.extend((f"bias_{i}", value) for i, value in enumerate(model.biases))
    for name, value in arrays:
        normalized = np.array(value, dtype="<f4", order="C", copy=True)
        normalized[normalized == 0.] = 0.  # Numeric +0/-0 do not create a new racer.
        h.update(canonical({"name": name, "shape": list(normalized.shape), "dtype": "float32"}))
        h.update(normalized.tobytes(order="C"))
    return {"numerical_model_sha256": h.hexdigest(), "checkpoint_sha256": sha(path),
            "contract": contract, "array_shapes": {name: list(value.shape) for name, value in arrays},
            "scope": "Exact numerical inference-array/encoding identity; not a proof against every neural-network reparameterization."}


def cem_validation_gate(state):
    """A callback admission gate only; never collects rewards or changes CEM."""
    issues = []
    if state.get("status") != "complete" or state.get("reason") != "development_gain_verified":
        issues.append("CEM_not_complete_with_verified_development_gain")
    generations = state.get("generations", [])
    if len(generations) != 2 or any(g.get("index") != i or not g.get("completed") for i, g in enumerate(generations)):
        issues.append("full_two_generations_not_complete")
    if any(len(g.get("candidates", [])) != 6 or any("reward" not in c for c in g.get("candidates", [])) for g in generations):
        issues.append("all_twelve_fitness_records_not_preserved")
    validation = state.get("validation", {})
    winner = state.get("winner", {})
    parameters = np.asarray(winner.get("parameters", []), dtype=float)
    if parameters.shape != (5,) or not np.isfinite(parameters).all() or np.allclose(parameters, IDENTITY, atol=1e-10, rtol=0.):
        issues.append("winner_missing_or_identity_calibration")
    if (validation.get("seed") != 211 or validation.get("rotation") != 1
            or validation.get("used_for_cem_updates") is not False
            or validation.get("development_improvement") is not True
            or validation.get("winner", {}).get("candidate_id") != winner.get("candidate_id")):
        issues.append("paired_seed211_development_validation_not_verified")
    results = validation.get("results", {})
    before, after = results.get("baseline", {}), results.get("winner", {})
    a, b = before.get("mean_finish_time_s"), after.get("mean_finish_time_s")
    finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)
    if (before.get("passed") is not True or after.get("passed") is not True
            or not finite(a) or not finite(b) or a - b < 1.):
        issues.append("paired_seed211_clean_gain_below_one_second")
    return sorted(set(issues))


def proposed_cem_selection_jobs(state):
    """Return new8/20 gate jobs only after the complete CEM validation gate.

    The caller must independently re-audit CEM identities/results before root
    authorizes enqueue. This function never appends jobs or modifies state.
    """
    issues = cem_validation_gate(state)
    if issues:
        return {"ready": False, "issues": issues, "jobs": []}
    winner = state["winner"]
    if numerical_model_identity(winner["model_path"])["numerical_model_sha256"] == numerical_model_identity(state["foundation_model"])["numerical_model_sha256"]:
        return {"ready": False, "issues": ["winner_is_numerical_foundation_clone"], "jobs": []}
    common = ["--source-root", winner["snapshot_path"], "--reference", str(Path(winner["snapshot_path"]) / "reference.npz"),
        "--candidate-id", winner["candidate_id"], "--single-policy", winner["policy"],
        "--vision-url", "http://127.0.0.1:8770", "--detector-sha", OFFICIAL_DETECTOR_SHA256,
        "--stage", "engineering", "--split", "development", "--visibility", "70", "--no-video",
        "--pointcloud-stride", "0", "--scenario", "race", "--seconds", "480", "--spacing", "14",
        "--seed", "83", "--grid-jitter", ".3", "--rotation", "0"]
    jobs = []
    for cars in (8, 20):
        identity = winner["candidate_id"] + f"-selection-field{cars}"
        dependencies = (state["validation"]["jobs"] if cars == 8 else [jobs[0]["job_id"]])
        jobs.append({"job_id": identity, "output": f"runs/improvement-12h/engineering/{identity}",
            "estimated_wall_seconds": 480 if cars == 8 else 900, "max_wall_seconds": 1200 if cars == 8 else 1800,
            "args": common + ["--cars", str(cars), "--secondary-cars", "0" if cars == 8 else "8"],
            "requires": [{"job_id": identity, "true_fields": ["audit_verified", "all_clean_finishes"]} for identity in dependencies]})
    return {"ready": True, "issues": [], "jobs": jobs, "enqueue_authorized_by_this_script": False,
            "feedback_to_CEM_distribution_or_parameters": False}


def build_plan():
    campaign = ROOT / "runs/improvement-12h"
    state = Ledger(campaign / "ledger.jsonl").state()
    new_formal = [a["attempt_id"] for a in state["attempts"] if a.get("candidate_id") == "reference-fast70-suite2"]
    if new_formal:
        raise ValueError("Selection rules must be prepared before new Fast70 formal attempts")
    source = campaign / "rich-driver-006"
    reference_result = json.loads((campaign / "engineering/rich005-s17-field20/cars/R01/result.json").read_text())
    selection = scenario_identity({"scenario": "race", "cars": 20, "rotation": 0, "seed": 83,
        "spacing": 14., "seconds": 480., "grid_jitter": .3})
    registrations = state["candidates"]
    candidates = []
    for policy, model_hash, alias, run in (
        ("neural_rich005_seed17", MODEL_SHA["neural_rich005_seed17"], "rich005-seed17-fast70", "rich005-s17-field20"),
        ("neural_rich006_seed17", MODEL_SHA["neural_rich006_seed17"], "rich00617-original-fast70", "rich006-original-field20-recovery1"),
        ("neural_rich005_seed7", MODEL_SHA["neural_rich005_seed7"], "rich005-seed7-fast70", "rich005-s7-field20-recovery1")):
        registration = registrations[alias]
        snapshot = Path(registration["snapshot_path"])
        manifest = json.loads((snapshot / "manifest.json").read_text())
        spec = next(row for row in manifest["metadata"]["policies"] if row["policy"] == policy)
        model = snapshot / spec["model_relpath"]
        identity = numerical_model_identity(model)
        if identity["checkpoint_sha256"] != model_hash:
            raise ValueError("Registered planned model hash changed")
        candidates.append({"policy_key": policy, "model_sha256": model_hash, "model_path": str(model),
            "registered_engineering_id": alias, "snapshot_path": str(snapshot),
            "snapshot_sha256": registration["snapshot_sha256"], "numeric_identity": identity,
            "selection20_run": str(campaign / "engineering" / run),
            "seed7_recovery_requirement": "Old535-frame/car infrastructure-interrupted20 attempt remains retained; only explicitly linked fresh same-treatment retry may supply its missing full-lap objective." if policy.endswith("seed7") else None})
    cem_path = campaign / "training/cem005-r0-bound2-recovered/state.json"
    cem = json.loads(cem_path.read_text())
    identity_clone = cem["generations"][0]["candidates"][0]
    identity_comparison = {"foundation": numerical_model_identity(cem["foundation_model"]),
                           "generation0_identity": numerical_model_identity(identity_clone["model_path"])}
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DRAFT_FOR_ROOT_REVIEW_NO_QUEUE_LEDGER_CEM_OR_RUNTIME_MUTATIONS",
        "new_Fast70_formal_attempts_at_preparation": new_formal,
        "common_detector_sha256": OFFICIAL_DETECTOR_SHA256,
        "common_control_parameters": {"visible_distance_m": 70.},
        "selection20_scenario": selection, "selection20_scenario_sha256": digest(selection),
        "selection20_reference_sha256": reference_result["reference_sha256"],
        "common_actual_runtime_source8_sha256": reference_result["runtime_source_sha256"],
        "primary_candidates": candidates,
        "CEM_candidate": {"state_path": str(cem_path), "generations": 2, "candidates_per_generation": 6,
            "entry_requires": ["Completed unchanged two-generation engineering search; every proposal/fitness and failed attempt retained.",
                "Selected winner is nonidentity and numerically distinct from all already selected driver inference arrays/encoding.",
                "Both authoritative whole seed211/rotation1 validation field4 runs pass exact source/model/physics/world audit, all cars clean, winner mean at least1s faster than unchanged baseline.",
                "Fresh winner-only engineering fields8and20 pass;20 uses exact selection seed83/spacing14/jitter.3/rotation0 and all20 full clean valid laps.",
                "Neither8nor20 evidence updates CEM parameters, distribution or checkpoint; only frozen engineering eligibility/selection uses it."],
            "callback": "proposed_cem_selection_jobs(state) prepares only; root must independently audit and explicitly authorize enqueue after validation. No callback watcher/queue mutation is launched by this plan.",
            "current_callback_preview": proposed_cem_selection_jobs(cem)},
        "common_eligibility": ["Only explicitly inventoried development engineering outputs; never read or use formal/confirmation/championship/freshgeometry outcomes for model choice.",
            "All four policy-specific known engineering hazards pass verified zero-overlap/violation/reset progress+20m objectives; complete clean fields4and8 precede selection20.",
            "All20 selection entrants are homogeneous, source/weight/detector/control-bound, closed, valid and clean; common seed, per-gridID/seed/start/lane, reference/physical rules and actual eight control sources match.",
            "Each selection20 car starts at nonpositive ordered progress and physically traverses the full reference route, crossing the common finish without reset; authoritative supervisor incident audit and measured-perception/own-pose/clock checks pass.",
            "Use every scheduled engineering gate in the inventory. Genuine failures remain failures; missing cases do not qualify. Infrastructure retries require explicit classification/provenance and retain prior incomplete attempts."],
        "ranking": {"eligible_only": True, "primary_key": "Mean common elapsed simulation finish time over all20 exact paired grid IDs; lower is better.",
            "tiebreak": "Exact equal mean: lexical numerical_model_sha256, then lexical policy_key; no favorable subset or best-of retries.",
            "distinctness": "Inference float32 normalization, all weight/bias arrays, actual lane summary mode and validated feature/output schema. NPZ zip bytes/metadata-only SHA or name changes cannot create distinct racers.",
            "identity_clone_proof": identity_comparison,
            "interpretation": "Engineering selection on a known reused scene, not independent superiority or formal promotion."},
        "finalist": {"count": 1, "rule": "Lowest mean among fully eligible numerically distinct learned drivers; select using engineering/CEM only.",
            "freeze_target_utc": "2026-11-14T18:00:00Z", "freeze_before_any_confirmation_schedule": True,
            "no_candidate_if_none_eligible": True, "post_freeze_updates_or_substitutions": False},
        "roster": {"policies": 4, "rule": "Reference plus THREE FASTEST distinct fully eligible learned policies; no fixed005/006 anchor preference over a faster0057/CEM candidate.",
            "ordered_slots": "Reference first, then the three learned drivers in the declared deterministic engineering rank order.",
            "finalist_always_included": True, "perception": "One common official9409 checkpoint and70m horizon for every entrant.",
            "freeze_before_any_championship_or_geometric_schedule": True,
            "insufficient_distinct_drivers": "Do not insert identity or renamed clones. Activate only the predeclared006seed7 fallback if primary outcomes establish fewer than3 eligible distinct learned drivers; otherwise leave championship incomplete if time/physical qualification fails."},
        "conditional_backup006seed7": {"activation": "Only once primary00517/00617/0057/CEM outcomes are known and fewer than3 numerically distinct learned policies meet all gates; never from formal feedback.",
            "not_created_or_trained_at_preparation": True, "model_sha256": None,
            "fitting_examples_path": str(source / "fitting_examples.npz"), "fitting_examples_sha256": sha(source / "fitting_examples.npz"),
            "fitter_path": str(source / "source/frozen005_fitter.py"), "fitter_sha256": sha(source / "source/frozen005_fitter.py"),
            "warm_start_model_sha256": MODEL_SHA["neural_rich005_seed7"], "seed": 7, "epochs": 80, "batch_size": 1024,
            "learning_rate": .0003, "device": "CPU4", "architecture": [48, 192, 192, 96, 6],
            "recipe": "Reuse exact frozen006 dataset/splits/selection/loss and frozen fitter; warm-start corresponding005seed7. No data regeneration/new labels/formal feedback. Preserve actual changed-weight, Torch/NumPy/export parity and source/checkpoint audits.",
            "physical_gate_sequence": ["All four known official-CV engineering hazards", "clean field4", "clean field8", "exact matched clean field20"],
            "qualification": "Same complete selection eligibility and distinctness rules; no gate is skipped merely because it is a backup."},
        "final_tests_unchanged": {"baseline_bootstrap": "Full15 canonicalFast70holdout cases.",
            "fresh_confirmation": "Exactly one frozen learned finalist and incumbent, all15cases each, complete paired identity/safety/pace gates through Ledger.promote(split='confirmation').",
            "championship": "Four unchanged balanced20-car cases and all scheduled attempts; each policy occupies each exact slot once.",
            "fresh_geometry": "All6 predeclared new layouts for each of4 roster drivers; no fitting/threshold/model selection from results.",
            "claims": "Keep engineering-chosen finalist, verified formal-promotion status, per-heat first finisher, and championship observed/qualified winner separate. A raw race winner need not be the formally promoted finalist."},
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(__file__),
            "stage_schedule_amendment": sha(campaign / "plans/final_stage_amendment_20261114T1405Z/amendment.json"),
            "canonical_confirmation": sha(campaign / "formal-cases/confirmation.json"),
            "championship20": sha(campaign / "final-evaluation-plan/championship20.json"),
            "geometric_confirmation": sha(campaign / "final-evaluation-plan/geometric-confirmation.json")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Retain the earlier draft and use a new path")
    report = build_plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
        "formal_attempts_seen": report["new_Fast70_formal_attempts_at_preparation"],
        "identity_CEM_is_numeric_clone": report["ranking"]["identity_clone_proof"]["foundation"]["numerical_model_sha256"] ==
                                       report["ranking"]["identity_clone_proof"]["generation0_identity"]["numerical_model_sha256"],
        "CEM_callback_ready": report["CEM_candidate"]["current_callback_preview"]["ready"]}))


if __name__ == "__main__":
    main()
