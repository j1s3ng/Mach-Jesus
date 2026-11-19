"""Conditional006 seed7 support; no activation, fit or queue action on import.

The backup is admitted only after the predeclared primary pool is closed and
has fewer than three eligible numerical identities. Training uses the existing
frozen006 examples/fitter; formal results never determine activation or labels.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from scripts.improvement_ledger import Ledger, digest, OFFICIAL_DETECTOR_SHA256
from scripts.prepare_finalist_selection import numerical_model_identity

ROOT = Path(__file__).resolve().parents[1]
BACKUP_ID = "rich006-seed7-conditional-fast70"
POLICY = "neural_rich006_seed7"
PREFIX = "rich006-s7-conditional"
FOLDER = "conditional-rich006-seed7-001"
FIT_FOLDER = "training/conditional-rich006-seed7-fit-001"
SNAPSHOT = "snapshots/conditional-rich006-seed7-fast70"
ACTIVATION_EVENT = "conditional_rich006_seed7_backup_activated"
AUTHORIZATION_EVENT = "conditional_rich006_seed7_activation_authorized"
PROTOCOL_SHA = "6f45474e2b227741a2c3fa10354f4999f0636b0123f7712c5203e58ce8f91faa"
DRAFT_SHA = "cb49428cef009166fa485e6d1a0b8a74eb86d183caa653e443f7db6022e592f6"
DRAFT = "plans/finalist_roster_selection_draft_20261114T1415Z.json"
SOURCE_FILES = ("scripts/conditional_rich006_backup.py", "scripts/prepare_conditional_rich006_backup.py",
    "scripts/fit_conditional_rich006_backup.py", "scripts/train_rich_dagger.py", "scripts/train_rich_driver.py",
    "scripts/rich_measured_data.py", "scripts/training_sources.py", "src/race_lidar/rich_driver.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def finite(value): return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def source_hashes(): return {p: sha(ROOT / p) for p in SOURCE_FILES}


def trigger(known, rank):
    if known is not True:
        raise ValueError("Every primary and CEM outcome must be known before backup activation")
    identities = [r["numerical_model_sha256"] for r in rank]
    if len(set(identities)) != len(identities):
        raise ValueError("Primary ranking did not normalize numerical clones")
    if len(identities) >= 3:
        raise ValueError("Three eligible primary identities exist; backup activation is forbidden")


def recipe(campaign):
    campaign = Path(campaign).resolve()
    path = campaign / DRAFT
    if sha(path) != DRAFT_SHA: raise ValueError("Prospective backup recipe changed")
    result = read(path)["conditional_backup006seed7"]
    if {k: result[k] for k in ("seed", "epochs", "batch_size", "learning_rate", "device", "architecture")} != {
        "seed": 7, "epochs": 80, "batch_size": 1024, "learning_rate": .0003,
        "device": "CPU4", "architecture": [48, 192, 192, 96, 6]}:
        raise ValueError("Prospective backup numeric recipe differs")
    for name, key in (("fitting_examples_path", "fitting_examples_sha256"), ("fitter_path", "fitter_sha256")):
        if sha(result[name]) != result[key]: raise ValueError("Frozen backup data/fitter changed")
    initial = campaign / "rich-driver-005/seed-7/model.npz"
    if sha(initial) != result["warm_start_model_sha256"]:
        raise ValueError("Corresponding005seed7 warm start changed")
    return {**result, "warm_start_path": str(initial), "source_sha256": source_hashes()}


def primary_context_sha(primary, cem, rank, sources, inventory):
    return digest({"primary_candidates": primary, "CEM": cem, "primary_eligible_rank": rank,
        "source_sha256": sources, "primary_closed_file_inventory": inventory,
        "selection_protocol_sha256": PROTOCOL_SHA, "prospective_draft_sha256": DRAFT_SHA})


def require_new_numeric_identity(identity, warm, primary):
    if identity == warm or identity in {p["numerical_model_sha256"] for p in primary}:
        raise ValueError("Backup is unchanged or a metadata-only numerical clone")


def check_activation(campaign, context_sha, known, rank, state, records):
    """Return the exact activation, None if unused; unsolicited backups reject."""
    campaign = Path(campaign).resolve()
    path = campaign / FOLDER / "activation_receipt.json"
    events = [r for r in records if r["event_type"] == ACTIVATION_EVENT]
    relevant_jobs = [a for a in state["attempts"] if a.get("candidate_id") == BACKUP_ID]
    if not path.exists():
        if events or BACKUP_ID in state["candidates"] or relevant_jobs or (campaign / FIT_FOLDER).exists():
            raise ValueError("Unsolicited backup weights, registration or attempts without activation")
        return None
    trigger(known, rank)
    receipt = read(path); declared = recipe(campaign)
    if (receipt.get("status") != "ROOT_ACTIVATED_PREDECLARED_006SEED7_BACKUP"
        or receipt.get("candidate_id") != BACKUP_ID or receipt.get("policy_key") != POLICY
        or receipt.get("primary_selection_context_sha256") != context_sha
        or receipt.get("primary_eligible_rank") != rank
        or receipt.get("recipe") != declared or receipt.get("recipe_sha256") != digest(declared)
        or receipt.get("formal_outcomes_used") is not False):
        raise ValueError("Backup activation does not bind the actual closed engineering deficit")
    if len(events) != 1 or events[0]["payload"] != {"candidate_id": BACKUP_ID,
        "activation_receipt_path": str(path), "activation_receipt_sha256": sha(path),
        "primary_selection_context_sha256": context_sha, "recipe_sha256": digest(declared),
        "formal_outcomes_used": False}:
        raise ValueError("Unique actual backup activation ledger receipt is required")
    saved_path = Path(receipt["primary_engineering_report"]["path"])
    if sha(saved_path) != receipt["primary_engineering_report"]["sha256"]:
        raise ValueError("Activation's original engineering evidence bytes changed")
    saved = read(saved_path)
    if (saved.get("primary_selection_context_sha256") != context_sha
        or saved.get("all_primary_and_CEM_outcomes_known") is not True
        or saved.get("formal_outcomes_used") is not False
        or saved.get("provisional_eligible_rank") != rank
        or saved.get("selection_semantic_sha256") != digest({k: v for k, v in saved.items() if k != "selection_semantic_sha256"})):
        raise ValueError("Activation is not bound to the actual pre-backup deficit report")
    approval = Path(receipt["root_authorization"]["path"])
    if sha(approval) != receipt["root_authorization"]["sha256"]:
        raise ValueError("Backup root authorization bytes changed")
    authority = read(approval)
    if (authority.get("status") != "ROOT_APPROVED_CONDITIONAL_006SEED7_ACTIVATION"
        or authority.get("approved") is not True or authority.get("formal_feedback_used") is not False
        or authority.get("primary_selection_context_sha256") != context_sha
        or authority.get("recipe_sha256") != digest(declared)):
        raise ValueError("Actual root authorization does not bind this backup")
    authorization = [r for r in records if r["event_type"] == AUTHORIZATION_EVENT
        and r["payload"].get("file_sha256") == sha(approval)
        and r["payload"].get("primary_selection_context_sha256") == context_sha]
    if len(authorization) != 1 or authorization[0]["sequence"] >= events[0]["sequence"]:
        raise ValueError("Root must authorize the exact backup before activation")
    if relevant_jobs and any(a["ledger_scheduled_at_utc"] < events[0]["utc"] for a in relevant_jobs):
        raise ValueError("Backup physical attempts predate actual trigger activation")
    return {"path": str(path), "sha256": sha(path), "receipt": receipt,
        "activation_ledger_sequence": events[0]["sequence"]}


def verified_fit_lifecycle(campaign, activation, proof_path, proof, *, require_completed_event=True):
    """Files alone cannot fabricate the actual, once-only authorized CPU fit."""
    records = Ledger(Path(campaign) / 'ledger.jsonl').records()
    started = [r for r in records if r['event_type'] == 'conditional_rich006_seed7_fit_started']
    completed = [r for r in records if r['event_type'] == 'conditional_rich006_seed7_fit_verified']
    failures = [r for r in records if r['event_type'] == 'conditional_rich006_seed7_fit_failed_preserved']
    if len(started) != 1 or failures:
        raise ValueError('A unique actual fit start without hidden failed attempts is required')
    start = started[0]; payload = start['payload']
    expected_recipe = digest(activation['receipt']['recipe'])
    pid = proof.get('fit_pid')
    if (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            or payload != {'activation_receipt_sha256': activation['sha256'], 'recipe_sha256': expected_recipe,
                'pid': pid, 'device': 'cpu', 'output': str(Path(campaign).resolve() / FIT_FOLDER)}
            or proof.get('fit_started_record_sha256') != start['sha256']
            or start['sequence'] <= activation['activation_ledger_sequence']):
        raise ValueError('Actual fit start does not bind activation, recipe, PID and output')
    if not require_completed_event:
        if completed or pid != os.getpid():
            raise ValueError('Pre-completion validation is only for the active fitter before its verified event')
        return
    expected = {'fit_proof_sha256': sha(proof_path), 'model_sha256': proof['model_sha256'],
        'activation_receipt_sha256': activation['sha256'], 'recipe_sha256': expected_recipe,
        'fit_started_record_sha256': start['sha256'], 'physical_gates_passed': False}
    if (len(completed) != 1 or completed[0]['payload'] != expected
            or completed[0]['sequence'] <= start['sequence']):
        raise ValueError('Actual fit must have a unique matching start-to-verified ledger lifecycle')


def verified_fit(campaign, activation, state, primary, *, require_completed_event=True):
    """Actual fit/parity/changed weights proof; metadata renaming cannot qualify."""
    campaign = Path(campaign).resolve(); folder = campaign / FIT_FOLDER
    path = folder / "fit_proof.json"
    if not path.exists():
        if BACKUP_ID in state["candidates"]: raise ValueError("Backup registered before verified actual fitting")
        return None
    proof = read(path); rec = activation["receipt"]; declared = rec["recipe"]
    verified_fit_lifecycle(campaign, activation, path, proof, require_completed_event=require_completed_event)
    if declared["source_sha256"] != source_hashes(): raise ValueError("Activated fitter dependencies changed")
    if (proof.get("status") != "ACTUAL_FROZEN006_SEED7_FIT_AND_PARITY_VERIFIED"
        or proof.get("activation_receipt_sha256") != activation["sha256"]
        or proof.get("recipe_sha256") != digest(declared)
        or proof.get("formal_data_used") is not False or proof.get("dataset_regenerated") is not False
        or proof.get("actual_device") != "cpu" or proof.get("cuda_initialized") is not False
        or proof.get("source_sha256") != source_hashes()):
        raise ValueError("Actual backup fit provenance differs")
    required = {"fitting_examples.npz", "seed-7/model.npz", "seed-7/training_report.json",
        "fit-stage/seed-7/model.npz", "fit-stage/seed-7/training_report.json", "fit-stage/seed-7/epochs.json", "gradient_audit.json"}
    if set(proof.get("file_sha256", {})) != required:
        raise ValueError("Every actual fitter/epoch/gradient/export artifact must be bound")
    for filename, expected in proof["file_sha256"].items():
        p = (folder / filename).resolve()
        if not p.is_relative_to(folder) or sha(p) != expected: raise ValueError("Backup fitting artifact changed")
    if sha(folder / "fitting_examples.npz") != declared["fitting_examples_sha256"]:
        raise ValueError("Backup fit did not use the exact frozen006 data")
    summary = read(folder / "seed-7/training_report.json")
    epochs = read(folder / "fit-stage/seed-7/epochs.json")
    measures = ("maximum_learned_parameter_change", "warm_normalization_max_abs_error", "numpy_torch_max_abs_error", "stage_to_final_weight_parity")
    if any(not finite(summary.get(k)) for k in measures) or any(not finite(e.get("selection_loss")) for e in epochs):
        raise ValueError("Actual fit/parity/epoch measurements must be finite")
    if (summary.get("seed") != 7 or len(epochs) != 80
        or not 1 <= summary.get("selected_epoch", 0) <= 80
        or summary.get("maximum_learned_parameter_change", 0) <= 1e-8
        or summary.get("warm_normalization_max_abs_error", math.inf) > 1e-4
        or summary.get("numpy_torch_max_abs_error", math.inf) > 1e-4
        or summary.get("stage_to_final_weight_parity") != 0.
        or proof.get("optimizer_steps", 0) <= 0 or proof.get("finite_nonzero_first_step_gradient_tensors") != 8):
        raise ValueError("Backup actual updates/80epochs/parity proof is incomplete")
    model = folder / "seed-7/model.npz"
    identity = numerical_model_identity(model)
    if any(identity["array_shapes"].get("weight_" + str(i)) != shape for i, shape in enumerate(([192, 48], [192, 192], [96, 192], [6, 96]))):
        raise ValueError("Backup architecture differs from frozen006 recipe")
    from race_lidar.rich_driver import RichDriver
    metadata = RichDriver.from_npz(model).metadata
    if (metadata.get("seed") != 7 or metadata.get("trained") is not True
        or metadata.get("conditional_activation_receipt_sha256") != activation["sha256"]
        or metadata.get("recipe_sha256") != digest(declared)
        or metadata.get("initial_checkpoint_sha256") != declared["warm_start_model_sha256"]
        or metadata.get("training_dataset_sha256") != declared["fitting_examples_sha256"]
        or metadata.get("fitter_script_sha256") != declared["fitter_sha256"]):
        raise ValueError("Backup checkpoint metadata does not bind seed7/exact actual fit inputs")
    import numpy as np
    with np.load(folder / "fitting_examples.npz", allow_pickle=False) as dataset:
        splits = dataset["splits"]
        draws = sum(int((splits == name).sum()) * repeat for name, repeat in
            (("synthetic_train", 1), ("actual_train", 4), ("commit_train", 2), ("failure_train", 30)))
    gradient = read(folder / "gradient_audit.json")
    if (draws <= 0 or proof["optimizer_steps"] != 80 * math.ceil(draws / 1024)
        or gradient != {k: proof[k] for k in ("optimizer_steps", "finite_nonzero_first_step_gradient_tensors")}):
        raise ValueError("Actual optimizer budget/gradient log differs from the frozen fitter")
    if identity["checkpoint_sha256"] != summary.get("checkpoint_sha256") or identity["checkpoint_sha256"] != proof.get("model_sha256"):
        raise ValueError("Actual backup checkpoint bytes disagree")
    warm = numerical_model_identity(declared["warm_start_path"])["numerical_model_sha256"]
    require_new_numeric_identity(identity["numerical_model_sha256"], warm, primary)
    if BACKUP_ID in state["candidates"]:
        reg = state["candidates"][BACKUP_ID]
        if (reg.get("conditional_backup_activation_sha256") != activation["sha256"]
            or reg.get("conditional_backup_fit_proof_sha256") != sha(path)
            or reg.get("descriptor") != {"policy_key": POLICY, "model_sha256": identity["checkpoint_sha256"],
                "detector_sha256": OFFICIAL_DETECTOR_SHA256, "control_parameters": {"visible_distance_m": 70.}}):
            raise ValueError("Backup registration is not linked to actual activated fit")
    return {"proof_path": str(path), "proof_sha256": sha(path), "model_path": str(model), **identity}


def preview_jobs(campaign, snapshot, model_sha256):
    """Return the original seven engineering gates; never append them."""
    common = ["--source-root", str(snapshot), "--reference", str(Path(snapshot) / "reference.npz"),
        "--candidate-id", BACKUP_ID, "--single-policy", POLICY, "--vision-url", "http://127.0.0.1:8770",
        "--detector-sha", OFFICIAL_DETECTOR_SHA256, "--visibility", "70", "--stage", "engineering",
        "--split", "development", "--no-video", "--pointcloud-stride", "0"]
    jobs = []
    for gate in ("visibility-entry", "right", "left", "flying-straight", "field4", "field8", "field20"):
        args = list(common)
        if gate.startswith("field"):
            cars = int(gate[5:]); args += ["--scenario", "race", "--cars", str(cars), "--seconds", "480",
                "--seed", "83", "--spacing", "14", "--grid-jitter", ".3", "--secondary-cars", "8" if cars == 20 else "0"]
        else:
            blocked = gate in ("left", "right")
            obstacle, ego = (3000, 2900) if gate == "left" else (1000, 900) if blocked else (300, 0) if gate == "visibility-entry" else (1000, 0)
            args += ["--scenario", "stopped", "--cars", "3" if blocked else "2", "--seconds", "120", "--seed", "7",
                "--spacing", "12", "--obstacle-progress", str(obstacle), "--ego-progress", str(ego), "--blocked-side", "left" if gate == "left" else "right"]
        identity = PREFIX + "-" + gate
        dependencies = [] if len(jobs) < 4 else [j["job_id"] for j in jobs[:4]] + ([] if len(jobs) == 4 else [jobs[-1]["job_id"]])
        jobs.append({"job_id": identity, "output": "runs/improvement-12h/engineering/" + identity,
            "estimated_wall_seconds": 90 if not gate.startswith("field") else {"field4": 170, "field8": 270, "field20": 560}[gate],
            "max_wall_seconds": 600 if not gate.startswith("field") else 1800, "args": args,
            "requires": [{"job_id": aid, "true_fields": ["audit_verified", "hazard_pass" if index < 4 else "all_clean_finishes"]}
                         for aid in dependencies for index in [next(i for i, j in enumerate(jobs) if j["job_id"] == aid)]],
            "conditional_backup_model_sha256": model_sha256})
    return jobs
