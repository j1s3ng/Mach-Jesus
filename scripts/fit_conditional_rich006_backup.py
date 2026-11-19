#!/usr/bin/env python3
"""Future explicit CPU-only fit/register of the activated predeclared backup."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from scripts import conditional_rich006_backup as backup
from scripts.improvement_ledger import Ledger, digest, freeze_snapshot, verify_snapshot


def fit_budget(campaign, now=None):
    deadline = datetime.fromisoformat(backup.read(Path(campaign) / "campaign.json")["deadline_utc"].replace("Z", "+00:00"))
    remaining = (deadline - (now or datetime.now(timezone.utc))).total_seconds() - 300.
    if remaining < 60.: raise ValueError("Finite CPU fit cannot fit before campaign cleanup")
    return min(300., remaining)


def context(campaign):
    from scripts.audit_engineering_finalist_selection import Auditor
    audit = Auditor(campaign); basis = audit.primary_evidence()
    activation = backup.check_activation(campaign, basis["context_sha256"], basis["known"], basis["rank"], audit.state, audit.records)
    if activation is None: raise ValueError("No root-authorized actual trigger activation; fitting is forbidden")
    return audit, basis, activation


def fit(campaign, *, execute=False):
    if not execute: raise ValueError("Explicit --fit is required; no model is created during preparation")
    campaign = Path(campaign).resolve(); audit, basis, activation = context(campaign)
    fit_budget(campaign)
    declared = activation["receipt"]["recipe"]; folder = campaign / backup.FIT_FOLDER
    if folder.exists(): raise ValueError("Preserve previous actual fit; automatic replacement is forbidden")
    import numpy as np
    import torch
    from race_lidar.rich_driver import RichDriver, save_rich_model
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized(): raise ValueError("Backup recipe is CPU-only")
    folder.mkdir(parents=True)
    shutil.copyfile(declared["fitting_examples_path"], folder / "fitting_examples.npz")
    with np.load(folder / "fitting_examples.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    stage = folder / "fit-stage"; stage.mkdir(); shutil.copyfile(folder / "fitting_examples.npz", stage / "fitting_examples.npz")
    spec = importlib.util.spec_from_file_location("exact_frozen006_backup_fitter", declared["fitter_path"])
    fitter = importlib.util.module_from_spec(spec); spec.loader.exec_module(fitter)
    # ROOT is metadata-only in fit(); labels/splits already reside in the exact
    # frozen006 NPZ. No build()/teacher/data-generation entry point is called.
    fitter.ROOT = ROOT
    args = argparse.Namespace(output=stage, initial=campaign / "rich-driver-005", epochs=80,
        batch_size=1024, learning_rate=.0003,
        failure_run=campaign / "engineering/cv425-nn00517-left",
        evaluation_run=campaign / "engineering/cv425-nn00517-right")
    gradient = {"optimizer_steps": 0, "finite_nonzero_first_step_gradient_tensors": 0}
    original_step = torch.optim.AdamW.step
    def measured_step(optimizer, *a, **kw):
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if any(p.device.type != "cpu" or p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError("Actual CPU optimizer has missing/nonfinite gradients")
        if gradient["optimizer_steps"] == 0:
            gradient["finite_nonzero_first_step_gradient_tensors"] = sum(bool(torch.count_nonzero(p.grad)) for p in parameters)
        gradient["optimizer_steps"] += 1
        return original_step(optimizer, *a, **kw)
    ledger = Ledger(campaign / "ledger.jsonl")
    started = ledger.append("conditional_rich006_seed7_fit_started", {"activation_receipt_sha256": activation["sha256"],
        "recipe_sha256": digest(declared), "pid": os.getpid(), "device": "cpu", "output": str(folder)})
    def expired(*_): raise TimeoutError("Finite300s CPU fit budget expired; partial outputs preserved")
    previous_alarm = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, fit_budget(campaign))
        torch.optim.AdamW.step = measured_step
        summary = fitter.fit(7, args, data)
    except BaseException as error:
        (folder / "fit_failure.json").write_text(json.dumps({"error": repr(error), "outputs_preserved": True}) + "\n")
        ledger.append("conditional_rich006_seed7_fit_failed_preserved", {"output": str(folder), "error": repr(error), "automatic_retry": False})
        raise
    finally:
        torch.optim.AdamW.step = original_step
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_alarm)
    stage_model = RichDriver.from_npz(summary["checkpoint"])
    metadata = {**stage_model.metadata, "coverage_version": "rich006_conditional_seed7",
        "method": "predeclared_frozen006_seed7_backup", "conditional_activation_receipt_sha256": activation["sha256"],
        "frozen006_dataset_sha256": declared["fitting_examples_sha256"], "fitter_script_sha256": declared["fitter_sha256"],
        "recipe_sha256": digest(declared), "dataset_regenerated": False, "formal_data_used": False}
    final = folder / "seed-7"; final.mkdir()
    model_path = save_rich_model(final / "model.npz", stage_model.weights, stage_model.biases,
                                stage_model.mean, stage_model.scale, metadata)
    model = RichDriver.from_npz(model_path)
    parity = float(np.max(np.abs(model.forward_features(data["features"][:512]) - stage_model.forward_features(data["features"][:512]))))
    if parity != 0.: raise RuntimeError("Backup metadata export changed numerical weights")
    summary.update(checkpoint=str(model_path), checkpoint_sha256=backup.sha(model_path), stage_to_final_weight_parity=parity)
    (final / "training_report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (folder / "gradient_audit.json").write_text(json.dumps(gradient, indent=2) + "\n")
    names = ("fitting_examples.npz", "seed-7/model.npz", "seed-7/training_report.json",
        "fit-stage/seed-7/model.npz", "fit-stage/seed-7/training_report.json", "fit-stage/seed-7/epochs.json", "gradient_audit.json")
    proof = {"status": "ACTUAL_FROZEN006_SEED7_FIT_AND_PARITY_VERIFIED",
        "activation_receipt_sha256": activation["sha256"], "recipe_sha256": digest(declared),
        "model_sha256": backup.sha(model_path), "actual_device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),
        "fit_pid": os.getpid(), "fit_started_record_sha256": started['sha256'],
        "formal_data_used": False, "dataset_regenerated": False, **gradient,
        "file_sha256": {name: backup.sha(folder / name) for name in names}, "source_sha256": backup.source_hashes()}
    (folder / "fit_proof.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    backup.verified_fit(campaign, activation, audit.state, basis["candidates"], require_completed_event=False)
    ledger.append("conditional_rich006_seed7_fit_verified", {"fit_proof_sha256": backup.sha(folder / "fit_proof.json"),
        "model_sha256": proof["model_sha256"], "activation_receipt_sha256": activation["sha256"],
        "recipe_sha256": digest(declared), "fit_started_record_sha256": started['sha256'], "physical_gates_passed": False})
    backup.verified_fit(campaign, activation, audit.state, basis["candidates"])
    return {"model": str(model_path), "model_sha256": proof["model_sha256"], "physical_gates_passed": False}


def register(campaign, *, execute=False):
    if not execute: raise ValueError("Explicit --register is required")
    campaign = Path(campaign).resolve(); audit, basis, activation = context(campaign)
    verified = backup.verified_fit(campaign, activation, audit.state, basis["candidates"])
    if verified is None: raise ValueError("Actual fitting has not completed")
    if backup.BACKUP_ID in audit.state["candidates"]: raise ValueError("Keep the original backup registration")
    base = campaign / "snapshots/formal-v1-fast70"; manifest = verify_snapshot(base)
    files = {r["path"]: base / r["path"] for r in manifest["files"] if not r["path"].startswith("models/")}
    model_name = "models/" + backup.POLICY + ".npz"; files[model_name] = Path(verified["model_path"])
    snapshot = campaign / backup.SNAPSHOT
    frozen = freeze_snapshot(snapshot, files, metadata={"candidate_id": backup.BACKUP_ID,
        "policies": [{"policy": "reference", "label": "Reference"}, {"policy": backup.POLICY,
            "label": "Conditional Rich006 seed7", "model_relpath": model_name}],
        "policy_parameters": {"visible_distance_m": 70.}, "protocol": "race-improvement-v2",
        "conditional_backup_activation_sha256": activation["sha256"], "conditional_backup_fit_proof_sha256": verified["proof_sha256"]})
    Ledger(campaign / "ledger.jsonl").register_candidate(backup.BACKUP_ID, snapshot,
        {"policy_key": backup.POLICY, "model_sha256": verified["checkpoint_sha256"],
         "detector_sha256": backup.OFFICIAL_DETECTOR_SHA256, "control_parameters": {"visible_distance_m": 70.}},
        conditional_backup_activation_sha256=activation["sha256"], conditional_backup_fit_proof_sha256=verified["proof_sha256"])
    return {"snapshot": str(snapshot), "snapshot_sha256": frozen["snapshot_sha256"],
        "jobs_preview": backup.preview_jobs(campaign, snapshot, verified["checkpoint_sha256"]), "queue_changed": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--campaign", type=Path, default=ROOT / "runs/improvement-12h")
    group = p.add_mutually_exclusive_group(required=True); group.add_argument("--fit", action="store_true"); group.add_argument("--register", action="store_true")
    a = p.parse_args(); print(json.dumps(fit(a.campaign, execute=True) if a.fit else register(a.campaign, execute=True), indent=2))
