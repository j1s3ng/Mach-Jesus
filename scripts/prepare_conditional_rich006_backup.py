#!/usr/bin/env python3
"""Prepare conditional0067 support; explicit root authorization activates later."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import conditional_rich006_backup as backup
from scripts.improvement_ledger import Ledger, digest


def prepare(campaign):
    contract = backup.recipe(campaign)
    return {"status": "PREPARED_ONLY_NO_ACTIVATION_MODEL_FIT_OR_QUEUE",
        "candidate_id": backup.BACKUP_ID, "policy_key": backup.POLICY,
        "prospective_protocol_sha256": backup.PROTOCOL_SHA, "prospective_draft_sha256": backup.DRAFT_SHA,
        "recipe": contract, "recipe_sha256": digest(contract),
        "trigger": "All primary00517/00617/0057 plus declaredCEM outcomes known and fewer than3 eligible numerical identities.",
        "root_activation_required": True, "source_sha256": backup.source_hashes(),
        "future_fit_command": [str(Path(sys.executable)), str(ROOT / "scripts/fit_conditional_rich006_backup.py"),
            "--campaign", str(Path(campaign).resolve()), "--fit"],
        "future_gate_order": ["visibility-entry", "right", "left", "flying-straight", "field4", "field8", "field20"],
        "formal_data_used": False, "model_created": False, "training_started": False, "queue_changed": False}


def activate(campaign, report_path, authorization_path, *, execute=False):
    if not execute: raise ValueError("Root explicit --activate is required")
    from scripts.audit_engineering_finalist_selection import verify_selection_report
    from scripts.freeze_final_engineering_selection import require_first_freeze, immutable_json
    campaign = Path(campaign).resolve(); folder = campaign / backup.FOLDER
    with (campaign / "conditional_backup.lock").open("a+") as own, (campaign / "queue_edit.lock").open("a+") as writer:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(writer, fcntl.LOCK_EX)
        if folder.exists(): raise ValueError("Preserve the earlier backup activation; no silent retry")
        report = verify_selection_report(report_path, campaign, require_complete=False)
        backup.trigger(report["all_primary_and_CEM_outcomes_known"], report["provisional_eligible_rank"])
        if report.get("conditional_backup_candidate") is not None:
            raise ValueError("A backup is already present")
        ledger = Ledger(campaign / "ledger.jsonl")
        require_first_freeze(campaign, campaign / "final-selection-001", ledger.records())
        contract = backup.recipe(campaign); context = report["primary_selection_context_sha256"]
        authority = backup.read(authorization_path)
        if (authority.get("status") != "ROOT_APPROVED_CONDITIONAL_006SEED7_ACTIVATION"
            or authority.get("approved") is not True or authority.get("formal_feedback_used") is not False
            or authority.get("primary_selection_context_sha256") != context
            or authority.get("recipe_sha256") != digest(contract)):
            raise ValueError("Exact root-reviewed activation authority is required")
        matches = [r for r in ledger.records() if r["event_type"] == backup.AUTHORIZATION_EVENT
            and r["payload"].get("file_sha256") == backup.sha(authorization_path)
            and r["payload"].get("primary_selection_context_sha256") == context]
        if len(matches) != 1: raise ValueError("Root activation authorization must be recorded first")
        folder.mkdir()
        saved = folder / "primary_engineering_report.json"; saved.write_bytes(Path(report_path).read_bytes()); saved.chmod(0o444)
        receipt = {"status": "ROOT_ACTIVATED_PREDECLARED_006SEED7_BACKUP", "candidate_id": backup.BACKUP_ID,
            "policy_key": backup.POLICY, "activated_utc": datetime.now(timezone.utc).isoformat(),
            "primary_selection_context_sha256": context, "primary_eligible_rank": report["provisional_eligible_rank"],
            "primary_engineering_report": {"path": str(saved), "sha256": backup.sha(saved)},
            "root_authorization": {"path": str(Path(authorization_path).resolve()), "sha256": backup.sha(authorization_path)},
            "recipe": contract, "recipe_sha256": digest(contract), "formal_outcomes_used": False,
            "model_created": False, "training_or_physics_started": False}
        path = folder / "activation_receipt.json"; immutable_json(path, receipt)
        ledger.append(backup.ACTIVATION_EVENT, {"candidate_id": backup.BACKUP_ID,
            "activation_receipt_path": str(path), "activation_receipt_sha256": backup.sha(path),
            "primary_selection_context_sha256": context, "recipe_sha256": digest(contract), "formal_outcomes_used": False})
        return {"activation_receipt": str(path), "sha256": backup.sha(path), "training_started": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=ROOT / "runs/improvement-12h")
    p.add_argument("--plan-output", type=Path)
    p.add_argument("--activate", action="store_true"); p.add_argument("--primary-report", type=Path)
    p.add_argument("--root-authorization", type=Path)
    a = p.parse_args()
    if a.activate:
        if not a.primary_report or not a.root_authorization: p.error("Actual trigger report and root authorization required")
        result = activate(a.campaign, a.primary_report, a.root_authorization, execute=True)
    else:
        result = prepare(a.campaign)
        if a.plan_output:
            with a.plan_output.open("x") as f: json.dump(result, f, indent=2); f.write("\n")
    print(json.dumps(result, indent=2))
