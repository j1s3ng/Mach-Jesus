#!/usr/bin/env python3
"""Freeze the verified engineering finalist/roster only under explicit --freeze.

Preparation is CPU-only and read-only. No queue jobs, processes or simulations
are started. Baseline bootstrap may still be pending; later dispatch uses its
separate unchanged normal/descriptive confirmation gate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.audit_engineering_finalist_selection import (APPROVAL, APPROVAL_SHA, DRAFT_SHA,
    PRIMARY, eligible_ranking, verify_selection_report)
from scripts.analyze_final_championship import declare_roster
from scripts.generate_improvement_cases import formal_cases
from scripts.exact_legacy_demo_scope import verified_legacy_demo_id, PLAN_SHA256 as LEGACY_PLAN_SHA
from scripts.improvement_ledger import Ledger, OFFICIAL_DETECTOR_SHA256, candidate_descriptor, digest, verify_snapshot
from scripts.prepare_final_race_jobs import RUNTIME, freeze_common_bundle
from scripts.prepare_finalist_selection import numerical_model_identity
from scripts.conditional_rich006_backup import BACKUP_ID

BASELINE = "reference-fast70-suite2"
RESERVED = {"confirmation", "championship", "geometric-confirmation"}
FREEZE_EVENTS = {"final_selection_freeze_started", "finalist_frozen_for_confirmation", "championship_roster_predeclared"}
SOURCE_FILES = ("scripts/freeze_final_engineering_selection.py", "scripts/audit_engineering_finalist_selection.py",
    "scripts/prepare_finalist_selection.py", "scripts/prepare_final_race_jobs.py", "scripts/analyze_final_championship.py",
    "scripts/generate_improvement_cases.py", "scripts/improvement_ledger.py")
SOURCE_FILES += ("scripts/exact_legacy_demo_scope.py",)
SOURCE_FILES += ("scripts/final_roster_binding.py",)
SOURCE_FILES += ("scripts/conditional_rich006_backup.py", "scripts/prepare_conditional_rich006_backup.py",
                 "scripts/fit_conditional_rich006_backup.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def immutable_json(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def verify_engineering_selection(path, campaign):
    """Tests may replace this boundary; production always calls the real audit."""
    return verify_selection_report(path, campaign, require_complete=True)


def protocol(campaign):
    approval_path = Path(campaign) / APPROVAL
    if sha(approval_path) != APPROVAL_SHA:
        raise ValueError("Approved prospective selection protocol changed")
    approval = read(approval_path)
    draft_path = ROOT / approval["draft_path"]
    if approval.get("draft_sha256") != DRAFT_SHA or sha(draft_path) != DRAFT_SHA:
        raise ValueError("Approved prospective rank/pool draft changed")
    return read(draft_path)


def require_no_reserved_schedule(campaign, records):
    jobs = read(Path(campaign) / "queue.json")
    legacy = verified_legacy_demo_id(campaign, records=records, jobs=jobs)
    if any(r["event_type"] == "attempt_scheduled" and r["payload"].get("split") in RESERVED
           and not (legacy is not None and r["payload"].get("attempt_id") == legacy) for r in records):
        raise ValueError("Fresh confirmation/championship/geometry was scheduled before final freeze")
    selected = []
    for job in jobs:
        args = job["args"]
        split = args[args.index("--split") + 1] if "--split" in args else "development"
        if split in RESERVED and job["job_id"] != legacy:
            selected.append(job["job_id"])
    if selected:
        raise ValueError("Fresh confirmation/championship/geometry is already queued before final freeze")
    return {"ignored_exact_demo_id": legacy, "authority_sha256": LEGACY_PLAN_SHA if legacy else None,
        "old_demo_counts_as_fresh_selection_training_or_promotion": False}


def require_first_freeze(campaign, destination, records):
    if Path(destination).exists() or any(r["event_type"] in FREEZE_EVENTS for r in records):
        raise ValueError("A final freeze or retained partial freeze already exists")
    return require_no_reserved_schedule(campaign, records)


def selection_blueprint(campaign, report_path, destination):
    campaign, report_path, destination = map(lambda p: Path(p).resolve(), (campaign, report_path, destination))
    if destination.parent != campaign:
        raise ValueError("Final freeze directory must be a new direct campaign child")
    report = verify_engineering_selection(report_path, campaign)
    if (report.get("status") != "VERIFIED_ENGINEERING_SELECTION"
        or report.get("formal_outcomes_used") is not False
        or report.get("selection_protocol_sha256") != APPROVAL_SHA
        or report.get("prospective_draft_sha256") != DRAFT_SHA):
        raise ValueError("Complete approved engineering-only selection is required")
    declared = protocol(campaign)
    ledger = Ledger(campaign / "ledger.jsonl")
    state, records = ledger.state(), ledger.records()
    legacy_scope = require_first_freeze(campaign, destination, records)
    candidates = list(report["primary_candidates"])
    if report.get("CEM", {}).get("candidate_id"):
        candidates.append(report["CEM"])
    conditional = report.get("conditional_backup_candidate")
    if conditional is not None:
        fallback = report.get("fallback006seed7", {})
        if (len(eligible_ranking(candidates)) >= 3
            or conditional.get("candidate_id") != BACKUP_ID or fallback.get("required") is not True
            or fallback.get("activation_verified") is not True or fallback.get("status") != "PHYSICALLY_QUALIFIED"
            or conditional.get("known") is not True or conditional.get("eligible") is not True):
            raise ValueError("Conditional backup lacks actual authorized fitting and all seven physical gates")
        candidates.append(conditional)
    allowed = set(PRIMARY)
    if conditional is not None:
        allowed.add(BACKUP_ID)
    cem = report.get("CEM", {})
    if cem.get("candidate_id") and cem.get("known") is True:
        if cem.get("search_id") != Path(declared["CEM_candidate"]["state_path"]).parent.name:
            raise ValueError("CEM candidate belongs to a different, unapproved search")
        allowed.add(cem["candidate_id"])
    if any(c["candidate_id"] not in allowed for c in candidates):
        raise ValueError("Engineering candidate lies outside the approved primary/CEM pool")
    ranked = eligible_ranking(candidates)
    if (len(ranked) < 3 or ranked != report.get("provisional_eligible_rank")
        or report.get("all_primary_and_CEM_outcomes_known") is not True
        or report.get("selected_finalist_id") != ranked[0]["candidate_id"]
        or report.get("selected_roster_learned_ids") != [r["candidate_id"] for r in ranked[:3]]):
        raise ValueError("Finalist/roster is not the exact prospective numeric-distinct rank")
    if {c["registered_engineering_id"] for c in declared["primary_candidates"]} != set(PRIMARY):
        raise ValueError("Approved primary pool is incomplete or altered")
    planned = {c["registered_engineering_id"]: c for c in declared["primary_candidates"]}
    if BASELINE not in state["candidates"]:
        raise ValueError("The exact reference-fast70-suite2 baseline must already be registered")
    base = campaign / "snapshots/formal-v1-fast70"
    reference = state["candidates"][BASELINE]
    descriptor = candidate_descriptor(reference)
    base_manifest = verify_snapshot(base)
    if (descriptor != {"policy_key": "reference", "model_sha256": None,
            "detector_sha256": OFFICIAL_DETECTOR_SHA256, "control_parameters": {"visible_distance_m": 70.}}
        or Path(reference["snapshot_path"]).resolve() != base.resolve()
        or reference["snapshot_sha256"] != base_manifest["snapshot_sha256"]):
        raise ValueError("Registered baseline differs from the frozen common reference")
    actual_base = {r["path"]: r["sha256"] for r in base_manifest["files"]}
    if (any(actual_base.get(name) != declared["common_actual_runtime_source8_sha256"].get(name) for name in RUNTIME)
        or actual_base.get("reference.npz") != declared["selection20_reference_sha256"]):
        raise ValueError("Common base differs from the engineering runtime8/reference")
    confirmation = campaign / "formal-cases/confirmation.json"
    if read(confirmation) != formal_cases("confirmation"):
        raise ValueError("Exactly the unchanged fifteen confirmation definitions are required")
    roster, models, bindings, chosen = [descriptor], {}, {BASELINE: {
        "snapshot_sha256": reference["snapshot_sha256"], "descriptor": descriptor}}, []
    for rank in ranked[:3]:
        cid = rank["candidate_id"]
        reg = state["candidates"].get(cid)
        if not reg:
            raise ValueError("Select an existing registered alias; no final clone is allowed")
        desc = candidate_descriptor(reg)
        if (desc["policy_key"] != rank["policy_key"] or desc["model_sha256"] != rank["model_sha256"]
            or desc["detector_sha256"] != OFFICIAL_DETECTOR_SHA256 or desc.get("control_parameters") != {"visible_distance_m": 70.}):
            raise ValueError("Ranked candidate registration changed")
        if cid in planned and (desc["model_sha256"] != planned[cid]["model_sha256"]
            or reg["snapshot_sha256"] != planned[cid]["snapshot_sha256"]):
            raise ValueError("Selected primary differs from its prospective frozen weights/source")
        snap = Path(reg["snapshot_path"])
        manifest = verify_snapshot(snap)
        if manifest["snapshot_sha256"] != reg["snapshot_sha256"]:
            raise ValueError("Ranked candidate snapshot changed")
        spec = next((p for p in manifest["metadata"]["policies"] if p["policy"] == desc["policy_key"]), None)
        if not spec or not spec.get("model_relpath"):
            raise ValueError("Ranked policy has no frozen runnable model")
        relative = Path(spec["model_relpath"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Ranked model path escaped its frozen snapshot")
        model = snap / relative
        identity = numerical_model_identity(model)
        if identity["checkpoint_sha256"] != rank["model_sha256"] or identity["numerical_model_sha256"] != rank["numerical_model_sha256"]:
            raise ValueError("Ranked actual numerical weights differ")
        roster.append(desc)
        models[desc["policy_key"]] = str(model.resolve())
        bindings[cid] = {"snapshot_sha256": reg["snapshot_sha256"], "descriptor": desc}
        chosen.append({**rank, "registered_snapshot_path": str(snap.resolve()), "model_path": str(model.resolve())})
    return {"schema_version": 1, "status": "VERIFIED_BLUEPRINT_NOT_FROZEN_OR_DECLARED",
        "campaign": str(campaign), "destination": str(destination), "baseline_id": BASELINE,
        "finalist_id": ranked[0]["candidate_id"], "learned_ids_in_rank_order": [r["candidate_id"] for r in chosen],
        "selected_models": chosen, "all_eligible_rank": ranked, "policies": roster, "models": models,
        "base_snapshot_path": str(base.resolve()), "registered_bindings": bindings,
        "engineering_selection_report": {"path": str(report_path), "sha256": sha(report_path),
            "selection_semantic_sha256": report["selection_semantic_sha256"]},
        "selection_protocol_sha256": APPROVAL_SHA, "prospective_draft_sha256": DRAFT_SHA,
        "exact_legacy_presentation_scope": legacy_scope,
        "confirmation_cases_sha256": sha(confirmation), "source_sha256": sources(),
        "formal_outcomes_used_for_selection": False,
        "baseline_bootstrap_is_not_a_selection_prerequisite": True,
        "queues_processes_and_tests_started": False}


def freeze_selection(campaign, report_path, destination, *, execute=False):
    if not execute:
        raise ValueError("Explicit --freeze is required; preparation does not create a final freeze")
    campaign, destination = Path(campaign).resolve(), Path(destination).resolve()
    # Use the existing producer lock directly. edit_queue() would rewrite the
    # queue even for a read-only critical section, which this helper never does.
    with (campaign / "final_selection_freeze.lock").open("a+") as own_lock, (campaign / "queue_edit.lock").open("a+") as producer_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(producer_lock, fcntl.LOCK_EX)
        blueprint = selection_blueprint(campaign, report_path, destination)
        destination.mkdir()
        immutable_json(destination / "selection_blueprint.json", blueprint)
        ledger = Ledger(campaign / "ledger.jsonl")
        ledger.append("final_selection_freeze_started", {"selection_blueprint_path": str(destination / "selection_blueprint.json"),
            "selection_blueprint_sha256": sha(destination / "selection_blueprint.json"),
            "selected_existing_finalist_id": blueprint["finalist_id"], "formal_outcomes_used": False})
        try:
            snapshot = destination / "common-bundle"
            manifest = freeze_common_bundle(blueprint["base_snapshot_path"], blueprint["policies"], blueprint["models"], snapshot)
            expected_roster = {"schema_version": 1, "snapshot_path": str(snapshot),
                "snapshot_sha256": manifest["snapshot_sha256"], "policies": blueprint["policies"], "excluded_from_training": True}
            roster_binding = {"schema_version": 1, "status": "ACTUAL_ROSTER_BOUND_TO_ENGINEERING_BEFORE_DECLARATION",
                "engineering_selection_report": blueprint["engineering_selection_report"],
                "selection_protocol_sha256": APPROVAL_SHA, "prospective_draft_sha256": DRAFT_SHA,
                "all_eligible_rank": blueprint["all_eligible_rank"],
                "learned_ids_in_rank_order": blueprint["learned_ids_in_rank_order"],
                "finalist_id": blueprint["finalist_id"], "registered_bindings": blueprint["registered_bindings"],
                "roster": expected_roster, "roster_canonical_sha256": digest(expected_roster),
                "source_sha256": blueprint["source_sha256"], "formal_outcomes_used_for_selection": False}
            roster_binding["exact_legacy_presentation_scope"] = blueprint["exact_legacy_presentation_scope"]
            binding_path = destination / "roster_selection_receipt.json"
            immutable_json(binding_path, roster_binding)
            ledger.append("final_roster_engineering_selection_bound", {"receipt_path": str(binding_path),
                "receipt_sha256": sha(binding_path), "roster_sha256": digest(expected_roster),
                "engineering_selection_report_sha256": blueprint["engineering_selection_report"]["sha256"],
                "engineering_rank_sha256": digest(blueprint["all_eligible_rank"])})
            require_no_reserved_schedule(campaign, ledger.records())
            roster_path = destination / "roster.json"
            declared = declare_roster(campaign, snapshot, blueprint["policies"], roster_path)
            if read(roster_path) != expected_roster or declared["roster_sha256"] != digest(expected_roster):
                raise ValueError("Existing roster API did not declare the exact pre-bound ordered roster")
            # Check current actual closed bytes/rank again after copying. No
            # candidate registration is replaced or duplicated by this helper.
            verified = verify_engineering_selection(report_path, campaign)
            if (verified["selection_semantic_sha256"] != blueprint["engineering_selection_report"]["selection_semantic_sha256"]
                or sha(report_path) != blueprint["engineering_selection_report"]["sha256"]
                or sources() != blueprint["source_sha256"]):
                raise ValueError("Engineering evidence or freeze sources changed during commit")
            receipt = {"schema_version": 1, "status": "FINALIST_FROZEN_FROM_ENGINEERING_ONLY",
                "frozen_utc": datetime.now(timezone.utc).isoformat(), "baseline_id": BASELINE,
                "finalist_id": blueprint["finalist_id"], "registered_bindings": blueprint["registered_bindings"],
                "engineering_selection_report": blueprint["engineering_selection_report"],
                "selection_protocol_sha256": APPROVAL_SHA, "prospective_draft_sha256": DRAFT_SHA,
                "confirmation_cases_sha256": blueprint["confirmation_cases_sha256"],
                "formal_outcomes_used_for_selection": False,
                "selected_roster_learned_ids": blueprint["learned_ids_in_rank_order"],
                "all_eligible_rank": blueprint["all_eligible_rank"],
                "actual_roster": {"path": str(roster_path), "file_sha256": sha(roster_path), "canonical_sha256": digest(expected_roster)},
                "roster_selection_receipt": {"path": str(binding_path), "sha256": sha(binding_path)},
                "selection_blueprint_sha256": sha(destination / "selection_blueprint.json"),
                "source_sha256": blueprint["source_sha256"], "fresh_jobs_prepared_or_enqueued": False}
            receipt["exact_legacy_presentation_scope"] = blueprint["exact_legacy_presentation_scope"]
            require_no_reserved_schedule(campaign, ledger.records())
            receipt_path = destination / "selection_receipt.json"
            immutable_json(receipt_path, receipt)
            ledger.append("finalist_frozen_for_confirmation", {"selection_receipt_path": str(receipt_path),
                "selection_receipt_sha256": sha(receipt_path), "baseline_id": BASELINE,
                "candidate_id": blueprint["finalist_id"], "roster_sha256": digest(expected_roster),
                "engineering_selection_report_sha256": blueprint["engineering_selection_report"]["sha256"],
                "selection_protocol_sha256": APPROVAL_SHA, "formal_outcomes_used_for_selection": False})
            return {"status": receipt["status"], "selection_receipt": str(receipt_path),
                "selection_receipt_sha256": sha(receipt_path), "roster_path": str(roster_path),
                "snapshot_path": str(snapshot), "snapshot_sha256": manifest["snapshot_sha256"],
                "finalist_id": blueprint["finalist_id"], "selected_existing_learned_ids": blueprint["learned_ids_in_rank_order"],
                "queue_or_process_actions": False}
        except BaseException as error:
            ledger.append("final_selection_freeze_failed_preserved", {"destination": str(destination), "error": repr(error),
                "historical_files_or_registrations_rewritten": False, "automatic_retry_or_substitution_allowed": False})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=ROOT / "runs/improvement-12h")
    p.add_argument("--selection-report", type=Path, required=True)
    p.add_argument("--destination", type=Path)
    p.add_argument("--freeze", action="store_true", help="Explicit root-reviewed commit; no jobs or processes are started")
    args = p.parse_args()
    destination = args.destination or args.campaign / "final-selection-001"
    result = freeze_selection(args.campaign, args.selection_report, destination, execute=True) if args.freeze else selection_blueprint(args.campaign, args.selection_report, destination)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__": main()
