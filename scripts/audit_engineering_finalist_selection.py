#!/usr/bin/env python3
"""Read-only, prospective-plan-bound engineering finalist eligibility.

Only development engineering runs enter selection. Supervisor poses are used
offline by the inherited case/finish audit, never as policy or fitting inputs.
The only writes are new reports and independent CPU evidence caches.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.cem_evidence import expected_case, scenario_from_job
from scripts.improvement_ledger import (Ledger, OFFICIAL_DETECTOR_SHA256, digest,
                                       scenario_identity, verify_snapshot)
from scripts.prepare_finalist_selection import (cem_validation_gate,
    numerical_model_identity, proposed_cem_selection_jobs)
from scripts.race_reward_cem import (outcome, candidate_reward, IDENTITY, INITIAL_STD,
    sample_generation, update_distribution, job_batch, bind_jobs)
from race_lidar.rich_driver import RichDriver

APPROVAL = "plans/finalist_roster_selection_approved_20261114T1435Z.json"
APPROVAL_SHA = "6f45474e2b227741a2c3fa10354f4999f0636b0123f7712c5203e58ce8f91faa"
DRAFT_SHA = "cb49428cef009166fa485e6d1a0b8a74eb86d183caa653e443f7db6022e592f6"
VERSION = "prospective_engineering_selection_v1"
GATES = ("visibility-entry", "right", "left", "flying-straight", "field4", "field8", "field20")
PRIMARY = {
    "rich005-seed17-fast70": "rich005-s17",
    "rich00617-original-fast70": "rich006-original",
    "rich005-seed7-fast70": "rich005-s7",
}
DEPENDENCIES = (
    "scripts/audit_engineering_finalist_selection.py", "scripts/prepare_finalist_selection.py",
    "scripts/race_reward_cem.py", "scripts/cem_evidence.py", "scripts/improvement_ledger.py",
    "scripts/evaluate_improvement_attempt.py", "scripts/training_sources.py",
    "src/race_lidar/race_results.py", "src/race_lidar/hotlap_driver.py",
    "src/race_lidar/hotlap_geometry.py", "src/race_lidar/rich_driver.py",
    "src/race_lidar/learned_driver.py", "src/race_lidar/lidar_hazards.py",
    "src/race_lidar/race_policy.py", "src/race_lidar/race_control.py", "src/race_lidar/remote.py",
    "scripts/conditional_rich006_backup.py", "scripts/prepare_conditional_rich006_backup.py",
    "scripts/fit_conditional_rich006_backup.py",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flag(job, key, default=None):
    args = job["args"]
    if args.count(key) > 1:
        raise ValueError("Duplicate queue flag: " + key)
    return args[args.index(key) + 1] if key in args else default


def gate_scene(gate):
    if gate.startswith("field"):
        return scenario_identity({"scenario": "race", "cars": int(gate[5:]), "rotation": 0,
            "seed": 83, "spacing": 14., "seconds": 480., "grid_jitter": .3})
    values = {"scenario": "stopped", "cars": 2, "rotation": 0, "seed": 7,
              "spacing": 12., "seconds": 120., "grid_jitter": 0., "ego_lane": 0.}
    values.update({
        "visibility-entry": {"obstacle_progress": 300., "ego_progress": 0.},
        "right": {"cars": 3, "obstacle_progress": 1000., "ego_progress": 900.},
        "left": {"cars": 3, "obstacle_progress": 3000., "ego_progress": 2900., "blocked_side": "left"},
        "flying-straight": {"obstacle_progress": 1000., "ego_progress": 0.},
    }[gate])
    return scenario_identity(values)


def eligible_ranking(candidates):
    """No best-of runs, metadata clones, arbitrary anchor or tolerance tie."""
    ranked, seen = [], set()
    eligible = [r for r in candidates if r.get("eligible") is True]
    if any(not finite(r.get("mean_finish_time_s")) or r["mean_finish_time_s"] <= 0 for r in eligible):
        raise ValueError("Eligible ranking contains an invalid twenty-car mean")
    for row in sorted(eligible,
                      key=lambda r: (r["mean_finish_time_s"], r["numerical_model_sha256"], r["policy_key"])):
        if row["numerical_model_sha256"] in seen:
            continue
        seen.add(row["numerical_model_sha256"])
        ranked.append({k: row[k] for k in ("candidate_id", "policy_key", "model_sha256",
            "numerical_model_sha256", "mean_finish_time_s", "selection20_run")})
    return ranked


def reduce_gate(rows):
    active = [row for row in rows if not row.get("infrastructure_exclusion")]
    if not active:
        return "PENDING"
    if any(row["status"] == "FAILED" for row in active):
        return "FAILED"
    if any(row["status"] == "PENDING" for row in active):
        return "PENDING"
    # Two genuine attempts are not an authorized best-of or a matched replicate.
    return "PASSED" if len(active) == 1 and active[0]["status"] == "PASSED" else "FAILED"


def cem_gate_order_issues(attempts, winner_id, validation_ids):
    """Authoritative closure order; queue dependency text alone is insufficient."""
    by_id = {a["attempt_id"]: a for a in attempts}
    field8, field20 = winner_id + "-selection-field8", winner_id + "-selection-field20"
    predecessors = {field8: list(validation_ids), field20: [field8, *validation_ids,
        *(winner_id + "-" + g for g in GATES[:5])]}
    issues = []
    parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    for aid, required in predecessors.items():
        if aid not in by_id:
            continue  # Not-yet-scheduled validation/selection remains pending.
        for parent in required:
            before, after = by_id.get(parent, {}), by_id[aid]
            try:
                finish, launch = parse(before["ledger_finished_at_utc"]), parse(after["ledger_scheduled_at_utc"])
                valid = (before.get("state") == "finished" and finish.tzinfo is not None
                    and launch.tzinfo is not None and finish <= launch)
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                issues.append("CEM_selection_launched_before_closed_gate:" + parent + "->" + aid)
    return issues


def undeclared_attempts(attempts, allowed_ids, job_by_id):
    return [a for a in attempts if a["attempt_id"] not in allowed_ids or a["attempt_id"] not in job_by_id]


def cem_candidate_spec(candidate):
    return {"registered_engineering_id": candidate["candidate_id"], "policy_key": candidate["policy"],
        **{k: candidate[k] for k in ("model_sha256", "model_path", "snapshot_path", "snapshot_sha256")}}


def cem_fold_matches(foundation_path, candidate):
    """Verify actual inference arrays against the frozen bounded output fold."""
    foundation, fitted = RichDriver.from_npz(foundation_path), RichDriver.from_npz(candidate["model_path"])
    parameters = np.asarray(candidate["parameters"], dtype=float)
    weights, biases = [a.copy() for a in foundation.weights], [a.copy() for a in foundation.biases]
    weights[-1][:3] *= np.float32(parameters[0])
    biases[-1][:3] *= np.float32(parameters[0])
    biases[-1][:3] += np.float32(parameters[1] / 60.)
    biases[-1][3:] += parameters[2:].astype(np.float32)
    return (len(weights) == len(fitted.weights) and len(biases) == len(fitted.biases)
        and all(np.array_equal(a, b) for a, b in zip(weights + biases + [foundation.mean, foundation.scale],
                                                    fitted.weights + fitted.biases + [fitted.mean, fitted.scale]))
        and all(fitted.metadata.get(k, "nearest" if k == "lane_summary_mode" else None)
                == foundation.metadata.get(k, "nearest" if k == "lane_summary_mode" else None)
                for k in ("format", "feature_names", "output_names", "lane_summary_mode"))
        and fitted.metadata.get("calibration_parameters") == candidate["parameters"]
        and fitted.metadata.get("calibration_foundation_sha256") == sha(foundation_path)
        and fitted.metadata.get("calibration_generation") == candidate["generation"])


def cem_validation_predicate(before, after):
    """The unchanged protected step predicate; failures are known negatives."""
    return bool(before and after and before.get("passed") is True and after.get("passed") is True
        and finite(before.get("mean_finish_time_s")) and finite(after.get("mean_finish_time_s"))
        and after["mean_finish_time_s"] <= before["mean_finish_time_s"] - 1.)


class Auditor:
    def __init__(self, campaign):
        self.campaign = Path(campaign).resolve()
        self.ledger = Ledger(self.campaign / "ledger.jsonl")
        self.records = self.ledger.records()
        self.state = self.ledger.state()
        self.jobs = load(self.campaign / "queue.json")
        if not isinstance(self.jobs, list) or len({r["job_id"] for r in self.jobs}) != len(self.jobs):
            raise ValueError("Queue job IDs are not unique")
        self.job_by_id = {r["job_id"]: r for r in self.jobs}
        self.completions = load(self.campaign / "queue_status.json").get("completed", {})
        self.inventory = {}
        self.sources = {p: sha(ROOT / p) for p in DEPENDENCIES}
        self.cache_dir = self.campaign / "analysis/finalist-selection-audit-cache" / digest(self.sources)
        self.approval, self.plan = self.approved_plan()
        self.exclusions = self.infrastructure_exclusions()
        self.base_inventory = dict(self.inventory)

    def remember(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError("Bound artifact changed: " + str(path))
        self.inventory[str(path)] = {"sha256": actual, "bytes": path.stat().st_size}
        return load(path) if path.suffix == ".json" else actual

    def event(self, event_type, predicate):
        found = [r for r in self.records if r["event_type"] == event_type and predicate(r["payload"])]
        if len(found) != 1:
            raise ValueError("Missing/ambiguous explicit authority: " + event_type)
        return {k: found[0][k] for k in ("event_type", "sequence", "utc", "sha256", "payload")}

    def approved_plan(self):
        approval = self.remember(self.campaign / APPROVAL, APPROVAL_SHA)
        if (approval.get("status") != "ROOT_APPROVED_PROSPECTIVE_SELECTION_PROTOCOL"
                or approval.get("draft_sha256") != DRAFT_SHA
                or approval.get("new_Fast70_formal_attempts_at_declaration") != []):
            raise ValueError("Prospective plan approval contract differs")
        plan = self.remember(ROOT / approval["draft_path"], DRAFT_SHA)
        self.protocol_event = self.event("finalist_roster_selection_protocol_approved",
            lambda p: p.get("sha256") == APPROVAL_SHA and p.get("draft_sha256") == DRAFT_SHA)
        if {c["registered_engineering_id"] for c in plan["primary_candidates"]} != set(PRIMARY):
            raise ValueError("Approved primary pool differs")
        for relative, expected in approval["source_sha256"].items():
            self.remember(ROOT / relative, expected)
        return approval, plan

    def infrastructure_exclusions(self):
        """Only the two declared outages, with exact immutable retry authority."""
        incident_dir = self.campaign / "incidents/detector-outage-20261114T131138Z"
        incident_sha = "557b9f74b397377b0234d416838037684653b42fb73aa1322aeb3940d456548f"
        recovery_sha = "5395993b742c68bcaedf40ba28f70922a2c9edb2d1ffe8b4964d5d5f2a38a1a4"
        incident = self.remember(incident_dir / "incident.json", incident_sha)
        # This one historical event used improvement_ledger.digest(parsed_plan).
        # Root reviewed the exact canonical-vs-byte proof; every other authority
        # below still binds file bytes. Both identities remain explicit here.
        recovery_path = incident_dir / "rich006_recovery1_plan.json"
        recovery_byte_sha = "f1e75cf0abfa7afeef5374a8eeeaf7568f750df37e6ca9dd7a88d5224817b6a2"
        recovery = self.remember(recovery_path, recovery_byte_sha)
        if digest(recovery) != recovery_sha:
            raise ValueError("Historical recovery canonical payload changed")
        ievent = self.event("infrastructure_incident_classified",
            lambda p: p.get("incident_sha256") == incident_sha and p.get("policy_performance_assessable") is False)
        revent = self.event("infrastructure_recovery_retry_plan",
            lambda p: p.get("plan_sha256") == recovery_sha and p.get("model_changed") is False
                      and p.get("original_attempts_preserved") is True)
        parse_time = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        created = parse_time(recovery["created_utc"])
        modified = datetime.fromtimestamp(recovery_path.stat().st_mtime, timezone.utc)
        if not (abs((modified - created).total_seconds()) < 1. and created < parse_time(revent["utc"])):
            raise ValueError("Historical recovery creation/declaration ordering differs")
        revent["historical_hash_contract"] = {"canonical_parsed_plan_sha256": recovery_sha,
            "exact_file_bytes_sha256": recovery_byte_sha,
            "scope": "Only this evidenced historical record uses canonical JSON digest; root explicitly reviewed."}
        for job in recovery["retry_jobs"]:
            old = self.job_by_id[job["job_id"].removesuffix("-recovery1")]
            if self.job_by_id.get(job["job_id"]) != job or old["args"] != job["args"]:
                raise ValueError("One of fourteen historical retry vectors changed")
        self.infrastructure_authorities = [ievent, revent]
        result = {}
        incident_rows = {r["attempt_id"]: r for r in incident["all_attempts"]}
        retry_jobs = {r["job_id"]: r for r in recovery["retry_jobs"]}
        for aid in recovery["superseded_attempt_ids"]:
            if not aid.startswith("rich006-original-"):
                continue  # Never consume the unrelated CV/formal outcomes.
            original = self.job_by_id[aid]
            retry = retry_jobs[aid + "-recovery1"]
            if original["args"] != retry["args"] or self.job_by_id.get(retry["job_id"]) != retry:
                raise ValueError("Infrastructure retry treatment/job changed: " + aid)
            row = incident_rows.get(aid)
            if row is not None:
                if (row["classification"] != "INFRASTRUCTURE_UNINITIALIZED"
                    or row["policy_performance_assessable"] is not False
                    or row["authored_result_files"] != 0 or row["sensor_frames"] != 0
                    or row["runtime_initialized_results"] != 0 or row["health_connection_refused"] is not True
                    or aid not in ievent["payload"]["affected_attempt_ids"]):
                    raise ValueError("Invalid uninitialized infrastructure classification: " + aid)
                run = Path(row["output_dir"])
                moving = [r["car_id"] for r in load(run / "config.json")["entrants"] if not r.get("is_static", False)]
                if len(moving) != row["expected_moving_cars"] or any((run / "cars" / car / "result.json").exists() for car in moving):
                    raise ValueError("An uninitialized exception acquired authored results: " + aid)
                for name, meta in incident["source_metadata_sha256"].items():
                    p = Path(name)
                    if p.is_relative_to(run) or p == Path(row["log_path"]):
                        self.remember(p, meta["sha256"])
                result[aid] = {"classification": row["classification"], "authority_sha256": incident_sha,
                    "retry_job_id": retry["job_id"], "original_evidence_preserved": True}
            else:
                completion = self.completions.get(aid, {})
                ancestor = completion.get("dependency")
                if (completion.get("status") != "skipped_failed_dependency"
                        or ancestor not in recovery["superseded_attempt_ids"]):
                    raise ValueError("Unproven outage dependency skip: " + aid)
                result[aid] = {"classification": "INFRASTRUCTURE_DEPENDENCY_SKIP",
                    "authority_sha256": recovery_sha, "outage_dependency": ancestor,
                    "retry_job_id": retry["job_id"], "original_evidence_preserved": True}
        # Every dependency-only exclusion must trace to an actually classified
        # uninitialized ancestor, not merely another arbitrary skipped job.
        for aid, exclusion in result.items():
            seen, cursor = set(), aid
            while result[cursor]["classification"] == "INFRASTRUCTURE_DEPENDENCY_SKIP":
                if cursor in seen:
                    raise ValueError("Cycle in infrastructure dependency exclusions")
                seen.add(cursor)
                cursor = result[cursor]["outage_dependency"]
                if cursor not in result:
                    raise ValueError("Skipped gate does not trace to classified detector outage")
        addendum_sha = "3bc8ffc7531484b0ad6651026e6f9bcb471fac0fe306b391d651c1b80e51cac6"
        retry_sha = "ffcad992b8b5e33cfe725600137812c637a2a406b3ef9b46d511825a642d30c6"
        addendum = self.remember(incident_dir / "rich005_seed7_midrun_addendum.json", addendum_sha)
        retry_plan = self.remember(self.campaign / "plans/final_stage_amendment_20261114T1405Z/rich005_seed7_field20_recovery1_plan.json", retry_sha)
        event = self.event("midrun_infrastructure_retry_queued_after_fits",
            lambda p: p.get("plan_sha256") == retry_sha and p.get("same_treatment_args") is True
                      and p.get("old_attempt_retained") is True)
        self.infrastructure_authorities.append(event)
        if (addendum.get("classification") != "INFRASTRUCTURE_INTERRUPTED_AFTER_INITIALIZATION"
            or addendum.get("classification_verified") is not True
            or addendum.get("original_attempt") != "rich005-s7-field20"
            or addendum.get("history_mutated") is not False
            or retry_plan.get("incident_addendum_sha256") != addendum_sha
            or retry_plan.get("treatment_args_identical") is not True
            or retry_plan["old_job"]["args"] != retry_plan["retry_job"]["args"]
            or self.job_by_id.get("rich005-s7-field20") != retry_plan["old_job"]
            or self.job_by_id.get("rich005-s7-field20-recovery1") != retry_plan["retry_job"]):
            raise ValueError("Interrupted seed7 retry authority/treatment differs")
        for relative, expected in addendum["file_sha256"].items():
            path = (ROOT / relative).resolve()
            sidecars = {self.campaign / "logs/rich005-s7-field20.log", ROOT / "scripts/audit_rich005_seed7_interruption.py"}
            if not path.is_relative_to(self.campaign / "engineering/rich005-s7-field20") and path not in sidecars:
                raise ValueError("Midrun exception file escaped its declared engineering run")
            self.remember(path, expected)
        result["rich005-s7-field20"] = {"classification": addendum["classification"],
            "authority_sha256": addendum_sha, "retry_job_id": "rich005-s7-field20-recovery1",
            "original_evidence_preserved": True, "partial_run_is_not_full_lap_evidence": True}
        return result

    def registered(self, spec):
        cid = spec["registered_engineering_id"]
        reg = self.state["candidates"].get(cid)
        if not reg:
            raise ValueError("Candidate alias was not registered: " + cid)
        descriptor = {"policy_key": spec["policy_key"], "model_sha256": spec["model_sha256"],
            "detector_sha256": OFFICIAL_DETECTOR_SHA256, "control_parameters": {"visible_distance_m": 70.}}
        if (reg.get("descriptor") != descriptor or reg.get("snapshot_sha256") != spec["snapshot_sha256"]
                or Path(reg["snapshot_path"]).resolve() != Path(spec["snapshot_path"]).resolve()):
            raise ValueError("Registered candidate differs from approved pool: " + cid)
        manifest = verify_snapshot(spec["snapshot_path"])
        if manifest["snapshot_sha256"] != spec["snapshot_sha256"]:
            raise ValueError("Frozen snapshot changed: " + cid)
        identity = numerical_model_identity(spec["model_path"])
        if identity["checkpoint_sha256"] != spec["model_sha256"]:
            raise ValueError("Actual candidate checkpoint changed: " + cid)
        if spec.get("numeric_identity") and identity != spec["numeric_identity"]:
            raise ValueError("Approved numerical model identity changed: " + cid)
        self.remember(spec["model_path"], spec["model_sha256"])
        return reg, identity

    def relevant_attempts(self, cid):
        return [a for a in self.state["attempts"] if a.get("candidate_id") == cid
                and a.get("stage") == "engineering" and a.get("split") == "development"]

    def relevant_jobs(self, cid):
        return [j for j in self.jobs if flag(j, "--candidate-id") == cid
                and flag(j, "--stage", "engineering") == "engineering"
                and flag(j, "--split", "development") == "development"]

    def audit_run(self, spec, job, expected=None):
        run = (ROOT / job["output"]).resolve()
        if run.parent != self.campaign / "engineering" or run.name != job["job_id"]:
            raise ValueError("Selection run is not an exact engineering output")
        expected = expected or expected_case(job, snapshot_path=spec["snapshot_path"],
            snapshot_sha256=spec["snapshot_sha256"], policy=spec["policy_key"], model_sha256=spec["model_sha256"])
        # Re-derive stored CEM expected cases from their original immutable jobs.
        derived = expected_case(job, snapshot_path=spec["snapshot_path"], snapshot_sha256=spec["snapshot_sha256"],
                                policy=spec["policy_key"], model_sha256=spec["model_sha256"])
        if derived != expected:
            raise ValueError("Expected case differs from actual declared queue job")
        result = outcome(run, expected_case=expected, cache_dir=self.cache_dir,
            model_sha256=spec["model_sha256"], policy=spec["policy_key"],
            kind="hazard" if expected["scenario"]["scenario"] == "stopped" else "field",
            snapshot_sha256=spec["snapshot_sha256"], field_size=expected["scenario"]["cars"])
        if result is None:
            return None
        issues = list(result["rejection_reasons"])
        binding = result.get("current_evidence_binding", {})
        if not binding.get("independent_current_audit"):
            issues.append("no_independent_current_closed_audit")
        else:
            cache = load(binding["cache_path"])
            for row in cache["file_sha256"]:
                path = Path(row["path"])
                self.inventory[str(path)] = {"sha256": row["sha256"], "bytes": path.stat().st_size}
        if expected["reference_sha256"] != self.plan["selection20_reference_sha256"]:
            issues.append("reference_differs_from_approved_common_track")
        full_rows, paired = [], []
        for entry in expected["entrants"]:
            if entry["is_static"]:
                continue
            path = run / "cars" / entry["car_id"] / "result.json"
            if not path.is_file():
                issues.append("missing_authored_result:" + entry["car_id"])
                continue
            row = self.remember(path)
            if row.get("runtime_source_sha256") != self.plan["common_actual_runtime_source8_sha256"]:
                issues.append("actual_runtime_source8_differs:" + entry["car_id"])
            if row.get("reference_sha256") != self.plan["selection20_reference_sha256"]:
                issues.append("actual_reference_differs:" + entry["car_id"])
            if row.get("opponent_ground_truth_used_for_control") is not False:
                issues.append("opponent_GT_control_not_explicitly_false:" + entry["car_id"])
            if expected["scenario"]["scenario"] == "race":
                target = expected["rules"]["reference_length_m"]
                if (row.get("valid_lap") is not True or row.get("completed") is not True
                    or row.get("start_progress_m") != entry["start_progress_m"] or entry["start_progress_m"] > 0
                    or row.get("laps") != 1 or row.get("finish_progress_m") != target
                    or not finite(row.get("ordered_progress_m")) or row["ordered_progress_m"] < target - 1e-3):
                    issues.append("not_a_full_common_ordered_lap:" + entry["car_id"])
                full_rows.append({"car_id": entry["car_id"], "finish_time_s": row.get("finish_time_s"),
                                  "ordered_progress_m": row.get("ordered_progress_m")})
                paired.append({k: entry[k] for k in ("car_id", "grid_index", "seed", "start_progress_m",
                    "start_lane_m", "expected_initial_xy_m", "expected_initial_yaw_rad")})
        # A source-verified DNF/collision remains usable negative evidence.
        # Ordered-lap failure is physical, whereas a changed source/model is not.
        source_issues = [v for v in issues if not v.startswith("not_a_full_common_ordered_lap:")]
        return {"passed": result["passed"] is True and not issues, "issues": sorted(set(issues)),
            "usable": result.get("usable") is True, "incident": result.get("incident"),
            "source_contract_verified": not source_issues,
            "audit_verified": result.get("audit_verified"), "mean_finish_time_s": result.get("mean_finish_time_s"),
            "outcome_sha256": result["outcome_sha256"], "expected_case_sha256": digest(expected),
            "audit_cache_identity": binding.get("identity"), "full_laps": full_rows,
            "paired_grid": paired, "paired_grid_sha256": digest(paired) if paired else None}

    def primary(self, spec):
        cid = spec["registered_engineering_id"]
        reg, identity = self.registered(spec)
        jobs, attempts = self.relevant_jobs(cid), self.relevant_attempts(cid)
        aids = {a["attempt_id"]: a for a in attempts}
        gates = {gate: [] for gate in GATES}
        unexpected = []
        for job in jobs:
            aid = job["job_id"]
            scene = scenario_from_job(job)
            gate = next((g for g in GATES if scene == gate_scene(g)), None)
            row = {"attempt_id": aid, "job": job, "ledger_attempt": aids.get(aid),
                   "queue_completion": self.completions.get(aid), "status": "PENDING"}
            if gate is None:
                row.update(status="FAILED", issues=["unexpected_engineering_scenario_for_declared_candidate"])
                unexpected.append(row)
                continue
            if aid in self.exclusions:
                row.update(status="EXCLUDED_INFRASTRUCTURE", infrastructure_exclusion=self.exclusions[aid])
            elif aid in aids and aids[aid].get("state") == "finished":
                attempt = aids[aid]
                if (attempt.get("snapshot_sha256") != spec["snapshot_sha256"]
                    or Path(attempt["output_dir"]).resolve() != (ROOT / job["output"]).resolve()
                    or attempt.get("scenario_sha256") != digest(scene)):
                    row.update(status="FAILED", issues=["ledger_attempt_identity_differs"])
                else:
                    evidence = self.audit_run(spec, job)
                    row.update(status="PASSED" if evidence and evidence["passed"] else "FAILED", evidence=evidence)
            elif aid in self.completions:
                row.update(status="FAILED", issues=["scheduled_gate_closed_without_authoritative_driver_evidence"])
            gates[gate].append(row)
        orphan = [a for a in attempts if a["attempt_id"] not in {j["job_id"] for j in jobs}]
        statuses = {g: reduce_gate(rows) for g, rows in gates.items()}
        selected = [r for r in gates["field20"] if r["status"] == "PASSED"]
        selected = selected[0] if len(selected) == 1 else None
        if selected and (ROOT / selected["job"]["output"]).resolve() != Path(spec["selection20_run"]).resolve():
            statuses["field20"] = "FAILED"
            unexpected.append({"issues": ["twenty_car_objective_not_predeclared_run"]})
        eligible = not unexpected and not orphan and all(v == "PASSED" for v in statuses.values())
        known = all(v != "PENDING" for v in statuses.values()) and all(a.get("state") == "finished" for a in orphan)
        if selected and eligible:
            launch_time = selected["ledger_attempt"]["ledger_scheduled_at_utc"]
            for gate in GATES[:-1]:
                successful = [r for r in gates[gate] if r["status"] == "PASSED"]
                if (len(successful) != 1 or successful[0]["ledger_attempt"].get("ledger_finished_at_utc", "~") > launch_time):
                    eligible = False
                    unexpected.append({"issues": ["selection20_launched_before_closed_gate:" + gate]})
        return {"candidate_id": cid, "policy_key": spec["policy_key"], "model_sha256": spec["model_sha256"],
            "numerical_model_sha256": identity["numerical_model_sha256"], "registration": reg,
            "known": known, "eligible": eligible, "gate_status": statuses, "gates": gates,
            "unexpected_jobs": unexpected, "orphan_engineering_attempts": orphan,
            "mean_finish_time_s": selected["evidence"]["mean_finish_time_s"] if selected else None,
            "selection20_run": spec["selection20_run"],
            "paired_grid_sha256": selected["evidence"]["paired_grid_sha256"] if selected else None}

    def cem_closed_job(self, candidate, job, expected):
        """Bind one consumed case to its exact queue, ledger and current bytes."""
        aid = job["job_id"]
        if self.job_by_id.get(aid) != job:
            raise ValueError("CEM original job vector differs: " + aid)
        attempts = [a for a in self.relevant_attempts(candidate["candidate_id"]) if a["attempt_id"] == aid]
        if (len(attempts) != 1 or attempts[0].get("state") != "finished"
                or aid not in self.completions):
            raise ValueError("CEM scheduled case not authoritatively closed: " + aid)
        attempt = attempts[0]
        if (attempt.get("snapshot_sha256") != candidate["snapshot_sha256"]
                or Path(attempt["output_dir"]).resolve() != (ROOT / job["output"]).resolve()
                or attempt.get("scenario_sha256") != digest(scenario_from_job(job))):
            raise ValueError("CEM ledger/world binding differs: " + aid)
        evidence = self.audit_run(cem_candidate_spec(candidate), job, expected)
        if (evidence is None or evidence.get("usable") is not True
                or evidence.get("audit_verified") is not True
                or evidence.get("source_contract_verified") is not True):
            raise ValueError("CEM current physical evidence is missing or unusable: " + aid)
        return evidence

    def cem_fitness(self, state):
        """Reproduce the actual proposal sequence and every preserved reward."""
        issues, fitness, candidates = [], [], []
        generations = state.get("generations", [])
        if (state.get("generation_limit") != 2 or state.get("candidates_per_generation") != 6
                or state.get("formal_sources_for_learning", []) != [] or "preparing_generation" in state
                or len(generations) > 2):
            issues.append("CEM_search_contract_changed")
        try:
            foundation = self.plan["ranking"]["identity_clone_proof"]["foundation"]
            self.remember(state["foundation_model"], foundation["checkpoint_sha256"])
            if (state["foundation_sha256"] != foundation["checkpoint_sha256"]
                    or numerical_model_identity(state["foundation_model"]) != foundation):
                raise ValueError("CEM foundation identity changed")
            for name, expected in state["source_sha256"].items():
                self.remember(ROOT / name, expected)
                self.remember(Path(state["output"]) / "source" / name, expected)
            if set(state["source_sha256"]) != {"scripts/race_reward_cem.py", "scripts/cem_evidence.py", "scripts/improvement_queue_edit.py",
                    "scripts/training_sources.py", "scripts/evaluate_improvement_attempt.py", "scripts/improvement_ledger.py"}:
                raise ValueError("CEM protected source inventory changed")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(str(exc))
        mean, std = IDENTITY.copy(), INITIAL_STD.copy()
        for index, generation in enumerate(generations):
            rows = generation.get("candidates", [])
            if generation.get("index") != index or generation.get("completed") is not True or len(rows) != 6:
                issues.append("CEM_generation_not_complete_six:" + str(index))
                continue
            expected_parameters = sample_generation(mean, std, index, state["seed"])
            current_rows = []
            for number, candidate in enumerate(rows):
                cid = candidate.get("candidate_id", "missing")
                candidates.append(candidate)
                record = {"candidate_id": cid, "all_scored_cases_reaudited": False,
                          "current_agrees_with_preserved_fitness": False}
                try:
                    if (cid != f"{state['search_id']}-g{index}-c{number}" or candidate.get("generation") != index
                            or candidate.get("index") != number
                            or not np.array_equal(candidate.get("parameters"), expected_parameters[number])):
                        raise ValueError("CEM preserved proposal sequence changed: " + cid)
                    self.registered(cem_candidate_spec(candidate))
                    if not cem_fold_matches(state["foundation_model"], candidate):
                        raise ValueError("CEM actual parameter-to-weight fold differs: " + cid)
                    immutable = self.remember(Path(candidate["model_path"]).parent / "candidate.json")
                    if any(candidate.get(k) != v for k, v in immutable.items()):
                        raise ValueError("CEM preserved candidate artifact differs: " + cid)
                    jobs = job_batch(candidate)
                    expected = bind_jobs(candidate, jobs)
                    if candidate.get("expected_cases") != expected:
                        raise ValueError("CEM preserved fitness worlds differ: " + cid)
                    current = candidate_reward(candidate, self.campaign, self.cache_dir)
                    if current is None or any(c.get("usable") is not True for c in current.get("cases", [])):
                        raise ValueError("CEM fitness missing/active/infrastructure-invalid: " + cid)
                    keys = ("feasible", "reward", "reason", "mean_finish_time_s")
                    preserved = self.remember(Path(candidate["model_path"]).parent / "physical_reward.json")
                    agrees = all(current.get(k) == candidate["reward"].get(k) == preserved.get(k) for k in keys)
                    if not agrees:
                        raise ValueError("CEM_fitness_changed:" + cid)
                    consumed = {Path(case["run"]).name for case in current["cases"]}
                    required = {j["job_id"] for j in jobs[:4]}
                    if all(c["passed"] for c in current["cases"] if c["kind"] == "hazard"):
                        required.add(jobs[-1]["job_id"])
                    if consumed != required:
                        raise ValueError("CEM actual scored case set differs: " + cid)
                    evidence = {j["job_id"]: self.cem_closed_job(candidate, j, expected[j["job_id"]])
                                for j in jobs if j["job_id"] in consumed}
                    # A field skipped after genuine failed hazards is a known
                    # dependency skip, never an invented field result.
                    if jobs[-1]["job_id"] not in consumed:
                        completion = self.completions.get(jobs[-1]["job_id"], {})
                        if (completion.get("status") != "skipped_failed_dependency"
                                or completion.get("dependency") not in required):
                            raise ValueError("CEM unproven failed-hazard field skip: " + cid)
                    current_rows.append({**candidate, "reward": current})
                    record.update(all_scored_cases_reaudited=True, current_agrees_with_preserved_fitness=agrees,
                        current_fitness={k: current.get(k) for k in keys}, case_evidence=evidence,
                        scored_run_paths=[c["run"] for c in current["cases"]])
                except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                    issues.append(str(exc)); record["issue"] = str(exc)
                fitness.append(record)
            if len(current_rows) == 6:
                try:
                    mean, std, elites = update_distribution(current_rows, mean, std)
                    if generation.get("elite_ids") != elites:
                        issues.append("CEM_actual_elites_differ:" + str(index))
                except ValueError as exc:
                    issues.append(str(exc))
            else:
                # Later proposal parameters cannot be justified from an
                # unverified earlier reward distribution.
                issues.append("CEM_generation_reward_distribution_unverified:" + str(index))
        if (not np.array_equal(state.get("mean"), mean) or not np.array_equal(state.get("std"), std)):
            issues.append("CEM_actual_final_distribution_differs")
        return candidates, fitness, sorted(set(issues))

    def cem_validation(self, state, winner, candidates):
        validation = state.get("validation") or {}
        baseline = candidates[0]
        if (validation.get("baseline") != baseline or validation.get("winner") != winner
                or validation.get("seed") != 211 or validation.get("rotation") != 1
                or validation.get("used_for_cem_updates") is not False):
            raise ValueError("CEM paired validation proposal/seed contract changed")
        jobs = [j for c in (baseline, winner) for j in job_batch(c, seed=211, rotation=1, validation=True)]
        expected = {k: v for c in (baseline, winner)
                    for k, v in bind_jobs(c, job_batch(c, seed=211, rotation=1, validation=True)).items()}
        if validation.get("jobs") != [j["job_id"] for j in jobs] or validation.get("expected_cases") != expected:
            raise ValueError("CEM exact paired seed211 validation worlds changed")
        evidence = {}
        for role, candidate, job in zip(("baseline", "winner"), (baseline, winner), jobs):
            evidence[role] = self.cem_closed_job(candidate, job, expected[job["job_id"]])
            saved = validation.get("results", {}).get(role)
            if (not saved or any(saved.get(k) != evidence[role].get(k)
                                  for k in ("usable", "passed", "audit_verified", "mean_finish_time_s", "outcome_sha256"))):
                raise ValueError("CEM saved/current validation differs: " + role)
        improved = cem_validation_predicate(evidence["baseline"], evidence["winner"])
        if validation.get("development_improvement") is not improved:
            raise ValueError("CEM saved validation predicate differs from actual outcomes")
        return evidence, improved

    def cem(self):
        path = Path(self.plan["CEM_candidate"]["state_path"])
        state = self.remember(path)
        # Include only this declared search's engineering registrations/attempts.
        search = state["search_id"]
        candidates = [c for g in state.get("generations", []) for c in g.get("candidates", [])]
        ids = {c["candidate_id"] for c in candidates}
        declared_ids = set(ids)
        validation = state.get("validation") or {}
        ids.update(c["candidate_id"] for c in (validation.get("baseline"), validation.get("winner")) if c)
        # An omitted whole proposal is still a same-search engineering trial.
        # Do not limit the inventory to IDs asserted by mutable state.json.
        prefix = search + "-g"
        actual_ids = {cid for cid in self.state["candidates"] if cid.startswith(prefix)}
        actual_ids.update(flag(j, "--candidate-id") for j in self.jobs
                          if str(flag(j, "--candidate-id", "")).startswith(prefix))
        actual_ids.update(a["candidate_id"] for a in self.state["attempts"]
                          if str(a.get("candidate_id", "")).startswith(prefix)
                          and a.get("stage") == "engineering" and a.get("split") == "development")
        ids.update(actual_ids)
        records = {cid: {"registration": self.state["candidates"].get(cid), "jobs": self.relevant_jobs(cid),
                         "attempts": self.relevant_attempts(cid)} for cid in sorted(ids)}
        terminal = state.get("status") in ("complete", "stopped")
        result = {"search_id": search, "state_path": str(path), "state_sha256": sha(path),
            "status": state.get("status"), "reason": state.get("reason"), "known": terminal,
            "eligible": False, "search_engineering_inventory": records, "formal_outcomes_used": False}
        if not terminal:
            result["issues"] = ["declared_CEM_search_not_terminal"]
            return result
        terminal_reasons = {"complete": {"identity_foundation_remains_best", "no_verified_development_gain", "development_gain_verified"},
            "stopped": {"deadline_reserve_prevents_complete_generation", "deadline_reserve_prevents_validation"}}
        if state.get("reason") not in terminal_reasons[state["status"]]:
            result.update(known=False, issues=["undeclared_CEM_terminal_reason"])
            return result
        candidates, fitness, issues = self.cem_fitness(state)
        if actual_ids - declared_ids:
            issues.append("CEM_whole_same_search_proposal_omitted_from_state")
        result["independent_twelve_proposal_fitness"] = fitness
        result["completed_proposal_count"] = len(candidates)
        reason = state["reason"]
        complete_required = reason != "deadline_reserve_prevents_complete_generation"
        if complete_required and (len(state.get("generations", [])) != 2 or len(candidates) != 12):
            issues.append("CEM_full_two_generations_of_six_not_verified")
        if not complete_required and len(state.get("generations", [])) not in (0, 1):
            issues.append("CEM_generation_deadline_stop_is_not_a_completed_prefix")
        # Every same-search attempt is retained, including an unlisted genuine
        # failure or an active orphan. A saved terminal flag cannot hide it.
        allowed_jobs = {j["job_id"] for c in candidates for j in job_batch(c)}
        validation_jobs = set((state.get("validation") or {}).get("jobs", []))
        allowed_jobs.update(validation_jobs)
        if reason == "development_gain_verified" and state.get("winner"):
            allowed_jobs.update(state["winner"]["candidate_id"] + "-selection-" + g for g in ("field8", "field20"))
        undeclared = [j for cid in ids for j in self.relevant_jobs(cid) if j["job_id"] not in allowed_jobs]
        orphan = [a for cid in ids for a in undeclared_attempts(self.relevant_attempts(cid), allowed_jobs, self.job_by_id)]
        result.update(unexpected_engineering_jobs=undeclared, orphan_engineering_attempts=orphan)
        if undeclared or orphan:
            issues.append("CEM_undeclared_same_candidate_engineering_attempt")
        feasible = [c for c in candidates if c.get("reward", {}).get("feasible") is True]
        winner = max(feasible, key=lambda c: c["reward"]["reward"]) if feasible else None
        if winner and state.get("winner") != winner and complete_required:
            issues.append("CEM_preserved_winner_is_not_actual_stable_maximum_reward")
        result["independently_selected_fitness_winner"] = winner["candidate_id"] if winner else None
        if reason.startswith("deadline_reserve_"):
            # Protected step saved only aggregate queue costs, not the queue
            # and completion snapshot used to compute them. Arithmetic alone
            # cannot establish an actual historical backlog. Fail closed until
            # independently bound scheduling evidence exists; do not fabricate
            # validation or a second generation that was never run.
            check = state.get("last_budget_check") or {}
            requested = (6 * sum(j["estimated_wall_seconds"] for j in job_batch(
                {"candidate_id": "estimate", "policy": "neural_estimate", "snapshot_path": "/estimate"}))
                if reason.endswith("complete_generation") else 600)
            try:
                campaign = self.remember(self.campaign / "campaign.json")
                deadline = datetime.fromisoformat(campaign["deadline_utc"]).timestamp()
                budget_valid = (all(finite(check.get(k)) for k in ("checked_epoch_s", "pending_queue_estimate_s",
                    "requested_estimate_s", "deadline_epoch_s")) and check["requested_estimate_s"] == requested
                    and check["deadline_epoch_s"] == deadline and check["pending_queue_estimate_s"] >= 0.
                    and check["checked_epoch_s"] < deadline - 300.
                    and check["checked_epoch_s"] + check["pending_queue_estimate_s"] + requested + 300. >= deadline)
            except (OSError, ValueError, KeyError, TypeError):
                budget_valid = False
            if state.get("validation") or validation_jobs:
                issues.append("CEM_deadline_stop_has_unexpected_validation_continuation")
            result.update(known=False, eligible=False, preserved_budget_check=check,
                saved_budget_rejection_arithmetic_verified=budget_valid,
                issues=sorted(set(issues + (["CEM_saved_budget_rejection_not_justified"] if not budget_valid else [])
                    + ["CEM_historical_budget_queue_completion_inputs_not_preserved"])))
            return result
        if issues or winner is None:
            result.update(known=False, issues=sorted(set(issues + ([] if winner else ["CEM_no_verified_feasible_winner"]))))
            return result
        spec = cem_candidate_spec(winner)
        reg, identity = self.registered(spec)
        result.update(candidate_id=winner["candidate_id"], policy_key=winner["policy"], registration=reg,
            model_sha256=winner["model_sha256"], numerical_model_sha256=identity["numerical_model_sha256"])
        if reason == "identity_foundation_remains_best":
            foundation_numeric = self.plan["ranking"]["identity_clone_proof"]["foundation"]["numerical_model_sha256"]
            if (not np.array_equal(winner["parameters"], IDENTITY) or
                    identity["numerical_model_sha256"] != foundation_numeric or validation or validation_jobs):
                result.update(known=False, issues=["CEM_identity_terminal_not_a_functional_foundation_clone_or_has_validation"])
            else:
                result.update(known=True, issues=["CEM_identity_foundation_remains_best_independently_verified"])
            return result
        if np.allclose(winner["parameters"], IDENTITY, atol=1e-10):
            result.update(known=False, issues=["CEM_validation_terminal_uses_identity_skipped_by_protected_step"])
            return result
        try:
            validation_evidence, improved = self.cem_validation(state, winner, candidates)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.update(known=False, issues=[str(exc)])
            return result
        result["validation_evidence"] = validation_evidence
        before, after = validation_evidence["baseline"], validation_evidence["winner"]
        result["independent_development_improvement"] = improved
        result["paired_validation_gain_s"] = (before["mean_finish_time_s"] - after["mean_finish_time_s"]
            if finite(before.get("mean_finish_time_s")) and finite(after.get("mean_finish_time_s")) else None)
        if improved != (reason == "development_gain_verified"):
            result.update(known=False, issues=["CEM_terminal_reason_disagrees_with_independent_validation_predicate"])
            return result
        if not improved:
            result.update(known=True, eligible=False,
                issues=["CEM_no_verified_nonidentity_development_gain_independently_verified"])
            return result
        issues = cem_validation_gate(state)
        preview = proposed_cem_selection_jobs(state)
        if not preview["ready"]:
            issues.extend(preview["issues"])
        gates, pending = {}, False
        expected_cases = winner.get("expected_cases") or next(
            (c["expected_cases"] for c in candidates if c["candidate_id"] == winner["candidate_id"]), {})
        for gate in GATES:
            if gate in ("field8", "field20"):
                job = next((j for j in preview["jobs"] if j["job_id"].endswith("-" + "selection-" + gate)), None)
                expected = None
            else:
                aid = winner["candidate_id"] + "-" + gate
                job = self.job_by_id.get(aid)
                expected = expected_cases.get(aid)
            if not job or self.job_by_id.get(job["job_id"]) != job:
                gates[gate] = {"status": "PENDING", "issues": ["required_frozen_CEM_gate_not_scheduled"]}
                pending = True
                continue
            a = next((a for a in self.relevant_attempts(winner["candidate_id"]) if a["attempt_id"] == job["job_id"]), None)
            if not a or a.get("state") != "finished":
                gates[gate] = {"status": "PENDING"}; pending = True; continue
            if gate in ("field8", "field20") and scenario_from_job(job) != gate_scene(gate):
                issues.append("CEM_selection_field_not_exact_paired_case:" + gate)
            evidence = self.audit_run(spec, job, expected)
            gates[gate] = {"status": "PASSED" if evidence and evidence["passed"] else "FAILED", "evidence": evidence}
        # A later additional same-candidate gate is never a favorable substitute.
        expected_ids = {winner["candidate_id"] + "-" + g for g in GATES[:5]}
        expected_ids.update(j["job_id"] for j in preview["jobs"])
        expected_ids.add(winner["candidate_id"] + "-validation-field4")
        extra = [j for j in self.relevant_jobs(winner["candidate_id"]) if j["job_id"] not in expected_ids]
        if extra:
            issues.append("extra_same_candidate_engineering_attempt_not_predeclared")
        winner_attempts = self.relevant_attempts(winner["candidate_id"])
        orphan = undeclared_attempts(winner_attempts, expected_ids, self.job_by_id)
        if orphan:
            issues.append("orphan_same_candidate_engineering_attempt_not_predeclared")
        validation_ids = [validation[role]["candidate_id"] + "-validation-field4" for role in ("baseline", "winner")]
        all_order_attempts = winner_attempts + [a for a in self.relevant_attempts(validation["baseline"]["candidate_id"])
                                               if a["attempt_id"] in validation_ids]
        issues.extend(cem_gate_order_issues(all_order_attempts, winner["candidate_id"], validation_ids))
        selected = gates.get("field20", {}).get("evidence") or {}
        result.update(known=not pending, eligible=not pending and not issues and all(g["status"] == "PASSED" for g in gates.values()),
            issues=sorted(set(issues)), validation_evidence=validation_evidence, gates=gates,
            orphan_engineering_attempts=orphan,
            mean_finish_time_s=selected.get("mean_finish_time_s"),
            selection20_run=str(self.campaign / "engineering" / (winner["candidate_id"] + "-selection-field20")),
            paired_grid_sha256=selected.get("paired_grid_sha256"))
        return result

    def primary_evidence(self):
        """Stable trigger basis excludes later backup state and formal outcomes."""
        from scripts.conditional_rich006_backup import primary_context_sha
        self.inventory = dict(self.base_inventory)
        primary = [self.primary(spec) for spec in self.plan["primary_candidates"]]
        cem = self.cem()
        candidates = list(primary)
        if cem.get("candidate_id"):
            candidates.append(cem)
        rank = eligible_ranking(candidates)
        known = all(c["known"] for c in candidates) and cem["known"]
        context = primary_context_sha(primary, cem, rank, self.sources, dict(self.inventory))
        return {"primary": primary, "CEM": cem, "candidates": candidates, "rank": rank,
                "known": known, "context_sha256": context}

    def conditional_backup(self, basis):
        from scripts import conditional_rich006_backup as backup
        activation = backup.check_activation(self.campaign, basis["context_sha256"], basis["known"],
                                             basis["rank"], self.state, self.records)
        if activation is None:
            return None, {"activation_verified": False, "status": "NOT_ACTIVATED", "known": True}
        self.remember(activation["path"], activation["sha256"])
        rec = activation["receipt"]
        for name in ("primary_engineering_report", "root_authorization"):
            self.remember(rec[name]["path"], rec[name]["sha256"])
        fit = backup.verified_fit(self.campaign, activation, self.state, basis["candidates"])
        status = {"activation_verified": True, "activation_receipt_sha256": activation["sha256"],
            "primary_selection_context_sha256": basis["context_sha256"], "status": "ACTIVATED_FIT_OR_REGISTRATION_PENDING", "known": False}
        if fit is None:
            return None, status
        self.remember(fit["proof_path"], fit["proof_sha256"])
        for name, expected in load(fit["proof_path"])["file_sha256"].items():
            self.remember(self.campaign / backup.FIT_FOLDER / name, expected)
        if backup.BACKUP_ID not in self.state["candidates"]:
            return None, {**status, "status": "VERIFIED_FIT_REGISTRATION_PENDING", "fit_proof_sha256": fit["proof_sha256"]}
        reg = self.state["candidates"][backup.BACKUP_ID]
        spec = {"registered_engineering_id": backup.BACKUP_ID, "policy_key": backup.POLICY,
            "model_path": fit["model_path"], "model_sha256": fit["checkpoint_sha256"],
            "snapshot_path": reg["snapshot_path"], "snapshot_sha256": reg["snapshot_sha256"],
            "selection20_run": str(self.campaign / "engineering" / (backup.PREFIX + "-field20"))}
        candidate = self.primary(spec)
        return candidate, {**status, "status": "PHYSICALLY_QUALIFIED" if candidate["eligible"] else "ENGINEERING_GATES_PENDING_OR_FAILED",
            "known": candidate["known"], "fit_proof_sha256": fit["proof_sha256"]}

    def report(self):
        basis = self.primary_evidence()
        candidates, cem = list(basis["candidates"]), basis["CEM"]
        conditional, backup_status = self.conditional_backup(basis)
        if conditional is not None:
            candidates.append(conditional)
        grids = {c["paired_grid_sha256"] for c in candidates if c.get("eligible")}
        if len(grids) > 1 or None in grids:
            raise ValueError("Eligible twenty-car evidence is not paired on the exact same twenty starts")
        known = basis["known"]
        ranked = eligible_ranking(candidates)
        fallback_required = known and len(basis["rank"]) < 3
        complete = known and backup_status["known"] and len(ranked) >= 3
        body = {"schema_version": 1, "audit_version": VERSION,
            "status": "VERIFIED_ENGINEERING_SELECTION" if complete else "INCOMPLETE",
            "campaign": str(self.campaign), "formal_outcomes_used": False,
            "selection_protocol_sha256": APPROVAL_SHA, "prospective_draft_sha256": DRAFT_SHA,
            "protocol_approval_ledger_record": self.protocol_event,
            "source_sha256": self.sources, "primary_candidates": basis["primary"], "CEM": cem,
            "primary_selection_context_sha256": basis["context_sha256"],
            "conditional_backup_candidate": conditional,
            "all_primary_and_CEM_outcomes_known": known, "provisional_eligible_rank": ranked,
            "selected_finalist_id": ranked[0]["candidate_id"] if complete else None,
            "selected_roster_learned_ids": [r["candidate_id"] for r in ranked[:3]] if complete else [],
            "fallback006seed7": {"required": fallback_required, **backup_status,
                "predeclared_contract": self.plan["conditional_backup006seed7"]},
            "infrastructure_authority_records": self.infrastructure_authorities,
            "closed_file_inventory": self.inventory,
            "limits": ["Known reused engineering scene, not independent superiority or formal promotion.",
                "Large-log CPU audit caches use size/mtime invalidation; receipt verification rehashes every bound file.",
                "Exact numerical arrays/schema identity does not rule out all neural reparameterizations.",
                "Supervisor poses are offline scene/ordered-finish evidence only; never policy or fitting inputs."],
            "no_queue_ledger_runtime_or_training_mutations": True}
        body["selection_semantic_sha256"] = digest(body)
        return body


def audit_selection(campaign):
    return Auditor(campaign).report()


def verify_selection_report(report_path, campaign, *, require_complete=True):
    """Recompute selection and bind all current relevant engineering evidence.

    Appending unrelated formal/capacity events is harmless. A new matching
    engineering attempt or changed source/closed file invalidates the receipt.
    """
    saved = load(report_path)
    supplied_hash = saved.get("selection_semantic_sha256")
    if supplied_hash != digest({k: v for k, v in saved.items() if k != "selection_semantic_sha256"}):
        raise ValueError("Selection report semantic hash is invalid")
    if saved.get("formal_outcomes_used") is not False:
        raise ValueError("Selection report does not exclude formal outcomes")
    for path, meta in saved.get("closed_file_inventory", {}).items():
        if sha(path) != meta["sha256"] or Path(path).stat().st_size != meta["bytes"]:
            raise ValueError("Selection receipt closed-file contents changed: " + path)
    current = audit_selection(campaign)
    if current != saved:
        raise ValueError("Current engineering selection/evidence differs from the frozen report")
    if require_complete and current["status"] != "VERIFIED_ENGINEERING_SELECTION":
        raise ValueError("Engineering primary/CEM/fallback eligibility remains incomplete")
    return current


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=ROOT / "runs/improvement-12h")
    p.add_argument("--output", type=Path)
    p.add_argument("--verify", type=Path)
    args = p.parse_args()
    if args.verify:
        report = verify_selection_report(args.verify, args.campaign)
    else:
        if args.output is None or args.output.exists():
            p.error("A new --output path is required; earlier evidence remains immutable")
        report = audit_selection(args.campaign)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "selected_finalist_id",
        "all_primary_and_CEM_outcomes_known", "selection_semantic_sha256")}))


if __name__ == "__main__":
    main()
