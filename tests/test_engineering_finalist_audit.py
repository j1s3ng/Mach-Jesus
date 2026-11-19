"""Prospective ranking, non-substitutable failures and current evidence binding."""
import copy
import json
from pathlib import Path

import pytest

from scripts import audit_engineering_finalist_selection as audit


def racer(cid, mean, numeric=None, eligible=True):
    return {"candidate_id": cid, "policy_key": cid, "model_sha256": "weights-" + cid,
        "numerical_model_sha256": numeric or cid, "mean_finish_time_s": mean,
        "eligible": eligible, "selection20_run": "/engineering/" + cid}


def test_faster_nonanchor_wins_and_failed_driver_cannot_win():
    ranked = audit.eligible_ranking([racer("anchor005", 280), racer("new006", 270),
                                     racer("0057", 260), racer("unsafe", 200, eligible=False)])
    assert [r["candidate_id"] for r in ranked] == ["0057", "new006", "anchor005"]


def test_numeric_clones_do_not_fill_roster_and_exact_tie_is_deterministic():
    ranked = audit.eligible_ranking([racer("zip-renamed", 260, "a"), racer("original", 260, "a"),
                                     racer("other", 260, "b"), racer("faster", 259, "z")])
    assert [r["candidate_id"] for r in ranked] == ["faster", "original", "other"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, 0, True])
def test_eligible_mean_requires_every_positive_finite_clock(value):
    with pytest.raises(ValueError, match="invalid twenty-car mean"):
        audit.eligible_ranking([racer("bad", value)])


def test_failure_cannot_be_replaced_by_later_success_or_majority_vote():
    assert audit.reduce_gate([{"status": "FAILED"}, {"status": "PASSED"}, {"status": "PASSED"}]) == "FAILED"
    assert audit.reduce_gate([{"status": "PASSED"}, {"status": "PASSED"}]) == "FAILED"


def test_only_explicit_infrastructure_exclusion_allows_fresh_retry():
    failed = {"status": "FAILED", "infrastructure_exclusion": {"classification": "INFRASTRUCTURE_UNINITIALIZED"}}
    assert audit.reduce_gate([failed]) == "PENDING"
    assert audit.reduce_gate([failed, {"status": "PASSED"}]) == "PASSED"
    assert audit.reduce_gate([{"status": "PENDING"}]) == "PENDING"


def test_known_gate_contracts_include_flying_straight_and_exact_paired_grid():
    assert len(audit.GATES) == 7
    assert audit.gate_scene("left")["ego_progress"] == 2900
    assert audit.gate_scene("flying-straight")["ego_progress"] == 0
    field = audit.gate_scene("field20")
    assert {k: field[k] for k in ("cars", "seed", "spacing", "grid_jitter", "rotation")} == {
        "cars": 20, "seed": 83, "spacing": 14., "grid_jitter": .3, "rotation": 0}


def write_report(path, **fields):
    report = {"status": "VERIFIED_ENGINEERING_SELECTION", "formal_outcomes_used": False,
              "closed_file_inventory": {}, "selected_finalist_id": "00517", **fields}
    report["selection_semantic_sha256"] = audit.digest(report)
    path.write_text(json.dumps(report))
    return report


def test_minimal_three_field_claim_is_not_accepted(tmp_path):
    path = tmp_path / "fake.json"
    path.write_text(json.dumps({"status": "VERIFIED_ENGINEERING_SELECTION",
                               "selected_finalist_id": "00517", "formal_outcomes_used": False}))
    with pytest.raises(ValueError, match="semantic hash"):
        audit.verify_selection_report(path, tmp_path)


def test_report_recomputes_engineering_outcomes_and_rejects_new_matching_attempt(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    report = write_report(path, matching_engineering_attempt_ids=["first"])
    monkeypatch.setattr(audit, "audit_selection", lambda _c: copy.deepcopy(report))
    assert audit.verify_selection_report(path, tmp_path) == report
    changed = copy.deepcopy(report)
    changed["matching_engineering_attempt_ids"].append("new-same-driver-run")
    monkeypatch.setattr(audit, "audit_selection", lambda _c: changed)
    with pytest.raises(ValueError, match="differs from the frozen"):
        audit.verify_selection_report(path, tmp_path)


def test_receipt_rehashes_large_log_even_if_mtime_and_size_are_unchanged(tmp_path, monkeypatch):
    import os
    trace = tmp_path / "control_trace.jsonl"
    trace.write_text("original\n")
    stat = trace.stat()
    path = tmp_path / "report.json"
    report = write_report(path, closed_file_inventory={str(trace): {"sha256": audit.sha(trace), "bytes": stat.st_size}})
    monkeypatch.setattr(audit, "audit_selection", lambda _c: copy.deepcopy(report))
    trace.write_text("changed!\n")
    os.utime(trace, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="closed-file contents changed"):
        audit.verify_selection_report(path, tmp_path)


def test_incomplete_report_never_qualifies_for_final_dispatch(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    report = write_report(path, status="INCOMPLETE", selected_finalist_id=None)
    monkeypatch.setattr(audit, "audit_selection", lambda _c: copy.deepcopy(report))
    with pytest.raises(ValueError, match="remains incomplete"):
        audit.verify_selection_report(path, tmp_path)
    assert audit.verify_selection_report(path, tmp_path, require_complete=False)["status"] == "INCOMPLETE"


def run_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    campaign = tmp_path / "campaign"
    run = campaign / "engineering" / "case"
    (run / "cars/R01").mkdir(parents=True)
    reference_sha, source8 = "track", {"controller": "frozen"}
    expected = {"job_id": "case", "output": str(run), "scenario": {"scenario": "race", "cars": 1},
        "reference_sha256": reference_sha, "rules": {"reference_length_m": 100.},
        "entrants": [{"car_id": "R01", "is_static": False, "grid_index": 0, "seed": 83,
            "start_progress_m": -.1, "start_lane_m": -2.5,
            "expected_initial_xy_m": [1., 2.], "expected_initial_yaw_rad": 0.}]}
    row = {"valid_lap": True, "completed": True, "start_progress_m": -.1, "laps": 1,
        "finish_progress_m": 100., "ordered_progress_m": 100.5, "finish_time_s": 10.,
        "runtime_source_sha256": source8, "reference_sha256": reference_sha,
        "opponent_ground_truth_used_for_control": False}
    path = run / "cars/R01/result.json"
    path.write_text(json.dumps(row))
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"file_sha256": []}))
    monkeypatch.setattr(audit, "expected_case", lambda *_a, **_k: copy.deepcopy(expected))
    monkeypatch.setattr(audit, "outcome", lambda *_a, **_k: {"passed": True, "audit_verified": True,
        "rejection_reasons": [], "outcome_sha256": "authored", "mean_finish_time_s": 10.,
        "current_evidence_binding": {"independent_current_audit": True, "cache_path": str(cache), "identity": "fresh"}})
    obj = audit.Auditor.__new__(audit.Auditor)
    obj.campaign, obj.inventory, obj.cache_dir = campaign, {}, tmp_path
    obj.plan = {"selection20_reference_sha256": reference_sha, "common_actual_runtime_source8_sha256": source8}
    spec = {"snapshot_path": "snapshot", "snapshot_sha256": "source", "policy_key": "nn", "model_sha256": "weights"}
    return obj, spec, {"job_id": "case", "output": str(run), "args": []}, path, row


@pytest.mark.parametrize("key,value,issue", [
    ("runtime_source_sha256", {"controller": "different"}, "actual_runtime_source8_differs"),
    ("reference_sha256", "different", "actual_reference_differs"),
    ("opponent_ground_truth_used_for_control", True, "opponent_GT_control"),
    ("ordered_progress_m", 50., "not_a_full_common_ordered_lap"),
    ("start_progress_m", 1., "not_a_full_common_ordered_lap"),
])
def test_actual_runtime_path_and_no_gt_contracts_are_required(tmp_path, monkeypatch, key, value, issue):
    obj, spec, job, path, row = run_fixture(tmp_path, monkeypatch)
    assert obj.audit_run(spec, job)["passed"]
    row[key] = value
    path.write_text(json.dumps(row))
    result = obj.audit_run(spec, job)
    assert not result["passed"] and any(issue in x for x in result["issues"])


def test_stored_cem_expected_case_cannot_drift_from_original_job(tmp_path, monkeypatch):
    obj, spec, job, _path, _row = run_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="differs from actual declared"):
        obj.audit_run(spec, job, {"different": "expected_world"})


def test_unrelated_formal_and_capacity_events_do_not_enter_engineering_fingerprint():
    obj = audit.Auditor.__new__(audit.Auditor)
    original = {"candidate_id": "chosen", "attempt_id": "engineering-one", "stage": "engineering", "split": "development"}
    obj.state = {"attempts": [original]}
    job = {"job_id": "engineering-one", "args": ["--candidate-id", "chosen"]}
    obj.jobs = [job]
    assert obj.relevant_attempts("chosen") == [original]
    assert obj.relevant_jobs("chosen") == [job]
    obj.state["attempts"] += [
        {"candidate_id": "chosen", "attempt_id": "confirmation", "stage": "promotion", "split": "confirmation", "outcome": {"DO_NOT_USE": True}},
        {"candidate_id": "capacity", "attempt_id": "capacity", "stage": "engineering", "split": "development"}]
    obj.jobs += [
        {"job_id": "confirmation", "args": ["--candidate-id", "chosen", "--stage", "promotion", "--split", "confirmation"]},
        {"job_id": "capacity", "args": ["--candidate-id", "capacity"]}]
    assert obj.relevant_attempts("chosen") == [original]
    assert obj.relevant_jobs("chosen") == [job]
    added = {**original, "attempt_id": "new-matching-engineering"}
    obj.state["attempts"].append(added)
    assert obj.relevant_attempts("chosen") == [original, added]


def test_cem_requires_actual_validation_to8_to20_closure_order():
    winner, validation = "winner", ["identity-validation-field4", "winner-validation-field4"]
    parents = validation + [winner + "-" + g for g in audit.GATES[:5]]
    attempts = [{"attempt_id": p, "state": "finished", "ledger_finished_at_utc": "2026-11-14T16:00:00Z"} for p in parents]
    attempts += [{"attempt_id": "winner-selection-field8", "state": "finished",
        "ledger_scheduled_at_utc": "2026-11-14T16:01:00Z", "ledger_finished_at_utc": "2026-11-14T16:05:00Z"},
        {"attempt_id": "winner-selection-field20", "state": "finished",
         "ledger_scheduled_at_utc": "2026-11-14T16:06:00Z", "ledger_finished_at_utc": "2026-11-14T16:10:00Z"}]
    assert audit.cem_gate_order_issues(attempts, winner, validation) == []
    changed = copy.deepcopy(attempts)
    changed[0]["ledger_finished_at_utc"] = "2026-11-14T16:02:00Z"
    assert any("identity-validation-field4->winner-selection-field8" in x
               for x in audit.cem_gate_order_issues(changed, winner, validation))
    changed = copy.deepcopy(attempts)
    changed[-2]["ledger_finished_at_utc"] = "2026-11-14T16:07:00Z"
    assert any("winner-selection-field8->winner-selection-field20" in x
               for x in audit.cem_gate_order_issues(changed, winner, validation))


def test_cem_pending_unscheduled_fields_do_not_invent_chronology_failure():
    assert audit.cem_gate_order_issues([], "winner", ["baseline-validation-field4", "winner-validation-field4"]) == []


def test_orphan_genuine_failure_is_not_ignored_even_when_all_declared_gates_pass():
    passed = {"attempt_id": "winner-left", "state": "finished", "outcome": {"hazard_pass": True}}
    orphan = {"attempt_id": "winner-unlisted-left-retry", "state": "finished", "outcome": {"hazard_pass": False}}
    assert audit.undeclared_attempts([passed, orphan], {"winner-left"}, {"winner-left": {}}) == [orphan]
    assert audit.undeclared_attempts([passed], {"winner-left"}, {}) == [passed]
