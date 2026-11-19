import copy
import json
from pathlib import Path

import pytest

from scripts.improvement_ledger import (
    OFFICIAL_DETECTOR_SHA256, Ledger, can_schedule, compare_candidates, cyclic_grid, freeze_snapshot,
    environment_rules, environment_rules_hash, scenario_suite, verify_snapshot,
)


def trials(candidate, cases, *, clean=True, time=100., hazard_pass=True):
    output = []
    for case in cases:
        expected = case["evaluated_entrants"]
        output.append({"attempt_id": f"{candidate}-{case['scenario_id']}", "candidate_id": candidate,
            "scenario_id": case["scenario_id"], "split": case["split"], "stage": "promotion",
            "comparison_key": case["scenario_id"], "scenario_sha256": case["scenario_sha256"],
            "rules_sha256": "common-protocol-v2", "snapshot_sha256": candidate + "-frozen",
            "tier": case["tier"], "is_hazard": case.get("is_hazard", False),
            "expected_entrants": expected, "state": "finished", "status": "completed",
            "outcome": {"audit_verified": True, "snapshot_verified": True, "hazard_pass": hazard_pass,
                "snapshot_sha256": candidate+"-frozen", "rules_sha256": "common-protocol-v2",
                "homogeneous_field": True,
                "entrants": [{"car_id": f"R{i:02}", "valid_finish": True,
                              "policy_key": "reference", "model_sha256": None,
                              "runtime_detector_checkpoint_sha256": OFFICIAL_DETECTOR_SHA256,
                              "clean_finish": clean, "finish_time_s": time} for i in range(expected)]}})
    return output


def test_frozen_snapshot_survives_source_edits_and_detects_snapshot_edits(tmp_path):
    source = tmp_path / "policy.py"
    source.write_text("version=1\n")
    first = freeze_snapshot(tmp_path / "first", {"src/policy.py": source}, {"candidate": "one"})
    second = freeze_snapshot(tmp_path / "second", {"src/policy.py": source}, {"candidate": "one"})
    assert first["snapshot_sha256"] == second["snapshot_sha256"]
    source.write_text("version=2\n")
    assert verify_snapshot(tmp_path / "first")["snapshot_sha256"] == first["snapshot_sha256"]
    frozen = tmp_path / "first/src/policy.py"
    frozen.chmod(0o644)
    frozen.write_text("changed\n")
    with pytest.raises(ValueError, match="Frozen file changed"):
        verify_snapshot(tmp_path / "first")
    with pytest.raises(ValueError, match="Unsafe"):
        freeze_snapshot(tmp_path / "bad", {"../escape.py": source})


def test_ledger_hash_chain_recovery_and_lifecycle_preserve_failed_attempt(tmp_path):
    ledger = Ledger(tmp_path / "events.jsonl")
    ledger.append("run_initialized", {"incumbent_id": "old"})
    ledger.append("attempt_scheduled", {"attempt_id": "a", "candidate_id": "new", "stage": "promotion"})
    ledger.append("attempt_started", {"attempt_id": "a", "pid": 123})
    with pytest.raises(ValueError, match="identity cannot change"):
        ledger.append("attempt_finished", {"attempt_id": "a", "stage": "engineering"})
    ledger.append("attempt_finished", {"attempt_id": "a", "status": "failed", "outcome": {}})
    with pytest.raises(ValueError, match="cannot be overwritten"):
        ledger.append("attempt_finished", {"attempt_id": "a", "status": "completed"})
    with ledger.path.open("ab") as stream:
        stream.write(b'{"interrupted":')
    with pytest.raises(ValueError, match="Incomplete ledger tail"):
        ledger.state()
    backup = ledger.recover_tail()
    assert Path(backup).read_bytes() == b'{"interrupted":'
    assert ledger.state()["attempts"][0]["status"] == "failed"
    assert ledger.state()["record_count"] == 5
    with ledger.path.open("ab") as stream:
        stream.write(b'{"interrupted":')
    assert ledger.recover_tail() == backup  # An identical subsequent interruption reuses its immutable backup.
    assert ledger.state()["record_count"] == 6
    rows = ledger.path.read_text().splitlines()
    corrupted = json.loads(rows[0]); corrupted["payload"]["incumbent_id"] = "forged"
    rows[0] = json.dumps(corrupted)
    ledger.path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="hash chain"):
        ledger.state()


def test_four_rotations_cover_each_policy_at_every_grid_slot():
    cases = scenario_suite("holdout")
    assert len(cases) == 14
    for count in (4, 8, 20):
        seen = {policy: [] for policy in range(4)}
        for case in [c for c in cases if c["cars"] == count and c["kind"] == "race"]:
            for slot, identity in enumerate(cyclic_grid(count, case["grid_rotation"], case["reverse_grid"])):
                seen[identity % 4].append(slot)
        assert all(sorted(slots) == list(range(count)) for slots in seen.values())
    assert {c["seed"] for c in cases}.isdisjoint({c["seed"] for c in scenario_suite("development")})
    assert {c["seed"] for c in cases}.isdisjoint({c["seed"] for c in scenario_suite("confirmation")})


def test_strict_clean_improvement_and_all_hazards_pass_can_promote():
    cases = scenario_suite("holdout")
    old = trials("old", cases, clean=False)
    new = trials("new", cases)
    report = compare_candidates(old + new, "old", "new", cases)
    assert report["approved"]
    assert report["scores"]["old"]["clean_rate"] == 0
    assert report["scores"]["new"]["clean_rate"] == 1
    assert report["scores"]["new"]["expected"] == 128


def test_missing_scheduled_attempts_and_retries_cannot_be_dropped():
    cases = scenario_suite("holdout")
    old, new = trials("old", cases, clean=False), trials("new", cases)
    missing = copy.deepcopy(new)
    missing[0]["state"] = "started"; missing[0].pop("outcome")
    report = compare_candidates(old + missing, "old", "new", cases)
    assert not report["approved"]
    assert any("attempt_not_completed" in issue for issue in report["issues"])
    assert report["scores"]["new"]["expected"] == 128  # Missing entrants still count.
    retry = copy.deepcopy(new[0]); retry["attempt_id"] += "-retry"
    report = compare_candidates(old + new + [retry], "old", "new", cases)
    assert not report["approved"]
    assert any("duplicate_comparison_key" in issue for issue in report["issues"])


@pytest.mark.parametrize("defect", ["hazard", "audit", "caps", "snapshot", "case", "tier_regression"])
def test_safety_provenance_or_stratum_regressions_block_promotion(defect):
    cases = scenario_suite("holdout")
    old, new = trials("old", cases, clean=False), trials("new", cases)
    if defect == "hazard":new[-1]["outcome"]["hazard_pass"] = False
    elif defect == "audit":new[0]["outcome"]["audit_verified"] = False
    elif defect == "caps":new[0]["rules_sha256"] = "faster-car"
    elif defect == "snapshot":new[0]["snapshot_sha256"] = "changed-weights"
    elif defect == "case":new[0]["scenario_sha256"] = "easier-start"
    elif defect == "tier_regression":
        old = trials("old", cases)
        new[0]["outcome"]["entrants"][0]["clean_finish"] = False
    assert not compare_candidates(old + new, "old", "new", cases)["approved"]


def test_pace_promotion_requires_matched_clean_identities_and_no_slow_pairs():
    cases = scenario_suite("holdout")
    old, new = trials("old", cases, time=100), trials("new", cases, time=97)
    report = compare_candidates(old + new, "old", "new", cases)
    assert report["approved"]
    assert report["clean_pace_pairs"] == 128
    assert report["verified_improvements"] == ["paired_clean_pace_improvement"]
    new[0]["outcome"]["entrants"][0]["finish_time_s"] = 101
    assert not compare_candidates(old + new, "old", "new", cases)["approved"]
    new = trials("new", cases, time=97)
    # Exchanging which car is clean can bias a clean-only time average.
    old[0]["outcome"]["entrants"][0]["clean_finish"] = False
    new[0]["outcome"]["entrants"][1]["clean_finish"] = False
    assert not compare_candidates(old + new, "old", "new", cases)["approved"]


def test_engineering_runs_remain_visible_but_do_not_contaminate_confirmation():
    cases = scenario_suite("holdout")
    old, new = trials("old", cases, clean=False), trials("new", cases)
    engineering = copy.deepcopy(new[0]); engineering["attempt_id"] = "engineering-failure"
    engineering["stage"] = "engineering"; engineering["status"] = "failed"
    report = compare_candidates(old + new + [engineering], "old", "new", cases)
    assert report["approved"]
    assert report["non_scoring_attempt_ids_retained"] == ["engineering-failure"]


def test_ledger_promote_verifies_real_snapshots_before_switching_incumbent(tmp_path):
    source = tmp_path / "source.py"; source.write_text("policy=1\n")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("run_initialized", {"incumbent_id": "old"})
    cases = scenario_suite("holdout")
    attempts = trials("old", cases, clean=False) + trials("new", cases)
    for cid in ("old", "new"):
        path = tmp_path / cid
        manifest = freeze_snapshot(path, {"policy.py": source}, {"candidate_id": cid})
        ledger.register_candidate(cid, path, {"policy_key": "reference", "model_sha256": None})
        for attempt in [a for a in attempts if a["candidate_id"] == cid]:
            attempt["snapshot_sha256"] = manifest["snapshot_sha256"]
            attempt["outcome"]["snapshot_sha256"] = manifest["snapshot_sha256"]
            specification = {k: v for k, v in attempt.items() if k not in {"state", "status", "outcome"}}
            ledger.append("attempt_scheduled", specification)
            ledger.append("attempt_finished", {"attempt_id": attempt["attempt_id"], "status": "completed", "outcome": attempt["outcome"]})
    assert ledger.promote("new", cases)["approved"]
    assert ledger.state()["incumbent_id"] == "new"


def test_promotion_cannot_reuse_another_incumbents_approved_decision(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("run_initialized", {"incumbent_id": "current"})
    source = tmp_path/"source.py"; source.write_text("# source\n")
    bundle = tmp_path/"bundle"; freeze_snapshot(bundle, {"source.py":source})
    for cid in ("current", "new"):
        ledger.register_candidate(cid, bundle, {"policy_key":"reference", "model_sha256":None})
    ledger.append("promotion_decision", {"incumbent_id": "other", "candidate_id": "new",
                                          "approved": True, "decision_sha256": "old-evidence"})
    with pytest.raises(ValueError, match="approved decision"):
        ledger.append("candidate_promoted", {"incumbent_id": "current", "candidate_id": "new",
                                              "decision_sha256": "old-evidence"})


def test_generic_promotion_event_cannot_bypass_registered_descriptor_proof(tmp_path):
    source = tmp_path/"source.py"; source.write_text("# source\n")
    bundle = tmp_path/"bundle"; freeze_snapshot(bundle, {"source.py":source})
    ledger = Ledger(tmp_path/"ledger.jsonl")
    ledger.append("run_initialized", {"incumbent_id":"old"})
    for cid in ("old", "new"):
        ledger.register_candidate(cid, bundle, {"policy_key":"reference", "model_sha256":None})
    ledger.append("promotion_decision", {"incumbent_id":"old", "candidate_id":"new",
        "approved":True, "decision_sha256":"unbound-diagnostic-report"})
    with pytest.raises(ValueError, match="approved decision"):
        ledger.append("candidate_promoted", {"incumbent_id":"old", "candidate_id":"new",
                                               "decision_sha256":"unbound-diagnostic-report"})


def test_registered_policy_and_model_are_bound_even_when_bundle_contains_multiple_models(tmp_path):
    policy = tmp_path/"policy.py"; policy.write_text("# immutable controller\n")
    model_a = tmp_path/"a.npz"; model_a.write_bytes(b"model-a")
    model_b = tmp_path/"b.npz"; model_b.write_bytes(b"model-b")
    bundle = tmp_path/"bundle"
    manifest = freeze_snapshot(bundle, {"policy.py": policy, "models/a.npz": model_a, "models/b.npz": model_b})
    hashes = {row["path"]: row["sha256"] for row in manifest["files"]}
    ledger = Ledger(tmp_path/"ledger.jsonl")
    ledger.append("run_initialized", {"incumbent_id": "old"})
    ledger.register_candidate("old", bundle, {"policy_key": "reference", "model_sha256": None})
    ledger.register_candidate("new", bundle, {"policy_key": "neural_a", "model_sha256": hashes["models/a.npz"]})
    registrations = ledger.state()["candidates"]
    assert registrations["old"]["snapshot_sha256"] == registrations["new"]["snapshot_sha256"]
    cases = scenario_suite("holdout")
    old, new = trials("old", cases, clean=False), trials("new", cases)
    for attempt in old+new:
        attempt["snapshot_sha256"] = manifest["snapshot_sha256"]
        attempt["outcome"]["snapshot_sha256"] = manifest["snapshot_sha256"]
    for attempt in new:
        for entrant in attempt["outcome"]["entrants"]:
            entrant.update(policy_key="neural_a", model_sha256=hashes["models/b.npz"])
    report = compare_candidates(old+new, "old", "new", cases, snapshot_registry=registrations)
    assert not report["approved"]
    assert any("checkpoint_not_registered_candidate" in issue for issue in report["issues"])
    for attempt in new:
        for entrant in attempt["outcome"]["entrants"]:
            entrant["model_sha256"] = hashes["models/a.npz"]
    assert compare_candidates(old+new, "old", "new", cases, snapshot_registry=registrations)["approved"]
    new[0]["outcome"]["entrants"][0]["policy_key"] = "different-policy-label"
    assert not compare_candidates(old+new, "old", "new", cases, snapshot_registry=registrations)["approved"]


def test_unmatched_car_identities_cannot_win_on_completion_counts():
    cases = scenario_suite("holdout")
    old, new = trials("old", cases, clean=False), trials("new", cases)
    new[0]["outcome"]["entrants"][0]["car_id"] = "easier-starting-car"
    report = compare_candidates(old+new, "old", "new", cases)
    assert not report["approved"]
    assert any("mismatched_entrant_identities" in issue for issue in report["issues"])


def test_cv_candidate_binds_detector_even_with_same_policy_and_driver_weights(tmp_path):
    source = tmp_path/"policy.py"; source.write_text("# reference policy\n")
    detector = tmp_path/"detector.pth"; detector.write_bytes(b"adapted-detector")
    bundle = tmp_path/"bundle"
    manifest = freeze_snapshot(bundle, {"policy.py": source, "models/detector.pth": detector})
    adapted_hash = next(row["sha256"] for row in manifest["files"] if row["path"].endswith(".pth"))
    ledger = Ledger(tmp_path/"ledger.jsonl")
    ledger.register_candidate("old", bundle, {"policy_key": "reference", "model_sha256": None})
    ledger.register_candidate("cv", bundle, {"policy_key": "reference", "model_sha256": None,
                                             "detector_sha256": adapted_hash})
    cases = scenario_suite("holdout")
    attempts = trials("old", cases, clean=False) + trials("cv", cases)
    for attempt in attempts:
        attempt["snapshot_sha256"] = manifest["snapshot_sha256"]
        attempt["outcome"]["snapshot_sha256"] = manifest["snapshot_sha256"]
    report = compare_candidates(attempts, "old", "cv", cases, snapshot_registry=ledger.state()["candidates"])
    assert not report["approved"]
    assert any("entrant_detector_not_registered_candidate" in issue for issue in report["issues"])
    for attempt in attempts:
        if attempt["candidate_id"] == "cv":
            for entrant in attempt["outcome"]["entrants"]:
                entrant["runtime_detector_checkpoint_sha256"] = adapted_hash
    assert compare_candidates(attempts, "old", "cv", cases,
                              snapshot_registry=ledger.state()["candidates"])["approved"]


def test_bootstrap_requires_fresh_complete_verified_set_and_preserves_initialization(tmp_path):
    source = tmp_path/"policy.py"; source.write_text("# frozen source\n")
    bundle = tmp_path/"bundle"; manifest = freeze_snapshot(bundle, {"policy.py": source})
    ledger = Ledger(tmp_path/"ledger.jsonl")
    ledger.append("run_initialized", {"started_at": "2020-01-01T00:00:00Z"})
    initialization = ledger.records()[0]
    ledger.register_candidate("base", bundle, {"policy_key": "reference", "model_sha256": None})
    cases = scenario_suite("holdout")
    data = trials("base", cases)
    for index, attempt in enumerate(data):
        attempt["snapshot_sha256"] = manifest["snapshot_sha256"]
        attempt["outcome"]["snapshot_sha256"] = manifest["snapshot_sha256"]
        ledger.append("attempt_scheduled", {k:v for k,v in attempt.items() if k not in {"state","status","outcome"}})
        if index != len(data)-1:
            ledger.append("attempt_finished", {"attempt_id":attempt["attempt_id"], "status":"completed", "outcome":attempt["outcome"]})
    rejected = ledger.bootstrap_incumbent("base", cases, "holdout")
    assert not rejected["approved"]
    assert ledger.state()["incumbent_id"] is None
    last = data[-1]
    ledger.append("attempt_finished", {"attempt_id":last["attempt_id"], "status":"completed", "outcome":last["outcome"]})
    assert ledger.bootstrap_incumbent("base", cases, "holdout")["approved"]
    assert ledger.state()["incumbent_id"] == "base"
    assert ledger.records()[0] == initialization
    assert not ledger.bootstrap_incumbent("base", cases, "holdout")["approved"]


def test_retrospective_registration_cannot_bootstrap_an_incumbent(tmp_path):
    source = tmp_path/"policy.py"; source.write_text("# source\n")
    bundle = tmp_path/"bundle"; manifest = freeze_snapshot(bundle, {"policy.py":source})
    ledger = Ledger(tmp_path/"ledger.jsonl")
    ledger.append("run_initialized", {})
    cases = scenario_suite("holdout")[:1]
    attempt = trials("late", cases)[0]
    attempt["snapshot_sha256"] = manifest["snapshot_sha256"]
    attempt["outcome"]["snapshot_sha256"] = manifest["snapshot_sha256"]
    ledger.append("attempt_scheduled", {k:v for k,v in attempt.items() if k not in {"state","status","outcome"}})
    ledger.append("attempt_finished", {"attempt_id":attempt["attempt_id"], "status":"completed", "outcome":attempt["outcome"]})
    ledger.register_candidate("late", bundle, {"policy_key":"reference", "model_sha256":None})
    report = ledger.bootstrap_incumbent("late", cases, "holdout")
    assert not report["approved"]
    assert any("not_fresh_preregistered_trial" in issue for issue in report["issues"])


def test_controller_horizon_is_frozen_treatment_while_physical_rules_stay_matched(tmp_path):
    source = tmp_path/"policy.py"; source.write_text("# same controller implementation\n")
    ledger = Ledger(tmp_path/"ledger.jsonl")
    baseline_rules = {"max_speed_mps":60., "lidar_range_m":100., "sensor_period_ms":200,
                      "control_version":"v2", "visible_distance_m":45., "post_finish_mode":"park"}
    tuned_rules = {**baseline_rules, "control_version":"v3", "visible_distance_m":70.}
    assert environment_rules_hash(baseline_rules) == environment_rules_hash(tuned_rules)
    assert environment_rules(baseline_rules)["lidar_range_m"] == 100.
    assert environment_rules_hash({**tuned_rules,"max_speed_mps":61.}) != environment_rules_hash(baseline_rules)
    cases = scenario_suite("holdout")
    attempts = trials("old",cases,clean=False)+trials("new",cases)
    for cid,distance in (("old",45.),("new",70.)):
        bundle = tmp_path/cid
        parameters = {"visible_distance_m":distance}
        manifest = freeze_snapshot(bundle,{"policy.py":source},metadata={"policy_parameters":parameters})
        with pytest.raises(ValueError,match="parameters"):
            ledger.register_candidate(cid+"-unbound",bundle,{"policy_key":"reference","model_sha256":None})
        ledger.register_candidate(cid,bundle,{"policy_key":"reference","model_sha256":None,
                                              "control_parameters":parameters})
        for attempt in [a for a in attempts if a["candidate_id"]==cid]:
            attempt["snapshot_sha256"] = manifest["snapshot_sha256"]
            attempt["rules_sha256"] = environment_rules_hash(baseline_rules)
            attempt["outcome"].update(snapshot_sha256=manifest["snapshot_sha256"],rules_sha256=attempt["rules_sha256"])
            for entrant in attempt["outcome"]["entrants"]:
                entrant["runtime_policy_parameters"] = parameters
    registry = ledger.state()["candidates"]
    assert compare_candidates(attempts,"old","new",cases,snapshot_registry=registry)["approved"]
    first = next(a for a in attempts if a["candidate_id"]=="new")
    first["outcome"]["entrants"][0]["runtime_policy_parameters"] = {"visible_distance_m":45.}
    rejected = compare_candidates(attempts,"old","new",cases,snapshot_registry=registry)
    assert not rejected["approved"]
    assert any("control_parameters_not_registered" in issue for issue in rejected["issues"])


def test_deadline_reserves_cleanup_time():
    assert can_schedule("2026-11-14T20:50:47Z", "2026-11-14T21:50:47Z", 3000, 300)
    assert not can_schedule("2026-11-14T21:40:47Z", "2026-11-14T21:50:47Z", 400, 300)
