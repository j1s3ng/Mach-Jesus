import copy
import datetime as dt
import json

import numpy as np
import pytest

from race_lidar.rich_driver import FEATURE_NAMES, RichDriver, save_rich_model
from scripts.improvement_ledger import OFFICIAL_DETECTOR_SHA256, environment_rules_hash
from scripts.improvement_queue_edit import validate_jobs
from scripts.race_reward_cem import (FIXED_RULES, IDENTITY, export_calibrated, foundation_ready,
                                    job_batch, outcome, project_parameters, race_reward,
                                    sample_generation, step, update_distribution)

MODEL_SHA = "1" * 64
SNAPSHOT_SHA = "2" * 64
POLICY = "neural_fixture"


@pytest.fixture(autouse=True)
def isolate_reward_unit_tests_from_the_separately_tested_closed_auditor(monkeypatch):
    # These tests isolate ranking/parameter search. test_cem_evidence exercises
    # real expected-world/closed-artifact validation and cache invalidation.
    import scripts.race_reward_cem as cem
    monkeypatch.setattr(cem, "closed_issues", lambda run, config, expected: [])
    monkeypatch.setattr(cem, "fresh_audit", lambda run, expected, cache: (json.loads((run / "ledger_outcome.json").read_text()), {"unit_fixture": True}))


def fixture_run(tmp_path, name, *, kind="field", size=None, incident=0, finish=240., passed=True):
    size = size or (2 if kind == "hazard" else 4)
    run = tmp_path / "engineering" / name
    run.mkdir(parents=True)
    config = {"dataset_split": "development", "excluded_from_training": False,
              "visible_distance_m": 70., "scenario": "stopped" if kind == "hazard" else "race",
              "rules": {**FIXED_RULES, "field_size": size, "reference_length_m": 1000.},
              "output_dir": str(run), "seed": 7, "grid_rotation": 0, "reference_length_m": 1000., "laps": 1,
              "start_time_s": 1., "max_sim_seconds": 481., "source_root": "/frozen", "reference_path": "/frozen/reference.npz",
              "snapshot_sha256": SNAPSHOT_SHA, "policy_parameters": {"visible_distance_m": 70.},
              "mode": "capacity" if kind == "hazard" else "race"}
    config["entrants"] = [{"car_id": f"R{index + 1:02d}", "grid_index": index, "seed": 7 + index,
        "start_progress_m": 300. if kind == "hazard" and index else -12. * index,
        "start_lane_m": 0. if kind == "hazard" else (-2.5 if index % 2 == 0 else 2.5),
        "is_static": kind == "hazard" and index > 0, "policy": "stationary_obstacle" if kind == "hazard" and index else POLICY,
        "model_path": None if kind == "hazard" and index else "/frozen/models/fixture.npz"} for index in range(size)]
    config["cars"] = {row["car_id"]: row for row in config["entrants"]}
    if kind == "hazard":
        config["hazard"] = {"ego_car_id": "R01", "progress_m": 300.}
    entries = [{"car_id": f"R{index:02d}", "model_sha256": MODEL_SHA, "policy_key": POLICY,
                "runtime_detector_checkpoint_sha256": OFFICIAL_DETECTOR_SHA256,
                "runtime_policy_parameters": {"visible_distance_m": 70.}, "incident_audit_complete": True,
                "collision_samples": incident, "track_violation_steps": 0, "position_resets_after_spawn": 0,
                "valid_finish": passed, "clean_finish": passed and incident == 0, "finish_time_s": finish,
                "progress_m": 1000. if passed else 500., "target_progress_m": 1000.}
               for index in range(1, (1 if kind == "hazard" else size) + 1)]
    audit = {"audit_verified": True, "snapshot_verified": True, "snapshot_sha256": SNAPSHOT_SHA,
             "rules_sha256": environment_rules_hash(config["rules"]),
             "entrants": entries, "all_clean_finishes": passed and incident == 0,
             "hazard_pass": passed and incident == 0,
             "hazard_evidence": {"ego_progress_m": 400. if passed else 290., "required_ego_progress_m": 320.,
                                 "zero_incidents": incident == 0}}
    (run / "config.json").write_text(json.dumps(config))
    (run / "ledger_outcome.json").write_text(json.dumps(audit))
    expected = {"job_id": name, "output": str(run), "scenario": {"scenario": config["scenario"], "seed": 7,
        "rotation": 0, "seconds": 480., "obstacle_progress": 300.}, "rules": config["rules"],
        "snapshot_path": "/frozen", "snapshot_sha256": SNAPSHOT_SHA, "reference_path": "/frozen/reference.npz",
        "entrants": config["entrants"]}
    (run / "expected_case.json").write_text(json.dumps(expected))
    return run


def score(run, *, kind="field", size=4):
    expected = json.loads((run / "expected_case.json").read_text())
    return outcome(run, expected_case=expected, cache_dir=run.parent.parent / "evidence_cache", model_sha256=MODEL_SHA,
                   policy=POLICY, kind=kind, snapshot_sha256=SNAPSHOT_SHA, field_size=size)


def test_parameter_projection_is_bounded_and_exactly_zero_mean():
    rng = np.random.default_rng(13)
    for values in rng.normal(size=(120, 5)) * 10.:
        result = project_parameters(values)
        assert .92 <= result[0] <= 1.08 and -1. <= result[1] <= 1.
        assert np.max(np.abs(result[2:])) <= .2 + 1e-12
        assert abs(result[2:].sum()) < 1e-12
    np.testing.assert_allclose(project_parameters(IDENTITY), IDENTITY, atol=1e-15)


def test_export_folds_calibration_into_weights_without_runtime_change(tmp_path):
    rng = np.random.default_rng(3)
    weight = rng.normal(0., .01, (6, len(FEATURE_NAMES))).astype(np.float32)
    bias = np.array([.3, .4, .5, -.1, .2, .3], np.float32)
    source = save_rich_model(tmp_path / "foundation.npz", [weight], [bias], np.zeros(48), np.ones(48), {})
    identity = export_calibrated(source, tmp_path / "identity.npz", IDENTITY, generation=0)
    parameters = project_parameters([1.05, .7, .2, -.1, -.1])
    calibrated = export_calibrated(source, tmp_path / "calibrated.npz", parameters, generation=1)
    values = rng.normal(size=(20, 48)).astype(np.float32)
    old = RichDriver.from_npz(source).forward_features(values)
    np.testing.assert_array_equal(old, RichDriver.from_npz(identity).forward_features(values))
    expected = old.copy(); expected[:, :3] = expected[:, :3] * 1.05 + .7 / 60.; expected[:, 3:] += parameters[2:]
    np.testing.assert_allclose(RichDriver.from_npz(calibrated).forward_features(values), expected, atol=1e-7)
    metadata = RichDriver.from_npz(calibrated).metadata
    assert metadata["calibration_search_status"] == "immutable_unscored_proposal"
    assert metadata["runtime_contract_changed"] is False


def test_hard_incident_rejection_cannot_be_offset_by_a_fast_finish(tmp_path):
    safe = score(fixture_run(tmp_path, "safe", finish=260.))
    fast_crash = score(fixture_run(tmp_path, "fast-crash", finish=100., incident=1))
    assert race_reward([safe])["feasible"]
    rejected = race_reward([fast_crash])
    assert rejected["reward"] == -1000. and not rejected["feasible"]
    assert race_reward([safe])["reward"] > rejected["reward"]


def test_short_hazards_use_passing_evidence_not_lap_completion_or_elapsed_time(tmp_path):
    run = fixture_run(tmp_path, "hazard", kind="hazard")
    path = run / "ledger_outcome.json"; audit = json.loads(path.read_text())
    audit["entrants"][0].update(valid_finish=False, clean_finish=False, finish_time_s=None)
    path.write_text(json.dumps(audit))
    hazard = score(run, kind="hazard")
    assert hazard["passed"] and hazard["mean_finish_time_s"] is None
    assert not race_reward([hazard])["feasible"]  # Required field4 has not run.
    field = score(fixture_run(tmp_path, "field"))
    assert race_reward([hazard, field])["feasible"]


@pytest.mark.parametrize("change", [{"model_sha256": "3" * 64}, {"collision_samples": None},
                                    {"runtime_policy_parameters": {"visible_distance_m": 90.}}])
def test_invalid_runtime_identity_and_missing_audit_cannot_be_elites(tmp_path, change):
    run = fixture_run(tmp_path, "invalid")
    path = run / "ledger_outcome.json"; audit = json.loads(path.read_text()); audit["entrants"][0].update(change)
    path.write_text(json.dumps(audit))
    reward = race_reward([score(run)])
    assert reward["reward"] == -2000. and not reward["feasible"]


def test_reserved_config_is_rejected_before_physical_reward_opens(tmp_path):
    run = fixture_run(tmp_path, "innocuous")
    path = run / "config.json"; config = json.loads(path.read_text()); config["dataset_split"] = "championship"
    path.write_text(json.dumps(config)); (run / "ledger_outcome.json").write_text("must not open")
    with pytest.raises(ValueError, match="excluded"):
        score(run)


def test_changed_physical_budget_rejects_even_with_audit_true(tmp_path):
    run = fixture_run(tmp_path, "budget")
    path = run / "config.json"; config = json.loads(path.read_text()); config["rules"]["max_speed_mps"] = 65.
    path.write_text(json.dumps(config)); result = score(run)
    assert not result["usable"] and "fixed_physical_rule_changed:max_speed_mps" in result["rejection_reasons"]


def test_cem_update_uses_only_feasible_actual_reward_elites():
    candidates = [{"candidate_id": "unsafe", "parameters": [1.08, 1., 0., 0., 0.], "reward": {"feasible": False, "reward": 1000.}},
                  {"candidate_id": "safe", "parameters": [1.02, .2, 0., 0., 0.], "reward": {"feasible": True, "reward": 110.}}]
    _, _, elites = update_distribution(candidates, IDENTITY, np.ones(5) * .1)
    assert elites == ["safe"]
    with pytest.raises(ValueError, match="rejected"):
        update_distribution(candidates[:1], IDENTITY, np.ones(5))


def test_generated_queue_skips_field4_unless_all_four_hazards_pass():
    candidate = {"candidate_id": "cem-g0-c0", "snapshot_path": "/frozen", "policy": POLICY}
    jobs = job_batch(candidate); validate_jobs(jobs)
    assert len(jobs) == 5
    assert len(jobs[-1]["requires"]) == 4
    assert all(item["true_fields"] == ["audit_verified", "hazard_pass"] for item in jobs[-1]["requires"])
    assert all("--stage" in job["args"] and "engineering" in job["args"] for job in jobs)
    validation = job_batch(candidate, validation=True, seed=211, rotation=1)
    assert len(validation) == 1 and validation[0]["requires"] == []


def test_initial_search_is_bounded_and_includes_exact_identity_baseline():
    generation = sample_generation(IDENTITY, np.ones(5), 0, 7)
    assert len(generation) == 6
    np.testing.assert_array_equal(generation[0], IDENTITY)
    with pytest.raises(ValueError):
        sample_generation(IDENTITY, np.ones(5), 2, 7)


def test_no_jobs_can_enqueue_before_all_foundation_gates(tmp_path):
    run = fixture_run(tmp_path, "foundation-field4")
    state = {"foundation_runs": [str(run)], "foundation_sha256": MODEL_SHA, "foundation_policy": POLICY,
             "foundation_snapshot_sha256": SNAPSHOT_SHA, "output": str(tmp_path),
             "foundation_expected_cases": {str(run): json.loads((run / "expected_case.json").read_text())}}
    ready, reason, _ = foundation_ready(state)
    assert not ready and "four_distinct_hazards" in reason
    campaign = tmp_path / "campaign"; campaign.mkdir()
    (campaign / "campaign.json").write_text(json.dumps({"deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)).isoformat()}))
    state.update(status="waiting_for_foundation", output=str(tmp_path), campaign=str(campaign), generations=[])
    result = step(state, enqueue=True)
    assert result["status"] == "waiting_for_foundation"
    assert not (campaign / "queue.json").exists()


def test_interrupted_generation_resumes_without_reregistering_or_duplicating_jobs(tmp_path, monkeypatch):
    import scripts.race_reward_cem as cem
    from scripts.improvement_ledger import Ledger, freeze_snapshot
    campaign = tmp_path / "campaign"; campaign.mkdir()
    (campaign / "campaign.json").write_text(json.dumps({"deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)).isoformat()}))
    output = tmp_path / "search"; output.mkdir()
    foundation = save_rich_model(tmp_path / "foundation.npz", [np.zeros((6, 48), np.float32)],
                                 [np.zeros(6, np.float32)], np.zeros(48), np.ones(48), {})
    reference = tmp_path / "reference.npz"
    angles = np.linspace(0., 2 * np.pi, 60, endpoint=False)
    np.savez(reference, x_m=100. * np.cos(angles), y_m=100. * np.sin(angles), speed_mps=np.full(60, 30.))
    snapshot = tmp_path / "foundation-snapshot"
    manifest = freeze_snapshot(snapshot, {"models/foundation.npz": foundation, "reference.npz": reference},
                               metadata={"policies": [{"policy": POLICY, "model_relpath": "models/foundation.npz"}],
                                         "policy_parameters": {"visible_distance_m": 70.}})
    state = {"status": "waiting_for_foundation", "campaign": str(campaign), "output": str(output),
             "foundation_model": str(foundation), "foundation_snapshot": str(snapshot), "foundation_sha256": cem.sha(foundation),
             "foundation_snapshot_sha256": manifest["snapshot_sha256"], "search_id": "fixture-cem", "seed": 7,
             "mean": IDENTITY.tolist(), "std": [.01] * 5, "generations": []}
    (output / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(cem, "foundation_ready", lambda state: (True, "fixture_only", []))
    monkeypatch.setattr(cem, "queue_budget", lambda state, jobs: (True, {}))
    original = cem.freeze_candidate
    def interrupted(state, parameters, generation, index):
        if index == 1:
            raise RuntimeError("simulated interruption after candidate0 was frozen")
        return original(state, parameters, generation, index)
    monkeypatch.setattr(cem, "freeze_candidate", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        cem.step(state, enqueue=True)
    assert len(Ledger(campaign / "ledger.jsonl").records()) == 1
    recovered = json.loads((output / "state.json").read_text())
    monkeypatch.setattr(cem, "freeze_candidate", original)
    recovered = cem.step(recovered, enqueue=True)
    assert len(recovered["generations"][0]["candidates"]) == 6
    assert len(Ledger(campaign / "ledger.jsonl").records()) == 6
    assert len(json.loads((campaign / "queue.json").read_text())) == 30
