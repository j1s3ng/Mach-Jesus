import copy
import json
import math

import numpy as np
import pytest

from race_lidar.rich_driver import save_rich_model
from scripts.cem_evidence import closed_issues, config_issues, expected_case, fresh_audit
from scripts.improvement_ledger import freeze_snapshot


def fixture(tmp_path, *, scenario="stopped"):
    reference = tmp_path / "reference.npz"
    angle = np.linspace(0., 2 * np.pi, 60, endpoint=False)
    np.savez(reference, x_m=100. * np.cos(angle), y_m=100. * np.sin(angle), speed_mps=np.full(60, 30.))
    model = save_rich_model(tmp_path / "model.npz", [np.zeros((6, 48), np.float32)], [np.zeros(6)], np.zeros(48), np.ones(48), {})
    snapshot = tmp_path / "snapshot"
    manifest = freeze_snapshot(snapshot, {"reference.npz": reference, "models/fixture.npz": model},
        metadata={"policies": [{"policy": "neural_fixture", "model_relpath": "models/fixture.npz"}],
                  "policy_parameters": {"visible_distance_m": 70.}})
    run = tmp_path / "engineering/testcase"; run.mkdir(parents=True)
    size = 3 if scenario == "stopped" else 4
    job = {"job_id": "testcase", "output": str(run), "args": ["--source-root", str(snapshot), "--reference", str(snapshot / "reference.npz"),
        "--single-policy", "neural_fixture", "--scenario", scenario, "--cars", str(size), "--seed", "101", "--seconds", "120",
        "--obstacle-progress", "400", "--ego-progress", "300", "--blocked-side", "right", "--grid-jitter", ".3", "--spacing", "14"]}
    model_sha = next(row["sha256"] for row in manifest["files"] if row["path"] == "models/fixture.npz")
    expected = expected_case(job, snapshot_path=snapshot, snapshot_sha256=manifest["snapshot_sha256"], policy="neural_fixture", model_sha256=model_sha)
    entries = [{key: value for key, value in row.items() if not key.startswith("expected_")} for row in expected["entrants"]]
    config = {"dataset_split": "development", "excluded_from_training": False, "output_dir": str(run),
        "scenario": scenario, "seed": 101, "grid_rotation": 0, "reference_length_m": expected["rules"]["reference_length_m"],
        "laps": 1, "start_time_s": 1., "max_sim_seconds": 121., "source_root": str(snapshot),
        "reference_path": str(snapshot / "reference.npz"), "snapshot_sha256": manifest["snapshot_sha256"],
        "policy_parameters": {"visible_distance_m": 70.}, "mode": "capacity" if scenario == "stopped" else "race",
        "rules": expected["rules"], "entrants": entries, "cars": {row["car_id"]: row for row in entries}}
    if scenario == "stopped":
        config["hazard"] = {"ego_car_id": "R01", "progress_m": 400.}
    (run / "config.json").write_text(json.dumps(config))
    heat = {"status": "time_limit", "mode": config["mode"], "rules": config["rules"], "car_count": size, "cars": entries,
            "collisions_by_car": {entry["car_id"]: 0 for entry in entries}}
    (run / "heat_results.json").write_text(json.dumps(heat))
    poses = []
    for entry in expected["entrants"]:
        folder = run / "cars" / entry["car_id"]; folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(entry))
        if not entry["is_static"]:
            for name in ("controller_summary.json", "control_trace.jsonl", "policy_actions.jsonl", "predictions.jsonl"):
                (folder / name).write_text("{}\n")
        yaw = entry["expected_initial_yaw_rad"]
        poses.append({"car_id": entry["car_id"], "position_m": entry["expected_initial_xy_m"] + [.3],
                      "rotation": [math.cos(yaw), -math.sin(yaw), 0., math.sin(yaw), math.cos(yaw), 0., 0., 0., 1.]})
    (run / "gt_poses.jsonl").write_text(json.dumps({"timestamp_ns": 200000000, "cars": poses}) + "\n")
    (run / "ledger_outcome.json").write_text(json.dumps({"audit_verified": True}))
    return run, config, expected


@pytest.mark.parametrize("key,value", [("seed", 102), ("scenario", "race"), ("laps", 2),
                                      ("max_sim_seconds", 481.), ("reference_path", "/different/reference.npz")])
def test_same_model_and_caps_do_not_allow_a_different_world(tmp_path, key, value):
    _, config, expected = fixture(tmp_path)
    assert config_issues(config, expected) == []
    altered = copy.deepcopy(config); altered[key] = value
    assert "expected_case_config_mismatch:" + key in config_issues(altered, expected)


def test_exact_seeded_grid_and_entrant_count_are_bound(tmp_path):
    _, config, expected = fixture(tmp_path, scenario="race")
    rng = np.random.default_rng(101)
    assert expected["entrants"][0]["start_progress_m"] == -float(rng.uniform(0., .3))
    assert expected["entrants"][0]["start_lane_m"] == -2.5 + float(rng.uniform(-.3, .3))
    config["entrants"][1]["start_lane_m"] += .2
    assert any("spawn_or_policy_mismatch" in issue for issue in config_issues(config, expected))
    config["entrants"].pop()
    assert "expected_case_entrant_ids_mismatch" in config_issues(config, expected)


def test_changed_spacing_cutoff_and_hazard_progress_are_rejected(tmp_path):
    _, config, expected = fixture(tmp_path)
    config["rules"]["grid_spacing_m"] = 12.
    # Detach expected so this is a real config-only mutation.
    expected = copy.deepcopy(expected); expected["rules"]["grid_spacing_m"] = 14.
    assert "expected_case_rules_mismatch" in config_issues(config, expected)
    config["hazard"]["progress_m"] = 450.
    assert "expected_hazard_identity_mismatch" in config_issues(config, expected)


def test_closed_heat_and_actual_initial_poses_must_match(tmp_path):
    run, config, expected = fixture(tmp_path)
    assert closed_issues(run, config, expected) == []
    path = run / "gt_poses.jsonl"; poses = json.loads(path.read_text())
    poses["cars"][1]["position_m"][0] += 1.
    path.write_text(json.dumps(poses))
    assert "initial_physical_spawn_mismatch:R02" in closed_issues(run, config, expected)
    (run / "cars/R01/controller_summary.json").unlink()
    assert "controller_not_closed:R01" in closed_issues(run, config, expected)


def test_reused_heat_with_different_actual_entrants_cannot_be_scored(tmp_path):
    run, config, expected = fixture(tmp_path)
    path = run / "heat_results.json"; heat = json.loads(path.read_text()); heat["cars"].pop()
    path.write_text(json.dumps(heat))
    assert "closed_heat_entrant_ids_mismatch" in closed_issues(run, config, expected)


def test_fresh_cpu_audit_is_cached_and_any_closed_log_change_invalidates_it(tmp_path, monkeypatch):
    import scripts.evaluate_improvement_attempt as evaluator
    run, _, expected = fixture(tmp_path)
    calls = []
    def cpu_audit(*args, **kwargs):
        calls.append(kwargs)
        return {"audit_verified": True, "entrants": [{"clean_finish": True, "valid_finish": True}]}
    monkeypatch.setattr(evaluator, "evaluate_attempt", cpu_audit)
    first, check1 = fresh_audit(run, expected, tmp_path / "cache")
    second, check2 = fresh_audit(run, expected, tmp_path / "cache")
    assert first == second and not check1["cached"] and check2["cached"] and len(calls) == 1
    with (run / "cars/R01/control_trace.jsonl").open("a") as stream:
        stream.write("{}\n")
    _, check3 = fresh_audit(run, expected, tmp_path / "cache")
    assert not check3["cached"] and len(calls) == 2
    assert calls[-1]["expected_rules_sha256"] == expected["rules_sha256"]


def test_forged_cached_audit_hash_cannot_skip_a_fresh_audit(tmp_path, monkeypatch):
    import scripts.evaluate_improvement_attempt as evaluator
    run, _, expected = fixture(tmp_path); calls = []
    monkeypatch.setattr(evaluator, "evaluate_attempt", lambda *args, **kwargs: calls.append(1) or {"audit_verified": True, "entrants": []})
    _, check = fresh_audit(run, expected, tmp_path / "cache")
    cache = json.loads(open(check["cache_path"]).read()); cache["audit"]["audit_verified"] = False
    from pathlib import Path
    Path(check["cache_path"]).write_text(json.dumps(cache))
    fresh_audit(run, expected, tmp_path / "cache")
    assert len(calls) == 2
