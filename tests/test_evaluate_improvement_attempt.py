import hashlib
import json

import pytest

from scripts.evaluate_improvement_attempt import evaluate_attempt
from scripts.improvement_ledger import OFFICIAL_DETECTOR_SHA256, freeze_snapshot


def dump(path, value):
    path.write_text(json.dumps(value))


def lines(path, values):
    path.write_text("".join(json.dumps(value)+"\n" for value in values))


def setup_attempt(tmp_path, *, hazard=False, forbidden_source=False, learned_models=False,
                  policy_parameters=None, new_imports=False, optional_remote=False):
    tmp_path.mkdir(parents=True,exist_ok=True)
    source = tmp_path / "racer.py"
    code = "driver.getFromDef('OPPONENT')\n" if forbidden_source else "driver.getSelf()\n"
    if new_imports:
        code = "from race_lidar.lidar_hazards import LidarHazardChannel\nfrom race_lidar import rich_driver as rich\n"+code
    source.write_text(code)
    reference = tmp_path / "reference.npz"; reference.write_bytes(b"frozen-reference-fixture")
    snapshot = tmp_path / "snapshot"
    sources = {"webots/controllers/tournament_racer/tournament_racer.py": source, "reference.npz": reference}
    for name in ("race_policy.py", "race_control.py", "hotlap_driver.py", "learned_driver.py"):
        stub = tmp_path/name; stub.write_text("# frozen runtime fixture\n")
        sources["src/race_lidar/"+name] = stub
    extras = (["lidar_hazards.py", "rich_driver.py"] if new_imports else []) + (["remote.py"] if optional_remote else [])
    for name in extras:
        stub = tmp_path/name; stub.write_text("# frozen optional runtime module\n")
        sources["src/race_lidar/"+name] = stub
    if learned_models:
        for name in ("a", "b"):
            model = tmp_path/f"{name}.npz"; model.write_bytes(f"model-{name}".encode())
            sources[f"models/{name}.npz"] = model
        detector = tmp_path/"detector.pth"; detector.write_bytes(b"adapted-detector")
        sources["models/detector.pth"] = detector
    manifest = freeze_snapshot(snapshot, sources,metadata={"policy_parameters":policy_parameters} if policy_parameters is not None else {})
    run = tmp_path / "attempt"; car = run / "cars/EGO"; car.mkdir(parents=True)
    rules = {"reference_length_m": 1000., "laps": 1, "sensor_period_ms": 200,
             "physics_step_ms": 20, "max_speed_mps": 60., "lateral_accel_budget_mps2": 8.,
             "friction_coefficient": 1.5, "vehicle_model": "physical Car",
             "brake_decel_mps2": 8., "acceleration_mps2": 5.}
    config = {"reference_path": str(snapshot / "reference.npz"), "reference_length_m": 1000.,
              "laps": 1, "start_time_s": 0., "rules": rules,
              "entrants": [{"car_id": "EGO", "policy": "reference", "grid_index": 0}]}
    if policy_parameters is not None:
        config["policy_parameters"] = policy_parameters
        config["rules"].update(policy_parameters,control_version="predictive_v2")
    result = {"car_id": "EGO", "policy": "reference", "status": "completed", "completed": True, "valid_lap": True,
              "progress_m": 1001., "finish_time_s": 50.05, "simulation_time_s": 50.1,
              "collision_samples": None, "track_violation_steps": 0, "position_resets_after_spawn": 0,
              "opponent_ground_truth_used_for_control": False, "perception_enabled": True,
              "sensor_period_ms": 200, "physics_step_ms": 20, "neural_driver_used": False,
              "physical_limits": {"max_speed_mps": 60., "lateral_accel_mps2": 8.,
                                  "brake_decel_mps2": 8., "acceleration_mps2": 5.},
              "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "sensor_frames": 1, "policy_decisions": 1}
    result["runtime_detector_checkpoint_sha256"] = OFFICIAL_DETECTOR_SHA256
    if policy_parameters is not None:
        result["runtime_policy_parameters"] = policy_parameters
    result["runtime_source_root"] = str(snapshot)
    result["runtime_source_sha256"] = {item["path"]: item["sha256"] for item in manifest["files"] if item["path"].endswith(".py")}
    trace = [{"simulation_time_s": 50., "progress_m": 999., "position_m": [999., 0., .35]},
             {"simulation_time_s": 50.1, "progress_m": 1001., "position_m": [1001., 0., .35]}]
    frame_time = 49.8
    if hazard:
        config["hazard"] = {"ego_car_id": "EGO", "progress_m": 200.}
        result.update(status="incomplete", completed=False, valid_lap=False, finish_time_s=None,
                      progress_m=230., simulation_time_s=10.)
        trace = [{"simulation_time_s": 9., "progress_m": 200., "position_m": [200., 0., .35]},
                 {"simulation_time_s": 10., "progress_m": 230., "position_m": [230., 0., .35]}]
        frame_time = 9.8
    aggregate = {"status": "time_limit" if hazard else "all_cars_reported",
                 "cars": [{"car_id": "EGO", "progress_m": result["progress_m"]}], "collisions_by_car": {"EGO": 0}}
    dump(run / "config.json", config); dump(run / "heat_results.json", aggregate)
    dump(car / "result.json", result); dump(car / "controller_summary.json", {"closed": True})
    lines(car / "control_trace.jsonl", trace)
    lines(car / "predictions.jsonl", [{"car_id": "EGO", "frame_id": "0", "simulation_time_s": frame_time,
                                       "points": 32768, "boxes": []}])
    lines(car / "policy_actions.jsonl", [{"car_id": "EGO", "vision_frame_id": "0", "simulation_time_s": frame_time}])
    return run, snapshot, manifest


def test_full_lap_outcome_requires_independent_snapshot_collision_and_crossing_evidence(tmp_path):
    run, snapshot, manifest = setup_attempt(tmp_path)
    outcome = evaluate_attempt(run, snapshot, manifest["snapshot_sha256"])
    assert outcome["audit_verified"] and outcome["snapshot_verified"]
    assert outcome["audit_issues"] == []
    assert outcome["entrants"][0]["valid_finish"]
    assert outcome["entrants"][0]["clean_finish"]
    assert outcome["control_source_review"]["verified"]
    assert outcome["hazard_pass"] is None
    assert outcome["homogeneous_field"]
    assert outcome["observed_candidate_descriptors"] == [{"policy_key": "reference", "model_sha256": None,
                                                          "detector_sha256": OFFICIAL_DETECTOR_SHA256}]


def test_hazard_pass_is_progress_plus_clean_audit_and_does_not_fabricate_lap(tmp_path):
    run, snapshot, _ = setup_attempt(tmp_path, hazard=True)
    outcome = evaluate_attempt(run, snapshot)
    assert outcome["audit_verified"] and outcome["hazard_pass"]
    assert not outcome["entrants"][0]["valid_finish"]
    assert not outcome["entrants"][0]["clean_finish"]
    assert outcome["hazard_evidence"]["required_ego_progress_m"] == 220
    result_path = run / "cars/EGO/result.json"
    result = json.loads(result_path.read_text()); result["progress_m"] = 219
    dump(result_path, result)
    outcome = evaluate_attempt(run, snapshot)
    assert outcome["audit_verified"]  # An unsuccessful objective is still an auditable result.
    assert not outcome["hazard_pass"]


@pytest.mark.parametrize("defect", ["missing_collision", "wrong_snapshot", "clock", "future_vision", "ground_truth", "not_closed", "live_source", "source_changed"])
def test_unverified_or_contradictory_artifacts_block_outcome_verification(tmp_path, defect):
    run, snapshot, manifest = setup_attempt(tmp_path)
    expected = manifest["snapshot_sha256"]
    if defect == "missing_collision":
        value = json.loads((run / "heat_results.json").read_text()); value["collisions_by_car"] = {}
        dump(run / "heat_results.json", value)
    elif defect == "wrong_snapshot":expected = "another-snapshot"
    elif defect == "future_vision":
        lines(run / "cars/EGO/predictions.jsonl", [{"car_id": "EGO", "frame_id": "0", "simulation_time_s": 49.9}])
    elif defect == "not_closed":(run / "cars/EGO/controller_summary.json").unlink()
    else:
        path = run / "cars/EGO/result.json"; value = json.loads(path.read_text())
        if defect == "clock":value["finish_time_s"] = 53.
        elif defect == "ground_truth":value["opponent_ground_truth_used_for_control"] = True
        elif defect == "live_source":value["runtime_source_root"] = str(tmp_path/"live-project")
        elif defect == "source_changed":value["runtime_source_sha256"]["src/race_lidar/race_control.py"] = "different-code"
        dump(path, value)
    assert not evaluate_attempt(run, snapshot, expected)["audit_verified"]


def test_static_source_review_detects_opponent_node_lookup_in_frozen_controller(tmp_path):
    run, snapshot, _ = setup_attempt(tmp_path, forbidden_source=True)
    outcome = evaluate_attempt(run, snapshot)
    assert not outcome["audit_verified"]
    assert outcome["snapshot_verified"]  # Valid content hashes do not make cheating control legitimate.
    assert outcome["control_source_review"]["forbidden_control_calls"]


def test_hazard_overlap_is_recorded_failure_even_with_enough_progress(tmp_path):
    run, snapshot, _ = setup_attempt(tmp_path, hazard=True)
    value = json.loads((run / "heat_results.json").read_text()); value["collisions_by_car"]["EGO"] = 5
    dump(run / "heat_results.json", value)
    outcome = evaluate_attempt(run, snapshot)
    assert outcome["audit_verified"]
    assert not outcome["hazard_pass"]


def test_wrong_selected_model_is_rejected_even_if_its_hash_is_valid_in_the_bundle(tmp_path):
    run, snapshot, _ = setup_attempt(tmp_path, learned_models=True)
    config = json.loads((run/"config.json").read_text())
    config["entrants"][0].update(policy="neural_a", model_path=str(snapshot/"models/b.npz"))
    dump(run/"config.json", config)
    result = json.loads((run/"cars/EGO/result.json").read_text())
    result.update(policy="neural_a", neural_driver_used=True,
                  model_sha256=hashlib.sha256((snapshot/"models/b.npz").read_bytes()).hexdigest())
    dump(run/"cars/EGO/result.json", result)
    assert evaluate_attempt(run, snapshot)["audit_verified"]  # Valid frozen bundle, wrong registered selection.
    expected_hash = hashlib.sha256((snapshot/"models/a.npz").read_bytes()).hexdigest()
    outcome = evaluate_attempt(run, snapshot, expected_policy="neural_a", expected_model_sha256=expected_hash)
    assert not outcome["audit_verified"]
    assert "EGO:checkpoint_not_expected_candidate" in outcome["audit_issues"]
    assert outcome["snapshot_verified"]


def test_runtime_detector_health_hash_must_match_selected_frozen_adaptation(tmp_path):
    run, snapshot, _ = setup_attempt(tmp_path, learned_models=True)
    adapted = hashlib.sha256((snapshot/"models/detector.pth").read_bytes()).hexdigest()
    config = json.loads((run/"config.json").read_text())
    config["expected_detector_checkpoint_sha256"] = adapted
    dump(run/"config.json", config)
    outcome = evaluate_attempt(run, snapshot, expected_policy="reference")
    assert not outcome["audit_verified"]
    assert "EGO:runtime_detector_not_expected_checkpoint" in outcome["audit_issues"]
    result = json.loads((run/"cars/EGO/result.json").read_text())
    result["runtime_detector_checkpoint_sha256"] = adapted
    dump(run/"cars/EGO/result.json", result)
    assert evaluate_attempt(run, snapshot, expected_policy="reference")["audit_verified"]


def test_missing_detector_binding_is_allowed_only_without_formal_expectation(tmp_path):
    run, snapshot, _ = setup_attempt(tmp_path)
    path = run/"cars/EGO/result.json"; result = json.loads(path.read_text())
    result.pop("runtime_detector_checkpoint_sha256"); dump(path, result)
    assert evaluate_attempt(run, snapshot)["audit_verified"]  # Legacy engineering data stays inspectable.
    outcome = evaluate_attempt(run, snapshot, expected_policy="reference")
    assert not outcome["audit_verified"]
    assert "EGO:runtime_detector_not_expected_checkpoint" in outcome["audit_issues"]


def test_policy_parameters_bind_actual_execution_without_changing_physical_environment_hash(tmp_path):
    run45,snapshot45,_ = setup_attempt(tmp_path/"45",policy_parameters={"visible_distance_m":45.})
    run70,snapshot70,_ = setup_attempt(tmp_path/"70",policy_parameters={"visible_distance_m":70.})
    first = evaluate_attempt(run45,snapshot45,expected_policy="reference")
    second = evaluate_attempt(run70,snapshot70,expected_policy="reference")
    assert first["audit_verified"] and second["audit_verified"]
    assert first["rules_sha256"] == second["rules_sha256"]
    assert first["raw_rules_sha256"] != second["raw_rules_sha256"]
    path = run70/"cars/EGO/result.json"; result = json.loads(path.read_text())
    result["runtime_policy_parameters"] = {"visible_distance_m":45.}; dump(path,result)
    rejected = evaluate_attempt(run70,snapshot70,expected_policy="reference")
    assert not rejected["audit_verified"]
    assert "EGO:runtime_control_parameters_mismatch" in rejected["audit_issues"]
    result.pop("runtime_policy_parameters"); dump(path,result)
    assert not evaluate_attempt(run70,snapshot70,expected_policy="reference")["audit_verified"]


@pytest.mark.parametrize("module",["lidar_hazards.py","rich_driver.py"])
def test_new_modules_are_required_when_imported_by_frozen_racer(tmp_path,module):
    run,snapshot,_ = setup_attempt(tmp_path,new_imports=True)
    verified = evaluate_attempt(run,snapshot)
    assert verified["audit_verified"]
    assert "src/race_lidar/"+module in verified["per_car_audits"][0]["required_runtime_sources"]
    path = run/"cars/EGO/result.json"; result = json.loads(path.read_text())
    result["runtime_source_sha256"].pop("src/race_lidar/"+module)
    dump(path,result)
    rejected = evaluate_attempt(run,snapshot)
    assert not rejected["audit_verified"]
    assert "EGO:runtime_source_checksum_mismatch:src/race_lidar/"+module in rejected["audit_issues"]


def test_every_reported_hash_is_verified_even_if_module_is_not_mandatory(tmp_path):
    run,snapshot,_ = setup_attempt(tmp_path,optional_remote=True)
    assert evaluate_attempt(run,snapshot)["audit_verified"]
    path = run/"cars/EGO/result.json"; result = json.loads(path.read_text())
    result["runtime_source_sha256"]["src/race_lidar/remote.py"] = "wrong-reported-hash"
    dump(path,result)
    assert not evaluate_attempt(run,snapshot)["audit_verified"]
    result["runtime_source_sha256"].pop("src/race_lidar/remote.py"); dump(path,result)
    assert evaluate_attempt(run,snapshot)["audit_verified"]  # Historic optional omission remains legitimate.
    result["runtime_source_sha256"]["src/race_lidar/not_in_bundle.py"] = "unregistered-source"
    dump(path,result)
    assert not evaluate_attempt(run,snapshot)["audit_verified"]
