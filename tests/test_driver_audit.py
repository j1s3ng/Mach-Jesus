import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from race_lidar.hotlap_driver import RacingReference
from race_lidar.learned_driver import FEATURE_NAMES, LearnedDriver, save_numpy_model
from race_lidar.race_policy import RacePolicy, TournamentTraffic, policy_state


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_tournament_drivers.py"
spec = importlib.util.spec_from_file_location("driver_auditor", SCRIPT)
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def recorded_heat(tmp_path):
    reference_path = tmp_path / "reference.npz"
    angle = np.linspace(0., 2. * np.pi, 200, endpoint=False)
    np.savez(reference_path, x_m=100. * np.cos(angle), y_m=100. * np.sin(angle),
             speed_mps=np.full(len(angle), 40.))
    reference = RacingReference.from_npz(reference_path)
    model_path = tmp_path / "model.npz"
    save_numpy_model(model_path, [np.zeros((2, len(FEATURE_NAMES)))], [np.array([.5, 0.])],
                     np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), {})
    model = LearnedDriver.from_npz(model_path)
    heat = tmp_path / "heat-1"
    folder = heat / "cars/R01"
    folder.mkdir(parents=True)
    entry = {"car_id": "R01", "policy": "neural_v3", "model_path": str(model_path)}
    config = {"reference_path": str(reference_path), "entrants": [entry], "perception_enabled": True,
              "output_dir": str(heat), "physics_threads": 8}
    (heat / "config.json").write_text(json.dumps(config))
    (heat / "heat_results.json").write_text("{}")
    (folder / "controller_summary.json").write_text("{}")
    (folder / "result.json").write_text(json.dumps({"status": "completed", "model_sha256": model.checkpoint_sha256}))
    (tmp_path / "tournament.json").write_text(json.dumps({"heats": [{"heat_id": "heat-1", "path": "heat-1"}]}))
    traffic = TournamentTraffic(reference)
    policy = RacePolicy(reference, model=model, name="neural_v3")
    traces, predictions, actions = [], [], []
    for index in range(4):
        now = 1. + .2 * index
        progress = 2. * index
        position, yaw, _, _ = reference.sample(progress)
        position = [*position, .3]
        boxes = [{"center": [20., 0., .6], "size": [5., 2.1, 1.1], "score": .9}]
        traffic.update(boxes, position, yaw, progress, now, ego_speed=10.)
        observed = traffic.observation(progress, 10., now)
        state = policy_state(reference, progress, 0., yaw, 10., 0., observed)
        decision = policy.predict(state, observed, 35.)
        traces.append({"simulation_time_s": now, "time_s": now - 1., "position_m": position,
                       "yaw_rad": yaw, "progress_m": progress, "speed_mps": 10.,
                       "cross_track_error_m": 0., "target_lane_offset_m": 0.})
        predictions.append({"simulation_time_s": now, "boxes": boxes, "inference_ms": 20.})
        actions.append({"simulation_time_s": now, "inputs": state, **decision,
                        "confirmed_tracks": observed["confirmed_tracks"], "feasible_speed_cap_mps": 35.})
    write_rows(folder / "control_trace.jsonl", traces)
    write_rows(folder / "predictions.jsonl", predictions)
    write_rows(folder / "policy_actions.jsonl", actions)
    return tmp_path, folder, actions


def test_audit_reconstructs_closed_perception_and_driver_stream(recorded_heat):
    root, _, _ = recorded_heat
    audit = auditor.audit_directory(root)
    assert audit["final"] and audit["audit_passed"]
    assert audit["totals"]["measurement_frames_replayed"] == 4
    assert audit["totals"]["neural_actions_replayed"] == 4
    assert audit["totals"]["raw_action_mismatches"] == 0


def test_audit_detects_tampered_perception_feature(recorded_heat):
    root, folder, actions = recorded_heat
    actions[2]["inputs"]["front_distance_m"] += 1.
    write_rows(folder / "policy_actions.jsonl", actions)
    audit = auditor.audit_directory(root)
    assert not audit["audit_passed"]
    assert audit["totals"]["traffic_feature_mismatches"] == 1


def test_audit_detects_tampered_neural_output_and_counterfactual(recorded_heat):
    root, folder, actions = recorded_heat
    actions[0]["raw_action"]["target_speed_mps"] += 2.
    actions[1]["no_traffic_counterfactual"]["target_offset_m"] += 1.
    write_rows(folder / "policy_actions.jsonl", actions)
    audit = auditor.audit_directory(root)
    assert not audit["audit_passed"]
    assert audit["totals"]["raw_action_mismatches"] == 1
    assert audit["totals"]["counterfactual_mismatches"] == 1


def test_audit_refuses_unclosed_tournament_and_wrong_checkpoint(recorded_heat):
    root, folder, _ = recorded_heat
    (folder / "result.json").write_text(json.dumps({"status": "completed", "model_sha256": "wrong"}))
    assert not auditor.audit_directory(root)["audit_passed"]
    (folder / "controller_summary.json").unlink()
    with pytest.raises(ValueError, match="not closed"):
        auditor.audit_directory(root)
