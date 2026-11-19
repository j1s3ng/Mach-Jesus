import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from race_lidar.rich_driver import FEATURE_NAMES, RichDriver, lane_summaries, rich_feature_vector, save_rich_model
from scripts.train_rich_driver import lateral_rollout, trajectory_costs, trajectory_teacher


def state():
    return {"progress_fraction": .05, "body_speed_mps": 20., "heading_error_rad": 0.,
            "cross_track_m": 0., "curvature_ahead": [0.] * 7, "last_target_offset_m": 0.,
            "front_distance_m": 20., "front_lateral_m": 0., "front_speed_mps": 45.,
            "left_clearance_m": 10., "right_clearance_m": 10., "hazard_tracks": [
                {"relative_distance_m": 20., "lateral_m": 0., "speed_mps": 45., "speed_std_mps": 1.},
                {"relative_distance_m": 80., "lateral_m": 0., "speed_mps": 0., "speed_std_mps": 1.}]}


def test_critical_front_exposes_stopped_car_hidden_by_a_nearer_fast_car():
    nearest = lane_summaries(state(), mode="nearest")[1]
    critical = lane_summaries(state(), mode="stopping_critical")[1]
    assert nearest["front_gap_m"] == 20. and nearest["front_speed_mps"] == 45.
    assert critical["front_gap_m"] == 80. and critical["front_speed_mps"] == 0.


def test_nearest_aliases_two_scenes_that_require_different_trajectory_choices():
    blocked = state()
    blocked["hazard_tracks"].append({"relative_distance_m": 80., "lateral_m": -3.2,
                                     "speed_mps": 0., "speed_std_mps": 1.})
    clear = {**blocked, "hazard_tracks": [blocked["hazard_tracks"][0], blocked["hazard_tracks"][2]]}
    np.testing.assert_array_equal(rich_feature_vector(clear), rich_feature_vector(blocked))
    assert not np.array_equal(rich_feature_vector(clear, mode="stopping_critical"),
                              rich_feature_vector(blocked, mode="stopping_critical"))
    assert trajectory_teacher(clear)["target_offset_m"] == 0.
    assert trajectory_teacher(blocked)["target_offset_m"] == 3.2


def test_checkpoint_metadata_selects_summary_without_changing_old_default(tmp_path):
    weight = np.zeros((6, len(FEATURE_NAMES)), np.float32)
    bias = np.array([.3, .3, .3, 0., -1., -.4], np.float32)
    weight[5, 32] = 1.  # Left utility depends on the center lane's front gap.
    args = ([weight], [bias], np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
    old_path = save_rich_model(tmp_path / "nearest.npz", *args, {})
    new_path = save_rich_model(tmp_path / "critical.npz", *args, {"lane_summary_mode": "stopping_critical"})
    assert RichDriver.from_npz(old_path).predict(state())["target_offset_m"] == -3.2
    assert RichDriver.from_npz(new_path).predict(state())["target_offset_m"] == 3.2


def test_zero_speed_cannot_slide_sideways_and_moving_slew_is_delayed():
    times = np.linspace(0., 4.5, 19)
    velocity = np.array([np.zeros(len(times)), np.full(len(times), 10.)])
    path, _ = lateral_rollout(-3.2, [3.2, 3.2], velocity, times)
    np.testing.assert_array_equal(path[0], np.full(len(times), -3.2))
    assert path[1, 1] == -3.2  # Before the command delay.
    assert abs(path[1, -1] - .95) < 1e-6


def test_unreached_free_destination_does_not_erase_an_actual_blocked_endpoint():
    observed = state()
    observed.update(body_speed_mps=0., cross_track_m=-3.2, last_target_offset_m=-3.2,
        hazard_tracks=[{"relative_distance_m": 10., "lateral_m": -3.2, "speed_mps": 0., "speed_std_mps": 1.}])
    cost, _ = trajectory_costs(observed, [-3.2, 3.2], [0., 0.])
    assert cost[1] > cost[0]  # A stationary plan cannot claim a completed escape.


def test_actual_001_models_match_the_frozen_nearest_runtime_exactly():
    root = Path(__file__).resolve().parents[1]
    frozen = root / "runs/improvement-12h/rich-driver-001-source/src/race_lidar/rich_driver.py"
    observations = root / "runs/improvement-12h/rich-driver-001/training_observations.jsonl"
    if not frozen.exists() or not observations.exists():
        pytest.skip("Local frozen001 training artifacts are not installed")
    spec = importlib.util.spec_from_file_location("race_lidar._frozen_rich_001", frozen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states = [json.loads(line) for index, line in enumerate(observations.open()) if index < 1200 and index % 23 == 0]
    for seed in (7, 17):
        checkpoint = root / f"runs/improvement-12h/rich-driver-001/seed-{seed}/model.npz"
        old, current = module.RichDriver.from_npz(checkpoint), RichDriver.from_npz(checkpoint)
        for observed in states:
            assert current.predict(observed) == old.predict(observed)
