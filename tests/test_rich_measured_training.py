import json

import numpy as np
import pytest

from race_lidar.rich_driver import FEATURE_NAMES, rich_feature_vector
from scripts.rich_measured_data import ALLOWED_RUNS, load_run, reference_counterfactual, validate_run
from scripts.train_rich_dagger import actual_split, measured_label, renormalize_first_layer


def state():
    return {"progress_fraction": .05, "body_speed_mps": 20., "heading_error_rad": 0.,
            "cross_track_m": 0., "curvature_ahead": [0.] * 7, "last_target_offset_m": 0.,
            "front_distance_m": 60., "front_lateral_m": 0., "front_speed_mps": 20.,
            "left_clearance_m": 10., "right_clearance_m": 10.}


def test_loader_keeps_only_measured_feature_fields_and_the_separate_label(tmp_path):
    run = tmp_path / "engineering/rich001-field4-r0"
    folder = run / "cars/R01"
    folder.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"dataset_split": "engineering", "excluded_from_training": False}))
    row = {"inputs": {**state(), "reference_speed_mps": 999., "opponent_gt": [1, 2, 3]},
           "hazard_tracks": [{"relative_distance_m": 40., "lateral_m": 0., "speed_mps": 0.,
                              "ground_truth_label": "must not be copied"}],
           "feasible_speed_cap_mps": 35., "simulation_time_s": 3., "phase": "racing",
           "raw_action": {"target_speed_mps": 0., "target_offset_m": -3.2}, "leaderboard": "ignored"}
    (folder / "policy_actions.jsonl").write_text(json.dumps(row) + "\n")
    samples, _ = load_run(run)
    observed = samples[0]["state"]
    assert "reference_speed_mps" not in observed and "opponent_gt" not in observed
    assert "ground_truth_label" not in observed["hazard_tracks"][0]
    assert samples[0]["label_feasible_speed_mps"] == 35.
    assert samples[0]["split_role"] == "actual_fit"
    assert len(rich_feature_vector(observed)) == len(FEATURE_NAMES)


def test_only_the_two_engineering_runs_can_be_opened(tmp_path):
    for prefix in ["formal_cases", "bootstrap", "holdout", "confirmation"]:
        path = tmp_path / prefix / "engineering/rich001-field4-r0"
        with pytest.raises(ValueError, match="sealed"):
            validate_run(path)
    with pytest.raises(ValueError, match="authorized"):
        validate_run(tmp_path / "engineering/new-unknown-run")
    assert ALLOWED_RUNS["rich001-field20-r0"] == "actual_run_holdout"


@pytest.mark.parametrize("metadata", [{"excluded_from_training": True}, {"dataset_split": "cv_qualification"},
                                      {"dataset_split": "confirmation"}, {"dataset_split": "bootstrap"}])
def test_authorized_name_does_not_override_config_exclusion(tmp_path, metadata):
    run = tmp_path / "engineering/rich001-field4-r0"
    folder = run / "cars/R01"
    folder.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(metadata))
    # If eligibility is checked too late, JSON parsing this file would fail.
    (folder / "policy_actions.jsonl").write_text("must never be opened")
    with pytest.raises(ValueError, match="excluded"):
        load_run(run)


def test_expert_cap_changes_only_the_label_not_the_student_input():
    observed = {**state(), "hazard_tracks": []}
    before = rich_feature_vector(observed)
    assert reference_counterfactual(observed, 15.)["target_speed_mps"] == 15.
    assert reference_counterfactual(observed, 45.)["target_speed_mps"] == 45.
    np.testing.assert_array_equal(before, rich_feature_vector(observed))


def test_ambiguous_partial_reference_lane_is_not_a_hard_student_lane_label():
    observed = {**state(), "last_target_offset_m": 1.1,
                "hazard_tracks": [{"relative_distance_m": 8., "lateral_m": 3.2, "speed_mps": 10.}]}
    action, _, _, unambiguous = measured_label(observed, 35.)
    assert action["target_offset_m"] == 1.1
    assert not unambiguous


def test_temporal_validation_has_a_gap_from_actual_fit_windows():
    assert actual_split({"simulation_time_s": 35.}) == "actual_train"
    assert actual_split({"simulation_time_s": 39.}) == "actual_gap"
    assert actual_split({"simulation_time_s": 45.}) == "actual_validation"
    assert actual_split({"simulation_time_s": 51.}) == "actual_gap"
    assert actual_split({"simulation_time_s": 53.}) == "actual_train"


def test_warm_start_reparameterization_preserves_the_old_function():
    rng = np.random.default_rng(4)
    weight, bias = rng.normal(size=(5, 48)).astype(np.float32), rng.normal(size=5).astype(np.float32)
    old_mean, new_mean = rng.normal(size=(2, 48)).astype(np.float32)
    old_scale, new_scale = rng.uniform(.2, 2., size=(2, 48)).astype(np.float32)
    x = rng.normal(size=(12, 48)).astype(np.float32)
    weights, biases = renormalize_first_layer([weight], [bias], old_mean, old_scale, new_mean, new_scale)
    before = ((x - old_mean) / old_scale) @ weight.T + bias
    after = ((x - new_mean) / new_scale) @ weights[0].T + biases[0]
    np.testing.assert_allclose(before, after, atol=1e-5, rtol=1e-5)
