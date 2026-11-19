import copy

import numpy as np

from scripts.train_rich_driver import intent_teacher, trajectory_teacher


def state():
    return {"progress_fraction": .05, "body_speed_mps": 27., "heading_error_rad": 0.,
            "cross_track_m": .01, "curvature_ahead": [0.] * 7, "last_target_offset_m": .2,
            "front_distance_m": 90., "front_lateral_m": 0., "front_speed_mps": 0.,
            "left_clearance_m": 10., "right_clearance_m": 10.,
            "hazard_tracks": [{"relative_distance_m": 90., "lateral_m": 0., "speed_mps": 0.,
                               "position_std_m": .25, "speed_std_mps": 45., "velocity_valid": False}]}


def test_emerging_point_two_metre_slew_retains_both_possible_pass_intents():
    positive = state()
    negative = {**state(), "last_target_offset_m": -.2}
    assert intent_teacher(positive)["target_offset_m"] == 3.2
    assert intent_teacher(negative)["target_offset_m"] == -3.2


def test_fresh_aged_covariance_cycle_does_not_erase_passing_intent():
    for uncertainty in (.25, 1.5, 2.6, 9.13):
        observed = state()
        observed["hazard_tracks"][0]["position_std_m"] = uncertainty
        assert intent_teacher(observed)["target_offset_m"] == 3.2


def test_default_side_is_consistent_under_tiny_midline_noise():
    for lateral in (-.02, -.001, .001, .02):
        observed = {**state(), "cross_track_m": lateral, "last_target_offset_m": 0.}
        assert intent_teacher(observed)["target_offset_m"] == 3.2


def test_a_clearly_blocked_intended_side_overrides_the_tie_regularizer():
    observed = state()
    observed["hazard_tracks"][0].update(velocity_valid=True, speed_std_mps=1.)
    observed["hazard_tracks"].append({"relative_distance_m": 75., "lateral_m": 3.2, "speed_mps": 0.,
                                      "position_std_m": .25, "speed_std_mps": 1., "velocity_valid": True})
    teacher = intent_teacher(observed)
    assert teacher["target_offset_m"] == -3.2
    assert not teacher["intent_regularization_applied"]


def test_clear_track_releases_intent_and_no_hidden_fields_change_labels():
    observed = {**state(), "hazard_tracks": []}
    before = copy.deepcopy(observed)
    assert intent_teacher(observed)["target_offset_m"] == 0.
    observed.update(opponent_gt=["ignored"], reference_speed_mps=999.)
    np.testing.assert_array_equal(intent_teacher(before)["lane_costs"], intent_teacher(observed)["lane_costs"])


def test_frozen_teacher_is_unchanged_and_005_reports_its_cost_adjustment():
    observed = state()
    old = trajectory_teacher(observed)
    new = intent_teacher(observed)
    np.testing.assert_array_equal(old["lane_costs"], new["raw_lane_costs"])
    assert new["lane_costs"][new["lane_index"]] == 0.
