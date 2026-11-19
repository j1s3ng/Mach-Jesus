import copy

import numpy as np

from scripts.rich006_teacher import BASE, mirror, perturb_measured, teacher


def state():
    return {"progress_fraction": .5, "body_speed_mps": 13., "heading_error_rad": .03,
        "cross_track_m": -.9, "curvature_ahead": [0.] * 7, "last_target_offset_m": -1.6,
        "front_distance_m": 60., "front_lateral_m": 0., "front_speed_mps": 13.,
        "left_clearance_m": 10., "right_clearance_m": 10., "hazard_tracks": []}


def test_partial_pass_intent_survives_missing_returns_in_both_directions():
    observed = state()
    assert BASE.intent_teacher(observed)["target_offset_m"] == 0.
    assert teacher(observed)["target_offset_m"] == -3.2
    assert teacher(mirror(observed))["target_offset_m"] == 3.2


def test_completed_clear_lane_and_neutral_command_are_not_held_forever():
    assert teacher({**state(), "cross_track_m": -3.1})["target_offset_m"] == 0.
    assert teacher({**state(), "last_target_offset_m": 0.})["target_offset_m"] == 0.


def test_nonempty_observations_keep_frozen005_safety_costs():
    observed = state(); observed["hazard_tracks"] = [{"relative_distance_m": 40., "lateral_m": 0., "speed_mps": 0.,
        "position_std_m": .25, "speed_std_mps": 2., "velocity_valid": True}]
    np.testing.assert_array_equal(teacher(observed)["lane_costs"], BASE.intent_teacher(observed)["lane_costs"])


def test_mirror_is_reversible_and_perturbation_never_mutates_source():
    observed = state(); observed["hazard_tracks"] = [{"relative_distance_m": 40., "lateral_m": 0., "speed_mps": 0.,
        "position_std_m": .25, "speed_std_mps": 2., "velocity_valid": True, "lateral_min_m": -1., "lateral_max_m": 1.}]
    before = copy.deepcopy(observed)
    assert mirror(mirror(observed)) == observed
    changed = perturb_measured(observed, np.random.default_rng(5))
    assert observed == before and len(changed["hazard_tracks"]) > len(observed["hazard_tracks"])
    assert all(row["velocity_valid"] is False for row in changed["hazard_tracks"][1:])
