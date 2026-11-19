import json

import numpy as np
import pytest

from race_lidar.learned_driver import feature_vector
from race_lidar.rich_driver import (FEATURE_NAMES, LANE_OFFSETS_M, MODEL_FORMAT, RichDriver,
                                    lane_summaries, rich_feature_vector, save_rich_model)
from scripts.train_rich_driver import nominal_pace, trajectory_teacher


def state(hazards=()):
    return {"progress_fraction": .05, "body_speed_mps": 20., "heading_error_rad": 0.,
            "cross_track_m": 0., "curvature_ahead": [0.] * 7, "last_target_offset_m": 0.,
            "front_distance_m": 60., "front_lateral_m": 0., "front_speed_mps": 20.,
            "left_clearance_m": 10., "right_clearance_m": 10., "hazard_tracks": list(hazards)}


def hazard(distance, lane, speed=0., **extra):
    return {"relative_distance_m": distance, "lateral_m": lane, "speed_mps": speed,
            "position_std_m": .35, "speed_std_mps": 1., "velocity_valid": True, **extra}


def test_far_lane_occupancy_is_visible_when_the_original_features_are_identical():
    clear = state([hazard(100., 0.)])
    blocked = state([hazard(100., 0.), hazard(100., -3.2)])
    np.testing.assert_array_equal(feature_vector(clear), feature_vector(blocked))
    old, new = rich_feature_vector(clear), rich_feature_vector(blocked)
    assert len(new) == len(FEATURE_NAMES) == 48
    np.testing.assert_array_equal(new[:24], feature_vector(blocked))
    assert not np.array_equal(old[24:], new[24:])
    lanes = lane_summaries(blocked)
    assert [row["front_gap_m"] for row in lanes] == [100., 100., 120.]


def test_signed_oncoming_motion_and_rear_closing_speed_remain_distinct():
    observed = state([hazard(70., 3.2, -15.), hazard(-25., 3.2, 45.)])
    lane = lane_summaries(observed)[2]
    assert lane["front_speed_mps"] == -15.
    assert lane["rear_gap_m"] == 25. and lane["rear_closing_speed_mps"] == 25.
    assert lane["front_speed_std_mps"] == 1.


def test_only_measured_hazard_contract_is_consumed_and_source_pace_is_ignored():
    observed = state([hazard(80., 0.)])
    original = rich_feature_vector(observed)
    changed = {**observed, "reference_speed_mps": 9999., "gt_boxes": [hazard(1., 3.2)],
               "opponent_ground_truth": [hazard(1., -3.2)]}
    np.testing.assert_array_equal(original, rich_feature_vector(changed))
    assert nominal_pace(observed) == nominal_pace(changed)
    absent = dict(observed)
    absent.pop("hazard_tracks")
    with pytest.raises(ValueError, match="hazard_tracks"):
        rich_feature_vector(absent)


def test_unknown_motion_and_sparse_geometry_do_not_claim_a_narrow_clear_lane():
    observed = state([hazard(40., 1.8, 60., velocity_valid=False, lateral_min_m=1.7, lateral_max_m=1.9)])
    center = lane_summaries(observed)[1]
    assert center["front_gap_m"] == 40.
    assert center["front_speed_mps"] == 0. and center["front_speed_std_mps"] >= 15.


def test_portable_checkpoint_decodes_a_lane_without_averaging_into_the_center(tmp_path):
    weight = np.zeros((6, len(FEATURE_NAMES)), np.float32)
    bias = np.asarray([.2, .3, .4, 0., -2., 3.], np.float32)
    path = save_rich_model(tmp_path / "rich.npz", [weight], [bias],
                           np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), {"trained": True})
    model = RichDriver.from_npz(path)
    predicted = model.predict(state())
    assert predicted["target_offset_m"] == 3.2
    assert abs(predicted["target_speed_mps"] - 24.) < 1e-5
    assert abs(sum(predicted["lane_probabilities"]) - 1.) < 1e-6
    assert model.metadata["format"] == MODEL_FORMAT
    assert not model.metadata["reference_speed_is_model_input"]
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        assert metadata["lane_offsets_m"] == list(LANE_OFFSETS_M)


@pytest.mark.parametrize("side", [-1., 1.])
@pytest.mark.parametrize("already_committed", [False, True])
def test_trajectory_teacher_escapes_the_reproduced_far_double_block(side, already_committed):
    observed = state([hazard(100., 0.), hazard(100., side * 3.2)])
    if already_committed:
        observed.update(cross_track_m=side * 3.2, last_target_offset_m=side * 3.2)
    target = trajectory_teacher(observed)
    assert target["target_offset_m"] == -side * 3.2
    assert target["target_speed_mps"] > 5.
    assert target["lane_costs"][target["lane_index"]] == min(target["lane_costs"])


def test_trajectory_teacher_brakes_for_three_blocked_lanes_and_uses_curvature():
    observed = state([hazard(50., lane) for lane in LANE_OFFSETS_M])
    target = trajectory_teacher(observed)
    assert target["target_speed_mps"] < observed["body_speed_mps"]
    curved = {**state(), "curvature_ahead": [.08] * 7}
    assert nominal_pace(curved) < nominal_pace(state())
