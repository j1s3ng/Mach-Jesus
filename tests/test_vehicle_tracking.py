"""Physical-motion scenarios; opponent truth is used only to make test boxes."""
import math

import numpy as np
import pytest

from race_lidar.hotlap_driver import RacingReference
from race_lidar.race_policy import TournamentTraffic


def circle(radius=600.):
    angle = np.linspace(0., 2. * np.pi, 1000, endpoint=False)
    return RacingReference(radius * np.cos(angle), radius * np.sin(angle), np.full(len(angle), 60.))


def box_for(reference, ego_progress, target_progress, lane=0., score=.95, noise=(0., 0.)):
    ego, yaw, _, _ = reference.sample(ego_progress)
    target, target_yaw, _, _ = reference.sample(target_progress)
    target += lane * np.array([-math.sin(target_yaw), math.cos(target_yaw)])
    target += np.asarray(noise)
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[c, s], [-s, c]]) @ (target - ego)
    return {"center": [*local, .7], "size": [5., 2.1, 1.1], "score": score}, target


def update(traffic, reference, ego_progress, now, targets, ego_speed=40.):
    ego, yaw, _, _ = reference.sample(ego_progress)
    boxes = [box_for(reference, ego_progress, *target)[0] for target in targets]
    traffic.update(boxes, ego, yaw, ego_progress, now, ego_speed=ego_speed)
    return traffic.observation(ego_progress, ego_speed, now)


def test_original_stationary_fifty_metre_fixture_confirms_and_warns_immediately():
    # Exact motion/range from the preserved failed-tournament reproduction.
    ref = circle(300.)
    traffic = TournamentTraffic(ref)
    first_id = None
    for index in range(6):
        now = index * .2
        observed = update(traffic, ref, 40. * now, now, [(50.,)])
        assert abs(observed["front_distance_m"] - (50. - 40. * now)) < .05
        assert len(observed["hazard_tracks"]) == 1
        row = observed["hazard_tracks"][0]
        first_id = row["track_id"] if first_id is None else first_id
        assert row["track_id"] == first_id
        assert len(observed["confirmed_tracks"]) == (1 if index >= 2 else 0)
        assert abs(row["speed_mps"]) < .1
    assert len(traffic.tracks) == 1
    assert traffic.tracks[0].hits == 6


@pytest.mark.parametrize("target_speed", [-75., -20., 0., 20., 60., 75.])
@pytest.mark.parametrize("ego_speed", [0., 40., 60.])
def test_signed_motion_acquires_without_ego_velocity_assumption(target_speed, ego_speed):
    ref = circle()
    traffic = TournamentTraffic(ref)
    for index in range(4):
        now = index * .2
        observed = update(traffic, ref, ego_speed * now, now, [(50. + target_speed * now,)], ego_speed)
    assert len(observed["confirmed_tracks"]) == 1
    row = observed["confirmed_tracks"][0]
    assert row["track_id"] == 1
    assert abs(row["speed_mps"] - target_speed) < 3.
    assert row["velocity_valid"] and row["speed_std_mps"] < 8.


def test_dense_lane_crossings_keep_identity_with_score_order_changes_and_occlusion():
    ref = circle()
    traffic = TournamentTraffic(ref, max_forward_m=250.)
    actors = [(25., -3., 40.), (65., -3., 40.),
              (45., 0., 20.), (85., 0., 20.),
              (65., 3., 0.), (105., 3., 0.)]
    rng = np.random.default_rng(742)
    identities = None
    for index in range(16):
        now = index * .2
        ego, yaw, _, _ = ref.sample(0.)
        boxes, positions = [], []
        for actor_index, (start, lane, speed) in enumerate(actors):
            noise = rng.normal(0., .12, 2)
            box, position = box_for(ref, 0., start + speed * now, lane,
                                    score=.72 + .035 * ((actor_index + index) % 6), noise=noise)
            positions.append(position)
            if not (actor_index == 2 and index in (5, 6)):
                boxes.append(box)
        rng.shuffle(boxes)
        traffic.update(boxes, ego, yaw, 0., now, ego_speed=0.)
        if identities is None:
            identities = [min(traffic.tracks, key=lambda track: np.linalg.norm(track.position - position)).identifier
                          for position in positions]
            assert len(set(identities)) == 6
        if index >= 2:
            tracks = {track.identifier: track for track in traffic.tracks}
            assert len(tracks) == 6
            for identity, position in zip(identities, positions):
                assert identity in tracks
                assert np.linalg.norm(tracks[identity].position - position) < 1.1
                assert tracks[identity].confirmed


def test_one_scan_hazard_expires_but_weak_detections_require_confirmation():
    ref = circle()
    traffic = TournamentTraffic(ref)
    first = update(traffic, ref, 0., 0., [(30., 0., .95)])
    assert first["hazard_tracks"] and not first["confirmed_tracks"]
    assert first["front_speed_mps"] == 0.
    assert not first["hazard_tracks"][0]["velocity_valid"]
    update(traffic, ref, 0., .2, [])
    missing = update(traffic, ref, 0., .4, [])
    assert not missing["hazard_tracks"] and missing["front_distance_m"] == 60.
    weak = TournamentTraffic(ref)
    for index in range(3):
        observation = update(weak, ref, 0., index * .2, [(30., 0., .45)])
        assert bool(observation["hazard_tracks"]) == (index >= 1)
        assert bool(observation["confirmed_tracks"]) == (index == 2)


def test_two_weak_stopped_measurements_hold_through_the_observed_dropout_sequence():
    ref = circle()
    traffic = TournamentTraffic(ref, rapid_hazard_confidence=.35)
    # Observed stopped-003 detector scores/timing. The car must not appear to
    # vanish between braking and the subsequent collision approach.
    sequence = [(6.4, .393), (6.6, None), (6.8, .439), (7.0, None),
                (7.2, None), (7.4, None), (7.6, None), (7.8, None), (8.0, None)]
    uncertainties = []
    for now, score in sequence:
        targets = [(40., 0., score)] if score is not None else []
        observed = update(traffic, ref, 0., now, targets)
        assert len(observed["hazard_tracks"]) == 1
        assert abs(observed["front_distance_m"] - 40.) < .1
        row = observed["hazard_tracks"][0]
        assert abs(row["speed_mps"]) < .1
        if now >= 6.8:
            assert row["control_evidence"] == "two_measured_hits"
            uncertainties.append(row["position_std_m"])
    assert not observed["confirmed_tracks"]
    assert row["measurement_hits"] == 2 and row["age_s"] > 1.1
    assert uncertainties[-1] > uncertainties[0]
    expired = update(traffic, ref, 0., 8.2, [])
    assert not expired["hazard_tracks"]


def test_scan_duplicates_do_not_confirm_and_bad_timestamps_are_rejected():
    ref = circle()
    traffic = TournamentTraffic(ref)
    for _ in range(5):
        observed = update(traffic, ref, 0., 1., [(30.,)])
    assert traffic.tracks[0].hits == 1
    assert not observed["confirmed_tracks"]
    with pytest.raises(ValueError, match="monotonic"):
        update(traffic, ref, 0., .8, [(30.,)])
    with pytest.raises(ValueError, match="latest"):
        traffic.observation(0., 0., .8)


def test_duplicate_boxes_and_self_returns_do_not_create_extra_opponents():
    ref = circle()
    traffic = TournamentTraffic(ref)
    ego, yaw, _, _ = ref.sample(0.)
    box, _ = box_for(ref, 0., 30.)
    duplicate = {**box, "score": .8}
    malformed = {"center": None, "size": [5., 2.1, 1.1], "score": .95}
    self_box = {"center": [1.6, 0., .7], "size": [5., 2.1, 1.1], "score": .95}
    for now in (0., .2, .4):
        traffic.update([box, duplicate, malformed, self_box, None], ego, yaw, 0., now)
    assert len(traffic.snapshot(0., .4)) == 1


def test_far_and_partly_offline_car_is_still_a_measured_hazard():
    ref = circle()
    traffic = TournamentTraffic(ref)
    observed = update(traffic, ref, 0., 0., [(90., 5.5)])
    assert len(observed["hazard_tracks"]) == 1
    assert abs(observed["hazard_tracks"][0]["relative_distance_m"] - 90.) < .1
    assert abs(observed["hazard_tracks"][0]["lateral_m"] - 5.5) < .1


def test_collision_like_abrupt_stop_remains_a_fresh_hazard_during_reassociation():
    ref = circle()
    traffic = TournamentTraffic(ref)
    # A contact impulse need not fit the tracker's normal acceleration model.
    # It may replace the identity, but it must not erase the measured obstacle.
    for index in range(5):
        now = index * .2
        update(traffic, ref, 0., now, [(20. + 75. * now,)])
    for index in range(5, 9):
        now = index * .2
        observed = update(traffic, ref, 0., now, [(80.,)])
        closest = min(observed["hazard_tracks"], key=lambda row: abs(row["relative_distance_m"] - 80.))
        assert abs(closest["relative_distance_m"] - 80.) < 1.
        assert abs(closest["speed_mps"]) < 2.
    assert any(row["confirmed"] and abs(row["speed_mps"]) < 2.
               for row in observed["hazard_tracks"])
