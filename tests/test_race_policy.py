import math

import numpy as np

from race_lidar.hotlap_driver import RacingReference
from race_lidar.race_policy import (RacePolicy, SignedProgressTracker, TournamentTraffic,
                                    policy_state, shared_guard)


def circle(radius=100.):
    angle = np.linspace(0., 2. * np.pi, 500, endpoint=False)
    return RacingReference(radius * np.cos(angle), radius * np.sin(angle),
                           np.full(len(angle), 60.), max_speed=60., lateral_accel=8.)


def observation(tracks=()):
    return {"front_distance_m": 60., "front_lateral_m": 0., "front_speed_mps": 30.,
            "left_clearance_m": 10., "right_clearance_m": 10., "confirmed_tracks": list(tracks)}


def test_staggered_grid_uses_shared_finish_and_preserves_negative_progress():
    ref = circle()
    tracker = SignedProgressTracker(ref, -24.)
    for s in np.arange(-24., ref.length + 1., .5):
        position = ref.sample(s)[0]
        state = tracker.update(position, 25., .02)
        assert abs(state.progress_m - s) < .6
        if -20. < s < -10.:
            assert state.progress_m < 0.
        if s < ref.length - 1.:
            assert state.progress_m < ref.length
    assert state.progress_m >= ref.length
    # Starting 24 m back adds 24 m of physical travel to the shared finish;
    # merely driving reference.length from one's own grid slot is insufficient.
    assert state.progress_m - (-24.) > ref.length + 23.


def test_signed_tracker_cannot_jump_across_track():
    ref = circle()
    tracker = SignedProgressTracker(ref, -12.)
    far = ref.sample(ref.length / 2.)[0]
    state = tracker.update(far, 0., .02)
    assert state.progress_m < -8.
    assert state.distance_m > 190.


def test_guard_brakes_from_all_measured_tracks_not_only_selected_lead():
    tracks = [{"relative_distance_m": 9., "lateral_m": 0., "speed_mps": 12.},
              {"relative_distance_m": 6., "lateral_m": 3.2, "speed_mps": 40.}]
    action = shared_guard({"target_speed_mps": 50., "target_offset_m": 0.}, observation(tracks),
                          body_speed=40., current_lane=0., feasible_speed=60., dt=.02)
    assert action["target_speed_mps"] == 12.
    assert "measured_vehicle_gap" in action["guards"]


def test_lane_guard_prevents_sweeping_through_alongside_car_but_allows_escape():
    tracks = [{"relative_distance_m": -1., "lateral_m": 2.4, "speed_mps": 30.}]
    towards = shared_guard({"target_speed_mps": 50., "target_offset_m": 3.2}, observation(tracks),
                          body_speed=30., current_lane=.2, feasible_speed=40., dt=.2)
    assert towards["target_offset_m"] == .2
    assert "alongside_lane_occupied" in towards["guards"]
    assert towards["target_speed_mps"] == 40.
    away = shared_guard({"target_speed_mps": 50., "target_offset_m": -3.2}, observation(tracks),
                       body_speed=30., current_lane=.2, feasible_speed=40., dt=.2)
    assert away["target_offset_m"] < .2
    assert "alongside_lane_occupied" not in away["guards"]


def test_tournament_tracks_detect_alongside_and_rear_and_ignore_self():
    ref = circle()
    traffic = TournamentTraffic(ref)
    ego, yaw, _, _ = ref.sample(0.)
    boxes = [{"center": [1.6, 0., .7], "size": [5., 2.1, 1.1], "score": .95},
             {"center": [-6., 2.5, .7], "size": [5., 2.1, 1.1], "score": .95}]
    for now in (0., .2, .4):
        traffic.update(boxes, ego, yaw, 0., now)
    observed = traffic.observation(0., 0., .4)
    assert len(observed["confirmed_tracks"]) == 1
    assert -7. < observed["front_distance_m"] < -5.
    assert 2. < observed["front_lateral_m"] < 3.


def test_new_fast_car_stays_associated_at_five_hz():
    ref = circle(300.)
    traffic = TournamentTraffic(ref)
    for tick in range(4):
        now = tick * .2
        ego_progress = 60. * now
        ego, yaw, _, _ = ref.sample(ego_progress)
        target = ref.sample(ego_progress + 20.)[0]
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([[c, s], [-s, c]]) @ (target - ego)
        box = {"center": [*local, .7], "size": [5., 2.1, 1.1], "score": .95}
        traffic.update([box], ego, yaw, ego_progress, now, ego_speed=60.)
    observed = traffic.observation(ego_progress, 60., now)
    assert len(observed["confirmed_tracks"]) == 1
    assert observed["confirmed_tracks"][0]["track_id"] == 1
    assert abs(observed["front_speed_mps"] - 60.) < 2.


def test_repassed_opponent_remains_available_to_policy():
    ref = circle()
    traffic = TournamentTraffic(ref)
    ego, yaw, _, _ = ref.sample(0.)
    for now in (0., .2, .4):
        traffic.update([{"center": [20., 0., .7], "size": [5., 2.1, 1.1], "score": .95}], ego, yaw, 0., now)
    traffic.observation(40., 30., .4)
    assert traffic.pass_count == 1
    observed = traffic.observation(0., 30., .4)
    assert observed["front_distance_m"] < 30.


def test_neural_decision_records_measured_features_and_counterfactual():
    class Model:
        def predict(self, state):
            return {"target_speed_mps": state["front_distance_m"], "target_offset_m": 1., "model_used": True}
    ref = circle()
    observed = observation([{"relative_distance_m": 10., "lateral_m": 0., "speed_mps": 15.}])
    observed["front_distance_m"] = 10.
    state = policy_state(ref, -12., -2.5, ref.sample(-12.)[1], 30., -2.5, observed)
    decision = RacePolicy(ref, model=Model(), name="neural").predict(state, observed, 40.)
    assert decision["raw_action"]["target_speed_mps"] == 10.
    assert decision["no_traffic_counterfactual"]["target_speed_mps"] == 60.
    assert decision["vision_changed_action"]
    assert len(state["curvature_ahead"]) == 7
    assert 0. < state["progress_fraction"] < 1.
    assert "confirmed_tracks" not in state  # Only the trained feature contract.


def test_classical_policy_chooses_clear_lane_without_extra_speed_budget():
    ref = circle()
    observed = observation([{"relative_distance_m": 20., "lateral_m": 0., "speed_mps": 15.}])
    observed["left_clearance_m"] = 0.
    policy = RacePolicy(ref, name="reference", seed=0)
    decision = policy.predict({}, observed, 27.)
    assert decision["raw_action"]["target_offset_m"] == -3.2
    assert decision["raw_action"]["target_speed_mps"] == 27.


def test_reference_pass_uses_far_lane_occupancy_beyond_saturated_summary():
    ref=circle()
    observed=observation([{'relative_distance_m':70.,'lateral_m':0.,'speed_mps':0.},
                          {'relative_distance_m':70.,'lateral_m':-3.2,'speed_mps':0.}])
    # Both compressed clearance features saturate at10m, but right is blocked.
    policy=RacePolicy(ref,name="reference",seed=1)
    decision=policy.predict({'body_speed_mps':20.},observed,40.)
    assert decision['raw_action']['target_offset_m']==3.2
