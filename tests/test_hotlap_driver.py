import math

import numpy as np

from race_lidar.hotlap_driver import RacingReference, ProgressTracker, PurePursuit, longitudinal_control, VisionTraffic


def circle(radius=50):
    angle = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    return RacingReference(radius * np.cos(angle), radius * np.sin(angle), np.full(len(angle), 80.),
                           max_speed=60, lateral_accel=8)


def test_grip_and_cyclic_braking_budget():
    ref = circle()
    assert abs(ref.length - 2 * math.pi * 50) < 1
    assert ref.speed.max() <= 20.1  # sqrt(8 m/s² * 50 m)
    assert np.all(ref.speed ** 2 <= np.roll(ref.speed, -1) ** 2 + 2 * 8 * ref.ds + 1e-7)
    assert np.all(np.roll(ref.speed, -1) ** 2 <= ref.speed ** 2 + 2 * 5 * ref.ds + 1e-7)


def test_ordered_progress_crosses_finish_only_after_full_lap():
    ref = circle()
    tracker = ProgressTracker(ref)
    for s in np.arange(0, ref.length + 3, .5):
        position, _, _, _ = ref.sample(s)
        state = tracker.update(position, speed=25., dt=.02)
        assert abs(state.progress_m - s) < .6
    assert state.progress_m >= ref.length


def test_progress_cannot_teleport_to_other_track_section():
    ref = circle()
    tracker = ProgressTracker(ref)
    far, _, _, _ = ref.sample(ref.length / 2)
    state = tracker.update(far, speed=0., dt=.02)
    assert state.progress_m < 5
    assert state.distance_m > 90


def test_pure_pursuit_bicycle_completes_circle():
    ref = circle()
    tracker, controller = ProgressTracker(ref), PurePursuit()
    position = ref.xy[0].copy()
    yaw, speed, dt = math.pi / 2, 15., .02
    maximum_error = 0.
    for _ in range(1400):
        state = tracker.update(position,speed,dt)
        steer, _ = controller.control(ref,state.progress_m,position,yaw,speed,dt)
        position += speed * np.array([math.cos(yaw),math.sin(yaw)]) * dt
        yaw += speed / controller.wheelbase * math.tan(steer) * dt
        maximum_error = max(maximum_error,state.distance_m)
    assert tracker.progress > ref.length
    assert maximum_error < 1


def test_longitudinal_feedback_brakes_and_reduces_spin():
    assert longitudinal_control(20,10,10)[0] > longitudinal_control(20,10,35)[0]
    throttle, brake = longitudinal_control(10,25,25)
    assert throttle == 0 and 0 < brake <= 1
    # Finite-difference wheel encoders under-report speed at coarse timesteps;
    # they must not suppress braking while the body is still moving quickly.
    assert longitudinal_control(10,25,0)[1] == brake


def test_brake_budget_matches_webots_constant_torque_model():
    _, brake = longitudinal_control(10,25,25,mass=880,wheel_radius=.35,brake_coefficient=700,max_decel=8)
    np.testing.assert_allclose(4*brake*700/(880*.35),8)


def test_braking_preview_catches_apex_and_wraps_finish():
    ref = circle()
    ref.speed[:] = 25.
    ref.speed[7] = 6.
    assert ref.minimum_speed(ref.s[4],ref.s[10]-ref.s[4]) == 6
    # Merely checking the interval endpoint would miss the low-speed sample.
    assert ref.sample(ref.s[10])[2] == 25
    ref.speed[1] = 5.
    assert ref.minimum_speed(ref.length-2,8) == 5


def test_vision_tracks_use_track_lateral_and_confirm_then_pass():
    ref = circle(100)
    traffic = VisionTraffic(ref)
    for tick in range(3):
        timestamp = tick*.1
        ego_progress = 10*timestamp
        ego,ego_yaw,_,_ = ref.sample(ego_progress)
        ego = ego+2.5*np.array([-math.sin(ego_yaw),math.cos(ego_yaw)])
        target,target_yaw,_,_ = ref.sample(30+5*timestamp)
        target = target-2.5*np.array([-math.sin(target_yaw),math.cos(target_yaw)])
        c,s = math.cos(ego_yaw),math.sin(ego_yaw)
        relative = np.array([[c,s],[-s,c]])@(target-ego)
        box = {"center":[*relative,.6],"size":[4.8,2.1,1.1],"score":.85}
        traffic.update([box],ego,ego_yaw,ego_progress,timestamp)
        observed = traffic.observation(ego_progress,10,timestamp)
        assert len(observed["confirmed_tracks"]) == (1 if tick==2 else 0)
    assert abs(observed["front_lateral_m"]+2.5)<.1
    assert traffic.encounter_count == 1
    # Track remains associated through an alongside/behind phase and is counted
    # passed only after the ego has cleared it by twelve metres of arc length.
    observed = traffic.observation(45,10,.3)
    assert observed["front_distance_m"] == 60
    assert traffic.pass_count == 1


def test_vision_tracks_reject_small_cones_and_off_track_detections():
    ref = circle(100)
    traffic = VisionTraffic(ref)
    ego,yaw,_,_ = ref.sample(0)
    c,s = math.cos(yaw),math.sin(yaw)
    target,target_yaw,_,_ = ref.sample(20)
    target = target+8*np.array([-math.sin(target_yaw),math.cos(target_yaw)])
    relative = np.array([[c,s],[-s,c]])@(target-ego)
    off_track = {"center":[*relative,.6],"size":[4.8,2.1,1.1],"score":.9}
    cone = {"center":[20,0,.4],"size":[.5,.5,.8],"score":.99}
    for t in [0,.1,.2]:traffic.update([off_track,cone],ego,yaw,0,t)
    assert traffic.observation(0,10,.2)["confirmed_tracks"] == []
