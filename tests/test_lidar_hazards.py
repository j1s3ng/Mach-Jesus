import math

import numpy as np
import pytest

from race_lidar.hotlap_driver import RacingReference
from race_lidar.lidar_hazards import LidarHazardChannel


def circle():
    angle = np.linspace(0., 2. * np.pi, 1000, endpoint=False)
    return RacingReference(600. * np.cos(angle), 600. * np.sin(angle), np.full(len(angle), 60.))


def to_ego(reference, ego_progress, xyz):
    position, yaw, _, _ = reference.sample(ego_progress)
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.asarray(xyz, float).copy()
    result[:, :2] = (result[:, :2] - position) @ np.array([[c, -s], [s, c]])
    return np.column_stack((result, np.ones(len(result))))


def surface(reference, ego_progress, obstacle_progress, lane=0., sparse=False):
    center, yaw, _, _ = reference.sample(obstacle_progress)
    normal = np.array([-math.sin(yaw), math.cos(yaw)])
    widths = (-.6, 0., .6) if sparse else np.linspace(-.9, .9, 9)
    heights = (.65,) if sparse else (.45, .8, 1.15)
    xyz = [[*(center + (lane + width) * normal), z] for width in widths for z in heights]
    return to_ego(reference, ego_progress, xyz)


def observe(channel, reference, ego_progress, now, points):
    position, yaw, _, _ = reference.sample(ego_progress)
    return channel.update(points, [*position, .33], yaw, ego_progress, now)


def test_ground_curbs_and_self_body_do_not_become_obstacles():
    ref = circle()
    channel = LidarHazardChannel(ref)
    rng = np.random.default_rng(151)
    ground = np.array([[x, y, rng.normal(0., .025), 1.] for x in range(-10, 100, 2) for y in np.arange(-8., 8., .4)])
    curbs = np.array([[x, y, .18, 1.] for x in np.arange(5., 95., .2) for y in (-5., 5.)])
    body = np.array([[x, y, z, 1.] for x in np.arange(-1., 4.6, .3) for y in (-1., 0., 1.) for z in (.5, .9, 1.2)])
    result = observe(channel, ref, 0., 0., np.vstack((ground, curbs, body)))
    assert not result["geometry_hazards"]
    assert not result["free_space_verified"] and result["occluded_space"] == "unknown"


def test_three_supported_sparse_returns_warn_without_a_learned_car_score():
    ref = circle()
    channel = LidarHazardChannel(ref)
    result = observe(channel, ref, 0., 0., surface(ref, 0., 45., sparse=True))
    assert len(result["hazard_tracks"]) == 1
    row = result["hazard_tracks"][0]
    assert 44.8 < row["relative_distance_m"] < 45.2
    assert row["source"] == "raw_lidar_geometry" and row["object_class"] == "unclassified_obstacle"
    assert "score" not in row and row["confidence_kind"] == "uncalibrated_geometry_support"
    assert row["point_support"] == 3 and not row["persistent"] and not row["velocity_valid"]
    assert not result["free_space_verified"] and "visible_distance_m" not in result


@pytest.mark.parametrize("speed", [-15., 0., 25., 60.])
def test_motion_is_estimated_from_repeated_world_geometry_only(speed):
    ref = circle()
    channel = LidarHazardChannel(ref)
    first_id = None
    for index in range(6):
        now = index * .2
        ego_progress = 20. * now
        result = observe(channel, ref, ego_progress, now, surface(ref, ego_progress, 50. + speed * now))
        assert len(result["hazard_tracks"]) == 1
        row = result["hazard_tracks"][0]
        first_id = row["track_id"] if first_id is None else first_id
        assert row["track_id"] == first_id
    assert row["persistent"] and row["velocity_valid"]
    assert abs(row["speed_mps"] - speed) < 1.
    assert row["measurement_age_s"] == 0.


def test_range_corridor_and_overhead_filters_are_physical_geometry_limits():
    ref = circle()
    inside = LidarHazardChannel(ref)
    assert observe(inside, ref, 0., 0., surface(ref, 0., 98., sparse=True))["hazard_tracks"]
    outside = LidarHazardChannel(ref)
    assert not observe(outside, ref, 0., 0., surface(ref, 0., 110.))["hazard_tracks"]
    parked = LidarHazardChannel(ref)
    assert not observe(parked, ref, 0., 0., surface(ref, 0., 30., lane=8.5))["hazard_tracks"]
    overhead = surface(ref, 0., 30.)
    overhead[:, 2] += 5.
    assert not observe(LidarHazardChannel(ref), ref, 0., 0., overhead)["hazard_tracks"]


def test_one_beam_or_duplicate_noise_is_not_a_supported_surface():
    ref = circle()
    one = surface(ref, 0., 30., sparse=True)[:1]
    for points in (one, np.repeat(one, 10, axis=0)):
        assert not observe(LidarHazardChannel(ref), ref, 0., 0., points)["hazard_tracks"]


def test_stale_geometry_expires_and_missing_returns_do_not_prove_clear_space():
    ref = circle()
    channel = LidarHazardChannel(ref)
    observe(channel, ref, 0., 0., surface(ref, 0., 30.))
    coast = observe(channel, ref, 0., .2, np.empty((0, 4)))
    assert coast["hazard_tracks"][0]["measurement_age_s"] == .2
    expired = observe(channel, ref, 0., .8, np.empty((0, 4)))
    assert not expired["hazard_tracks"]
    assert not expired["free_space_verified"] and expired["occluded_space"] == "unknown"


def test_bad_rows_are_ignored_and_duplicate_timestamps_do_not_invent_persistence():
    ref = circle()
    channel = LidarHazardChannel(ref)
    points = np.vstack((surface(ref, 0., 30.), [[np.nan, 0., .5, 1.], [np.inf, 0., .5, 1.]]))
    for _ in range(3):
        result = observe(channel, ref, 0., 1., points)
    assert result["hazard_tracks"][0]["measurement_hits"] == 1
    with pytest.raises(ValueError, match="monotonic"):
        observe(channel, ref, 0., .8, points)
    with pytest.raises(ValueError, match="latest"):
        channel.observation(0., .8)
    with pytest.raises(ValueError, match="N,4"):
        observe(LidarHazardChannel(ref), ref, 0., 0., np.zeros((4, 3)))
