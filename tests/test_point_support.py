import math

import numpy as np
import pytest

from race_lidar.point_support import config_sha256, filter_supported_boxes, support_counts, wrap_predictor, xy_diameter_at_least


CONFIG = {"name": "measured_above_ground_box_support", "version": 1,
          "minimum_points": 2, "above_ground_z_m": .25, "box_margin_m": 0., "minimum_xy_diameter_m": 0.}


def box(**updates):
    return {"center": [0., 0., .5], "size": [4., 1., 1.], "yaw": math.pi/2, "score": .6, **updates}


def test_ground_is_not_support_and_exactly_two_sparse_returns_preserve_original_box():
    detection = box()
    cloud = np.array([[0., 0., 0., 1.], [0., 0., .25, 1.], [0., .8, .5, 1.]], dtype=np.float32)
    assert support_counts(cloud, [detection], CONFIG) == [1]
    assert filter_supported_boxes(cloud, [detection], CONFIG) == []
    cloud = np.vstack((cloud, [[0., -.8, .5, 1.]]))
    result = filter_supported_boxes(cloud, [detection], CONFIG)
    assert result == [detection] and result[0] is detection


def test_yaw_and_full_vertical_extent_are_respected():
    cloud = np.array([[0., 1.5, .5, 1.], [1.5, 0., .5, 1.], [0., 0., 2., 1.]], dtype=np.float32)
    assert support_counts(cloud, [box(), box(yaw=0.)], CONFIG) == [1, 1]
    assert support_counts(np.empty((0, 4), dtype=np.float32), [box()], CONFIG) == [0]


def test_batched_filter_does_not_share_another_cars_returns_and_health_binds_identity():
    detection = box()
    predictor = lambda clouds: [[detection] for _ in clouds]
    predictor.metadata = {"checkpoint_sha256": "old", "transfer": {"loaded_tensors": 542}}
    wrapped = wrap_predictor(predictor, CONFIG, artifact_sha256="artifact", parent_checkpoint_sha256="parent", state_sha256="state")
    cloud = np.array([[0., -.8, .5, 1.], [0., .8, .5, 1.]], dtype=np.float32)
    assert wrapped([cloud, np.empty((0, 4), dtype=np.float32)]) == [[detection], []]
    assert wrapped.metadata["checkpoint_sha256"] == "artifact"
    assert wrapped.metadata["weights_parent_checkpoint_sha256"] == "parent"
    assert wrapped.metadata["postprocess_sha256"] == config_sha256(dict(reversed(list(CONFIG.items()))))


def test_invalid_boxes_or_silently_changed_config_fail_closed():
    cloud = np.empty((0, 4), dtype=np.float32)
    with pytest.raises(ValueError): support_counts(cloud, [box(size=[4., -1., 1.])], CONFIG)
    with pytest.raises(ValueError): support_counts(cloud, [box()], {**CONFIG, "gt_fallback": True})


def test_xy_diameter_rejects_vertical_pole_and_preserves_sparse_sideview_under_rotation():
    config = {**CONFIG, "minimum_xy_diameter_m": .2}
    pole = np.array([[0., 0., .3, 1.], [0., 0., .8, 1.]], dtype=np.float32)
    assert filter_supported_boxes(pole, [box()], config) == []
    sideview = np.array([[0., -.15, .5, 1.], [0., .15, .5, 1.]], dtype=np.float32)
    assert filter_supported_boxes(sideview, [box()], config) == [box()]
    for yaw in np.linspace(0., 2*math.pi, 19):
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        assert xy_diameter_at_least(sideview[:, :2]@rotation.T, .2)
        assert not xy_diameter_at_least(sideview[:, :2]@rotation.T, .4)
