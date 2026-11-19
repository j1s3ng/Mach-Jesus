import numpy as np
import pytest

from race_lidar.hotlap_geometry import closed_geometry, signed_face_areas


def test_closed_track_preserves_coordinates_and_upward_faces():
    t = np.linspace(0, 2*np.pi, 100)
    xy = np.column_stack([100+60*np.cos(t), -80+60*np.sin(t)])
    data = closed_geometry(xy, spacing=3, half_width=8)
    assert np.linalg.norm(data['reference_xy'][0]-xy[0]) < 1e-8
    np.testing.assert_allclose(data['center_xy'].mean(0), [100,-80], atol=.1)
    assert abs(data['length_m']-2*np.pi*60) < .1
    normal, center = data['normal_xy'],data['center_xy']
    assert np.all(signed_face_areas(center+8*normal,center-8*normal)>0)
    assert np.max(np.abs(data['curvature_per_m'])) < .018


def test_reject_incomplete_or_nonfinite_track():
    with pytest.raises(ValueError):closed_geometry(np.array([[0,0],[1,np.nan]]))
    xy = np.column_stack([np.arange(20)*20,np.zeros(20)])
    with pytest.raises(ValueError,match='complete closed lap'):closed_geometry(xy)


def test_backmarker_spawn_uses_right_lane_and_separate_progress():
    from scripts.generate_hotlap_world import backmarker_specs,reference_spawn
    from race_lidar.hotlap_driver import RacingReference
    angle=np.linspace(0,2*np.pi,200,endpoint=False)
    reference=RacingReference(100*np.cos(angle),100*np.sin(angle))
    specs=backmarker_specs(3)
    assert [spec['start_progress_m'] for spec in specs]==[70.,260.,650.]
    point,yaw=reference_spawn(reference,70.,-2.5)
    center,heading,_,_=reference.sample(70.)
    normal=np.array([-np.sin(heading),np.cos(heading)])
    assert abs(np.dot(point-center,normal)+2.5)<1e-8
    assert yaw==heading
