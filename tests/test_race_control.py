import math

from race_lidar.hotlap_driver import longitudinal_control
from race_lidar.race_control import parking_plan, predictive_guard, stopping_speed


def guard(rows=(), **kwargs):
    arguments=dict(body_speed=25.,current_lane=0.,actual_lateral=0.,feasible_speed=60.,dt=.2)
    arguments.update(kwargs)
    return predictive_guard({'target_speed_mps':60.,'target_offset_m':arguments.pop('target_lane',0.)},
                            {'hazard_tracks':list(rows)},**arguments)


def test_zero_target_holds_stationary_and_reverse_creep_without_drive_torque():
    for body in (-.2,0.,.1,.3):
        assert longitudinal_control(0.,body,0.) == (0.,1.)
    throttle,brake=longitudinal_control(0.,20.,0.,mass=880.)
    assert throttle==0. and brake>0.


def test_visibility_budget_can_stop_before_newly_visible_stationary_obstacle():
    value=guard(visible_distance=45.)['target_speed_mps']
    assert value*.4+value**2/(2*7.)+8. <=45.+1e-8
    assert value<25.


def test_unconfirmed_first_scan_causes_braking_and_signed_oncoming_is_stricter():
    tentative={'relative_distance_m':12.,'lateral_m':0.,'speed_mps':40.,'velocity_valid':False}
    assert guard([tentative])['target_speed_mps']<5.
    stopped=stopping_speed(30.,0.)
    oncoming=stopping_speed(30.,-15.)
    assert oncoming<stopped
    assert oncoming*.4+oncoming**2/14.+15.*(.4+oncoming/7.)<=30.+1e-8


def test_lane_change_checks_rear_closing_car_and_actual_lateral_state():
    rear={'relative_distance_m':-20.,'lateral_m':3.2,'speed_mps':45.,'velocity_valid':True}
    result=guard([rear],target_lane=3.2,body_speed=25.)
    assert result['target_offset_m']==0.
    assert 'predicted_lane_conflict' in result['guards']
    result=guard([rear],target_lane=3.2,current_lane=2.,actual_lateral=.5,body_speed=25.)
    assert result['target_offset_m']==.5


def test_escape_from_stopped_front_car_remains_possible_while_braking():
    front={'relative_distance_m':15.,'lateral_m':0.,'speed_mps':0.,'velocity_valid':True}
    result=guard([front],target_lane=-3.2)
    assert result['target_offset_m']<0.
    assert result['target_speed_mps']<9.


def test_five_hz_closed_loop_stops_before_stationary_car_footprints_overlap():
    distance=45.;speed=stopping_speed(37.);observed=distance
    for tick in range(600):
        if tick%10==0:observed=distance
        action=guard([{'relative_distance_m':observed,'lateral_m':0.,'speed_mps':0.,
                       'velocity_valid':True,'speed_uncertainty_mps':0.}],body_speed=speed)
        throttle,brake=longitudinal_control(action['target_speed_mps'],speed,speed,mass=880.)
        decel=max(4*brake*700/(880*.35),action['minimum_decel_mps2'])
        previous=speed;speed=max(0.,speed+(.6*throttle-decel)*.02)
        distance-=.5*(previous+speed)*.02
        assert distance>6.6
    assert speed<.4 and distance>7.


def test_parking_is_staggered_outside_racing_corridor_and_holds_after_arrival():
    first=parking_plan(5300.,5200.,0,2.,10.)
    last=parking_plan(5300.,5200.,19,-2.,10.)
    assert first['target_offset_m']==8.5 and last['target_offset_m']==-8.5
    assert last['station_m']-first['station_m']==19*24.
    approach=parking_plan(first['station_m']-15.,5200.,0,8.5,3.)
    assert approach['target_offset_m']==13. and approach['target_speed_mps']<=3.
    arrived=parking_plan(first['station_m'],5200.,0,12.9,.1)
    assert arrived['parked'] and arrived['target_speed_mps']==0.
    missed=parking_plan(first['station_m']+3.,5200.,0,0.,.1)
    assert not missed['parked'] and missed['target_speed_mps']==2.


def test_wide_raw_surface_cannot_be_treated_as_a_single_narrow_car_center():
    wide={'relative_distance_m':15.,'lateral_m':0.,'lateral_min_m':-4.2,'lateral_max_m':4.2,
          'speed_mps':0.,'velocity_valid':True}
    result=guard([wide],current_lane=3.2,actual_lateral=3.2,target_lane=3.2)
    assert result['target_speed_mps']<9.
    assert 'predictive_stopping_gap' in result['guards']
