"""Measured-traffic trajectory guards and physical post-finish parking.

No opponent ground truth enters this module. A policy still chooses its passing
lane; the guard limits speed and prevents a predicted sweep through a vehicle.
"""
from __future__ import annotations

import math
import numpy as np


def combine_observations(vision, geometry):
    """Fuse measured safety rows without turning geometry into learned boxes."""
    result=dict(vision)
    raw=geometry.get('geometry_hazards',[])
    rows=[*vision.get('hazard_tracks',vision.get('confirmed_tracks',[])),*raw]
    result['hazard_tracks']=rows
    result['geometry_hazards']=raw
    relevant=[r for r in rows if abs(r['lateral_m'])<=6. and -12.<=r['relative_distance_m']<=100.]
    ahead=[r for r in relevant if r['relative_distance_m']>=0.]
    lead=min(ahead,key=lambda r:r['relative_distance_m']) if ahead else None
    if lead is not None:
        result.update(front_distance_m=lead['relative_distance_m'],front_lateral_m=lead['lateral_m'],
                      front_speed_mps=lead['speed_mps'],lead_track_id=lead['track_id'])
    for side,lane in [('left',2.5),('right',-2.5)]:
        distances=[max(0.,abs(r['relative_distance_m'])-5.) for r in relevant if abs(r['lateral_m']-lane)<1.7]
        result[f'{side}_clearance_m']=min(10.,min(distances,default=10.))
    return result


def stopping_speed(clearance, lead_speed=0., *, decel=7., reaction=.4):
    """Speed that leaves reaction and constant-deceleration stopping distance.

    A signed negative lead speed represents an oncoming obstacle that continues
    approaching while ego brakes. Forward traffic is conservatively assumed to
    brake at the same available deceleration.
    """
    clearance = max(0., float(clearance))
    if lead_speed >= 0.:
        return max(0., math.sqrt((decel * reaction) ** 2 + lead_speed ** 2 + 2 * decel * clearance) - decel * reaction)
    approach = -lead_speed
    constant = decel * reaction + approach
    return max(0., math.sqrt(constant ** 2 + 2 * decel * max(0., clearance - approach * reaction)) - constant)


def predictive_guard(raw_action, observation, *, body_speed, current_lane,
                     actual_lateral, feasible_speed, dt, lane_rate=1.,
                     maximum_lane=3.2, brake_decel=8., visible_distance=45.,
                     enforce_visibility_stopping_budget=True, racing_parameters=None):
    requested = float(np.clip(raw_action['target_offset_m'], -maximum_lane, maximum_lane))
    speed = float(np.clip(raw_action['target_speed_mps'], 0., feasible_speed))
    params = dict(racing_parameters or {})
    lane_rate = float(params.get('lane_rate_mps', lane_rate))
    half_width = float(params.get('ego_half_width_m', 1.3)) + float(params.get('lateral_margin_m', 0.))
    uncertainty_cap = float(params.get('position_uncertainty_cap_m', 1.5))
    sigma_factor = float(params.get('lead_speed_sigma_factor', 1.))
    reaction = float(params.get('reaction_seconds', .4))
    following_gap = float(params.get('following_gap_m', 8.))
    flags = []
    minimum_decel = 0.
    decel = min(7., .9 * brake_decel)
    visibility_cap = stopping_speed(max(0., visible_distance - 8.), decel=decel)
    if enforce_visibility_stopping_budget and speed > visibility_cap:
        speed = visibility_cap
        flags.append('visibility_stopping_budget')
    lane = current_lane + float(np.clip(requested - current_lane, -lane_rate * dt, lane_rate * dt))
    rows = observation.get('hazard_tracks', observation.get('confirmed_tracks', []))
    horizon = min(5., max(.6, abs(requested - actual_lateral) / lane_rate + .6))
    times = np.linspace(0., horizon, 18)
    planned = actual_lateral + np.sign(requested - actual_lateral) * np.minimum(
        abs(requested - actual_lateral), lane_rate * np.maximum(0., times - .35))
    def distance_to_obstacle(row,position):
        center=float(row['lateral_m'])
        lower=min(float(row.get('lateral_min_m',center-1.05)),center-1.05)
        upper=max(float(row.get('lateral_max_m',center+1.05)),center+1.05)
        return np.maximum(np.maximum(lower-position,position-upper),0.)
    for row in rows:
        distance, lateral = float(row['relative_distance_m']), float(row['lateral_m'])
        velocity = float(row.get('speed_mps', 0.))
        uncertainty = min(uncertainty_cap, max(.15, float(row.get('position_uncertainty_m', row.get('position_std_m', .35)))))
        width = half_width + uncertainty
        valid_velocity = bool(row.get('velocity_valid', True))
        sigma = max(1., float(row.get('speed_uncertainty_mps', row.get('speed_std_mps', 2.))))
        low, high = (velocity - sigma, velocity + sigma) if valid_velocity else (0., 70.)
        lo = distance + (low - body_speed) * times
        hi = distance + (high - body_speed) * times
        possible_near = (lo < 8.5) & (hi > -8.5)
        planned_gap=distance_to_obstacle(row,planned)
        approaches_lane = planned_gap < distance_to_obstacle(row,actual_lateral) - .02
        if np.any(possible_near & approaches_lane & (planned_gap < width)):
            lane = float(actual_lateral)
            flags.append('predicted_lane_conflict')
    for row in rows:
        distance, lateral = float(row['relative_distance_m']), float(row['lateral_m'])
        uncertainty = min(uncertainty_cap, max(.15, float(row.get('position_uncertainty_m', row.get('position_std_m', .35)))))
        if distance <= 0. or min(distance_to_obstacle(row,actual_lateral), distance_to_obstacle(row,lane)) >= half_width + uncertainty:
            continue
        velocity = float(row.get('speed_mps', 0.))
        sigma = max(0., float(row.get('speed_uncertainty_mps', row.get('speed_std_mps', 2.))))
        lead = velocity - sigma * sigma_factor if row.get('velocity_valid', True) else 0.
        cap = stopping_speed(distance - following_gap - uncertainty, lead, decel=decel, reaction=reaction)
        if cap < speed:
            speed = cap
            flags.append('predictive_stopping_gap')
        if cap < body_speed + .1:
            space = distance - following_gap - uncertainty - reaction * body_speed
            space += max(0., lead) ** 2 / (2. * decel)
            if lead < 0.:
                space -= -lead * (reaction + body_speed / decel)
            required = body_speed ** 2 / (2. * max(.1, space))
            minimum_decel = max(minimum_decel, min(brake_decel, required))
            flags.append('stopping_brake_feedforward')
    return {'target_speed_mps': float(speed), 'target_offset_m': float(lane),
            'minimum_decel_mps2': minimum_decel,
            'requested_offset_m': requested, 'guards': sorted(set(flags))}


def parking_plan(progress, finish_progress, grid_index, actual_lateral, speed, *,
                 parked=False, side=None):
    """Steer and brake into a staggered off-racing-line bay, without pose writes."""
    side = float(side if side is not None else (1. if actual_lateral >= 0. else -1.))
    station = finish_progress + 180. + 24. * int(grid_index)
    offset = 13. * side
    remaining = station - progress
    aligned = abs(actual_lateral - offset) < 1.3
    parked = bool(parked or (remaining < 2. and aligned and speed < .35))
    target = 0. if parked else min(8., math.sqrt(2. * 3. * max(0., remaining - 1.)))
    # Travel along an inner runoff corridor past occupied earlier bays, then
    # peel outward slowly only near this car's own staggered bay.
    if remaining > 28.:
        offset = 8.5 * side
    elif not aligned:
        target = min(target, 3.)
    if remaining < 1. and not aligned:
        target = 2.
    return {'target_speed_mps': target, 'target_offset_m': offset,
            'parked': parked, 'station_m': station, 'side': side}
