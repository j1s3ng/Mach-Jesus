"""Opt-in release of discretionary braking on an approved known route.

This is a receding-horizon command coordinator, not a tire-force estimator.
Traffic/stopping, road and spin guards remain outside it and authoritative.
"""
from __future__ import annotations

import math
import numpy as np


class TrailBrakeCoordinator:
    VERSION = 'approved-route-trail-release-v1'

    def __init__(self, config=None):
        c = dict(config or {})
        self.enabled = bool(c.get('enabled', False))
        self.strength = float(c.get('release_strength', 1.))
        self.headroom_band = float(c.get('release_headroom_band_mps', 0.))
        self.release_rate = float(c.get('release_rate_mps3', 40.))
        self.floor_margin = float(c.get('profile_floor_margin_mps2', .15))
        self.minimum_speed = float(c.get('minimum_speed_mps', 5.))
        self.preview_seconds = float(c.get('preview_seconds', 4.))
        self.minimum_preview = float(c.get('minimum_preview_m', 40.))
        self.maximum_preview = float(c.get('maximum_preview_m', 700.))
        self.mu = float(c.get('friction_coefficient', 1.5))
        self.grip_factor = float(c.get('grip_safety_factor', .95))
        limits = [(self.headroom_band, 0., 3.), (self.strength, 0., 1.), (self.release_rate, 1., 200.),
            (self.floor_margin, 0., 1.), (self.minimum_speed, 0., 15.),
            (self.preview_seconds, 1., 8.), (self.minimum_preview, 5., 100.),
            (self.maximum_preview, 50., 1500.), (self.mu, .5, 2.),
            (self.grip_factor, .5, 1.)]
        if any(not math.isfinite(v) or not lo <= v <= hi for v, lo, hi in limits):
            raise ValueError('Invalid trail-brake coordinator configuration')
        if self.maximum_preview < self.minimum_preview:
            raise ValueError('Maximum preview must cover minimum preview')
        self._route_key = None
        self._route_owner = None
        self._route_table = None
        self._previous_decel = None

    def _table(self, line):
        """Cache one immutable route; keep physical and original map arc apart."""
        if id(line) == self._route_key:
            return self._route_table
        required = ('s', 'base_ds', 'ds', 'speed', 'length')
        if line is None or not all(hasattr(line, k) for k in required):
            return None
        s, base_ds, ds, speed = (np.asarray(getattr(line, k), float).copy()
                                for k in ('s', 'base_ds', 'ds', 'speed'))
        length = float(line.length)
        if (s.ndim != 1 or len(s) < 3 or any(v.shape != s.shape for v in (base_ds, ds, speed))
                or not math.isfinite(length) or length <= 0.
                or not np.isfinite(np.r_[s, base_ds, ds, speed]).all()
                or abs(s[0]) > 1e-7 or np.any(np.diff(s) <= 0.)
                or np.any(ds <= 0.) or np.any(base_ds <= 0.) or np.any(speed <= 0.)
                or not np.allclose(np.diff(np.r_[s, length]), base_ds, rtol=1e-6, atol=1e-6)):
            return None
        arc = np.r_[0., np.cumsum(ds[:-1])]
        physical_length = float(ds.sum())
        table = (s, base_ds, ds, arc, physical_length,
                 np.r_[arc, physical_length], np.r_[speed, speed[0]])
        self._route_key, self._route_owner, self._route_table = id(line), line, table
        self._previous_decel = None
        return table

    @staticmethod
    def _piecewise_energy_floor(distance, cap, speed):
        """Exact constant-deceleration floor for a linear-speed cap envelope.

        Vertices alone can miss a maximum inside a falling speed segment when
        the vehicle initially lies below the profile. Include that stationary
        point analytically. These are command-model constraints, not forces.
        """
        distance, cap = np.asarray(distance, float), np.asarray(cap, float)
        positive = distance > 1e-8
        candidates = (speed * speed - cap[positive] ** 2) / (2. * distance[positive])
        where = distance[positive]
        caps = cap[positive]
        widths = np.diff(distance)
        slopes = np.diff(cap) / widths
        intercept = cap[:-1] - slopes * distance[:-1]
        possible = (intercept * intercept > speed * speed) & (abs(slopes) > 1e-10)
        critical = np.zeros(len(slopes))
        critical[possible] = np.sqrt((intercept[possible] ** 2 - speed * speed) / slopes[possible] ** 2)
        inside = possible & (critical > distance[:-1] + 1e-8) & (critical < distance[1:] - 1e-8)
        if np.any(inside):
            d = critical[inside]
            v = intercept[inside] + slopes[inside] * d
            candidates = np.r_[candidates, (speed * speed - v * v) / (2. * d)]
            where, caps = np.r_[where, d], np.r_[caps, v]
        i = int(np.argmax(candidates))
        return max(0., float(candidates[i])), float(where[i]), float(caps[i])

    def profile_braking_floor(self, line, progress_m, speed_mps, brake_decel):
        """Inspect unchanged map speed caps over a full braking-distance horizon."""
        table = self._table(line)
        if table is None:
            return None
        s, base_ds, ds, arc, length, extended_arc, extended_cap = table
        wrapped = float(progress_m) % float(line.length)
        index = min(max(int(np.searchsorted(s, wrapped, side='right') - 1), 0), len(s) - 1)
        fraction = float(np.clip((wrapped - s[index]) / base_ds[index], 0., 1.))
        center = float(arc[index] + fraction * ds[index])
        horizon = min(length - 1e-6, self.maximum_preview,
            max(self.minimum_preview, speed_mps * self.preview_seconds,
                speed_mps * speed_mps / (2. * brake_decel) + 10.))
        offsets = (arc - center) % length
        selected = (offsets > 1e-8) & (offsets < horizon - 1e-8)
        distance = np.r_[0., np.sort(offsets[selected]), horizon]
        cap = np.interp((center + distance) % length, extended_arc, extended_cap)
        floor, limiting_distance, limiting_cap = self._piecewise_energy_floor(distance, cap, speed_mps)
        return {'required_decel_mps2':floor, 'limiting_distance_m':limiting_distance,
            'limiting_speed_cap_mps':limiting_cap, 'current_map_cap_mps':float(cap[0]),
            'horizon_physical_m':float(horizon), 'profile_vertex_count':int(len(distance)),
            'physical_arc_center_m':center, 'original_map_progress_m':float(progress_m),
            'interior_energy_extrema_checked':True, 'speed_profile_modified':False,
            'assumption':'Constant deceleration on the approved unchanged piecewise-linear speed profile; updated every physics step'}

    def apply(self, throttle, brake, *, reference, line, progress_m, body_speed_mps,
              forward_speed_mps, target_speed_mps, current_profile_cap_mps,
              tracking_metadata, guard_flags=(), minimum_decel_mps2=0., dt=.02,
              mass=880., wheel_radius=.35, brake_coefficient=700.):
        meta = {'version':self.VERSION, 'enabled':self.enabled, 'shaping_applied':False,
            'input_throttle':float(throttle), 'input_brake':float(brake),
            'command_temporal_filter_used':False, 'steering_or_feedback_modified':False,
            'opponent_ground_truth_used':False, 'grip_proxy_is_measured_tire_force':False,
            'final_stability_and_minimum_decel_overrides_are_external':True}
        def bypass(reason):
            self._previous_decel = None
            meta.update(reason=reason, output_throttle=float(throttle), output_brake=float(brake))
            return throttle, brake, meta
        if not self.enabled:
            return bypass('disabled')
        values = [throttle, brake, progress_m, body_speed_mps, forward_speed_mps,
            target_speed_mps, current_profile_cap_mps, minimum_decel_mps2,
            dt, mass, wheel_radius, brake_coefficient]
        if not all(math.isfinite(float(v)) for v in values) or min(dt, mass, wheel_radius, brake_coefficient) <= 0.:
            return bypass('invalid_input')
        if minimum_decel_mps2 > 0. or guard_flags:
            return bypass('traffic_road_or_mandatory_deceleration_priority')
        m = tracking_metadata or {}
        if not all(math.isfinite(float(m.get(k, default))) for k, default in
                [('selected_path_heading_error_rad', math.inf),
                 ('selected_path_curvature_per_m', math.nan),
                 ('estimated_body_lateral_accel_command_mps2', math.nan),
                 ('path_tracking_speed_factor', 1.)]):
            return bypass('invalid_tracking_metadata')
        if (not all(hasattr(reference, k) for k in ('brake_decel', 'lateral_accel'))
                or not all(math.isfinite(float(getattr(reference, k))) and float(getattr(reference, k)) > 0.
                    for k in ('brake_decel', 'lateral_accel'))):
            return bypass('invalid_planning_budgets')
        ratio = forward_speed_mps / max(body_speed_mps, .1)
        if (body_speed_mps > 3. and (ratio < math.cos(.35)
                or abs(float(m.get('selected_path_heading_error_rad', math.inf))) > .4)):
            return bypass('spin_or_heading_recovery_priority')
        if (m.get('geometry_source') != 'variable_nominal_path'
                or not m.get('nominal_capture') or not m.get('mapped_nominal_route_approval_used')):
            return bypass('selected_route_not_explicitly_approved_captured_nominal')
        if brake <= 0.:
            return bypass('no_requested_brake_no_tail_added')
        if body_speed_mps < self.minimum_speed or target_speed_mps <= .15:
            return bypass('low_speed_or_stop_priority')
        if body_speed_mps >= target_speed_mps:
            return bypass('current_target_braking_priority')
        factor = float(m.get('path_tracking_speed_factor', 1.))
        if target_speed_mps < current_profile_cap_mps * factor - .05:
            return bypass('external_target_speed_constraint_priority')
        envelope = self.profile_braking_floor(line, progress_m, body_speed_mps,
                                            float(reference.brake_decel))
        if envelope is None:
            return bypass('invalid_or_missing_physical_route_table')
        meta['profile_braking_floor'] = envelope
        if (current_profile_cap_mps < envelope['current_map_cap_mps'] - .05
                or body_speed_mps > envelope['current_map_cap_mps']):
            return bypass('current_profile_or_external_cap_priority')
        conversion = 4. * brake_coefficient / (mass * wheel_radius)
        raw_decel = float(brake) * conversion
        floor = envelope['required_decel_mps2']
        if floor > 0.:
            floor += self.floor_margin
        raw_curvature = abs(float(m.get('selected_path_curvature_per_m', 0.)))
        lateral = max(body_speed_mps * body_speed_mps * raw_curvature,
            abs(float(m.get('estimated_body_lateral_accel_command_mps2', 0.))))
        budget = max(.1, float(reference.lateral_accel))
        fraction = float(np.clip(lateral / budget, 0., 1.))
        grip = self.mu * 9.81 * self.grip_factor
        remaining = math.sqrt(max(0., grip * grip - lateral * lateral))
        meta.update(input_decel_mps2=raw_decel, mandatory_profile_floor_mps2=floor,
            lateral_acceleration_proxy_mps2=lateral, corner_release_fraction=fraction,
            grip_proxy_budget_mps2=grip, grip_proxy_longitudinal_remaining_mps2=remaining,
            profile_floor_exceeds_grip_proxy=bool(floor > remaining),
            release_rate_mps3=self.release_rate)
        if floor >= raw_decel:
            return bypass('future_profile_floor_dominates_requested_brake')
        if floor > remaining:
            return bypass('future_profile_floor_cannot_be_relaxed_for_grip_proxy')
        if fraction <= .1 or self.strength <= 0.:
            self._previous_decel = raw_decel
            meta.update(reason='straight_or_no_discretionary_release', output_decel_mps2=raw_decel,
                output_throttle=float(throttle), output_brake=float(brake))
            return throttle, brake, meta
        effective_fraction = fraction
        if self.headroom_band > 0.:
            headroom = target_speed_mps - body_speed_mps
            normalized = float(np.clip(headroom / self.headroom_band, 0., 1.))
            weight = normalized * normalized * (3. - 2. * normalized)
            effective_fraction *= weight
            meta.update(target_speed_headroom_mps=float(headroom),
                release_headroom_band_mps=self.headroom_band,
                headroom_release_weight=weight,
                effective_corner_release_fraction=effective_fraction,
                headroom_taper_is_static=True)
        desired = min(remaining, floor + (raw_decel - floor) * (1. - self.strength * effective_fraction))
        previous = raw_decel if self._previous_decel is None else self._previous_decel
        # Increases/floors are immediate. Only discretionary release is slewed.
        output_decel = min(raw_decel, remaining,
            max(floor, desired, previous - self.release_rate * dt))
        self._previous_decel = output_decel
        output_brake = float(output_decel / conversion)
        output_throttle = 0. if output_brake > 0. else float(throttle)
        meta.update(reason='approved_turn_in_discretionary_brake_release',
            shaping_applied=bool(output_brake < brake - 1e-10),
            desired_decel_mps2=desired, output_decel_mps2=output_decel,
            release_limited=bool(output_decel > max(floor, desired) + 1e-10),
            output_throttle=output_throttle, output_brake=output_brake)
        return output_throttle, output_brake, meta
