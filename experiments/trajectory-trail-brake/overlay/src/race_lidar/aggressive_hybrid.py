"""Measured-corridor passing priority over a smooth nominal racing line.

This is an explicit engineering overlay, not a fitted policy. Original learned
outputs are retained. All traffic inputs are measured hazard rows; no opponent
truth is consulted. The final physical/occupancy guard remains authoritative.
"""
from __future__ import annotations

import math
import numpy as np

from .race_control import predictive_guard, stopping_speed
from .traffic_trajectories import TrafficTrajectoryMap

VERSION = 'measured-free-gap-pass-first-v2'


class AggressiveHybrid:
    def __init__(self, configuration, racing_line=None):
        self.configuration = dict(configuration)
        self.racing_line = racing_line
        self.racing_parameters = dict(configuration.get('racing_guard_parameters', {}))
        self.relax_visibility = bool(configuration.get('relax_blanket_visibility_cap', False))
        self.maximum_lane = float(configuration.get('maximum_lane_m', 4.0))
        self.commit_seconds = float(configuration.get('commit_seconds', 2.5))
        self.switch_gain = float(configuration.get('switch_gain_mps', 4.0))
        self.pass_gain = float(configuration.get('minimum_pass_gain_mps', 1.0))
        self.transient_hold = float(configuration.get('transient_conflict_hold_seconds', .6))
        self.adaptive_gaps = bool(configuration.get('adaptive_gap_passing', False))
        self.profile_preview = configuration.get('profile_preview_m')
        if self.profile_preview is not None:
            self.profile_preview = float(self.profile_preview)
            if not math.isfinite(self.profile_preview) or not 0. <= self.profile_preview <= 100.:
                raise ValueError('Invalid path-profile preview')
        self.gap_tracking_reserve = float(configuration.get('gap_tracking_reserve_m', .20))
        self.gap_minimum_width = float(configuration.get('gap_minimum_center_width_m', .25))
        self.nominal_capture = float(configuration.get('nominal_capture_m', 1.25))
        self.guard_tolerance = float(configuration.get('guard_approval_tolerance_m', .12))
        self.rank_reachable_pace = bool(configuration.get('rank_acceleration_reachable_pace', False))
        self.ranking_horizon = float(configuration.get('ranking_horizon_s', 2.5))
        self.coherent_clearance = bool(configuration.get('spatially_coherent_pass_clearance', False))
        if (not 0. < self.maximum_lane <= 12.0 or not 0. <= self.commit_seconds <= 6.
                or not 0. <= self.transient_hold <= 1. or not 0. <= self.switch_gain <= 10.
                or not .1 <= self.gap_tracking_reserve <= .6
                or not .1 <= self.gap_minimum_width <= 1.
                or not .25 <= self.nominal_capture <= 4.
                or not .01 <= self.guard_tolerance <= 1.
                or not .5 <= self.ranking_horizon <= 5.):
            raise ValueError('Invalid pass-first configuration')
        self.committed_lane = None
        self.committed_until = -math.inf
        self.blocked_since = None
        self.commitment_started = None
        self.blocker_id = None
        self.blocker_lateral = None
        self.blocker_missing_since = None
        self.last_mode = 'nominal_line'
        self.committed_gap = None
        self.lateral_history = {}
        self.clearance_bindings = {}
        trajectory_config = dict(configuration.get('trajectory_planning', {}))
        self.trajectory_planner = TrafficTrajectoryMap(trajectory_config,
            reference=getattr(racing_line, 'reference', None)) if trajectory_config.get('enabled', False) else None

    def _clearance_snapshot(self, row, progress, timestamp):
        point = row.get('position_m')
        point = np.asarray(point, float) if point is not None else None
        if point is not None and (point.shape not in ((2,), (3,)) or not np.isfinite(point).all()):
            point = None
        return {'timestamp': float(timestamp),
            'measurement_time': float(timestamp) - max(0., float(row.get('age_s', 0.))),
            'absolute_progress_m': float(progress + row['relative_distance_m']),
            'position_m': None if point is None else point[:2].tolist(),
            'lateral_m': row['lateral_m'], 'speed_mps': row['speed_mps'],
            'speed_std_mps': row['speed_std_mps'], 'velocity_valid': bool(row.get('velocity_valid', True)),
            'position_std_m': row['position_std_m']}

    def _bind_clearance_episode(self, rows, progress, timestamp):
        ids = set(self.committed_gap.get('bracket_track_ids', [])) if self.committed_gap else set()
        if self.blocker_id is not None:
            ids.add(self.blocker_id)
        self.clearance_bindings = {r['track_id']: {'aliases': [r['track_id']],
            'last': self._clearance_snapshot(r, progress, timestamp)}
            for r in rows if r['track_id'] in ids}

    def _clearance_alias_status(self, row, binding, progress, timestamp):
        last = binding['last']; elapsed = max(0., float(timestamp) - last['timestamp'])
        # A new observation may bridge a normal short tracker rename, not a
        # long unobserved journey to another same-lateral road marker.
        if elapsed > 1. or float(timestamp) - last['measurement_time'] > 1.8:
            return False, 'observation_gap_too_old'
        uncertainty = 4. + min(5., last['position_std_m'] + row['position_std_m'])
        if last['velocity_valid']:
            low, high = last['speed_mps'] - 2. * last['speed_std_mps'], last['speed_mps'] + 2. * last['speed_std_mps']
            motion_bound = max(3., abs(low), abs(high)) + 3.
        else:
            low, high, motion_bound = -100., 100., 100.
        absolute = float(progress + row['relative_distance_m'])
        if not last['absolute_progress_m'] + low * elapsed - uncertainty <= absolute <= last['absolute_progress_m'] + high * elapsed + uncertainty:
            return False, 'incoherent_absolute_progress'
        lateral_budget = 2.4 + 10. * elapsed + min(5., last['position_std_m'] + row['position_std_m'])
        if abs(row['lateral_m'] - last['lateral_m']) > lateral_budget:
            return False, 'incoherent_lateral_motion'
        point = row.get('position_m')
        if point is not None and last['position_m'] is not None:
            point = np.asarray(point, float)
            if point.shape not in ((2,), (3,)) or not np.isfinite(point).all():
                return False, 'invalid_observed_world_position'
            if np.linalg.norm(point[:2] - np.asarray(last['position_m'])) > uncertainty + motion_bound * elapsed:
                return False, 'incoherent_observed_world_motion'
        return True, 'spatially_coherent_observed_track'

    def _coherent_pass_status(self, rows, timestamp, progress):
        if not self.clearance_bindings:
            self._bind_clearance_episode(rows, progress, timestamp)
        matched, rejected = {}, []
        for root_id, binding in self.clearance_bindings.items():
            coherent_rows = []
            for row in rows:
                coherent, reason = self._clearance_alias_status(row, binding, progress, timestamp)
                if coherent:
                    coherent_rows.append(row)
                elif -12. <= row['relative_distance_m'] <= 25. and abs(row['lateral_m'] - binding['last']['lateral_m']) < 2.4:
                    rejected.append({'root_track_id': root_id, 'candidate_track_id': row['track_id'], 'reason': reason})
            if coherent_rows:
                # A stale old ID behind must not hide a fresh renamed car
                # still alongside. Clearance uses the most-ahead coherent row;
                # motion memory updates from the freshest coherent observation.
                matched[root_id] = max(coherent_rows, key=lambda r: r['relative_distance_m'])
                expected = binding['last']['absolute_progress_m'] + (
                    binding['last']['speed_mps'] if binding['last']['velocity_valid'] else 0.) * max(0., timestamp-binding['last']['timestamp'])
                freshest = min(coherent_rows, key=lambda r: (float(r.get('age_s', 0.)),
                    abs(progress+r['relative_distance_m']-expected), r['track_id'] not in binding['aliases']))
                if freshest['track_id'] not in binding['aliases']:
                    binding['aliases'].append(freshest['track_id'])
                binding['last'] = self._clearance_snapshot(freshest, progress, timestamp)
        ids = [r['track_id'] for r in matched.values()]
        if self.clearance_bindings and len(matched) == len(self.clearance_bindings) and all(r['relative_distance_m'] < -12. for r in matched.values()):
            self.blocker_missing_since = None
            return {'unresolved': False, 'physically_cleared': True,
                'reason': 'spatially_bound_pass_blockers_measured_behind', 'matching_track_ids': ids,
                'spatial_binding_used': True, 'rejected_lateral_only_aliases': rejected}
        if any(r['relative_distance_m'] >= -12. for r in matched.values()):
            self.blocker_missing_since = None
            return {'unresolved': True, 'reason': 'spatially_coherent_blocking_or_alongside_track',
                'matching_track_ids': ids, 'spatial_binding_used': True, 'rejected_lateral_only_aliases': rejected}
        self.blocker_missing_since = timestamp if self.blocker_missing_since is None else self.blocker_missing_since
        unresolved = timestamp < self.committed_until or timestamp - self.blocker_missing_since < 1.
        return {'unresolved': unresolved, 'reason': 'measurement_loss_grace' if unresolved else 'spatially_bound_measurement_grace_expired',
            'matching_track_ids': ids, 'spatial_binding_used': True, 'rejected_lateral_only_aliases': rejected}

    def _pass_status(self, rows, timestamp, progress=0.):
        if self.committed_lane is None:
            return {'unresolved': False, 'reason': 'no_committed_pass', 'matching_track_ids': []}
        if self.coherent_clearance:
            return self._coherent_pass_status(rows, timestamp, progress)
        ids_to_clear = set(self.committed_gap.get('bracket_track_ids', [])) if self.committed_gap else set()
        if self.blocker_id is not None:
            ids_to_clear.add(self.blocker_id)
        matching = [r for r in rows if r['track_id'] in ids_to_clear]
        # A measured clearance ends a commitment even before its hysteresis
        # timer: holding a completed pass would needlessly miss the next apex.
        if ids_to_clear and ids_to_clear <= {r['track_id'] for r in matching} and all(r['relative_distance_m'] < -12. for r in matching):
            self.blocker_missing_since = None
            return {'unresolved': False, 'physically_cleared': True,
                    'reason': 'all_measured_pass_blockers_clearly_behind',
                    'matching_track_ids': [r['track_id'] for r in matching]}
        # Vision/geometry can duplicate or rename the same car. Nearby measured
        # traffic in the original blocking corridor also delays a cut-back.
        blocker_laterals = self.committed_gap.get('bracket_lateral_m', []) if self.committed_gap else []
        blocker_laterals = [*blocker_laterals, *([self.blocker_lateral] if self.blocker_lateral is not None else [])]
        near = [r for r in rows if any(abs(r['lateral_m'] - x) < 2.4 for x in blocker_laterals)
                and -12. <= r['relative_distance_m'] <= 25.]
        ids = [r['track_id'] for r in [*matching, *near]]
        if any(r['relative_distance_m'] >= -12. for r in matching) or near:
            self.blocker_missing_since = None
            return {'unresolved': True, 'reason': 'blocking_or_alongside_traffic_not_cleared', 'matching_track_ids': ids}
        self.blocker_missing_since = timestamp if self.blocker_missing_since is None else self.blocker_missing_since
        unresolved = timestamp < self.committed_until or timestamp - self.blocker_missing_since < 1.
        return {'unresolved': unresolved, 'reason': 'measurement_loss_grace' if unresolved else 'corridor_clear_after_measurement_grace',
                'matching_track_ids': ids}

    def _lateral_motion(self, rows, timestamp):
        """Estimate lateral drift from repeated measured tracks, never truth."""
        motion = {}
        for row in rows:
            key = row['track_id']
            old = self.lateral_history.get(key)
            age = max(0., float(row.get('age_s', 0.)))
            stamp = float(timestamp) - min(2., age)
            if old is not None and .08 <= stamp - old[0] <= 1.5:
                rate = float((row['lateral_m'] - old[1]) / (stamp - old[0]))
                motion[key] = {'lateral_velocity_mps': rate, 'observed': True}
            elif old is not None and float(timestamp) - old[0] <= .8:
                motion[key] = {'lateral_velocity_mps': old[2], 'observed': old[3]}
            else:
                motion[key] = {'lateral_velocity_mps': 0., 'observed': False}
            if old is None or stamp > old[0]:
                self.lateral_history[key] = (stamp, row['lateral_m'],
                    motion[key]['lateral_velocity_mps'], motion[key]['observed'])
        self.lateral_history = {k: v for k, v in self.lateral_history.items() if float(timestamp) - v[0] <= 2.}
        return motion

    def _center_bounds(self, progress, preview=0.):
        if self.racing_line is not None and hasattr(self.racing_line, 'center_bounds'):
            lo, hi = self.racing_line.center_bounds(float(progress), preview_m=float(preview))
            lo, hi = max(-self.maximum_lane, float(lo)), min(self.maximum_lane, float(hi))
            if not math.isfinite(lo + hi) or lo >= hi:
                raise ValueError('Invalid mapped wheel-rule center bounds')
            return lo, hi
        return -self.maximum_lane, self.maximum_lane

    def _nominal_road_status(self, progress, lookahead, actual, ego_position, ego_yaw):
        """Bind a frozen variable path to exact map and measured own-wheel pose.

        Constant-heading center bounds conservatively prefilter parallel lanes.
        They do not define legality for a variable path with another tangent.
        """
        line = self.racing_line
        road = getattr(line, 'road', None)
        geometry_hash = getattr(line, 'road_geometry_sha256', None)
        bound = bool(road is not None and geometry_hash
            and getattr(road, 'geometry_sha256', None) == geometry_hash
            and getattr(line, 'plan_artifact_sha256', None)
            and getattr(line, 'optimization_executed_this_instance', None) is False
            and getattr(line, 'path_geometry_representation', None)
                == 'smooth_xy_reindexed_by_original_ordered_projection')
        status = {'frozen_road_binding_verified': bound, 'current_nominal_wheel_valid': False,
            'future_nominal_wheel_valid': False, 'own_wheel_pose_checked': False,
            'own_two_tyre_rule_valid': False, 'captured_legal_nominal': False,
            'nominal_offset_here_m': None, 'current_nominal_capture_error_m': None}
        if not bound:
            return status
        current_point, current_heading, _, _ = line.sample(progress)
        future_point, future_heading, _, _ = line.sample(progress + lookahead)
        current = road.tyre_contact_status(current_point, current_heading, progress_m=progress)
        future = road.tyre_contact_status(future_point, future_heading, progress_m=progress + lookahead)
        status.update(current_nominal_wheel_valid=bool(current['requested_two_tyre_rule_valid']),
            future_nominal_wheel_valid=bool(future['requested_two_tyre_rule_valid']),
            nominal_offset_here_m=float(line.nominal_offset_m(progress)))
        status['current_nominal_capture_error_m'] = abs(actual - status['nominal_offset_here_m'])
        if ego_position is not None and ego_yaw is not None:
            point = np.asarray(ego_position, float)
            if point.shape not in ((2,), (3,)) or not np.isfinite(point).all() or not math.isfinite(float(ego_yaw)):
                raise ValueError('Exact measured own pose must be finite XY and yaw')
            own = road.tyre_contact_status(point[:2], float(ego_yaw), progress_m=progress)
            status.update(own_wheel_pose_checked=True,
                own_two_tyre_rule_valid=bool(own['requested_two_tyre_rule_valid']),
                own_on_track_count=int(own['on_track_count']))
        status['captured_legal_nominal'] = bool(status['current_nominal_wheel_valid']
            and status['own_two_tyre_rule_valid']
            and status['current_nominal_capture_error_m'] <= self.nominal_capture)
        return status

    @staticmethod
    def _vehicle_row(row):
        # The detector is a car-only VoxelNeXt channel. Raw geometry, even a
        # persistent vehicle-shaped cluster, remains an unclassified obstacle.
        if row.get('source') == 'raw_lidar_geometry' or row.get('object_class') == 'unclassified_obstacle':
            return False
        try:
            score = float(row.get('score', 0.))
        except (TypeError, ValueError):
            return False
        return bool(math.isfinite(score) and score >= .35 and (
            row.get('object_class') in ('car', 'vehicle') or row.get('confirmed')
            or row.get('tentative') or row.get('control_evidence')))

    def _vehicle_bracket_evidence(self, brackets, rows):
        vehicles = [r for r in rows if self._vehicle_row(r)]
        return list(dict.fromkeys(v['track_id'] for b in brackets for v in vehicles
            if v['track_id'] == b['track_id'] or (
                abs(v['relative_distance_m'] - b['relative_distance_m']) <= 3.
                and abs(v['lateral_m'] - b['lateral_m']) <= .8)))

    def _gap_specs(self, rows, body, actual, nominal, bounds):
        """Free center intervals after vehicle width, uncertainty and reserve."""
        half = float(self.racing_parameters.get('ego_half_width_m', 1.05))
        half += float(self.racing_parameters.get('lateral_margin_m', .10)) + self.gap_tracking_reserve
        ucap = float(self.racing_parameters.get('position_uncertainty_cap_m', .5))
        horizon = min(100., max(40., body * 3. + 10.))
        lower_bound, upper_bound = bounds
        spans = []
        for row in rows:
            if not -12. <= row['relative_distance_m'] <= horizon:
                continue
            inflate = half + min(ucap, max(.15, row['position_std_m']))
            lo, hi = row['lateral_min_m'] - inflate, row['lateral_max_m'] + inflate
            if hi < lower_bound or lo > upper_bound:
                continue
            spans.append({'lo': max(lower_bound, lo), 'hi': min(upper_bound, hi),
                          'lower_rows': [row], 'upper_rows': [row]})
        spans.sort(key=lambda x: x['lo'])
        merged = []
        for span in spans:
            if merged and span['lo'] <= merged[-1]['hi']:
                previous = merged[-1]
                if span['lo'] == previous['lo']:
                    previous['lower_rows'].extend(span['lower_rows'])
                if span['hi'] > previous['hi']:
                    previous['hi'], previous['upper_rows'] = span['hi'], span['upper_rows']
                elif span['hi'] == previous['hi']:
                    previous['upper_rows'].extend(span['upper_rows'])
            else:
                merged.append(span)
        free, edge, right_rows = [], lower_bound, []
        for span in merged:
            if span['lo'] - edge >= self.gap_minimum_width:
                free.append((edge, span['lo'], right_rows, span['lower_rows']))
            edge, right_rows = span['hi'], span['upper_rows']
        if merged and upper_bound - edge >= self.gap_minimum_width:
            free.append((edge, upper_bound, right_rows, []))
        specs = []
        for index, (lo, hi, right, left) in enumerate(free):
            lane = .5 * (lo + hi)
            brackets = [*right, *left]
            alongside_pair = bool(right and left and min(abs(r['relative_distance_m'] - l['relative_distance_m'])
                for r in right for l in left) <= 12.)
            right_vehicles = self._vehicle_bracket_evidence(right, rows)
            left_vehicles = self._vehicle_bracket_evidence(left, rows)
            vehicle_pair = bool(alongside_pair and right_vehicles and left_vehicles)
            metadata = {'center_interval_m': [float(lo), float(hi)],
                'center_interval_width_m': float(hi - lo),
                'right_bracket_track_ids': list(dict.fromkeys(r['track_id'] for r in right)),
                'left_bracket_track_ids': list(dict.fromkeys(r['track_id'] for r in left)),
                'bracket_track_ids': list(dict.fromkeys(r['track_id'] for r in brackets)),
                'bracket_lateral_m': [r['lateral_m'] for r in brackets],
                'between_longitudinally_near_pair': alongside_pair,
                'right_vehicle_track_ids': right_vehicles, 'left_vehicle_track_ids': left_vehicles,
                'vehicle_brackets_verified': vehicle_pair,
                'ego_tracking_reserve_m': self.gap_tracking_reserve,
                'mapped_center_bounds_m': [lower_bound, upper_bound],
                'planning_intent_only': True, 'completed_middle_pass_claimed': False}
            specs.append({'corridor_id': f'adaptive_gap_{index}',
                'mode': 'pass_between' if vehicle_pair else 'pass_gap',
                'lane': float(lane), 'profile_lane': float(lane), 'gap': metadata})
        return specs

    def _gap_sweep(self, lane, rows, motion, body, actual, cap):
        """Check a whole-car transition, including observed lateral closure.

        Longitudinal overlap uses an acceleration/braking envelope; a clear
        endpoint alone never authorizes crossing an occupied intermediate lane.
        This is a short-horizon measured check, not a free-space guarantee.
        """
        rate = float(self.racing_parameters.get('lane_rate_mps', 1.8))
        duration = min(4., max(1.5, abs(lane - actual) / rate + .6))
        times = np.linspace(0., duration, 33)
        planned = actual + np.sign(lane - actual) * np.minimum(abs(lane - actual), rate * np.maximum(0., times - .35))
        # The lower bound assumes hard braking; the upper bound allows normal
        # acceleration through the target speed, conservatively capped there.
        stop_time = body / 8.
        braking_t = np.minimum(times, stop_time)
        slow = body * braking_t - 4. * braking_t ** 2
        ceiling = max(body, cap)
        accel_t = np.minimum(times, max(0., (ceiling - body) / 5.))
        fast = body * accel_t + 2.5 * accel_t ** 2 + ceiling * np.maximum(0., times - accel_t)
        half = float(self.racing_parameters.get('ego_half_width_m', 1.05))
        half += float(self.racing_parameters.get('lateral_margin_m', .10)) + self.gap_tracking_reserve
        ucap = float(self.racing_parameters.get('position_uncertainty_cap_m', .5))
        conflicts = []
        observed = []
        for row in rows:
            d, velocity = row['relative_distance_m'], row['speed_mps']
            sigma = max(1., row['speed_std_mps'])
            low, high = (velocity - sigma, velocity + sigma) if row.get('velocity_valid', True) else (0., 70.)
            near = (d + low * times - fast <= 8.5) & (d + high * times - slow >= -8.5)
            m = motion[row['track_id']]
            drift = m['lateral_velocity_mps'] * times
            # Do not credit a guessed departure for opening a route. The
            # envelope includes the measured footprint and its extrapolation.
            lower = row['lateral_min_m'] + np.minimum(0., drift)
            upper = row['lateral_max_m'] + np.maximum(0., drift)
            clearance = np.maximum(np.maximum(lower - planned, planned - upper), 0.)
            required = half + min(ucap, max(.15, row['position_std_m']))
            if np.any(near & (clearance < required)):
                conflicts.append(row['track_id'])
            if m['observed']:
                observed.append(row['track_id'])
        return {'clear': not conflicts, 'conflicting_track_ids': conflicts,
                'observed_lateral_motion_track_ids': observed,
                'duration_s': duration, 'samples': len(times),
                'ego_tracking_reserve_m': self.gap_tracking_reserve,
                'acceleration_envelope_checked': True,
                'planning_intent_only': True}

    def _rows(self, state, *, trajectory_rows=False):
        rows = state.get('hazard_tracks')
        if not isinstance(rows, (list, tuple)):
            raise ValueError('Measured hazard_tracks are required')
        out = []
        for i, row in enumerate(rows):
            try:
                d, lateral = float(row['relative_distance_m']), float(row['lateral_m'])
                velocity = float(row.get('speed_mps', 0.))
                std = float(row.get('position_std_m', row.get('position_uncertainty_m', .35)))
                vstd = float(row.get('speed_std_mps', row.get('speed_uncertainty_mps', 2.)))
                lower = min(float(row.get('lateral_min_m', lateral - 1.05)), lateral - 1.05)
                upper = max(float(row.get('lateral_max_m', lateral + 1.05)), lateral + 1.05)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite([d, lateral, velocity, std, vstd, lower, upper]).all():
                continue
            if not trajectory_rows and (not -100. <= d <= 150. or abs(lateral) > max(10., self.maximum_lane + 3.)):
                continue
            out.append({**row, 'relative_distance_m': d, 'lateral_m': lateral,
                'speed_mps': velocity, 'position_std_m': max(0., std),
                'speed_std_mps': max(0., vstd), 'lateral_min_m': lower,
                'lateral_max_m': upper, 'track_id': row.get('track_id', f'row-{i}')})
        return out

    def _summary(self, lane, rows, cap, body_speed):
        half = float(self.racing_parameters.get('ego_half_width_m', 1.05))
        half += float(self.racing_parameters.get('lateral_margin_m', .10))
        ucap = float(self.racing_parameters.get('position_uncertainty_cap_m', .5))
        gap_margin = float(self.racing_parameters.get('following_gap_m', 5.))
        reaction = float(self.racing_parameters.get('reaction_seconds', .2))
        sigma_factor = float(self.racing_parameters.get('lead_speed_sigma_factor', .25))
        occupying = []
        for row in rows:
            uncertainty = min(ucap, max(.15, row['position_std_m']))
            if row['lateral_min_m'] - uncertainty <= lane + half and row['lateral_max_m'] + uncertainty >= lane - half:
                occupying.append(row)
        front = [r for r in occupying if r['relative_distance_m'] >= -5.]
        rear = [r for r in occupying if r['relative_distance_m'] < -5.]
        def budget(row):
            u = min(ucap, max(.15, row['position_std_m']))
            velocity = row['speed_mps'] - sigma_factor * row['speed_std_mps'] if row.get('velocity_valid', True) else 0.
            return stopping_speed(row['relative_distance_m'] - gap_margin - u,
                velocity, decel=7., reaction=reaction)
        lead = min(front, key=budget) if front else None
        behind = max(rear, key=lambda r: r['relative_distance_m']) if rear else None
        speed = min(cap, budget(lead)) if lead else cap
        return {'front_gap_m': max(0., lead['relative_distance_m']) if lead else 150.,
            'front_speed_mps': lead['speed_mps'] if lead else body_speed,
            'front_position_std_m': min(ucap, lead['position_std_m']) if lead else 0.,
            'front_track_id': lead['track_id'] if lead else None,
            'front_lateral_m': lead['lateral_m'] if lead else None,
            'rear_gap_m': abs(behind['relative_distance_m']) if behind else 100.,
            'rear_closing_speed_mps': max(0., behind['speed_mps'] - body_speed) if behind else 0.,
            'destination_speed_budget_mps': float(speed)}

    def _profile_cap(self, progress, preview, lane, supplied_cap):
        if self.racing_line is None:
            return float(supplied_cap)
        # Measured tracking overshoot is not a requested path. Query the edge
        # profile without concealing the true lateral coordinate from guards.
        profile_lane = None if lane is None else float(np.clip(lane, -self.maximum_lane, self.maximum_lane))
        return float(self.racing_line.feasible_speed(progress, preview, lateral_offset=profile_lane))

    def _constant_path_status(self, progress, body, lane):
        reference = getattr(self.racing_line, 'reference', None)
        if lane is None or reference is None:
            return {'checked': False, 'valid': True,
                    'reason': 'variable_nominal_path' if lane is None else 'no_reference_geometry'}
        preview = max(20., .8 * body)
        ahead = np.mod(np.asarray(reference.s, float) - float(progress), float(reference.length))
        curvatures = np.r_[np.asarray(reference.curvature, float)[ahead <= preview],
            reference.sample(progress)[3], reference.sample(progress + preview)[3]]
        scale = float(np.min(1. - float(lane) * curvatures))
        return {'checked': True, 'valid': math.isfinite(scale) and scale > .20,
                'minimum_tangent_scale': scale, 'minimum_accepted_scale': .20,
                'preview_m': preview, 'reason': 'constant_offset_parallel_curve_forward_orientation'}

    def guard_profile_cap(self, progress, preview, raw_action, actual_lateral, supplied_cap,
                          *, guarded_lane=None, guard_flags=(), mapped_nominal_route_approved=False):
        """Keep the actual lane's corner budget until a wide transition aligns."""
        target = float(raw_action.get('target_offset_m', actual_lateral))
        profile_lane = raw_action.get('path_profile_lane_m')
        destination = self._profile_cap(progress, preview, profile_lane, supplied_cap)
        current = self._profile_cap(progress, preview, actual_lateral, supplied_cap)
        nominal = (self.racing_line is not None and raw_action.get('path_mode') == 'nominal_line'
                   and profile_lane is None)
        capture_target = float(self.racing_line.nominal_offset_m(progress)) if nominal else target
        capture_error = abs(float(actual_lateral) - capture_target)
        scalar_approved = (guarded_lane is None or abs(float(guarded_lane) - target) <= self.guard_tolerance)
        mapped_approval = bool(nominal and capture_error <= self.nominal_capture
                               and mapped_nominal_route_approved)
        # A slewed scalar FUTURE target does not describe the mapped route's
        # current lateral velocity. Only an explicit independently checked
        # route proof may supersede that scalar mismatch; real vetoes remain.
        approved = scalar_approved or mapped_approval
        approved = approved and not any(f in guard_flags for f in (
            'predicted_lane_conflict', 'alongside_lane_occupied', 'mapped_nominal_route_conflict'))
        transitioning = capture_error > (self.nominal_capture if nominal else 1.) or not approved
        used = min(destination, current) if transitioning else destination
        return used, {'destination_path_cap_mps': destination,
            'actual_lane_path_cap_mps': current, 'transition_limit_active': transitioning,
            'used_path_cap_mps': used, 'actual_lateral_m': float(actual_lateral),
            'target_absolute_lane_m': target,
            'nominal_current_progress_capture_used': nominal,
            'capture_reference_offset_m': capture_target, 'capture_error_m': capture_error,
            'nominal_capture_threshold_m': self.nominal_capture if nominal else 1.,
            'guard_lane_approval_checked': guarded_lane is not None,
            'scalar_lane_parameter_approved': scalar_approved,
            'mapped_nominal_route_approval_supplied': bool(mapped_nominal_route_approved),
            'mapped_nominal_route_approval_used': bool(mapped_approval and approved),
            'selected_route_guard_approved': approved}

    def apply(self, decision, state, observation, feasible_speed, *, timestamp,
              dt, brake_decel=8., visible_distance=70., progress_m=None,
              ego_position=None, ego_yaw=None):
        original = dict(decision['raw_action'])
        speeds = np.asarray(original.get('lane_target_speeds_mps'), float)
        probabilities = np.asarray(original.get('lane_probabilities'), float)
        if speeds.shape != (3,) or probabilities.shape != (3,) or not np.isfinite(np.r_[speeds, probabilities, feasible_speed, timestamp]).all():
            raise ValueError('Finite three-lane learned outputs are required')
        current = float(state.get('last_target_offset_m', 0.))
        actual = float(state.get('cross_track_m', current))
        body = max(0., float(state.get('body_speed_mps', 0.)))
        progress = float(progress_m if progress_m is not None else state.get('progress_m', 0.))
        preview = 3. + body if self.profile_preview is None else self.profile_preview
        lookahead = float(np.clip(4.5 + body * .28, 5., 20.))
        nominal = float(self.racing_line.nominal_offset_m(progress + lookahead)) if self.racing_line is not None else 0.
        nominal_bounds = self._center_bounds(progress + lookahead)
        nominal_road = self._nominal_road_status(progress, lookahead, actual, ego_position, ego_yaw)
        if nominal_road['future_nominal_wheel_valid']:
            nominal = float(np.clip(nominal, -self.maximum_lane, self.maximum_lane))
        else:
            nominal = float(np.clip(nominal, *nominal_bounds))
        current_bounds = self._center_bounds(progress)
        passing_bounds = self._center_bounds(progress, min(100., max(15., body * 1.5 + 10.)))
        rows = self._rows(state)
        trajectory_prediction = None
        if self.trajectory_planner is not None:
            # Confirmed coasted identities also stay mapped, without being
            # promoted to fresh hazards or being used to open a corridor.
            measured = self._rows({'hazard_tracks': [*observation.get('hazard_tracks', state.get('hazard_tracks', [])),
                *observation.get('confirmed_tracks', [])]}, trajectory_rows=True)
            trajectory_prediction = self.trajectory_planner.update(measured,
                progress_m=progress, timestamp=timestamp)
        lateral_motion = self._lateral_motion(rows, float(timestamp)) if self.adaptive_gaps else {}
        nominal_cap = self._profile_cap(progress, preview, None, feasible_speed)
        current_cap = self._profile_cap(progress, preview, actual, feasible_speed)
        current_summary = self._summary(actual, rows, current_cap, body)
        nominal_summary = self._summary(nominal, rows, nominal_cap, body)
        relevant = min((current_summary, nominal_summary), key=lambda s: s['destination_speed_budget_mps'])
        anticipation = max(45., body * 4. + 10.)
        pass_needed = (relevant['front_track_id'] is not None
            and relevant['front_gap_m'] < min(150., anticipation)
            and relevant['destination_speed_budget_mps'] < nominal_cap - self.pass_gain)
        pass_status = self._pass_status(rows, float(timestamp), progress)

        right, left = passing_bounds
        specs = [{'corridor_id': 'nominal_line', 'mode': 'nominal_line', 'lane': nominal, 'profile_lane': None, 'gap': None},
            *[{'corridor_id': name, 'mode': name, 'lane': lane, 'profile_lane': lane, 'gap': None}
              for name, lane in [('pass_right', right), ('pass_center', float(np.clip(0., right, left))), ('pass_left', left)]]]
        gap_specs = self._gap_specs(rows, body, actual, nominal, passing_bounds) if self.adaptive_gaps else []
        for gap in gap_specs:
            existing = next((s for s in specs[1:] if abs(s['lane'] - gap['lane']) < .05), None)
            if existing is not None:
                existing.update(mode=gap['mode'], gap=gap['gap'])
            else:
                specs.append(gap)
        if self.committed_lane is not None:
            existing = next((s for s in specs[1:] if abs(s['lane'] - self.committed_lane) < .05), None)
            if existing is None:
                specs.append({'corridor_id': 'committed_pass', 'mode': self.last_mode,
                    'lane': self.committed_lane, 'profile_lane': self.committed_lane, 'gap': self.committed_gap})
            elif self.committed_gap is not None:
                existing['gap'] = existing['gap'] or self.committed_gap
                existing['mode'] = 'pass_between' if existing['gap'].get('vehicle_brackets_verified') else 'pass_gap'
        candidates = []
        for spec in specs:
            corridor, lane, profile_lane = spec['corridor_id'], spec['lane'], spec['profile_lane']
            geometry = self._constant_path_status(progress, body, profile_lane)
            cap = self._profile_cap(progress, preview, profile_lane, feasible_speed) if geometry['valid'] else 0.
            summary = self._summary(lane, rows, cap, body)
            request = {**original, 'target_speed_mps': cap, 'target_offset_m': lane}
            guard = predictive_guard(request, observation, body_speed=body,
                current_lane=current, actual_lateral=actual, feasible_speed=cap, dt=dt,
                brake_decel=brake_decel, visible_distance=visible_distance,
                maximum_lane=self.maximum_lane,
                enforce_visibility_stopping_budget=not self.relax_visibility,
                racing_parameters=self.racing_parameters)
            rejects = []
            if not geometry['valid']:
                rejects.append('collapsed_or_reversed_parallel_path')
            if not passing_bounds[0] <= lane <= passing_bounds[1] and spec['mode'] != 'nominal_line':
                rejects.append('outside_mapped_center_bounds')
            if 'predicted_lane_conflict' in guard['guards']:
                rejects.append('transition_sweep_occupied')
            transition_time = abs(lane - actual) / float(self.racing_parameters.get('lane_rate_mps', 1.8)) + .2
            rear_budget = 6. + transition_time * summary['rear_closing_speed_mps']
            if abs(lane - actual) > .8 and summary['rear_gap_m'] < rear_budget:
                rejects.append('closing_rear_vehicle')
            if summary['front_gap_m'] <= float(self.racing_parameters.get('following_gap_m', 5.)) + summary['front_position_std_m']:
                rejects.append('destination_front_occupied')
            sweep = self._gap_sweep(lane, rows, lateral_motion, body, actual, cap) if spec['gap'] is not None else None
            if sweep is not None and not sweep['clear']:
                rejects.append('gap_vehicle_sweep_or_measured_closure')
            legacy_eligible = not rejects
            trajectory_sweep = None
            if self.trajectory_planner is not None:
                trajectory_sweep = self.trajectory_planner.assess(lane, actual_lateral=actual,
                    body_speed=body, speed_cap=min(cap, summary['destination_speed_budget_mps']),
                    lane_rate=float(self.racing_parameters.get('lane_rate_mps', 1.8)),
                    nominal_line=self.racing_line if profile_lane is None else None,
                    nominal_captured=bool(profile_lane is None and nominal_road['captured_legal_nominal']),
                    acceleration=float(getattr(getattr(self.racing_line, 'reference', None), 'acceleration', 5.)),
                    brake_decel=brake_decel)
                if trajectory_sweep['hard_veto']:
                    rejects.append('fresh_measured_trajectory_imminent_conflict')
            learned_speed = float(np.interp(lane, [-3.2, 0., 3.2], speeds))
            learned_prior = float(np.interp(lane, [-3.2, 0., 3.2], probabilities))
            acceleration = float(getattr(getattr(self.racing_line, 'reference', None), 'acceleration', 5.))
            reachable = body + max(0., acceleration) * self.ranking_horizon
            ranking_speed = min(summary['destination_speed_budget_mps'], reachable) if self.rank_reachable_pace else summary['destination_speed_budget_mps']
            utility = (ranking_speed + .025 * learned_speed
                + .05 * learned_prior - .15 * abs(lane - actual) - .10 * abs(lane - nominal))
            if trajectory_sweep is not None:
                utility -= trajectory_sweep['risk_cost_mps']
            candidates.append({'corridor_id': corridor, 'lane_offset_m': float(lane),
                'mode': spec['mode'], 'measured_gap': spec['gap'], 'gap_sweep': sweep,
                'parallel_path_geometry': geometry,
                'profile_lane_m': profile_lane, 'path_speed_cap_mps': cap,
                'summary': summary, 'eligible': not rejects, 'rejections': rejects,
                'guard_preview': guard, 'utility': utility,
                'ranking_speed_budget_mps': ranking_speed,
                'acceleration_reachable_speed_mps': reachable,
                'learned_lane_speed_mps': learned_speed, 'learned_lane_probability': learned_prior})
            if trajectory_sweep is not None:
                candidates[-1].update(legacy_eligible_before_trajectory=legacy_eligible,
                    trajectory_sweep=trajectory_sweep)

        nominal_candidate = candidates[0]
        current_candidate = next((r for r in candidates if r['profile_lane_m'] is not None and self.committed_lane is not None
                                  and abs(r['lane_offset_m'] - self.committed_lane) < .05), None)
        eligible_passes = [c for c in candidates[1:] if c['eligible']]
        best_pass = max(eligible_passes, key=lambda c: (c['utility'], -abs(c['lane_offset_m'] - actual)), default=None)
        selected = None
        reasons = []
        pass_sequence_active = not pass_status.get('physically_cleared', False) and (
            timestamp < self.committed_until or pass_status['unresolved'] or pass_needed)
        if current_candidate is not None and pass_sequence_active:
            if current_candidate['eligible']:
                self.blocked_since = None
                new_blocker = (current_candidate['summary']['front_track_id'] is not None
                    and current_candidate['summary']['front_track_id'] != self.blocker_id
                    and current_candidate['summary']['front_gap_m'] < anticipation
                    and current_candidate['summary']['destination_speed_budget_mps'] < current_candidate['path_speed_cap_mps'] - self.pass_gain)
                if (best_pass is None or best_pass['utility'] < current_candidate['utility'] + self.switch_gain
                        or (pass_status['unresolved'] and not new_blocker)):
                    selected = current_candidate
                    reasons.append('retain_pass_until_measured_clearance' if pass_status['unresolved'] else 'retain_pass_commitment')
            elif current_candidate['rejections'] == ['transition_sweep_occupied']:
                self.blocked_since = timestamp if self.blocked_since is None else self.blocked_since
                if timestamp - self.blocked_since <= self.transient_hold:
                    selected = current_candidate
                    reasons.append('brief_guarded_wait_for_transition_clearance')
        if selected is None and pass_needed and best_pass is not None:
            if best_pass['summary']['destination_speed_budget_mps'] >= relevant['destination_speed_budget_mps'] + self.pass_gain:
                selected = best_pass
                reasons.append('clear_pass_preferred_to_following_brake')
        if selected is None and nominal_candidate['eligible']:
            selected = nominal_candidate
            reasons.append('follow_smooth_nominal_racing_line')
        if selected is None and best_pass is not None:
            selected = best_pass
            reasons.append('choose_available_measured_corridor')

        outside_constant_heading_bounds = actual < current_bounds[0] - .10 or actual > current_bounds[1] + .10
        nominal_road_exemption = bool(nominal_road['captured_legal_nominal'])
        outside_planning_corridor = outside_constant_heading_bounds and not nominal_road_exemption
        recovery_pace_factor = 1.
        if outside_planning_corridor:
            edge = float(np.clip(actual, *current_bounds))
            overshoot = abs(actual - edge)
            recovery_pace_factor = float(np.clip(1. - .20 * overshoot, .55, 1.))
            recovery_cap = self._profile_cap(progress, preview, edge, feasible_speed) * recovery_pace_factor
            if overshoot >= 1.4:
                recovery_cap = min(20., recovery_cap)
            selected = {'corridor_id': 'corridor_recovery', 'lane_offset_m': edge,
                'profile_lane_m': edge,
                'path_speed_cap_mps': recovery_cap}
            reasons.append('measured_outside_planning_corridor_use_edge_profile_recovery')
            if self.trajectory_planner is not None:
                recovery_sweep = self.trajectory_planner.assess(edge, actual_lateral=actual,
                    body_speed=body, speed_cap=recovery_cap,
                    lane_rate=float(self.racing_parameters.get('lane_rate_mps', 1.8)),
                    brake_decel=brake_decel)
                recovery_candidate = {**selected, 'mode':'corridor_recovery',
                    'eligible':not recovery_sweep['hard_veto'],
                    'legacy_eligible_before_trajectory':True,
                    'rejections':['fresh_measured_trajectory_imminent_conflict'] if recovery_sweep['hard_veto'] else [],
                    'trajectory_sweep':recovery_sweep,'utility':-1e6}
                candidates.append(recovery_candidate)
                if recovery_sweep['hard_veto']:
                    safe = [c for c in candidates[:-1] if c['eligible']]
                    selected = min(safe, key=lambda c:(abs(c['lane_offset_m']-actual),-c['utility']), default=None)
                    if selected is not None:
                        # Preserve the old recovery speed ceiling even when an
                        # independently assessed alternate recovery path wins.
                        selected = {**selected,'path_speed_cap_mps':min(recovery_cap,selected['path_speed_cap_mps'])}
                    reasons.append('recovery_trajectory_veto_use_assessed_safe_alternative_or_hold')

        modified = dict(original)
        if selected is None:
            # Holding the present lane avoids a blind reversal when every route
            # is blocked. Final stopping guard may still request braking.
            mode, selected_lane, selected_profile = 'follow_blocked', float(np.clip(current, *current_bounds)), float(np.clip(actual, *current_bounds))
            path_cap = current_cap
            reasons.append('no_clear_pass_hold_lane_and_brake_if_needed')
        else:
            selected_lane = selected['lane_offset_m']
            selected_profile = selected['profile_lane_m']
            path_cap = selected['path_speed_cap_mps']
            mode = selected.get('mode', selected['corridor_id'])
            if mode not in ('nominal_line', 'corridor_recovery'):
                if self.committed_lane is None or abs(self.committed_lane - selected_lane) > .05 or pass_status.get('physically_cleared', False):
                    self.committed_lane = selected_lane
                    self.commitment_started = float(timestamp)
                    self.committed_until = float(timestamp) + self.commit_seconds
                    self.blocker_id = relevant['front_track_id']
                    self.blocker_lateral = relevant['front_lateral_m']
                    self.blocker_missing_since = None
                    self.blocked_since = None
                    self.committed_gap = selected.get('measured_gap')
                    if self.coherent_clearance:
                        self._bind_clearance_episode(rows, progress, timestamp)
                elif self.committed_gap is None and selected.get('measured_gap') is not None:
                    self.committed_gap = selected['measured_gap']
                elif selected.get('measured_gap') is not None and selected['measured_gap'].get('vehicle_brackets_verified'):
                    self.committed_gap = selected['measured_gap']
            elif mode == 'corridor_recovery' or timestamp >= self.committed_until or pass_status.get('physically_cleared', False):
                self.committed_lane, self.blocker_id, self.blocker_lateral = None, None, None
                self.blocker_missing_since = None
                self.committed_gap = None
                self.clearance_bindings = {}
        modified.update(target_speed_mps=float(path_cap), target_offset_m=float(selected_lane),
            hybrid_overlay_used=True, path_mode=mode, path_profile_lane_m=selected_profile,
            path_speed_cap_mps=float(path_cap), nominal_racing_line_offset_m=nominal,
            measured_gap_intent=selected.get('measured_gap') if selected else None)
        trajectory_emergency = False
        trajectory_hold = None
        if self.trajectory_planner is not None and selected is None:
            previously_clear = [c for c in candidates if c['legacy_eligible_before_trajectory']]
            all_imminent = bool(previously_clear and all(c['trajectory_sweep']['hard_veto'] for c in previously_clear))
            if all_imminent:
                trajectory_hold = self.trajectory_planner.assess(actual, actual_lateral=actual,
                    body_speed=body, speed_cap=current_cap,
                    lane_rate=float(self.racing_parameters.get('lane_rate_mps', 1.8)),
                    brake_decel=brake_decel)
                # Holding the measured current lane avoids following an old,
                # distant target through a newly predicted crossing vehicle.
                selected_lane, selected_profile = actual, actual
                path_cap = current_cap
                trajectory_emergency = bool(trajectory_hold['hard_veto'])
                if trajectory_emergency:
                    path_cap = 0.
                    reasons.append('all_credible_imminent_routes_and_current_hold_conflict_brake')
                else:
                    reasons.append('imminent_transitions_blocked_hold_current_clear_corridor')
                modified.update(target_offset_m=float(actual), target_speed_mps=float(path_cap),
                    path_profile_lane_m=float(actual), path_speed_cap_mps=float(path_cap),
                    trajectory_hold_actual_lane_requested=True,
                    trajectory_emergency_brake_requested=trajectory_emergency,
                    trajectory_minimum_decel_mps2=min(8., brake_decel) if trajectory_emergency else 0.)
        self.last_mode = mode
        blocker_lateral = self.blocker_lateral if self.blocker_lateral is not None else relevant['front_lateral_m']
        relative_side = ('left' if selected_lane > blocker_lateral + .2 else 'right' if selected_lane < blocker_lateral - .2 else 'same') if blocker_lateral is not None else None
        log = {'version': VERSION, 'enabled': True, 'learned_checkpoint_only_result': False,
            'coordinate_frame': 'original_reference_frenet', 'free_space_verified': False,
            'adaptive_gap_passing_enabled': self.adaptive_gaps,
            'rank_acceleration_reachable_pace': self.rank_reachable_pace,
            'ranking_horizon_s': self.ranking_horizon,
            'spatially_coherent_pass_clearance_enabled': self.coherent_clearance,
            'selected_measured_gap': selected.get('measured_gap') if selected else None,
            'selected_gap_sweep': selected.get('gap_sweep') if selected else None,
            'profile_preview_m': preview, 'mapped_center_bounds_m': list(current_bounds),
            'nominal_road_binding': nominal_road,
            'outside_constant_heading_bounds': outside_constant_heading_bounds,
            'validated_variable_nominal_bounds_exemption': nominal_road_exemption,
            'both_sides_evaluated': True, 'pass_preferred_to_following_brake': pass_needed,
            'selected_mode': mode, 'selected_corridor_id': selected['corridor_id'] if selected else None,
            'selected_offset_m': selected_lane, 'nominal_offset_m': nominal,
            'actual_lateral_m': actual, 'outside_planning_corridor': outside_planning_corridor,
            'recovery_pace_factor': recovery_pace_factor,
            'nominal_preview_progress_m': progress + lookahead, 'pass_side_relative_to_measured_blocker': relative_side,
            'blocker_track_id': self.blocker_id, 'blocker_lateral_m': blocker_lateral,
            'pass_clearance_status': pass_status,
            'commitment_started_simulation_s': self.commitment_started,
            'committed_lane_m': self.committed_lane,
            'committed_until_simulation_s': self.committed_until if math.isfinite(self.committed_until) else None,
            'reasons': reasons, 'candidates': candidates, 'racing_guard_parameters': self.racing_parameters,
            'blanket_visibility_stopping_cap_relaxed': self.relax_visibility,
            'original_learned_request': original, 'modified_request': modified,
            'guards_retained': ['path_curvature_and_braking_profile', 'transition_sweep_occupancy',
                                'closing_rear_gap', 'measured_stopping_gap', 'physical_corridor']}
        if self.trajectory_planner is not None:
            log.update(trajectory_prediction=trajectory_prediction,
                trajectory_hold_current_sweep=trajectory_hold,
                trajectory_emergency_brake_requested=trajectory_emergency,
                trajectory_final_guard_can_only_restrict_further=True)
        return {**decision, 'learned_raw_action': original, 'raw_action': modified,
            'aggressive_hybrid': log}
