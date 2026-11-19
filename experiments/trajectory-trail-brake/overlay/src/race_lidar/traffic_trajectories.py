"""Bounded measured-track trajectory tubes for receding-horizon route scoring.

No simulator opponent state is accepted. World polylines are derived from the
supplied map and measured Frenet states; they are predictions, not free-space
proof or measured future vehicle motion. Existing guards remain authoritative.
"""
from __future__ import annotations

import math
import numpy as np


def _key(identifier):
    return type(identifier).__name__, str(identifier)


def is_measured_vehicle(row):
    if row.get('source') == 'raw_lidar_geometry' or row.get('object_class') == 'unclassified_obstacle':
        return False
    return bool(float(row.get('score', 0.)) >= .35 and (
        row.get('object_class') in ('car', 'vehicle') or row.get('confirmed')
        or row.get('tentative') or row.get('control_evidence')))


def apply_emergency_guard(raw_action, guarded, *, actual_lateral, brake_decel):
    """Apply a requested trajectory stop without relaxing any older guard.

    Inactive calls return the original object exactly. Invalid active inputs
    raise instead of emitting an unsafe/nonfinite actuator request; the caller
    must retain its existing full-brake exception handling.
    """
    emergency = bool(raw_action.get('trajectory_emergency_brake_requested', False))
    hold = bool(raw_action.get('trajectory_hold_actual_lane_requested', False))
    if not emergency and not hold:
        return guarded
    available = float(brake_decel)
    requested = float(raw_action.get('trajectory_minimum_decel_mps2', available))
    previous = float(guarded.get('minimum_decel_mps2', 0.))
    if (not np.isfinite([actual_lateral, available, requested, previous,
            guarded['target_offset_m'], guarded['target_speed_mps']]).all()
            or available <= 0. or requested < 0. or previous < 0.):
        raise ValueError('Active trajectory emergency guard requires finite physical inputs')
    result = dict(guarded)
    if hold:
        result['target_offset_m'] = float(actual_lateral)
    if emergency:
        result['target_speed_mps'] = 0.
        result['minimum_decel_mps2'] = max(previous, min(available, requested))
        result['guards'] = sorted(set([*guarded.get('guards', []), 'trajectory_imminent_collision_brake']))
    return result


def first_swept_overlap(times, separating_functions, *, maximum_time, minimum_time=.05):
    """First continuous overlap of linearly swept separating inequalities.

    Each row is <=0 when that axis overlaps. Intersect their interval solutions
    between adjacent predictions, so a short encounter between .1 s samples is
    not silently missed. The caller supplies already inflated body bounds.
    """
    times = np.asarray(times, float)
    values = np.asarray(separating_functions, float)
    start, end = values[:, :-1], values[:, 1:]
    delta = end - start
    crossing = np.divide(-start, delta, out=np.zeros_like(delta), where=abs(delta) > 1e-12)
    lower = np.max(np.where(delta < -1e-12, crossing, 0.), axis=0)
    upper = np.min(np.where(delta > 1e-12, crossing, 1.), axis=0)
    interval = np.diff(times)
    lower = np.maximum(lower, (minimum_time - times[:-1]) / interval)
    upper = np.minimum(upper, (maximum_time - times[:-1]) / interval)
    parallel_block = np.any((abs(delta) <= 1e-12) & (start > 0.), axis=0)
    valid = (~parallel_block) & (lower <= upper) & (lower <= 1.) & (upper >= 0.)
    if not np.any(valid):
        return None
    i = int(np.flatnonzero(valid)[0])
    return float(times[i] + interval[i] * max(0., lower[i]))


class TrafficTrajectoryMap:
    VERSION = 'measured-physical-metric-tubes-v1'

    def __init__(self, configuration=None, *, reference=None):
        self.config = dict(configuration or {})
        self.reference = reference
        self.horizon = float(self.config.get('horizon_s', 3.))
        self.step = float(self.config.get('step_s', .1))
        self.hard_horizon = float(self.config.get('hard_conflict_horizon_s', 1.2))
        self.max_age = float(self.config.get('history_max_age_s', 2.))
        self.risk_weight = float(self.config.get('risk_weight_mps', .75))
        self.maximum_cost = float(self.config.get('maximum_risk_cost_mps', 2.))
        values = [self.horizon, self.step, self.hard_horizon, self.max_age,
                  self.risk_weight, self.maximum_cost]
        if (not np.isfinite(values).all() or not 1. <= self.horizon <= 5.
                or not .05 <= self.step <= .25 or not .2 <= self.hard_horizon <= 1.5
                or not .5 <= self.max_age <= 3. or not 0. <= self.risk_weight <= 2.
                or not 0. <= self.maximum_cost <= 4.):
            raise ValueError('Invalid measured trajectory planning configuration')
        self.times = np.linspace(0., self.horizon, int(math.ceil(self.horizon / self.step)) + 1)
        self.history = {}
        self.tracks = []
        self.last_timestamp = None
        self.progress = 0.

    def _map(self, progress):
        if self.reference is None:
            return np.array([float(progress), 0.]), 0., 0.
        point, heading, _, curvature = self.reference.sample(float(progress))
        return np.asarray(point, float)[:2], float(heading), float(curvature)

    def _world(self, progress, lateral):
        point, heading, _ = self._map(progress)
        return point + float(lateral) * np.array([-math.sin(heading), math.cos(heading)])

    def _history_motion(self, row, progress, timestamp):
        key = _key(row['track_id'])
        history = self.history.get(key, [])
        age = max(0., float(row.get('age_s', row.get('measurement_age_s', 0.))))
        stamp = float(timestamp) - age
        absolute = float(progress + row['relative_distance_m'])
        if history:
            elapsed = stamp - history[-1][0]
            if elapsed > 0. and (abs(absolute - history[-1][1]) > 95. * elapsed + 8.
                    or abs(float(row['lateral_m']) - history[-1][2]) > 10. * elapsed + 3.):
                history = []  # A reused/renamed ID cannot inherit unrelated motion.
        # Predicted/coasted rows must not turn the filter's own extrapolation
        # into additional measured hits or double-integrate measurement age.
        if age <= .05 and (not history or stamp > history[-1][0] + 1e-6):
            history.append((stamp, absolute, float(row['lateral_m'])))
        history = [x for x in history if timestamp - x[0] <= self.max_age][-12:]
        self.history[key] = history
        measured, lateral_rate, residual = False, 0., 1.
        if len(history) >= 3 and history[-1][0] - history[0][0] >= .3:
            t = np.array([x[0] for x in history]); t -= t[-1]
            lateral = np.array([x[2] for x in history])
            design = np.c_[np.ones(len(t)), t]
            fit = np.linalg.lstsq(design, lateral, rcond=None)[0]
            lateral_rate = float(np.clip(fit[1], -8., 8.))
            residual = float(np.sqrt(np.mean((lateral - design @ fit) ** 2)))
            measured = residual <= .5
        return lateral_rate, measured, residual, age, stamp

    def update(self, rows, *, progress_m, timestamp):
        timestamp, self.progress = float(timestamp), float(progress_m)
        if not np.isfinite([timestamp, self.progress]).all() or (
                self.last_timestamp is not None and timestamp < self.last_timestamp - 1e-9):
            raise ValueError('Finite monotonic trajectory timestamps are required')
        self.last_timestamp = timestamp
        unique = {}
        for row in rows:
            key = _key(row['track_id'])
            if key not in unique or float(row.get('age_s', 0.)) < float(unique[key].get('age_s', 0.)):
                unique[key] = dict(row)
        self.tracks = []
        for row in unique.values():
            rate, observed, residual, age, stamp = self._history_motion(row, self.progress, timestamp)
            lateral = float(row['lateral_m']); absolute = self.progress + float(row['relative_distance_m'])
            _, heading, curvature = self._map(absolute)
            tangent = np.array([math.cos(heading), math.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            world_velocity = np.asarray(row.get('velocity_xy_mps', []), float)
            valid_velocity = bool(row.get('velocity_valid', False))
            lateral_std = float(row.get('lateral_speed_std_mps', 1.))
            if world_velocity.shape == (2,) and np.isfinite(world_velocity).all() and valid_velocity:
                # The state was projected into Frenet by the same supplied map.
                # World velocity supplies orientation without a body-yaw oracle.
                velocity = float(np.clip(np.dot(world_velocity, tangent), -85., 85.))
                measured_lateral = float(np.clip(np.dot(world_velocity, normal), -8., 8.))
                if not observed or abs(measured_lateral - rate) <= max(1., 2. * lateral_std):
                    rate = measured_lateral
                else:
                    observed = False
            else:
                velocity = float(np.clip(row.get('speed_mps', 0.), -85., 85.))
            sigma = max(0., float(row.get('speed_std_mps', 2.)))
            pos_std = max(.15, float(row.get('position_std_m', .35)))
            if not valid_velocity:
                low_speed, high_speed = -85., 85.
            else:
                low_speed, high_speed = velocity - max(1., sigma), velocity + max(1., sigma)
            center_s, low_s, high_s = [absolute], [absolute], [absolute]
            mean_lateral = lateral + rate * self.times
            credible_metric = True
            for index in range(1, len(self.times)):
                dt = self.times[index] - self.times[index - 1]
                for station, speed in ((center_s, velocity), (low_s, low_speed), (high_s, high_speed)):
                    _, _, kappa = self._map(station[-1])
                    metric = 1. - kappa * mean_lateral[index - 1]
                    credible_metric &= metric > .2
                    station.append(station[-1] + speed * dt / max(.2, metric))
            center_s, low_s, high_s = map(np.asarray, (center_s, low_s, high_s))
            # Position uncertainty grows with age/time. Guessed departure is
            # never credited as free space: retain the current footprint too.
            lateral_growth = min(.75, max(.15, residual if observed else (0. if valid_velocity else .75),
                min(1., lateral_std) * .25))
            uncertainty = min(1.5, pos_std) + lateral_growth * (self.times + age)
            half_width = max(1.05, .5 * float(row.get('vehicle_width_m', 2.1)))
            lower = float(row.get('lateral_min_m', lateral - half_width))
            upper = float(row.get('lateral_max_m', lateral + half_width))
            lateral_low = np.minimum(lower, lower + rate * self.times) - uncertainty
            lateral_high = np.maximum(upper, upper + rate * self.times) + uncertainty
            longitudinal_uncertainty = min(2., pos_std) + .5 * (self.times + age) ** 2
            low_s -= longitudinal_uncertainty / max(.2, 1. - curvature * lateral)
            high_s += longitudinal_uncertainty / max(.2, 1. - curvature * lateral)
            vehicle = is_measured_vehicle(row)
            credible = bool(vehicle and valid_velocity and observed and age <= .25
                and int(row.get('measurement_hits', 0)) >= 3 and lateral_std <= 1.5 and credible_metric)
            world = np.array([self._world(s, l) for s, l in zip(center_s, mean_lateral)])
            self.tracks.append({'track_id': row['track_id'], 'vehicle_evidence': vehicle,
                'source': row.get('source', 'measured_detector_track'), 'age_s': age,
                'measurement_time_s': stamp, 'velocity_valid': valid_velocity,
                'speed_mps': velocity, 'lateral_velocity_mps': rate,
                'lateral_motion_observed': observed, 'lateral_motion_residual_m': residual,
                'lateral_speed_std_mps': lateral_std, 'hard_veto_credible': credible,
                'physical_progress_metric_credible': bool(credible_metric),
                'current_absolute_progress_m': absolute, 'current_lateral_m': lateral,
                'center_progress_m': center_s, 'progress_lower_m': low_s, 'progress_upper_m': high_s,
                'center_lateral_m': mean_lateral, 'lateral_lower_m': lateral_low,
                'lateral_upper_m': lateral_high, 'world_center_xy_m': world})
        self.history = {k: v for k, v in self.history.items() if v and timestamp - v[-1][0] <= self.max_age}
        return self.snapshot()

    def snapshot(self):
        tracks = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()}
                  for row in self.tracks]
        return {'version': self.VERSION, 'coordinate_frame': 'original_reference_frenet',
            'timestamp_s': self.last_timestamp, 'ego_progress_m': self.progress,
            'horizon_s': self.horizon, 'time_samples_s': self.times.tolist(),
            'observed_track_count': len(tracks), 'vehicle_track_count': sum(r['vehicle_evidence'] for r in tracks),
            'unclassified_track_count': sum(not r['vehicle_evidence'] for r in tracks),
            'all_observed_rows_considered': True, 'top_n_truncation_used': False,
            'opponent_ground_truth_used': False, 'future_motion_is_prediction': True,
            'unknown_or_occluded_space_is_free': False, 'tracks': tracks}

    def assess(self, lane, *, actual_lateral, body_speed, speed_cap, lane_rate=1.8,
               nominal_line=None, nominal_captured=False, acceleration=5., brake_decel=8.):
        lateral, station, speed = [float(actual_lateral)], [self.progress], max(0., float(body_speed))
        ego_speeds = [speed]
        credible_ego_metric = True
        variable = bool(nominal_line is not None and nominal_captured)
        metric_table = None
        if variable and all(hasattr(nominal_line, x) for x in ('s', 'ds', 'base_ds', 'length')):
            ratio = np.asarray(nominal_line.ds, float) / np.asarray(nominal_line.base_ds, float)
            if np.isfinite(ratio).all() and np.all(ratio > 0.):
                metric_table = (np.asarray(nominal_line.s, float), ratio, float(nominal_line.length))
        def variable_metric(s):
            if metric_table is None:
                return None
            positions, ratio, length = metric_table
            return float(np.interp(float(s) % length, positions, ratio, period=length))
        initial_error = (actual_lateral - nominal_line.nominal_offset_m(self.progress)) if variable else 0.
        for index in range(1, len(self.times)):
            dt = self.times[index] - self.times[index - 1]
            target = float(speed_cap)
            if nominal_line is not None and hasattr(nominal_line, 'minimum_speed'):
                target = min(target, float(nominal_line.minimum_speed(station[-1], 0.)))
            delta = float(np.clip(target - speed, -brake_decel * dt, acceleration * dt))
            next_speed = max(0., speed + delta)
            body_distance = .5 * (speed + next_speed) * dt
            if variable and metric_table is not None:
                # The frozen mapped path's physical arc includes its changing
                # lateral offset. A parallel-curve 1-k*l metric would omit it.
                first_metric = variable_metric(station[-1])
                middle_metric = variable_metric(station[-1] + .5 * body_distance / max(.05, first_metric))
                credible_ego_metric &= middle_metric > .05
                next_station = station[-1] + body_distance / max(.05, middle_metric)
                next_lateral = float(nominal_line.nominal_offset_m(next_station)) + initial_error * math.exp(-self.times[index] / .6)
            else:
                _, _, curvature = self._map(station[-1])
                metric = 1. - curvature * lateral[-1]
                # Missing variable-path arc metadata is not credible for a
                # hard veto. It must never be optimistically treated as fixed.
                credible_ego_metric &= metric > .2 and not variable
                provisional = station[-1] + body_distance / max(.2, metric)
                desired = float(nominal_line.nominal_offset_m(provisional)) if nominal_line is not None else float(lane)
                active_dt = max(0., self.times[index] - .35) - max(0., self.times[index - 1] - .35)
                allowed = min(lane_rate * active_dt, body_distance)
                next_lateral = lateral[-1] + float(np.clip(desired - lateral[-1], -allowed, allowed))
                lateral_distance = next_lateral - lateral[-1]
                tangent_distance = math.sqrt(max(0., body_distance * body_distance - lateral_distance * lateral_distance))
                next_station = station[-1] + tangent_distance / max(.2, metric)
            station.append(next_station); lateral.append(next_lateral)
            speed = next_speed; ego_speeds.append(speed)
        station, lateral = np.asarray(station), np.asarray(lateral)
        minimum, risk, earliest = math.inf, 0., None
        hard_ids, near_ids = [], []
        for row in self.tracks:
            _, _, curvature = self._map(row['current_absolute_progress_m'])
            metric = max(.2, 1. - curvature * row['current_lateral_m'])
            long_low = (row['progress_lower_m'] - station) * metric - 8.5
            long_high = (row['progress_upper_m'] - station) * metric + 8.5
            longitudinal_overlap = (long_low <= 0.) & (long_high >= 0.)
            # Add a projected yaw allowance during ego lateral motion. This is
            # a conservative planning tube, not the independent actual-body audit.
            lateral_rate = np.r_[0., np.diff(lateral) / np.diff(self.times)]
            angle = np.arctan2(abs(lateral_rate), np.maximum(ego_speeds, 1.))
            ego_half_width = 1.15 + .20 + 2.5 * np.sin(angle)
            lateral_gap = np.maximum(row['lateral_lower_m'] - lateral - ego_half_width,
                                     lateral - row['lateral_upper_m'] - ego_half_width)
            longitudinal_gap = np.maximum(long_low, -long_high)
            distance = np.hypot(np.maximum(0., longitudinal_gap), np.maximum(0., lateral_gap))
            signed = np.where(longitudinal_overlap, lateral_gap, distance)
            minimum = min(minimum, float(signed.min()))
            encounter = longitudinal_overlap & (lateral_gap < .5)
            if np.any(encounter):
                near_ids.append(row['track_id'])
                pressure = np.clip((.5 - lateral_gap) / .75, 0., 8. / 3.)
                risk = max(risk, float(np.max(np.where(encounter, pressure * np.exp(-self.times / 1.5), 0.))))
            center_long = (row['center_progress_m'] - station) * metric
            mean_lateral_gap = np.abs(row['center_lateral_m'] - lateral)
            current_gap = abs(row['current_lateral_m'] - actual_lateral)
            lateral_closing = (abs(row['lateral_velocity_mps']) >= .4
                and np.any(mean_lateral_gap < current_gap - .1))
            closing_rear = (row['current_absolute_progress_m'] - self.progress < -5.
                and row['speed_mps'] > body_speed + 3. and abs(lane - actual_lateral) > .5)
            at = first_swept_overlap(self.times, np.array([center_long - 8.5, -center_long - 8.5,
                row['lateral_lower_m'] - lateral - ego_half_width,
                lateral - row['lateral_upper_m'] - ego_half_width]), maximum_time=self.hard_horizon)
            if credible_ego_metric and row['hard_veto_credible'] and (lateral_closing or closing_rear) and at is not None:
                hard_ids.append(row['track_id'])
                earliest = at if earliest is None else min(earliest, at)
        return {'clear': not hard_ids, 'hard_veto': bool(hard_ids),
            'conflicting_track_ids': list(dict.fromkeys(hard_ids)), 'first_conflict_time_s': earliest,
            'minimum_predicted_tube_clearance_m': None if not math.isfinite(minimum) else minimum,
            'risk_cost_mps': min(self.maximum_cost, self.risk_weight * risk),
            'near_encounter_track_ids': list(dict.fromkeys(near_ids)),
            'all_observed_rows_considered': True, 'trajectory_samples': len(self.times),
            'ego_progress_m': station.tolist(), 'ego_lateral_m': lateral.tolist(),
            'ego_world_xy_m': [(
                np.asarray(nominal_line.sample(s)[0], float)[:2]
                + initial_error * math.exp(-t / .6) * np.array([
                    -math.sin(nominal_line.sample(s)[1]), math.cos(nominal_line.sample(s)[1])])
                ).tolist() if variable else self._world(s, l).tolist()
                for s, l, t in zip(station, lateral, self.times)],
            'ego_geometry_source': 'captured_actual_variable_xy' if variable else 'guarded_frenet_slew',
            'ego_speed_model': 'candidate_cap_and_known_nominal_profile_acceleration_braking_bounded',
            'hard_veto_requires_fresh_reliable_motion': True,
            'continuous_between_sample_sweeps_checked': True,
            'ego_physical_progress_metric_credible': bool(credible_ego_metric),
            'planning_intent_only': True, 'actual_contact_or_clearance_claimed': False}
