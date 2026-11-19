"""Measured-corridor passing priority over a smooth nominal racing line.

This is an explicit engineering overlay, not a fitted policy. Original learned
outputs are retained. All traffic inputs are measured hazard rows; no opponent
truth is consulted. The final physical/occupancy guard remains authoritative.
"""
from __future__ import annotations

import math
import numpy as np

from .race_control import predictive_guard, stopping_speed

VERSION = 'pass-first-bidirectional-racing-line-v1'


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
        if (not 0. < self.maximum_lane <= 4.0 or not 0. <= self.commit_seconds <= 6.
                or not 0. <= self.transient_hold <= 1. or not 0. <= self.switch_gain <= 10.):
            raise ValueError('Invalid pass-first configuration')
        self.committed_lane = None
        self.committed_until = -math.inf
        self.blocked_since = None
        self.commitment_started = None
        self.blocker_id = None
        self.blocker_lateral = None
        self.blocker_missing_since = None
        self.last_mode = 'nominal_line'

    def _pass_status(self, rows, timestamp):
        if self.committed_lane is None:
            return {'unresolved': False, 'reason': 'no_committed_pass', 'matching_track_ids': []}
        matching = [r for r in rows if self.blocker_id is not None and r['track_id'] == self.blocker_id]
        # Vision/geometry can duplicate or rename the same car. Nearby measured
        # traffic in the original blocking corridor also delays a cut-back.
        near = [r for r in rows if self.blocker_lateral is not None
                and abs(r['lateral_m'] - self.blocker_lateral) < 2.4
                and -12. <= r['relative_distance_m'] <= 25.]
        ids = [r['track_id'] for r in [*matching, *near]]
        if any(r['relative_distance_m'] >= -12. for r in matching) or near:
            self.blocker_missing_since = None
            return {'unresolved': True, 'reason': 'blocking_or_alongside_traffic_not_cleared', 'matching_track_ids': ids}
        if matching and all(r['relative_distance_m'] < -12. for r in matching):
            self.blocker_missing_since = None
            return {'unresolved': timestamp < self.committed_until,
                    'reason': 'measured_blocker_clearly_behind', 'matching_track_ids': ids}
        self.blocker_missing_since = timestamp if self.blocker_missing_since is None else self.blocker_missing_since
        unresolved = timestamp < self.committed_until or timestamp - self.blocker_missing_since < 1.
        return {'unresolved': unresolved, 'reason': 'measurement_loss_grace' if unresolved else 'corridor_clear_after_measurement_grace',
                'matching_track_ids': ids}

    def _rows(self, state):
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
            if not -100. <= d <= 150. or abs(lateral) > 10.:
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

    def guard_profile_cap(self, progress, preview, raw_action, actual_lateral, supplied_cap):
        """Keep the actual lane's corner budget until a wide transition aligns."""
        target = float(raw_action.get('target_offset_m', actual_lateral))
        profile_lane = raw_action.get('path_profile_lane_m')
        destination = self._profile_cap(progress, preview, profile_lane, supplied_cap)
        current = self._profile_cap(progress, preview, actual_lateral, supplied_cap)
        transitioning = abs(float(actual_lateral) - target) > 1.
        used = min(destination, current) if transitioning else destination
        return used, {'destination_path_cap_mps': destination,
            'actual_lane_path_cap_mps': current, 'transition_limit_active': transitioning,
            'used_path_cap_mps': used, 'actual_lateral_m': float(actual_lateral),
            'target_absolute_lane_m': target}

    def apply(self, decision, state, observation, feasible_speed, *, timestamp,
              dt, brake_decel=8., visible_distance=70., progress_m=None):
        original = dict(decision['raw_action'])
        speeds = np.asarray(original.get('lane_target_speeds_mps'), float)
        probabilities = np.asarray(original.get('lane_probabilities'), float)
        if speeds.shape != (3,) or probabilities.shape != (3,) or not np.isfinite(np.r_[speeds, probabilities, feasible_speed, timestamp]).all():
            raise ValueError('Finite three-lane learned outputs are required')
        current = float(state.get('last_target_offset_m', 0.))
        actual = float(state.get('cross_track_m', current))
        body = max(0., float(state.get('body_speed_mps', 0.)))
        progress = float(progress_m if progress_m is not None else state.get('progress_m', 0.))
        preview = 3. + body
        lookahead = float(np.clip(4.5 + body * .28, 5., 20.))
        nominal = float(self.racing_line.nominal_offset_m(progress + lookahead)) if self.racing_line is not None else 0.
        nominal = float(np.clip(nominal, -self.maximum_lane, self.maximum_lane))
        rows = self._rows(state)
        nominal_cap = self._profile_cap(progress, preview, None, feasible_speed)
        current_cap = self._profile_cap(progress, preview, actual, feasible_speed)
        current_summary = self._summary(actual, rows, current_cap, body)
        nominal_summary = self._summary(nominal, rows, nominal_cap, body)
        relevant = min((current_summary, nominal_summary), key=lambda s: s['destination_speed_budget_mps'])
        anticipation = max(45., body * 4. + 10.)
        pass_needed = (relevant['front_track_id'] is not None
            and relevant['front_gap_m'] < min(150., anticipation)
            and relevant['destination_speed_budget_mps'] < nominal_cap - self.pass_gain)
        pass_status = self._pass_status(rows, float(timestamp))

        specs = [('nominal_line', nominal, None), ('pass_right', -self.maximum_lane, -self.maximum_lane),
                 ('pass_center', 0., 0.), ('pass_left', self.maximum_lane, self.maximum_lane)]
        if self.committed_lane is not None and all(abs(s[1] - self.committed_lane) > .05 or s[0] == 'nominal_line' for s in specs):
            specs.append(('committed_pass', self.committed_lane, self.committed_lane))
        candidates = []
        for corridor, lane, profile_lane in specs:
            cap = self._profile_cap(progress, preview, profile_lane, feasible_speed)
            summary = self._summary(lane, rows, cap, body)
            request = {**original, 'target_speed_mps': cap, 'target_offset_m': lane}
            guard = predictive_guard(request, observation, body_speed=body,
                current_lane=current, actual_lateral=actual, feasible_speed=cap, dt=dt,
                brake_decel=brake_decel, visible_distance=visible_distance,
                maximum_lane=self.maximum_lane,
                enforce_visibility_stopping_budget=not self.relax_visibility,
                racing_parameters=self.racing_parameters)
            rejects = []
            if 'predicted_lane_conflict' in guard['guards']:
                rejects.append('transition_sweep_occupied')
            transition_time = abs(lane - actual) / float(self.racing_parameters.get('lane_rate_mps', 1.8)) + .2
            rear_budget = 6. + transition_time * summary['rear_closing_speed_mps']
            if abs(lane - actual) > .8 and summary['rear_gap_m'] < rear_budget:
                rejects.append('closing_rear_vehicle')
            if summary['front_gap_m'] <= float(self.racing_parameters.get('following_gap_m', 5.)) + summary['front_position_std_m']:
                rejects.append('destination_front_occupied')
            learned_speed = float(np.interp(lane, [-3.2, 0., 3.2], speeds))
            learned_prior = float(np.interp(lane, [-3.2, 0., 3.2], probabilities))
            utility = (summary['destination_speed_budget_mps'] + .025 * learned_speed
                + .05 * learned_prior - .15 * abs(lane - actual) - .10 * abs(lane - nominal))
            candidates.append({'corridor_id': corridor, 'lane_offset_m': float(lane),
                'profile_lane_m': profile_lane, 'path_speed_cap_mps': cap,
                'summary': summary, 'eligible': not rejects, 'rejections': rejects,
                'guard_preview': guard, 'utility': utility,
                'learned_lane_speed_mps': learned_speed, 'learned_lane_probability': learned_prior})

        nominal_candidate = candidates[0]
        current_candidate = next((r for r in candidates if r['profile_lane_m'] is not None and self.committed_lane is not None
                                  and abs(r['lane_offset_m'] - self.committed_lane) < .05), None)
        eligible_passes = [c for c in candidates[1:] if c['eligible']]
        best_pass = max(eligible_passes, key=lambda c: (c['utility'], -abs(c['lane_offset_m'] - actual)), default=None)
        selected = None
        reasons = []
        pass_sequence_active = timestamp < self.committed_until or pass_status['unresolved'] or pass_needed
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

        outside_planning_corridor = abs(actual) > self.maximum_lane + .10
        recovery_pace_factor = 1.
        if outside_planning_corridor:
            edge = float(np.clip(actual, -self.maximum_lane, self.maximum_lane))
            overshoot = max(0., abs(actual) - self.maximum_lane)
            recovery_pace_factor = float(np.clip(1. - .20 * overshoot, .55, 1.))
            recovery_cap = self._profile_cap(progress, preview, edge, feasible_speed) * recovery_pace_factor
            if abs(actual) >= 5.4:
                recovery_cap = min(20., recovery_cap)
            selected = {'corridor_id': 'corridor_recovery', 'lane_offset_m': edge,
                'profile_lane_m': edge,
                'path_speed_cap_mps': recovery_cap}
            reasons.append('measured_outside_planning_corridor_use_edge_profile_recovery')

        modified = dict(original)
        if selected is None:
            # Holding the present lane avoids a blind reversal when every route
            # is blocked. Final stopping guard may still request braking.
            mode, selected_lane, selected_profile = 'follow_blocked', float(np.clip(current, -self.maximum_lane, self.maximum_lane)), float(np.clip(actual, -self.maximum_lane, self.maximum_lane))
            path_cap = current_cap
            reasons.append('no_clear_pass_hold_lane_and_brake_if_needed')
        else:
            selected_lane = selected['lane_offset_m']
            selected_profile = selected['profile_lane_m']
            path_cap = selected['path_speed_cap_mps']
            mode = selected['corridor_id']
            if mode not in ('nominal_line', 'corridor_recovery'):
                if self.committed_lane is None or abs(self.committed_lane - selected_lane) > .05:
                    self.committed_lane = selected_lane
                    self.commitment_started = float(timestamp)
                    self.committed_until = float(timestamp) + self.commit_seconds
                    self.blocker_id = relevant['front_track_id']
                    self.blocker_lateral = relevant['front_lateral_m']
                    self.blocker_missing_since = None
                    self.blocked_since = None
            elif mode == 'corridor_recovery' or timestamp >= self.committed_until:
                self.committed_lane, self.blocker_id, self.blocker_lateral = None, None, None
                self.blocker_missing_since = None
        modified.update(target_speed_mps=float(path_cap), target_offset_m=float(selected_lane),
            hybrid_overlay_used=True, path_mode=mode, path_profile_lane_m=selected_profile,
            path_speed_cap_mps=float(path_cap), nominal_racing_line_offset_m=nominal)
        self.last_mode = mode
        blocker_lateral = self.blocker_lateral if self.blocker_lateral is not None else relevant['front_lateral_m']
        relative_side = ('left' if selected_lane > blocker_lateral + .2 else 'right' if selected_lane < blocker_lateral - .2 else 'same') if blocker_lateral is not None else None
        log = {'version': VERSION, 'enabled': True, 'learned_checkpoint_only_result': False,
            'coordinate_frame': 'original_reference_frenet', 'free_space_verified': False,
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
        return {**decision, 'learned_raw_action': original, 'raw_action': modified,
            'aggressive_hybrid': log}
