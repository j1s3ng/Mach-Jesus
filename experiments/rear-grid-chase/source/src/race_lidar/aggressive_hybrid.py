"""Explicit demo-only hybrid requests from learned outputs and measured hazards.

This is not a newly trained model. An absent measured obstacle is not verified
free space. The configured closed-track race mode relaxes the blanket visibility
stop cap; measured stopping/occupancy guards and map/corner limits remain active.
"""
from __future__ import annotations
import math
import numpy as np
from .race_control import predictive_guard, stopping_speed
from .rich_driver import LANE_OFFSETS_M, lane_summaries

VERSION = 'aggressive-hybrid-demo-v2-racing-guard'

class AggressiveHybrid:
    def __init__(self, configuration):
        self.configuration = dict(configuration)
        self.commit_seconds = float(configuration.get('commit_seconds', 1.5))
        self.switch_gain_mps = float(configuration.get('switch_gain_mps', 2.))
        self.racing_parameters = dict(configuration.get('racing_guard_parameters', {}))
        self.relax_visibility = bool(configuration.get('relax_blanket_visibility_cap', False))
        if not 0. <= self.commit_seconds <= 5. or not 0. <= self.switch_gain_mps <= 10.:
            raise ValueError('Invalid aggressive hybrid commitment configuration')
        self.committed_lane = None
        self.committed_until = -math.inf

    def apply(self, decision, state, observation, feasible_speed, *, timestamp,
              dt, brake_decel=8., visible_distance=70.):
        original = dict(decision['raw_action'])
        learned_speeds = np.asarray(original.get('lane_target_speeds_mps'), dtype=float)
        probabilities = np.asarray(original.get('lane_probabilities'), dtype=float)
        if (learned_speeds.shape != (3,) or probabilities.shape != (3,)
                or not np.isfinite(np.r_[learned_speeds, probabilities, feasible_speed, timestamp]).all()):
            raise ValueError('Aggressive hybrid requires finite learned three-lane outputs')
        current = float(state.get('last_target_offset_m', 0.))
        actual = float(state.get('cross_track_m', current))
        body_speed = max(0., float(state.get('body_speed_mps', 0.)))
        cap = max(0., float(feasible_speed))
        uncertainty_cap = float(self.racing_parameters.get('position_uncertainty_cap_m', 2.))
        planning_state = {**state, 'hazard_tracks': [{**row, 'position_std_m': min(uncertainty_cap, float(row.get('position_std_m', row.get('position_uncertainty_m', .35))))} for row in state['hazard_tracks']]}
        summaries = lane_summaries(planning_state, mode='stopping_critical')
        candidates = []
        for index, summary in enumerate(summaries):
            lane = float(LANE_OFFSETS_M[index])
            front_gap = float(summary['front_gap_m'])
            position_std = min(2., float(summary['front_position_std_m']))
            lead = float(summary['front_speed_mps']) - min(8., float(summary['front_speed_std_mps']) * float(self.racing_parameters.get('lead_speed_sigma_factor', 1.)))
            # The summary's120m value means no reported front actor, not that
            # the entire lane is observed empty. Actual guard still sees all rows.
            destination_speed = cap
            if front_gap < 120.:
                destination_speed = min(cap, stopping_speed(front_gap - float(self.racing_parameters.get('following_gap_m', 8.)) - position_std,
                    lead, decel=min(7., .9 * brake_decel), reaction=float(self.racing_parameters.get('reaction_seconds', .4))))
            if not self.relax_visibility:
                destination_speed = min(destination_speed, stopping_speed(visible_distance - 8.,
                    decel=min(7., .9 * brake_decel)))
            request = {**original, 'target_speed_mps': cap, 'target_offset_m': lane}
            preview = predictive_guard(request, observation, body_speed=body_speed,
                current_lane=current, actual_lateral=actual, feasible_speed=cap,
                dt=dt, brake_decel=brake_decel, visible_distance=visible_distance,
                enforce_visibility_stopping_budget=not self.relax_visibility, racing_parameters=self.racing_parameters)
            rejects = []
            if 'predicted_lane_conflict' in preview['guards']:
                rejects.append('measured_predicted_lane_conflict')
            # A lane change may not cut across a recently passed or closing car.
            changing = abs(lane - actual) > .8
            rear_budget = 12. + 2. * float(summary['rear_closing_speed_mps'])
            if changing and float(summary['rear_gap_m']) < rear_budget:
                rejects.append('measured_rear_gap')
            if front_gap <= 8. + position_std:
                rejects.append('measured_front_gap')
            utility = (destination_speed + .10 * min(cap, max(0., learned_speeds[index]))
                + float(probabilities[index]) - .4 * abs(lane - current))
            candidates.append({'lane_offset_m': lane, 'eligible': not rejects,
                'rejections': rejects, 'destination_speed_budget_mps': destination_speed,
                'learned_lane_speed_mps': float(learned_speeds[index]),
                'learned_lane_probability': float(probabilities[index]),
                'utility': utility, 'summary': summary, 'guard_preview': preview})
        eligible = [row for row in candidates if row['eligible']]
        selected = None
        reasons = []
        modified = dict(original)
        if eligible:
            selected = max(eligible, key=lambda row: (row['utility'], -abs(row['lane_offset_m'] - current)))
            committed = next((row for row in eligible if row['lane_offset_m'] == self.committed_lane), None)
            if (committed is not None and timestamp < self.committed_until
                    and selected['utility'] < committed['utility'] + self.switch_gain_mps):
                selected = committed
                reasons.append('retain_credible_lane_commitment')
            if self.committed_lane != selected['lane_offset_m']:
                self.committed_lane = selected['lane_offset_m']
                self.committed_until = timestamp + self.commit_seconds
                reasons.append('commit_to_fastest_credible_lane')
            # Request the map/corner envelope. The unchanged measured-obstacle
            # guard can still require lifting or braking while crossing lanes.
            modified.update(target_speed_mps=cap, target_offset_m=selected['lane_offset_m'],
                hybrid_overlay_used=True)
            reasons.append('request_feasible_race_pace')
        else:
            modified['target_speed_mps'] = min(cap, max(0., float(original['target_speed_mps'])))
            reasons.append('no_credible_lane_preserve_learned_request')
        log = {'version': VERSION, 'enabled': True, 'racing_guard_parameters': self.racing_parameters, 'learned_checkpoint_only_result': False,
            'blanket_visibility_stopping_cap_relaxed': self.relax_visibility,
            'free_space_verified': False,
            'clearance_interpretation': 'absence of a reported measured hazard within the sensor/planning horizon',
            'selected_lane_m': selected['lane_offset_m'] if selected else None,
            'committed_until_simulation_s': self.committed_until if math.isfinite(self.committed_until) else None,
            'reasons': reasons, 'candidates': candidates,
            'original_learned_request': original, 'modified_request': modified,
            'guards_retained': ['map_corner_feasibility', 'predicted_lane_conflict',
                'measured_stopping_gap', 'stopping_brake_feedforward', 'physical_corridor']}
        return {**decision, 'learned_raw_action': original, 'raw_action': modified,
            'aggressive_hybrid': log}
