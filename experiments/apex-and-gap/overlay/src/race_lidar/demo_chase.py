"""Demo-only sustained total-progress lead/failure stop; no driver inputs."""
from __future__ import annotations
import math

class ChaseStop:
    def __init__(self, configuration, start_progress):
        self.hero = configuration['hero_car_id']
        self.hold = float(configuration.get('lead_hold_seconds', 1.))
        self.margin = float(configuration.get('lead_margin_m', 2.5))
        self.failure_hold = float(configuration.get('failure_hold_seconds', 1.))
        self.start_progress = float(start_progress)
        minimum = configuration.get('minimum_hero_progress_m')
        self.minimum_progress = None if minimum is None else float(minimum)
        if self.minimum_progress is not None and not math.isfinite(self.minimum_progress):
            raise ValueError('Minimum hero progress must be finite')
        self.lead_since = None
        self.failure_since = None
        self.last = None

    def update(self, rows, race_time, simulation_time):
        ordered = sorted(rows, key=lambda row: -float(row['progress_m']))
        hero = next(row for row in ordered if row['car_id'] == self.hero)
        rivals = [row for row in ordered if row['car_id'] != self.hero]
        gap = float(hero['progress_m']) - max(float(row['progress_m']) for row in rivals)
        failed = bool(hero.get('terminal') and not hero.get('finished'))
        progressed = float(hero['progress_m']) >= self.start_progress + 20.
        minimum_met = self.minimum_progress is None or float(hero['progress_m']) >= self.minimum_progress
        is_lead = not failed and progressed and minimum_met and ordered[0]['car_id'] == self.hero and gap > self.margin
        if is_lead:
            self.lead_since = race_time if self.lead_since is None else self.lead_since
        else:
            self.lead_since = None
        if failed:
            self.failure_since = race_time if self.failure_since is None else self.failure_since
        reason = None
        if failed and race_time - self.failure_since >= self.failure_hold - 1e-9:
            reason = 'hero_failed'
        elif is_lead and race_time - self.lead_since >= self.hold - 1e-9:
            reason = 'hero_took_lead'
        self.last = {'scope': 'CONTINUOUS_CHASE_NOT_FULL_DISTANCE_RACE',
            'stop_reason': reason, 'lead_achieved': reason == 'hero_took_lead',
            'hero_car_id': self.hero, 'hero_rank_by_total_progress': ordered.index(hero) + 1,
            'hero_total_progress_m': float(hero['progress_m']), 'hero_progressed_from_grid': progressed,
            'minimum_hero_progress_required_m': self.minimum_progress,
            'minimum_hero_progress_met': minimum_met,
            'gap_to_best_rival_m': gap, 'lead_margin_required_m': self.margin,
            'lead_hold_required_s': self.hold, 'lead_since_race_time_s': self.lead_since,
            'hero_failure_observed_race_time_s': self.failure_since,
            'race_time_s': float(race_time), 'simulation_time_s': float(simulation_time),
            'hero_terminal_status': hero.get('result_status'),
            'active_car_ids': [row['car_id'] for row in rows if not row.get('terminal')],
            'total_progress_standings': ordered,
            'full_twenty_lap_race_completed': False, 'formal_race_winner_claimed': False}
        return self.last

def flat_out_actuators(enabled, active_racing, throttle, brake):
    proposed = {'throttle': float(throttle), 'brake': float(brake)}
    overridden = bool(enabled and active_racing)
    return ({'throttle': 1., 'brake': 0.} if overridden else proposed), {
        'enabled': bool(enabled), 'active': overridden, 'proposed': proposed,
        'command': {'throttle': 1., 'brake': 0.} if overridden else dict(proposed),
        'reason': 'literal_flat_out_no_lift_demo' if overridden else 'normal_controller',
        'learned_policy_requested_override': False}
