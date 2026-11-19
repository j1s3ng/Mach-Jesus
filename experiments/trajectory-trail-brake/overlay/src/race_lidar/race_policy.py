"""Independent race policies and guards driven by measured detector tracks.

The simulator may provide an entrant's own pose for localization. Opponent
positions reach this module only as detector boxes; there is no world-node API.
All policies use the same physical speed envelope and collision guards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .hotlap_driver import (ProgressState, VisionTraffic,
                            project_near, wrap_angle)
from .learned_driver import CURVATURE_DISTANCES_M


class SignedProgressTracker:
    """Ordered arc length that also represents a grid behind the finish line."""
    def __init__(self, reference, start_progress_m=0.):
        if not math.isfinite(start_progress_m):
            raise ValueError("Start progress must be finite")
        self.reference = reference
        self.progress = float(start_progress_m)

    def update(self, position, speed, dt):
        ref = self.reference
        position = np.asarray(position, float)[:2]
        lap = math.floor(self.progress / ref.length)
        base = int(np.searchsorted(ref.s, self.progress % ref.length, side="right") - 1)
        base += lap * len(ref.xy)
        forward_window = max(10., abs(speed) * dt * 5 + 6.)
        maximum_advance = max(4., abs(speed) * dt * 4 + 2.)
        best = None
        for extended in range(base - 5, base + int(math.ceil(forward_window / np.min(ref.ds))) + 3):
            i = extended % len(ref.xy)
            fraction = float(np.clip(np.dot(position - ref.xy[i], ref.segment[i]) / ref.ds[i] ** 2, 0., 1.))
            projected = ref.xy[i] + fraction * ref.segment[i]
            progress = (extended // len(ref.xy)) * ref.length + ref.s[i] + fraction * ref.ds[i]
            if not self.progress - 8. <= progress <= self.progress + maximum_advance:
                continue
            distance = float(np.linalg.norm(position - projected))
            if best is None or distance < best[0]:
                normal = np.array([-ref.segment[i, 1], ref.segment[i, 0]]) / ref.ds[i]
                best = distance, progress, float(np.dot(position - projected, normal)), projected
        if best is None:
            point = ref.sample(self.progress)[0]
            return ProgressState(self.progress, 0., float(np.linalg.norm(position - point)), point)
        distance, progress, cte, point = best
        self.progress = max(self.progress, progress)
        return ProgressState(self.progress, cte, distance, point)


@dataclass
class UncertainVehicleTrack:
    identifier: int
    state: np.ndarray
    covariance: np.ndarray
    last_time: float
    filter_time: float
    last_position: np.ndarray
    size: np.ndarray
    last_lateral: float
    score: float
    hits: int = 1
    consecutive_hits: int = 1
    confirmed: bool = False
    encountered: bool = False
    passed: bool = False

    @property
    def position(self):
        return self.state[:2]

    @property
    def velocity(self):
        return self.state[2:]


class TournamentTraffic(VisionTraffic):
    """Measured vehicle tracking with uncertain birth motion and global matching.

    A first detection has unknown signed velocity, rather than assuming the
    opponent shares ego speed. Broad longitudinal uncertainty permits stopped,
    moving and oncoming cars to acquire velocity on the next scan. Separate
    fresh hazards allow braking before a stable identity is confirmed.
    """
    def __init__(self, reference, confidence=.35, confirmation_hits=3, max_age=2., *,
                 rapid_hazard_confidence=.60, max_forward_m=95., max_lateral_m=7., hazard_hold_s=1.25,
                 trajectory_motion_fields=False):
        super().__init__(reference, confidence, confirmation_hits, max_age)
        if (not np.isfinite([confidence, rapid_hazard_confidence, max_age, max_forward_m, max_lateral_m, hazard_hold_s]).all()
                or not (0. <= confidence <= rapid_hazard_confidence <= 1.)
                or confirmation_hits < 1 or int(confirmation_hits) != confirmation_hits
                or min(max_age, max_forward_m, max_lateral_m, hazard_hold_s) <= 0.):
            raise ValueError("Invalid track confidence, confirmation count or age")
        self.rapid_hazard_confidence = float(rapid_hazard_confidence)
        self.max_forward_m = float(max_forward_m)
        self.max_lateral_m = float(max_lateral_m)
        self.hazard_hold_s = min(float(hazard_hold_s), float(max_age))
        self.last_update_time = None
        self.trajectory_motion_fields = bool(trajectory_motion_fields)
        self.max_vehicle_speed = 85.
        self.acceleration_sigma = 8.

    def _predict(self, track, timestamp):
        dt = max(0., float(timestamp) - track.filter_time)
        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = dt
        acceleration = np.array([[.5 * dt * dt, 0.], [0., .5 * dt * dt], [dt, 0.], [0., dt]])
        covariance = transition @ track.covariance @ transition.T
        covariance += self.acceleration_sigma ** 2 * acceleration @ acceleration.T
        return transition @ track.state, covariance

    def _candidates(self, boxes, ego_position, ego_yaw, ego_progress):
        c, s = math.cos(ego_yaw), math.sin(ego_yaw)
        rotation = np.array([[c, -s], [s, c]])
        candidates = []
        for box in boxes:
            try:
                center = np.asarray(box.get("center", []), float)
                size = np.asarray(box.get("size", []), float)
                score = float(box.get("score", 0.))
            except (AttributeError, TypeError, ValueError):
                continue
            if center.shape != (3,) or size.shape != (3,) or not np.isfinite(np.r_[center, size, score]).all():
                continue
            if not self.confidence <= score <= 1. or not (3. <= size[0] <= 7.5 and 1.25 <= size[1] <= 2.9 and .55 <= size[2] <= 2.5):
                continue
            distance = float(np.linalg.norm(center[:2]))
            if distance < 3.:  # Own body returns, expressed from the rear axle.
                continue
            position = rotation @ center[:2] + np.asarray(ego_position)[:2]
            projection = project_near(self.reference, position, ego_progress, window=self.max_forward_m + 10.)
            relative = projection["progress_m"] - ego_progress
            if abs(projection["lateral_m"]) > self.max_lateral_m or not -45. <= relative <= self.max_forward_m:
                continue
            sigma = .35 + .5 * (1. - score) + .004 * distance
            candidates.append({"score": score, "position": position, "projection": projection,
                               "size": size, "measurement_covariance": np.eye(2) * sigma ** 2})
        # Suppress duplicate boxes from this scan only. An old unconfirmed track
        # must never prevent a fresh hazard from being observed.
        unique = []
        for candidate in sorted(candidates, key=lambda row: -row["score"]):
            if not any(np.linalg.norm(candidate["position"] - other["position"]) < 1.2 for other in unique):
                unique.append(candidate)
        return unique

    def _association_cost(self, track, candidate, timestamp):
        elapsed = max(0., timestamp - track.last_time)
        displacement = candidate["position"] - track.last_position
        if np.linalg.norm(displacement) > self.max_vehicle_speed * elapsed + 3.:
            return 1e6
        if abs(candidate["projection"]["lateral_m"] - track.last_lateral) > 1.5 + 8. * elapsed:
            return 1e6
        innovation = candidate["position"] - track.position
        covariance = track.covariance[:2, :2] + candidate["measurement_covariance"]
        distance = float(innovation @ np.linalg.solve(covariance, innovation))
        if distance > 16.:
            return 1e6
        size_error = float(np.sum(((candidate["size"] - track.size) / np.array([1.5, .5, .7])) ** 2))
        return distance + .35 * float(np.linalg.slogdet(covariance)[1]) + .3 * size_error

    def _correct(self, track, candidate, timestamp):
        measurement = candidate["measurement_covariance"]
        innovation_covariance = track.covariance[:2, :2] + measurement
        gain = np.linalg.solve(innovation_covariance, track.covariance[:2, :]).T
        track.state += gain @ (candidate["position"] - track.position)
        residual = np.eye(4)
        residual[:, :2] -= gain
        # Joseph form preserves a positive covariance through repeated updates.
        track.covariance = residual @ track.covariance @ residual.T + gain @ measurement @ gain.T
        track.covariance = .5 * (track.covariance + track.covariance.T)
        norm = np.linalg.norm(track.velocity)
        if norm > self.max_vehicle_speed:
            track.state[2:] *= self.max_vehicle_speed / norm
        interval = timestamp - track.last_time
        track.consecutive_hits = track.consecutive_hits + 1 if interval <= .45 else 1
        track.hits += 1
        track.confirmed |= track.consecutive_hits >= self.confirmation_hits
        track.last_time = timestamp
        track.last_position = candidate["position"].copy()
        track.last_lateral = candidate["projection"]["lateral_m"]
        track.size = .8 * track.size + .2 * candidate["size"]
        track.score = candidate["score"]

    def _birth(self, candidate, timestamp):
        heading = candidate["projection"]["heading_rad"]
        c, s = math.cos(heading), math.sin(heading)
        basis = np.array([[c, -s], [s, c]])
        covariance = np.zeros((4, 4))
        covariance[:2, :2] = candidate["measurement_covariance"]
        covariance[2:, 2:] = basis @ np.diag([45. ** 2, 4.5 ** 2]) @ basis.T
        track = UncertainVehicleTrack(self.next_identifier,
            np.r_[candidate["position"], 0., 0.], covariance, timestamp, timestamp,
            candidate["position"].copy(), candidate["size"].copy(),
            candidate["projection"]["lateral_m"], candidate["score"],
            confirmed=self.confirmation_hits <= 1)
        self.next_identifier += 1
        self.tracks.append(track)

    def update(self, boxes, ego_position, ego_yaw, ego_progress, timestamp, ego_speed=0.):
        # ego_speed is accepted for the existing controller API; it is never an
        # opponent-velocity prior. Only successive measured positions supply it.
        timestamp = float(timestamp)
        if not np.isfinite(np.r_[ego_position, ego_yaw, ego_progress, timestamp, ego_speed]).all():
            raise ValueError("Tracker ego state and timestamp must be finite")
        if self.last_update_time is not None:
            if timestamp < self.last_update_time - 1e-9:
                raise ValueError("Detector timestamps must be monotonic")
            if timestamp <= self.last_update_time + 1e-9:
                return
        self.last_update_time = timestamp
        self.tracks = [track for track in self.tracks if timestamp - track.last_time <= self.max_age]
        for track in self.tracks:
            track.state, track.covariance = self._predict(track, timestamp)
            track.filter_time = timestamp
        candidates = self._candidates(boxes, ego_position, ego_yaw, ego_progress)
        used = set()
        if self.tracks and candidates:
            count = len(self.tracks)
            costs = np.full((count, len(candidates) + count), 12.)
            for i, track in enumerate(self.tracks):
                for j, candidate in enumerate(candidates):
                    costs[i, j] = self._association_cost(track, candidate, timestamp)
            row_indices, column_indices = linear_sum_assignment(costs)
            for i, j in zip(row_indices, column_indices):
                if j < len(candidates) and costs[i, j] < 12.:
                    self._correct(self.tracks[i], candidates[j], timestamp)
                    used.add(int(j))
        for index, candidate in enumerate(candidates):
            if index not in used:
                self._birth(candidate, timestamp)
        for track in self.tracks:
            if track.confirmed and not track.encountered:
                projection = project_near(self.reference, track.position, ego_progress, window=self.max_forward_m + 15.)
                tangent = np.array([math.cos(projection["heading_rad"]), math.sin(projection["heading_rad"])])
                if projection["progress_m"] - ego_progress > 5. and np.dot(track.velocity, tangent) >= -2.:
                    track.encountered = True
                    self.encounter_count += 1

    def _rows(self, ego_progress, timestamp, include_tentative=False):
        if not np.isfinite([ego_progress, timestamp]).all():
            raise ValueError("Observation progress and timestamp must be finite")
        if self.last_update_time is not None and timestamp < self.last_update_time - 1e-9:
            raise ValueError("Cannot observe a tracker before its latest detector update")
        rows = []
        for track in self.tracks:
            age = max(0., timestamp - track.last_time)
            if age > self.max_age or (not include_tentative and not track.confirmed):
                continue
            state, covariance = self._predict(track, timestamp)
            projection = project_near(self.reference, state[:2], ego_progress, window=self.max_forward_m + 15.)
            relative = projection["progress_m"] - ego_progress
            tangent = np.array([math.cos(projection["heading_rad"]), math.sin(projection["heading_rad"])])
            speed = float(np.clip(np.dot(state[2:], tangent), -self.max_vehicle_speed, self.max_vehicle_speed))
            speed_std = math.sqrt(max(0., float(tangent @ covariance[2:, 2:] @ tangent)))
            if track.confirmed and track.encountered and not track.passed and relative < -12.:
                track.passed = True
                self.pass_count += 1
            rows.append({"track_id": track.identifier, "relative_distance_m": relative,
                "lateral_m": projection["lateral_m"], "speed_mps": speed,
                "speed_std_mps": speed_std, "velocity_valid": track.hits >= 2,
                "position_std_m": math.sqrt(max(0., float(np.linalg.eigvalsh(covariance[:2, :2]).max()))),
                "position_m": state[:2].tolist(), "age_s": age, "score": track.score,
                "passed": track.passed, "confirmed": track.confirmed, "tentative": not track.confirmed,
                "measurement_hits": track.hits})
            if self.trajectory_motion_fields:
                normal = np.array([-tangent[1], tangent[0]])
                rows[-1].update(velocity_xy_mps=state[2:].tolist(),
                    velocity_covariance_mps2=covariance[2:, 2:].tolist(),
                    position_covariance_m2=covariance[:2, :2].tolist(),
                    lateral_speed_mps=float(np.dot(state[2:], normal)),
                    lateral_speed_std_mps=math.sqrt(max(0., float(normal @ covariance[2:, 2:] @ normal))),
                    heading_from_velocity_rad=(math.atan2(state[3], state[2])
                        if track.hits >= 2 and np.linalg.norm(state[2:]) > .5 else None),
                    projection_heading_rad=float(projection['heading_rad']))
        return rows

    def snapshot(self, ego_progress, timestamp):
        return self._rows(ego_progress, timestamp)

    def observation(self, ego_progress, body_speed, timestamp):
        all_rows = self._rows(ego_progress, timestamp, include_tentative=True)
        rows = [row for row in all_rows if row["confirmed"]]
        hazards = []
        for row in all_rows:
            persistent_evidence = row["confirmed"] or row["measurement_hits"] >= 2
            if persistent_evidence and row["age_s"] <= self.hazard_hold_s:
                row["control_evidence"] = "confirmed_identity" if row["confirmed"] else "two_measured_hits"
                hazards.append(row)
            elif row["age_s"] <= .25 and row["score"] >= self.rapid_hazard_confidence:
                row["control_evidence"] = "single_high_confidence_measurement"
                hazards.append(row)
        active = [row for row in hazards if -12. <= row["relative_distance_m"] <= self.max_forward_m]
        ahead = [row for row in active if row["relative_distance_m"] >= 0.]
        lead = min(ahead, key=lambda row: row["relative_distance_m"]) if ahead else (
            max(active, key=lambda row: row["relative_distance_m"]) if active else None)
        def clearance(lane):
            distances = [max(0., abs(row["relative_distance_m"]) - 5.) for row in hazards
                         if abs(row["lateral_m"] - lane) < 1.7]
            return min(10., min(distances, default=10.))
        return {"front_distance_m": lead["relative_distance_m"] if lead else 60.,
                "front_lateral_m": lead["lateral_m"] if lead else 0.,
                "front_speed_mps": lead["speed_mps"] if lead else body_speed,
                "left_clearance_m": clearance(2.5), "right_clearance_m": clearance(-2.5),
                "confirmed_tracks": rows, "hazard_tracks": hazards,
                "lead_track_id": lead["track_id"] if lead else None,
                "lead_confirmed": lead["confirmed"] if lead else False}


def policy_state(reference, progress, cross_track, yaw, body_speed, lane_offset, observation):
    """Exact feature contract used by the learned checkpoint's runtime."""
    reference_yaw = reference.sample(progress)[1]
    return {"progress_fraction": (progress / reference.length) % 1.,
            "body_speed_mps": float(body_speed),
            "heading_error_rad": wrap_angle(reference_yaw - yaw),
            "cross_track_m": float(cross_track),
            "curvature_ahead": [reference.sample(progress + d)[3] for d in CURVATURE_DISTANCES_M],
            "last_target_offset_m": float(lane_offset),
            **{key: float(observation[key]) for key in ("front_distance_m", "front_lateral_m", "front_speed_mps",
                                                       "left_clearance_m", "right_clearance_m")}}


def neutral_traffic_state(state):
    return {**state, "front_distance_m": 60., "front_lateral_m": 0.,
            "front_speed_mps": state["body_speed_mps"], "left_clearance_m": 10., "right_clearance_m": 10.,
            **({'hazard_tracks':[]} if 'hazard_tracks' in state else {})}


def shared_guard(raw_action, observation, *, body_speed, current_lane, feasible_speed,
                 dt, lane_rate=1.2, maximum_lane=3.2, brake_decel=8.):
    """Identical physical envelope, lane occupancy and stopping-gap guard.

    A lane change cannot sweep through an associated car alongside the ego.
    This guard never selects an overtake or adds a speed advantage to a policy.
    """
    requested_lane = float(np.clip(raw_action["target_offset_m"], -maximum_lane, maximum_lane))
    target_speed = float(np.clip(raw_action["target_speed_mps"], 0., feasible_speed))
    flags = []
    if raw_action["target_speed_mps"] > feasible_speed:
        flags.append("physical_speed_envelope")
    lane = current_lane + float(np.clip(requested_lane - current_lane, -lane_rate * dt, lane_rate * dt))
    tracks = observation.get("confirmed_tracks", [])
    for row in tracks:
        relative = float(row["relative_distance_m"])
        lateral = float(row["lateral_m"])
        if abs(relative) >= 9.:
            continue
        # Only veto motion toward/crossing a nearby car. Moving away remains
        # possible if the current state already has insufficient clearance.
        old_gap, new_gap = abs(current_lane - lateral), abs(lane - lateral)
        if new_gap < 2.35 and new_gap < old_gap - 1e-7:
            lane = current_lane
            flags.append("alongside_lane_occupied")
    for row in tracks:
        gap = float(row["relative_distance_m"])
        lead_speed = float(row["speed_mps"])
        lateral_gap = abs(float(row["lateral_m"]) - lane)
        closing = max(0., body_speed - lead_speed)
        if 0. < gap < max(18., 8. + closing * closing / (2. * brake_decel)) and lateral_gap < 2.3:
            guarded = max(0., lead_speed + .35 * (gap - 9.))
            if guarded < target_speed:
                target_speed = guarded
                flags.append("measured_vehicle_gap")
    return {"target_speed_mps": target_speed, "target_offset_m": float(lane),
            "requested_offset_m": requested_lane, "guards": sorted(set(flags))}


class RacePolicy:
    """A neural checkpoint or a deterministic, perception-aware baseline."""
    def __init__(self, reference, *, model=None, name="reference", seed=0):
        self.reference, self.model, self.name = reference, model, str(name)
        self.preferred_side = 1. if int(seed) % 2 == 0 else -1.
        self.classic_lane = 0.

    def predict(self, state, observation, feasible_speed):
        if self.model is not None:
            raw = self.model.predict(state)
            counterfactual = self.model.predict(neutral_traffic_state(state))
            changed = (abs(raw["target_speed_mps"] - counterfactual["target_speed_mps"]) > .5 or
                       abs(raw["target_offset_m"] - counterfactual["target_offset_m"]) > .2)
            return {"raw_action": raw, "no_traffic_counterfactual": counterfactual,
                    "vision_changed_action": bool(changed and observation.get("hazard_tracks", observation.get("confirmed_tracks")))}
        # Conventional comparator: keep the chosen passing lane until the car
        # being passed is safely behind. It has the same speed envelope as ML.
        tracks = observation.get("hazard_tracks", observation.get("confirmed_tracks", []))
        lookahead=min(85.,max(45.,8.+3.8*float(state.get('body_speed_mps',0.))))
        blocking = [row for row in tracks if -12. < row["relative_distance_m"] < lookahead
                    and abs(row["lateral_m"]) < 2.3]
        if blocking:
            def passing_clearance(side):
                # The learned-v1 summary saturates at10m. A physical passing
                # decision also needs the measured destination-lane occupancy
                # farther ahead, before committing to a blocked lane.
                gaps=[max(0.,row['relative_distance_m']-7.) for row in tracks
                      if -12.<row['relative_distance_m']<100. and abs(row['lateral_m']-side*3.2)<2.35]
                summary=observation['left_clearance_m' if side>0 else 'right_clearance_m']
                return min(min(gaps,default=100.), summary if summary<9.9 else 100.)
            options = [(passing_clearance(1.), 1.), (passing_clearance(-1.), -1.)]
            options.sort(key=lambda option: (option[0], option[1] == self.preferred_side), reverse=True)
            clearance, side = options[0]
            if clearance >= 8.:
                self.classic_lane = 3.2 * side
        elif not any(abs(row["relative_distance_m"]) < 14. for row in tracks):
            self.classic_lane = 0.
        raw = {"target_speed_mps": float(feasible_speed), "target_offset_m": self.classic_lane, "model_used": False}
        return {"raw_action": raw, "no_traffic_counterfactual":
                {"target_speed_mps": float(feasible_speed), "target_offset_m": 0., "model_used": False},
                "vision_changed_action": abs(self.classic_lane) > .2}
