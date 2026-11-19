"""Closed-loop racing-line following with a generic tire-grip speed budget.

This planner follows a public F1 position reference, but its speed limits describe
the selected generic Webots car. They are not a fitted Formula 1 tire model.
All steering returned here is positive-left; Webots Driver expects its negative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class RacingReference:
    def __init__(self, x, y, source_speed=None, *, max_speed=60.0, lateral_accel=8.0,
                 brake_decel=8.0, acceleration=5.0, spacing=2.0):
        xy = np.column_stack((x, y)).astype(float)
        if xy.ndim != 2 or len(xy) < 12 or not np.isfinite(xy).all():
            raise ValueError("Reference needs at least twelve finite XY samples")
        if min(max_speed, lateral_accel, brake_decel, acceleration, spacing) <= 0:
            raise ValueError("Speed, grip, braking, acceleration and spacing must be positive")
        source_speed = np.full(len(xy), max_speed) if source_speed is None else np.asarray(source_speed, float)
        if source_speed.shape != (len(xy),) or not np.isfinite(source_speed).all():
            raise ValueError("Reference speeds must match the finite XY samples")
        # Consecutive duplicate position packets must not create zero-length edges.
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.05]
        xy, source_speed = xy[keep], source_speed[keep]
        if np.linalg.norm(xy[-1] - xy[0]) < 0.05:
            xy, source_speed = xy[:-1], source_speed[:-1]
        closure = np.linalg.norm(xy[-1] - xy[0])
        path_length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        if closure > max(80., .025 * path_length):
            raise ValueError(f"Reference does not form a complete lap: endpoint gap {closure:.1f} m")
        closed = np.vstack((xy, xy[0]))
        raw_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
        count = max(100, int(math.ceil(raw_s[-1] / spacing)))
        uniform_s = np.linspace(0, raw_s[-1], count, endpoint=False)
        resampled = np.column_stack([np.interp(uniform_s, raw_s, closed[:, i]) for i in range(2)])
        # Public timing positions are approximate. A short periodic spatial filter
        # prevents packet/interpolation noise becoming rapid steering commands.
        self.xy = gaussian_filter1d(resampled, sigma=1.25, axis=0, mode="wrap")
        self.segment = np.roll(self.xy, -1, axis=0) - self.xy
        self.ds = np.linalg.norm(self.segment, axis=1)
        if np.any(self.ds < .01):
            raise ValueError("Reference contains degenerate segments after resampling")
        self.s = np.r_[0., np.cumsum(self.ds[:-1])]
        self.length = float(self.ds.sum())
        self.heading = np.arctan2(self.segment[:, 1], self.segment[:, 0])
        dx = (np.roll(self.xy, -1, axis=0) - np.roll(self.xy, 1, axis=0)) / 2
        ddx = np.roll(self.xy, -1, axis=0) - 2 * self.xy + np.roll(self.xy, 1, axis=0)
        curvature = (dx[:, 0] * ddx[:, 1] - dx[:, 1] * ddx[:, 0]) / np.maximum(np.linalg.norm(dx, axis=1) ** 3, 1e-6)
        self.curvature = gaussian_filter1d(curvature, 1.2, mode="wrap")
        source = np.interp(uniform_s, raw_s, np.r_[source_speed, source_speed[0]])
        self.speed = np.minimum(np.maximum(source, 3.), max_speed)
        self.speed = np.minimum(self.speed, np.sqrt(lateral_accel / np.maximum(np.abs(self.curvature), 1e-5)))
        # Repeated cyclic passes also propagate constraints across start/finish.
        for _ in range(5):
            for i in range(count - 1, -1, -1):
                j = (i + 1) % count
                self.speed[i] = min(self.speed[i], math.sqrt(self.speed[j] ** 2 + 2 * brake_decel * self.ds[i]))
            for i in range(count):
                j = (i + 1) % count
                self.speed[j] = min(self.speed[j], math.sqrt(self.speed[i] ** 2 + 2 * acceleration * self.ds[i]))
        self.max_speed = max_speed
        self.lateral_accel = lateral_accel
        self.brake_decel = brake_decel
        self.acceleration = acceleration
        self.source_endpoint_gap_m = float(closure)
        self.estimated_flying_lap_s = float(np.sum(self.ds / np.maximum(self.speed, 1)))

    @classmethod
    def from_npz(cls, path, *, ignore_source_speed=False, **kwargs):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data["x_m"], data["y_m"], None if ignore_source_speed else data["speed_mps"], **kwargs)

    def sample(self, progress):
        progress = float(progress) % self.length
        index = min(int(np.searchsorted(self.s, progress, side="right") - 1), len(self.xy) - 1)
        fraction = (progress - self.s[index]) / self.ds[index]
        point = self.xy[index] + fraction * self.segment[index]
        nxt = (index + 1) % len(self.xy)
        speed = (1 - fraction) * self.speed[index] + fraction * self.speed[nxt]
        return point, float(self.heading[index]), float(speed), float(self.curvature[index])

    def minimum_speed(self, progress, distance):
        """Minimum over the entire forward interval, including narrow apexes."""
        if distance < 0 or not math.isfinite(distance):
            raise ValueError("Preview distance must be finite and nonnegative")
        distance = min(float(distance),self.length)
        wrapped = float(progress) % self.length
        first = int(np.searchsorted(self.s,wrapped,side="right"))
        count = int(math.ceil(distance/np.min(self.ds))) + 2
        indices = np.arange(first,first+count)
        sample_s = self.s[indices % len(self.xy)] + (indices // len(self.xy))*self.length
        include = (sample_s >= wrapped) & (sample_s <= wrapped+distance)
        values = self.speed[indices[include] % len(self.xy)]
        endpoints = (self.sample(progress)[2],self.sample(progress+distance)[2])
        return min(*endpoints,float(values.min()) if len(values) else min(endpoints))


@dataclass
class ProgressState:
    progress_m: float
    cross_track_m: float
    distance_m: float
    reference_xy: np.ndarray


class ProgressTracker:
    """Locally project onto ordered segments, preventing crossing-track jumps."""
    def __init__(self, reference):
        self.reference = reference
        self.progress = 0.0

    def update(self, position, speed, dt):
        ref = self.reference
        position = np.asarray(position, float)[:2]
        lap = math.floor(self.progress / ref.length)
        wrapped = self.progress % ref.length
        base = int(np.searchsorted(ref.s, wrapped, side="right") - 1) + lap * len(ref.xy)
        forward_window = max(10., abs(speed) * dt * 5 + 6.)
        candidates = range(base - 5, base + int(math.ceil(forward_window / np.min(ref.ds))) + 3)
        best = None
        max_advance = max(4., abs(speed) * dt * 4 + 2.)
        for extended in candidates:
            i = extended % len(ref.xy)
            cycle = extended // len(ref.xy)
            t = float(np.clip(np.dot(position - ref.xy[i], ref.segment[i]) / ref.ds[i] ** 2, 0, 1))
            projected = ref.xy[i] + t * ref.segment[i]
            s = cycle * ref.length + ref.s[i] + t * ref.ds[i]
            if s < self.progress - 8 or s > self.progress + max_advance:
                continue
            distance = float(np.linalg.norm(position - projected))
            if best is None or distance < best[0]:
                normal = np.array([-ref.segment[i, 1], ref.segment[i, 0]]) / ref.ds[i]
                best = (distance, s, float(np.dot(position - projected, normal)), projected)
        if best is None:
            point, _, _, _ = ref.sample(self.progress)
            return ProgressState(self.progress, 0., float(np.linalg.norm(position - point)), point)
        distance, s, cte, point = best
        self.progress = max(self.progress, s, 0.)
        return ProgressState(self.progress, cte, distance, point)


class PurePursuit:
    def __init__(self, wheelbase=3.2, lookahead_seconds=.28, max_steer=.50, steer_rate=1.0):
        self.wheelbase = wheelbase
        self.lookahead_seconds = lookahead_seconds
        self.max_steer = max_steer
        self.steer_rate = steer_rate
        self.previous_steer = 0.

    def control(self, reference, progress, position, yaw, speed, dt, lateral_offset=0.0):
        lookahead = float(np.clip(4.5 + abs(speed) * self.lookahead_seconds, 5., 20.))
        target, target_yaw, _, _ = reference.sample(progress + lookahead)
        target = target + float(lateral_offset) * np.array([-math.sin(target_yaw),math.cos(target_yaw)])
        delta = target - np.asarray(position)[:2]
        distance = max(float(np.linalg.norm(delta)), .5)
        alpha = wrap_angle(math.atan2(delta[1], delta[0]) - yaw)
        steer = math.atan2(2 * self.wheelbase * math.sin(alpha), distance)
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        steer = float(np.clip(steer, self.previous_steer - self.steer_rate * dt,
                              self.previous_steer + self.steer_rate * dt))
        self.previous_steer = steer
        return steer, lookahead


def longitudinal_control(target_speed, body_speed, wheel_speed, *, mass=800., wheel_radius=.35,
                         brake_coefficient=700., max_accel=5., max_decel=8.):
    """Torque-mode body-speed feedback with a constant-torque braking budget."""
    if target_speed <= .15:
        # A queued car must hold still rather than repeatedly applying the
        # rolling-resistance feed-forward throttle at zero requested speed.
        if body_speed < .4:
            return 0., 1.
        decel = min(max_decel, max(.8, 1.5 * body_speed))
        return 0., float(np.clip(decel * mass * wheel_radius / (4 * brake_coefficient), 0., 1.))
    error = target_speed - body_speed
    if error >= -.3:
        throttle = float(np.clip(.12 + .20 * error, 0., 1.))
        # Wheel rotation can over-report speed under torque; reduce wheelspin.
        slip_excess = wheel_speed - body_speed - max(3., .16 * body_speed)
        if slip_excess > 0:
            throttle *= max(.08, 1. / (1. + .3 * slip_excess))
        return throttle, 0.
    decel = min(max_decel, -1.2 * error)
    # Webots driver.c normalizes damping by wheel angular speed, making the
    # configured brakeCoefficient a per-wheel torque budget, not viscous damping.
    brake = decel * mass * wheel_radius / (4 * brake_coefficient)
    brake = float(np.clip(brake, 0., 1.))
    # Do not infer wheel lock from Driver's finite-difference wheel encoders:
    # at this physics timestep their estimate under-reads fast rolling wheels.
    # True body velocity drives braking feedback instead.
    return 0., brake


@dataclass
class VisionTrack:
    identifier: int
    position: np.ndarray
    velocity: np.ndarray
    last_time: float
    hits: int = 1
    confirmed: bool = False
    encountered: bool = False
    passed: bool = False
    score: float = 0.


def project_near(reference, position, progress, window=90.):
    """Project a measured global location onto the nearby ordered track section."""
    n = len(reference.xy)
    base = int(np.searchsorted(reference.s,progress % reference.length,side="right")-1)
    base += math.floor(progress/reference.length)*n
    radius = int(math.ceil(window/np.min(reference.ds)))+2
    extended = np.arange(base-radius,base+radius+1)
    indices = extended % n
    segments = reference.segment[indices]
    delta = np.asarray(position)[:2]-reference.xy[indices]
    fraction = np.clip(np.sum(delta*segments,axis=1)/reference.ds[indices]**2,0,1)
    projected = reference.xy[indices]+fraction[:,None]*segments
    distances = np.linalg.norm(np.asarray(position)[:2]-projected,axis=1)
    best = int(np.argmin(distances))
    index = indices[best]
    normal = np.array([-reference.segment[index,1],reference.segment[index,0]])/reference.ds[index]
    return {"progress_m":float(reference.s[index]+(extended[best]//n)*reference.length+fraction[best]*reference.ds[index]),
            "lateral_m":float(np.dot(np.asarray(position)[:2]-projected[best],normal)),
            "distance_m":float(distances[best]),"heading_rad":float(reference.heading[index])}


class VisionTraffic:
    """Vehicle tracks built only from the detector's boxes and measured ego pose.

    Lateral positions are relative to the track reference, not to the ego lane.
    No simulator vehicle nodes, labels or object recognition IDs enter this class.
    """
    def __init__(self, reference, confidence=.35, confirmation_hits=3, max_age=2.):
        self.reference = reference
        self.confidence = confidence
        self.confirmation_hits = confirmation_hits
        self.max_age = max_age
        self.tracks = []
        self.next_identifier = 1
        self.encounter_count = 0
        self.pass_count = 0

    def update(self, boxes, ego_position, ego_yaw, ego_progress, timestamp):
        c,s = math.cos(ego_yaw),math.sin(ego_yaw)
        rotation = np.array([[c,-s],[s,c]])
        candidates = []
        for box in boxes:
            center,size = np.asarray(box.get("center",[]),float),np.asarray(box.get("size",[]),float)
            score = float(box.get("score",0))
            if center.shape!=(3,) or size.shape!=(3,) or not np.isfinite(np.r_[center,size,score]).all(): continue
            if score < self.confidence or not (3.0 <= size[0] <= 7.5 and 1.25 <= size[1] <= 2.9 and .55 <= size[2] <= 2.5): continue
            measured = rotation @ center[:2] + np.asarray(ego_position)[:2]
            projection = project_near(self.reference,measured,ego_progress)
            if abs(projection["lateral_m"]) > 4.5 or abs(projection["progress_m"]-ego_progress)>65: continue
            candidates.append((score,measured,projection,center[0]))
        matched = set()
        for score,measured,projection,ego_forward in sorted(candidates,key=lambda row:-row[0]):
            choices = []
            for track in self.tracks:
                if track.identifier in matched or timestamp-track.last_time > self.max_age: continue
                predicted = track.position+track.velocity*max(0,timestamp-track.last_time)
                distance = float(np.linalg.norm(measured-predicted))
                if distance < 6: choices.append((distance,track))
            if choices:
                track = min(choices,key=lambda row:row[0])[1]
                dt = timestamp-track.last_time
                if dt > 1e-4:
                    velocity = (measured-track.position)/dt
                    norm = np.linalg.norm(velocity)
                    if norm > 85: velocity *= 85/norm
                    track.velocity = .65*track.velocity+.35*velocity
                track.hits = track.hits+1 if dt <= .35 else 1
                track.position,track.last_time,track.score = measured,timestamp,score
                if track.hits >= self.confirmation_hits: track.confirmed = True
            else:
                if ego_forward < 4: continue
                if any(np.linalg.norm(measured-track.position)<4 for track in self.tracks): continue
                track = VisionTrack(self.next_identifier,measured,np.zeros(2),timestamp,score=score)
                self.next_identifier += 1
                self.tracks.append(track)
            matched.add(track.identifier)
            relative = projection["progress_m"]-ego_progress
            if track.confirmed and relative > 5 and not track.encountered:
                track.encountered = True
                self.encounter_count += 1
        self.tracks = [track for track in self.tracks if timestamp-track.last_time <= self.max_age]

    def snapshot(self, ego_progress, timestamp):
        rows = []
        for track in self.tracks:
            age = max(0,timestamp-track.last_time)
            if not track.confirmed or age > self.max_age: continue
            position = track.position+track.velocity*age
            projection = project_near(self.reference,position,ego_progress)
            relative = projection["progress_m"]-ego_progress
            tangent = np.array([math.cos(projection["heading_rad"]),math.sin(projection["heading_rad"])])
            speed = float(np.clip(np.dot(track.velocity,tangent),0,80))
            if track.encountered and not track.passed and relative < -12:
                track.passed = True
                self.pass_count += 1
            rows.append({"track_id":track.identifier,"relative_distance_m":relative,"lateral_m":projection["lateral_m"],
                         "speed_mps":speed,"position_m":position.tolist(),"age_s":age,"score":track.score,
                         "passed":track.passed})
        return rows

    def observation(self, ego_progress, body_speed, timestamp):
        rows = self.snapshot(ego_progress,timestamp)
        active = [row for row in rows if not row["passed"] and -12 <= row["relative_distance_m"] <= 60]
        ahead = [row for row in active if row["relative_distance_m"] >= 0]
        lead = min(ahead,key=lambda row:row["relative_distance_m"]) if ahead else (max(active,key=lambda row:row["relative_distance_m"]) if active else None)
        def clearance(lane):
            distances = [max(0,abs(row["relative_distance_m"])-5) for row in rows
                         if abs(row["lateral_m"]-lane)<1.7]
            return min(10.,min(distances,default=10.))
        return {"front_distance_m":lead["relative_distance_m"] if lead else 60.,
                "front_lateral_m":lead["lateral_m"] if lead else 0.,
                "front_speed_mps":lead["speed_mps"] if lead else body_speed,
                "left_clearance_m":clearance(2.5),"right_clearance_m":clearance(-2.5),
                "confirmed_tracks":rows,"lead_track_id":lead["track_id"] if lead else None}
