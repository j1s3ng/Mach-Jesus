"""Map-only, periodic minimum-curvature line in the original Frenet frame.

The original map remains authoritative for progress, opponent projection and
track limits. This module does not use ground truth, learned predictions or
recorded racing pace. It plans a bounded simulator line, not an exact F1 line.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

VERSION = 'periodic-minimum-curvature-line-v1'


def _geometry(xy):
    edge = np.roll(xy, -1, axis=0) - xy
    ds = np.linalg.norm(edge, axis=1)
    if np.any(ds < .01) or not np.isfinite(xy).all():
        raise ValueError('Line contains non-finite or degenerate segments')
    incoming = xy - np.roll(xy, 1, axis=0)
    chord = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    denominator = np.linalg.norm(incoming, axis=1) * ds * np.linalg.norm(chord, axis=1)
    cross = incoming[:, 0] * edge[:, 1] - incoming[:, 1] * edge[:, 0]
    curvature = 2. * cross / np.maximum(denominator, 1e-9)
    heading = np.arctan2(chord[:, 1], chord[:, 0])
    return ds, heading, curvature


def _profile(xy, *, max_speed, lateral_accel, brake_decel, acceleration):
    ds, heading, curvature = _geometry(xy)
    # Match the original map's short spatial anti-noise treatment. The profile
    # is derived from this actual offset path, not the center-path curvature.
    smooth = gaussian_filter1d(curvature, 1.2, mode='wrap')
    speed = np.minimum(max_speed, np.sqrt(lateral_accel / np.maximum(np.abs(smooth), 1e-5)))
    for _ in range(5):
        for i in range(len(ds) - 1, -1, -1):
            j = (i + 1) % len(ds)
            speed[i] = min(speed[i], math.sqrt(speed[j] ** 2 + 2. * brake_decel * ds[i]))
        for i in range(len(ds)):
            j = (i + 1) % len(ds)
            speed[j] = min(speed[j], math.sqrt(speed[i] ** 2 + 2. * acceleration * ds[i]))
    return speed, ds, heading, smooth


def _residual(offset, xy, normal, base_ds, offset_weight, slope_weight, bend_weight):
    path = xy + offset[:, None] * normal
    ds, _, curvature = _geometry(path)
    local_ds = .5 * (ds + np.roll(ds, 1))
    slope = (np.roll(offset, -1) - offset) / base_ds
    bend = (np.roll(offset, -1) - 2. * offset + np.roll(offset, 1)) / np.maximum(base_ds ** 2, .01)
    return np.concatenate((curvature * np.sqrt(local_ds),
        math.sqrt(offset_weight) * offset * np.sqrt(base_ds),
        math.sqrt(slope_weight) * slope * np.sqrt(base_ds),
        math.sqrt(bend_weight) * bend * np.sqrt(base_ds)))


def _jacobian_sparsity(n):
    rows, cols = [], []
    for i in range(n):
        for j in ((i - 1) % n, i, (i + 1) % n):
            rows.append(i); cols.append(j)
            rows.append(3 * n + i); cols.append(j)
        rows.append(n + i); cols.append(i)
        for j in (i, (i + 1) % n):
            rows.append(2 * n + i); cols.append(j)
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(4 * n, n)).tocsr()


def _rate_project(offset, allowed_step, iterations=600):
    """Project onto periodic neighboring-offset rate bounds without new wiggles."""
    value = offset.copy()
    count = len(value)
    pairs = [np.arange(parity, count - 1, 2) for parity in (0, 1)]
    for _ in range(iterations):
        largest = 0.
        for first in pairs:
            second = first + 1
            difference = value[second] - value[first]
            correction = .5 * (difference - np.clip(difference, -allowed_step[first], allowed_step[first]))
            value[first] += correction
            value[second] -= correction
            largest = max(largest, float(np.max(np.abs(correction))))
        difference = value[0] - value[-1]
        correction = .5 * (difference - np.clip(difference, -allowed_step[-1], allowed_step[-1]))
        value[-1] += correction
        value[0] -= correction
        largest = max(largest, abs(float(correction)))
        if largest < 1e-8:
            break
    return value


class SmoothRacingLine:
    """Periodic line and speed profiles indexed by ORIGINAL reference progress.

    ``nominal_offset_m`` returns an absolute offset in the original map's normal
    direction. A policy may instead request either passing side; call
    ``feasible_speed(..., lateral_offset=d)`` for that absolute constant lane.
    Projection/ranking must continue to use the original reference.
    """
    def __init__(self, reference, *, offset_limit_m=2.8, passing_limit_m=4.,
                 optimize=True, max_nfev=300, smoothing_distances_m=(6., 12., 20.),
                 offset_weight=1e-6, slope_weight=.002, bend_weight=.2,
                 nominal_lateral_rate_mps=1.4):
        self.reference = reference
        self.length = float(reference.length)
        self.s = np.asarray(reference.s, float).copy()
        self.base_ds = np.asarray(reference.ds, float).copy()
        self.offset_limit_m = float(offset_limit_m)
        self.passing_limit_m = float(passing_limit_m)
        self.nominal_lateral_rate_mps = float(nominal_lateral_rate_mps)
        if not 0. < self.nominal_lateral_rate_mps <= 1.8:
            raise ValueError("Nominal lateral rate must fit the controller guard")
        if (not 0. < self.offset_limit_m <= self.passing_limit_m <= 4.
                or min(offset_weight, slope_weight, bend_weight) < 0.):
            raise ValueError('Invalid line bounds or regularization')
        self.normal = np.column_stack((-np.sin(reference.heading), np.cos(reference.heading)))
        self.limits = dict(max_speed=float(reference.max_speed),
            lateral_accel=float(reference.lateral_accel), brake_decel=float(reference.brake_decel),
            acceleration=float(reference.acceleration))
        base_profile = _profile(reference.xy, **self.limits)
        self.baseline_profile = base_profile[0]
        self.baseline_metrics = self._metrics(reference.xy, base_profile)
        n = len(self.s)
        solution = np.zeros(n)
        self.plan_artifact_sha256 = None
        self.optimization_executed_this_instance = bool(optimize)
        self.optimization = {'enabled': bool(optimize), 'method': 'bounded_sparse_periodic_true_curvature_least_squares',
            'known_map_only': True, 'max_nfev': int(max_nfev)}
        if optimize:
            result = least_squares(_residual, solution, bounds=(-self.offset_limit_m, self.offset_limit_m),
                args=(np.asarray(reference.xy, float), self.normal, self.base_ds,
                      offset_weight, slope_weight, bend_weight),
                jac_sparsity=_jacobian_sparsity(n), method='trf', tr_solver='lsmr',
                ftol=2e-7, xtol=2e-7, gtol=2e-7, max_nfev=int(max_nfev))
            solution = np.asarray(result.x, float)
            self.optimization.update(nfev=int(result.nfev), success=bool(result.success),
                status=int(result.status), message=str(result.message), cost=float(result.cost),
                solver_converged=bool(result.success), global_optimum_claimed=False)
        candidates = []
        # Smoothing choices are predeclared, map-only. Include the original line
        # as a fallback so a numerical optimization cannot worsen its estimate.
        for smoothing in (None, *smoothing_distances_m):
            if smoothing is None:
                offset = np.zeros(n)
            else:
                offset = gaussian_filter1d(solution, max(.5, float(smoothing) / np.mean(self.base_ds)), mode='wrap')
                offset = np.clip(offset, -self.offset_limit_m, self.offset_limit_m)
            xy = reference.xy + offset[:, None] * self.normal
            profile = _profile(xy, **self.limits)
            if smoothing is not None:
                fastest = np.maximum(self.baseline_profile, profile[0])
                for _ in range(4):
                    # Reserve a small margin for profile changes caused by the
                    # projection itself. Original progress is the rate frame.
                    edge_speed = np.maximum(fastest, np.roll(fastest, -1))
                    step = self.nominal_lateral_rate_mps * self.base_ds / np.maximum(1.05 * edge_speed, 1.)
                    offset = _rate_project(offset, step)
                    xy = reference.xy + offset[:, None] * self.normal
                    profile = _profile(xy, **self.limits)
                    fastest = np.maximum(fastest, profile[0])
            metrics = self._metrics(xy, profile)
            candidates.append((metrics['estimated_flying_lap_s'], offset, xy, profile, metrics, smoothing))
        _, self.offset, self.xy, profile, self.metrics, selected_smoothing = min(candidates, key=lambda x: x[0])
        self.speed, self.ds, self.heading, self.curvature = profile
        self.line_length_m = float(self.ds.sum())
        self.estimated_flying_lap_s = self.metrics['estimated_flying_lap_s']
        self.optimization['selected_smoothing_distance_m'] = selected_smoothing
        self.optimization['selected_zero_offset_fallback'] = selected_smoothing is None
        self.optimization['candidate_metrics'] = [dict(smoothing_distance_m=c[-1], **c[-2]) for c in candidates]
        self._lane_profiles = {}
        for offset in np.linspace(-self.passing_limit_m, self.passing_limit_m,
                                  int(round(2 * self.passing_limit_m / .5)) + 1):
            self._lane_profiles[round(float(offset), 8)] = _profile(
                reference.xy + offset * self.normal, **self.limits)[0]
        for offset in (-3.2, 0., 3.2):
            if abs(offset) <= self.passing_limit_m:
                self._lane_profiles[offset] = _profile(reference.xy + offset * self.normal, **self.limits)[0]
        self._lane_keys = np.array(sorted(self._lane_profiles))

    @staticmethod
    def _reference_digest(reference):
        digest = hashlib.sha256()
        for value in (reference.xy, reference.s, reference.ds, reference.heading):
            digest.update(np.ascontiguousarray(value, dtype='<f8').tobytes())
        digest.update(np.asarray([reference.length], dtype='<f8').tobytes())
        return digest.hexdigest()

    def save_plan(self, path):
        metadata = self.report()
        metadata.update(schema_version=1, reference_geometry_sha256=self._reference_digest(self.reference),
            module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        np.savez_compressed(path, metadata_json=json.dumps(metadata, sort_keys=True),
            original_s=self.s, offset=self.offset, xy=self.xy, speed=self.speed,
            ds=self.ds, heading=self.heading, curvature=self.curvature,
            baseline_profile=self.baseline_profile, lane_keys=self._lane_keys,
            lane_profiles=np.stack([self._lane_profiles[float(k)] for k in self._lane_keys]))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def from_npz(cls, reference, path, *, expected_sha256):
        """Load the exact offline-planned map artifact; never run an optimizer."""
        path = Path(path)
        artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if artifact_sha != expected_sha256:
            raise ValueError('Racing-line artifact SHA256 differs')
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json'].item()))
            arrays = {name: np.asarray(data[name], dtype=float).copy() for name in
                ('original_s', 'offset', 'xy', 'speed', 'ds', 'heading', 'curvature',
                 'baseline_profile', 'lane_keys', 'lane_profiles')}
        if metadata.get('schema_version') != 1 or metadata.get('version') != VERSION:
            raise ValueError('Unsupported racing-line artifact')
        if metadata['reference_geometry_sha256'] != cls._reference_digest(reference):
            raise ValueError('Racing-line artifact belongs to a different ranking map')
        if metadata['module_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('Racing-line artifact planner source differs')
        limits = dict(max_speed=float(reference.max_speed), lateral_accel=float(reference.lateral_accel),
            brake_decel=float(reference.brake_decel), acceleration=float(reference.acceleration))
        if limits != metadata['limits']:
            raise ValueError('Racing-line artifact physical profile limits differ')
        count = len(reference.s)
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError('Racing-line artifact has non-finite arrays: ' + name)
        for name in ('original_s', 'offset', 'speed', 'ds', 'heading', 'curvature', 'baseline_profile'):
            if arrays[name].shape != (count,):
                raise ValueError('Racing-line artifact array shape differs: ' + name)
        if (arrays['xy'].shape != (count, 2) or arrays['lane_profiles'].shape != (len(arrays['lane_keys']), count)
                or not np.array_equal(arrays['original_s'], reference.s)):
            raise ValueError('Racing-line artifact geometry shape/index differs')
        if (np.any(arrays['ds'] <= 0.) or np.any(arrays['speed'] <= 0.)
                or np.any(arrays['lane_profiles'] <= 0.) or np.any(np.diff(arrays['lane_keys']) <= 0.)):
            raise ValueError('Racing-line artifact has invalid segment/speed/cache values')
        self = cls.__new__(cls)
        self.reference, self.length = reference, float(reference.length)
        self.s, self.base_ds = arrays.pop('original_s'), np.asarray(reference.ds, float).copy()
        self.offset_limit_m = float(metadata['nominal_offset_limit_m'])
        self.passing_limit_m = float(metadata['passing_offset_limit_m'])
        self.nominal_lateral_rate_mps = float(metadata['nominal_lateral_rate_budget_mps'])
        self.normal = np.column_stack((-np.sin(reference.heading), np.cos(reference.heading)))
        if (np.max(np.abs(arrays['offset'])) > self.offset_limit_m + 1e-8
                or not np.allclose(arrays['xy'], reference.xy + arrays['offset'][:, None] * self.normal, rtol=0., atol=1e-9)):
            raise ValueError('Racing-line artifact leaves its bounded original-map corridor')
        for name in ('offset', 'xy', 'speed', 'ds', 'heading', 'curvature', 'baseline_profile'):
            setattr(self, name, arrays[name])
        self.limits = limits
        self.metrics, self.baseline_metrics = metadata['optimized'], metadata['baseline']
        self.line_length_m = float(self.ds.sum())
        self.estimated_flying_lap_s = float(self.metrics['estimated_flying_lap_s'])
        self.optimization = metadata['optimization']
        self.optimization_executed_this_instance = False
        self.plan_artifact_sha256 = artifact_sha
        self._lane_keys = arrays['lane_keys']
        self._lane_profiles = {float(k): arrays['lane_profiles'][i] for i, k in enumerate(self._lane_keys)}
        return self

    @staticmethod
    def _metrics(xy, profile):
        speed, ds, _, curvature = profile
        return {'line_length_m': float(ds.sum()),
            'integrated_squared_curvature': float(np.sum(curvature ** 2 * .5 * (ds + np.roll(ds, 1)))),
            'maximum_absolute_curvature_per_m': float(np.max(np.abs(curvature))),
            'estimated_flying_lap_s': float(np.sum(ds / np.maximum(speed, 1.))),
            'minimum_profile_speed_mps': float(np.min(speed)),
            'maximum_profile_speed_mps': float(np.max(speed))}

    def _interpolate(self, values, progress):
        wrapped = float(progress) % self.length
        return np.interp(wrapped, np.r_[self.s, self.length], np.r_[values, values[0]])

    def nominal_offset_m(self, progress):
        return float(self._interpolate(self.offset, progress))

    def sample(self, progress):
        # Points and angles are periodic, including the final start/finish edge.
        point = np.array([self._interpolate(self.xy[:, 0], progress),
                          self._interpolate(self.xy[:, 1], progress)])
        heading = math.atan2(self._interpolate(np.sin(self.heading), progress),
                             self._interpolate(np.cos(self.heading), progress))
        return point, heading, float(self._interpolate(self.speed, progress)), \
            float(self._interpolate(self.curvature, progress))

    def _minimum_profile(self, speed, progress, distance):
        if distance < 0 or not math.isfinite(distance):
            raise ValueError('Preview distance must be finite and nonnegative')
        distance = min(float(distance), self.length)
        wrapped = float(progress) % self.length
        count = int(math.ceil(distance / np.min(self.base_ds))) + 3
        first = int(np.searchsorted(self.s, wrapped, side='right'))
        indices = np.arange(first, first + count)
        sample_s = self.s[indices % len(self.s)] + (indices // len(self.s)) * self.length
        include = (sample_s >= wrapped) & (sample_s <= wrapped + distance)
        values = speed[indices[include] % len(self.s)]
        endpoints = [self._interpolate(speed, progress), self._interpolate(speed, progress + distance)]
        return float(min(*endpoints, values.min() if len(values) else math.inf))

    def minimum_speed(self, progress, distance):
        return self._minimum_profile(self.speed, progress, distance)

    def feasible_speed(self, progress, distance, lateral_offset=None):
        if lateral_offset is None:
            return self.minimum_speed(progress, distance)
        offset = float(lateral_offset)
        if not math.isfinite(offset) or abs(offset) > self.passing_limit_m + 1e-7:
            raise ValueError('Passing lane outside the configured corridor')
        upper = int(np.searchsorted(self._lane_keys, offset, side='left'))
        upper = min(upper, len(self._lane_keys) - 1)
        lower = max(0, upper - 1)
        keys = [self._lane_keys[upper]]
        if abs(self._lane_keys[upper] - offset) > 1e-7:
            keys.append(self._lane_keys[lower])
        # Adjacent absolute-offset profiles bound an intermediate offset rather
        # than inheriting the optimized line's possibly faster apex curvature.
        return min(self._minimum_profile(self._lane_profiles[float(k)], progress, distance) for k in keys)

    def report(self):
        return {'version': VERSION, 'optimization_executed_this_instance': self.optimization_executed_this_instance,
            'plan_artifact_sha256': self.plan_artifact_sha256, 'scope': 'known approximate simulator map; not a surveyed F1 racing line',
            'reference_length_m_for_ranking': self.length, 'original_reference_geometry_changed': False,
            'nominal_offset_limit_m': self.offset_limit_m, 'passing_offset_limit_m': self.passing_limit_m,
            'maximum_nominal_offset_m': float(np.max(np.abs(self.offset))),
            'maximum_offset_slope_m_per_m': float(np.max(np.abs(np.diff(np.r_[self.offset, self.offset[0]]) / self.base_ds))),
            'nominal_lateral_rate_budget_mps': self.nominal_lateral_rate_mps,
            'maximum_profile_lateral_rate_mps': float(np.max(np.abs(np.diff(np.r_[self.offset, self.offset[0]]) / self.base_ds) * np.maximum(self.speed, np.roll(self.speed, -1)))),
            'limits': self.limits, 'baseline': self.baseline_metrics, 'optimized': self.metrics,
            'optimization': self.optimization,
            'estimated_lap_time_is_model_only_not_a_simulation_result': True}
