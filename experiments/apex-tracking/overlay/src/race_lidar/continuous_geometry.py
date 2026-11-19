"""Continuous map geometry on the original reference-progress parameter.

The supplied XY vertices and speed limits stay unchanged. A periodic C2 curve
provides one consistent position, tangent and curvature between vertices. It is
not a filter of measured pose, traffic, steering or actuator commands.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.interpolate import CubicSpline


class ContinuousGeometry:
    VERSION = 'periodic-c2-map-geometry-v1'
    continuous_geometry = True

    def __init__(self, source, *, spacing_m=.25):
        self.source = source
        self.length = float(source.length)
        spacing_m = float(spacing_m)
        s = np.asarray(source.s, float)
        xy = np.asarray(source.xy, float)
        if (not math.isfinite(self.length + spacing_m) or self.length <= 0.
                or not .1 <= spacing_m <= 1. or len(s) < 4
                or xy.shape != (len(s), 2) or not np.isfinite(np.r_[s, xy.ravel()]).all()
                or abs(s[0]) > 1e-8 or np.any(np.diff(s) <= 0.) or s[-1] >= self.length):
            raise ValueError('Continuous geometry requires finite ordered periodic map vertices')
        self._curve = CubicSpline(np.r_[s, self.length], np.vstack([xy, xy[0]]),
                                  axis=0, bc_type='periodic', extrapolate='periodic')
        # Retain every original profile vertex while adding a fine arc table.
        count = max(4, int(math.ceil(self.length / spacing_m)))
        fine = np.linspace(0., self.length, count, endpoint=False)
        nearest = np.searchsorted(s, fine)
        distance = np.minimum(abs(fine - s[np.clip(nearest, 0, len(s)-1)]),
                              abs(fine - s[np.clip(nearest-1, 0, len(s)-1)]))
        self.s = np.sort(np.r_[s, fine[distance > 1e-8]])
        self.base_ds = np.diff(np.r_[self.s, self.length])
        nodes, weights = np.polynomial.legendre.leggauss(6)
        locations = self.s[:, None] + .5 * self.base_ds[:, None] * (nodes + 1.)
        self._arc_weights = .5 * self.base_ds[:, None] * weights
        self._quadrature_metric = self.metric(locations)
        self._quadrature_curvature = self.curvature_at(locations)
        self.ds = np.sum(self._arc_weights * self._quadrature_metric, axis=1)
        if not np.isfinite(self.ds).all() or np.any(self.ds <= 1e-10):
            raise ValueError('Continuous map has a degenerate physical arc')
        self.xy = self._curve(self.s)
        first = self.derivative(self.s)
        self.heading = np.arctan2(first[:, 1], first[:, 0])
        self.curvature = self.curvature_at(self.s)
        self.arc = np.r_[0., np.cumsum(self.ds[:-1])]
        self.physical_length = float(self.ds.sum())
        old_speed = np.asarray(getattr(source, 'speed', []), float)
        old_caps = (np.interp(self.s, np.r_[s, self.length], np.r_[old_speed, old_speed[0]])
                    if old_speed.shape == s.shape else
                    np.array([source.sample(float(q))[2] for q in self.s]))
        limits = getattr(source, 'reference', source)
        self.lateral_accel = float(getattr(limits, 'lateral_accel', 8.))
        self.brake_decel = float(getattr(limits, 'brake_decel', 8.))
        self.acceleration = float(getattr(limits, 'acceleration', 5.))
        self.max_speed = float(getattr(limits, 'max_speed', max(old_caps)))
        self.speed = self._speed_envelope(self.curvature, self.ds, old_caps)
        self._offset_cache = None

    def __getattr__(self, name):
        return getattr(self.source, name)

    def position(self, progress):
        return np.asarray(self._curve(progress), float)

    def derivative(self, progress, order=1):
        if order not in (1, 2, 3):
            raise ValueError('Geometry derivative order must be1,2or3')
        return np.asarray(self._curve(progress, nu=order), float)

    def metric(self, progress):
        result = np.linalg.norm(self.derivative(progress), axis=-1)
        if np.any(~np.isfinite(result)) or np.any(result <= 1e-10):
            raise ValueError('Continuous geometry has a zero or invalid tangent')
        return float(result) if result.ndim == 0 else result

    def curvature_at(self, progress):
        first, second = self.derivative(progress), self.derivative(progress, 2)
        norm = np.linalg.norm(first, axis=-1)
        if np.any(norm <= 1e-10):
            raise ValueError('Continuous geometry has a zero tangent')
        result = (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]) / norm**3
        return float(result) if result.ndim == 0 else result

    def sample(self, progress):
        progress = float(progress)
        point = self.position(progress)
        first = self.derivative(progress)
        heading = math.atan2(first[1], first[0])
        speed = min(float(self.source.sample(progress)[2]), self._interpolate(self.speed, progress))
        return point, heading, speed, self.curvature_at(progress)

    def minimum_speed(self, progress, distance):
        return min(float(self.source.minimum_speed(progress, distance)),
                   self._minimum_profile(self.speed, progress, distance))

    def _interpolate(self, values, progress):
        return float(np.interp(float(progress) % self.length,
                               np.r_[self.s, self.length], np.r_[values, values[0]]))

    def _minimum_profile(self, values, progress, distance):
        if not math.isfinite(float(distance)) or distance < 0.:
            raise ValueError('Profile interval must be finite and nonnegative')
        distance = min(float(distance), self.length)
        ahead = np.mod(self.s - float(progress), self.length)
        selected = values[ahead <= distance]
        endpoints = min(self._interpolate(values, progress),
                        self._interpolate(values, float(progress) + distance))
        return min(endpoints, float(selected.min()) if len(selected) else endpoints)

    def _speed_envelope(self, curvature, ds, old_caps=None, valid=None):
        """Periodic physical-distance energy envelope, tightened only.

        The shortest path on the directed cycle is either clockwise with the
        acceleration budget or anticlockwise with the braking budget. Prefix
        and suffix minima give the full cyclic closure without Python sweeps.
        """
        cap = np.minimum(self.max_speed, np.sqrt(self.lateral_accel /
                         np.maximum(np.abs(curvature), 1e-10)))
        if old_caps is not None:
            cap = np.minimum(cap, old_caps)
        if valid is not None:
            cap = np.where(valid, cap, 0.)
        q = cap * cap
        arc = np.r_[0., np.cumsum(ds[:-1])]
        length = float(np.sum(ds))
        forward = q - 2. * self.acceleration * arc
        prefix = np.minimum.accumulate(forward)
        suffix = np.minimum.accumulate(forward[::-1])[::-1]
        wrapped = np.r_[suffix[1:], np.inf] + 2. * self.acceleration * length
        q_forward = np.minimum(prefix, wrapped) + 2. * self.acceleration * arc
        backward = q + 2. * self.brake_decel * arc
        prefix = np.minimum.accumulate(backward)
        suffix = np.minimum.accumulate(backward[::-1])[::-1]
        wrapped = np.r_[np.inf, prefix[:-1]] + 2. * self.brake_decel * length
        q_backward = np.minimum(suffix, wrapped) - 2. * self.brake_decel * arc
        return np.sqrt(np.maximum(0., np.minimum(q, np.minimum(q_forward, q_backward))))

    def _arc_coordinate(self, progress):
        cycle = math.floor(float(progress) / self.length)
        wrapped = float(progress) - cycle * self.length
        return cycle * self.physical_length + float(np.interp(
            wrapped, np.r_[self.s, self.length], np.r_[self.arc, self.physical_length]))

    def physical_distance(self, start, end):
        return self._arc_coordinate(end) - self._arc_coordinate(start)

    def advance_by_distance(self, progress, distance):
        target = self._arc_coordinate(progress) + float(distance)
        cycle = math.floor(target / self.physical_length)
        wrapped = target - cycle * self.physical_length
        return cycle * self.length + float(np.interp(
            wrapped, np.r_[self.arc, self.physical_length], np.r_[self.s, self.length]))

    def offset_arc_table(self, lane):
        """Physical arc and curvature of the same analytic parallel curve.

        Any interval touching the old0.20 parallel-scale threshold is marked
        invalid. The caller refuses feedforward smoothing on such a window.
        """
        lane = float(lane)
        if not math.isfinite(lane):
            raise ValueError('Offset must be finite')
        if self._offset_cache is not None and self._offset_cache[0] == lane:
            return self._offset_cache[1]
        scale = 1. - lane * self.curvature
        quad_scale = 1. - lane * self._quadrature_curvature
        valid = (scale > .20) & (np.min(quad_scale, axis=1) > .20)
        ds = np.sum(self._arc_weights * self._quadrature_metric * np.abs(quad_scale), axis=1)
        safe_scale = np.maximum(scale, .20)
        table = {'ds': np.maximum(ds, 1e-10), 'curvature': self.curvature / safe_scale,
                 'valid': valid, 'scale': scale,
                 'interval_minimum_scale': np.minimum(scale, np.min(quad_scale, axis=1)),
                 'minimum_scale': float(min(scale.min(), quad_scale.min()))}
        table['speed'] = self._speed_envelope(table['curvature'], table['ds'], valid=valid)
        self._offset_cache = (lane, table)
        return table

    def offset_minimum_speed(self, progress, distance, lane):
        return self._minimum_profile(self.offset_arc_table(lane)['speed'], progress, distance)

    def parallel_status(self, progress, lane, preview_m=20., minimum_scale=.20):
        preview_m = min(self.length, max(0., float(preview_m)))
        table = self.offset_arc_table(lane)
        ahead = np.mod(self.s - float(progress), self.length)
        first = int(np.searchsorted(self.s, float(progress) % self.length, side='right') - 1)
        mask = ahead <= preview_m
        mask[first] = True
        endpoint_scale = 1. - float(lane) * np.asarray(self.curvature_at(
            np.array([progress, progress + preview_m])))
        scale = float(min(np.min(table['interval_minimum_scale'][mask]), endpoint_scale.min()))
        return {'trackable': scale > minimum_scale, 'minimum_parallel_scale': scale,
                'preview_m': preview_m, 'minimum_allowed_scale': minimum_scale,
                'continuous_geometry_checked': True}
