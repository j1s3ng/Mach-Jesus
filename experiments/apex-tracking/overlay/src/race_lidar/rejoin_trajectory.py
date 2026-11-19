"""Finite, measured-traffic-checked C2 return to a supplied mapped line.

The original progress parameter and actual road strip remain authoritative.
This module constructs geometry and command-model speed caps, not a guarantee
about unobserved traffic or measured tire forces. The caller retains all normal
traffic, braking, spin and road guards and caches an accepted route until it is
completed or invalidated. No opponent ground truth is accepted by this API.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

VERSION = 'finite-intrinsic-arc-c2-measured-rejoin-v2'


def _position(geometry, progress):
    values = np.asarray(progress, float)
    if hasattr(geometry, 'position'):
        return np.asarray(geometry.position(values), float)
    if values.ndim == 0:
        return np.asarray(geometry.sample(float(values))[0], float)
    return np.array([geometry.sample(float(s))[0] for s in values.ravel()]).reshape(values.shape + (2,))


def _wrap(angle):
    return (float(angle) + math.pi) % (2. * math.pi) - math.pi


def _first_overlap(times, inequalities):
    """Intersect linearly swept separating intervals between every sample."""
    values = np.asarray(inequalities, float)
    before, after = values[:, :-1], values[:, 1:]
    change = after - before
    crossing = np.divide(-before, change, out=np.zeros_like(change), where=abs(change) > 1e-12)
    lower = np.max(np.where(change < -1e-12, crossing, 0.), axis=0)
    upper = np.min(np.where(change > 1e-12, crossing, 1.), axis=0)
    impossible = np.any((abs(change) <= 1e-12) & (before > 0.), axis=0)
    valid = (~impossible) & (lower <= upper) & (lower <= 1.) & (upper >= 0.)
    if np.any(valid):
        i = int(np.flatnonzero(valid)[0])
        return float(times[i] + np.diff(times)[i] * max(0., lower[i]))
    return None


class PhysicalArcNominalGeometry:
    """Intrinsic C2 spline through unchanged XY vertices, indexed by chord arc.

    A monotone periodic PCHIP maps original progress to the physical-polyline
    arc parameter. Chain-rule derivatives retain that mapping explicitly;
    normalized heading and geometric curvature do not depend on its scale.
    This geometry is used inside a finite return only, not to retime or replace
    the already captured nominal controller after handover.
    """
    def __init__(self, source, *, analytic_fallback=None):
        self.source = source
        self.length = float(source.length)
        self.analytic = not all(hasattr(source, name) for name in ('s', 'xy'))
        if self.analytic:
            if not getattr(analytic_fallback, 'intrinsic_arc_parameter', False):
                raise ValueError('Nominal arc needs original XY vertices or an explicit intrinsic-arc geometry')
            self.geometry = analytic_fallback
            self.physical_length = self.length
            self.source_s = None
            return
        self.source_s = np.asarray(source.s, float).copy()
        xy = np.asarray(source.xy, float)
        if (xy.shape != (len(self.source_s), 2) or len(xy) < 4
                or not np.isfinite(np.r_[self.source_s, xy.ravel(), self.length]).all()
                or abs(self.source_s[0]) > 1e-8 or np.any(np.diff(self.source_s) <= 0.)
                or self.source_s[-1] >= self.length):
            raise ValueError('Invalid original nominal XY/parameter table')
        chord = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
        if np.any(chord <= 1e-6):
            raise ValueError('Nominal arc has a degenerate original segment')
        arc = np.r_[0., np.cumsum(chord)]
        self.physical_length = float(arc[-1])
        self._curve = CubicSpline(arc, np.vstack([xy, xy[0]]), axis=0,
                                  bc_type='periodic', extrapolate='periodic')
        extended_s = np.r_[self.source_s - self.length, self.source_s,
                           self.source_s + self.length, 2. * self.length]
        extended_arc = np.r_[arc[:-1] - arc[-1], arc[:-1],
                             arc[:-1] + arc[-1], 2. * arc[-1]]
        self._map = PchipInterpolator(extended_s, extended_arc)
        self._inverse_map = PchipInterpolator(extended_arc, extended_s)
        # Numerical physical arc of the spline itself (not chord distance).
        self._arc_grid = np.unique(np.r_[arc, np.linspace(0., arc[-1],
            int(math.ceil(arc[-1] / .25)) + 1)])
        nodes, weights = np.polynomial.legendre.leggauss(6)
        width = np.diff(self._arc_grid)
        location = self._arc_grid[:-1, None] + .5 * width[:, None] * (nodes + 1.)
        segments = .5 * width * np.sum(np.linalg.norm(self._curve(location, 1), axis=-1) * weights, axis=1)
        self._true_arc = np.r_[0., np.cumsum(segments)]
        self.true_physical_length = float(self._true_arc[-1])

    def arc_parameter(self, progress):
        progress = np.asarray(progress, float)
        if self.analytic:
            return progress
        cycle = np.floor(progress / self.length)
        return self._map(progress - cycle * self.length) + cycle * self.physical_length

    def parameter_derivative(self, progress, order=1):
        progress = np.asarray(progress, float)
        if self.analytic:
            return np.ones_like(progress) if order == 1 else np.zeros_like(progress)
        wrapped = np.mod(progress, self.length)
        return np.asarray(self._map(wrapped, nu=order), float)

    def arc_derivative(self, progress, order=1):
        if self.analytic:
            return np.asarray(self.geometry.derivative(progress, order), float)
        return np.asarray(self._curve(self.arc_parameter(progress), nu=order), float)

    def position(self, progress):
        if self.analytic:
            return _position(self.geometry, progress)
        return np.asarray(self._curve(self.arc_parameter(progress)), float)

    def derivative(self, progress, order=1):
        one = self.parameter_derivative(progress, 1)
        first = self.arc_derivative(progress, 1)
        if order == 1:
            return first * one[..., None]
        two = self.parameter_derivative(progress, 2)
        second = self.arc_derivative(progress, 2)
        if order == 2:
            return second * one[..., None]**2 + first * two[..., None]
        if order == 3:
            three = self.parameter_derivative(progress, 3)
            return (self.arc_derivative(progress, 3) * one[..., None]**3
                    + 3. * second * one[..., None] * two[..., None] + first * three[..., None])
        raise ValueError('Derivative order must be1,2or3')

    def metric(self, progress):
        return np.linalg.norm(self.derivative(progress, 1), axis=-1)

    def curvature_at(self, progress):
        first, second = self.arc_derivative(progress, 1), self.arc_derivative(progress, 2)
        return (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]) / np.linalg.norm(first, axis=-1)**3

    def sample(self, progress):
        first = self.arc_derivative(progress, 1)
        return self.position(progress), math.atan2(first[1], first[0]), float(self.source.minimum_speed(progress, 0.)), float(self.curvature_at(progress))

    def minimum_speed(self, progress, span):
        return float(self.source.minimum_speed(progress, span))

    def _true_arc_coordinate(self, progress):
        arc = self.arc_parameter(progress)
        if self.analytic:
            return arc
        cycle = np.floor(arc / self.physical_length)
        return np.interp(arc - cycle * self.physical_length, self._arc_grid,
                         self._true_arc) + cycle * self.true_physical_length

    def physical_distance(self, start, end):
        return float(self._true_arc_coordinate(end) - self._true_arc_coordinate(start))

    def advance_by_distance(self, progress, distance):
        target = self._true_arc_coordinate(progress) + np.asarray(distance, float)
        if self.analytic:
            return target
        cycle = np.floor(target / self.true_physical_length)
        arc = np.interp(target - cycle * self.true_physical_length,
                        self._true_arc, self._arc_grid)
        return self._inverse_map(arc) + cycle * self.length

    def snap_source_vertex(self, requested, minimum, maximum):
        if self.analytic:
            return float(np.clip(requested, minimum, maximum))
        cycle = math.floor(float(requested) / self.length)
        values = np.r_[self.source_s + (cycle - 1) * self.length,
                       self.source_s + cycle * self.length,
                       self.source_s + (cycle + 1) * self.length]
        values = values[(values >= minimum) & (values <= maximum)]
        if not len(values):
            raise ValueError('No original nominal vertex in the join interval')
        return float(values[np.argmin(abs(values - requested))])


_ARC_GEOMETRY_CACHE = {}


def _arc_geometry(source, fallback):
    cached = _ARC_GEOMETRY_CACHE.get(id(source))
    if cached is None or cached[0] is not source or cached[1] is not fallback:
        cached = (source, fallback, PhysicalArcNominalGeometry(source, analytic_fallback=fallback))
        _ARC_GEOMETRY_CACHE[id(source)] = cached
    return cached[2]


class RejoinTrajectory:
    """Nominal geometry plus a quintic XY correction on unwrapped progress."""
    VERSION = VERSION
    finite_domain = True

    def __init__(self, reference, nominal_geometry, nominal_profile, *, start_progress_m,
                 end_progress_m, ego_xy, ego_yaw, entry_curvature_per_m,
                 max_speed_mps=100., lateral_accel_mps2=8., brake_decel_mps2=8.,
                 acceleration_mps2=5., arc_step_m=.25, tail_progress_m=100.):
        self.reference = reference
        self.nominal_geometry = (nominal_geometry if isinstance(nominal_geometry, PhysicalArcNominalGeometry)
                                 else _arc_geometry(nominal_profile, nominal_geometry))
        nominal_geometry = self.nominal_geometry
        self.nominal_profile = nominal_profile
        self.start_progress_m = float(start_progress_m)
        self.end_progress_m = float(end_progress_m)
        self.span_m = self.end_progress_m - self.start_progress_m
        self.arc_start_m = float(nominal_geometry.arc_parameter(self.start_progress_m))
        self.arc_end_m = float(nominal_geometry.arc_parameter(self.end_progress_m))
        self.arc_span_m = self.arc_end_m - self.arc_start_m
        self.length = float(reference.length)  # Original map period, not route arc length.
        self.max_speed_mps = float(max_speed_mps)
        self.lateral_accel_mps2 = float(lateral_accel_mps2)
        self.brake_decel_mps2 = float(brake_decel_mps2)
        self.acceleration_mps2 = float(acceleration_mps2)
        self.ego_xy = np.asarray(ego_xy, float)[:2].copy()
        self.ego_yaw = float(ego_yaw)
        self.entry_curvature_per_m = float(entry_curvature_per_m)
        values = [self.start_progress_m, self.end_progress_m, self.ego_yaw,
                  self.entry_curvature_per_m, self.max_speed_mps, self.lateral_accel_mps2,
                  self.brake_decel_mps2, self.acceleration_mps2, arc_step_m, tail_progress_m]
        if (self.ego_xy.shape != (2,) or not np.isfinite(self.ego_xy).all()
                or not np.isfinite(values).all() or not 5. <= self.span_m <= 200.
                or min(self.max_speed_mps, self.lateral_accel_mps2,
                       self.brake_decel_mps2, self.acceleration_mps2) <= 0.
                or not .1 <= arc_step_m <= .5 or tail_progress_m < 20.):
            raise ValueError('Finite pose, positive original-progress span and physical budgets required')
        p0 = _position(nominal_geometry, self.start_progress_m)
        n1 = np.asarray(nominal_geometry.arc_derivative(self.start_progress_m, 1), float)
        n2 = np.asarray(nominal_geometry.arc_derivative(self.start_progress_m, 2), float)
        metric0 = float(np.linalg.norm(n1))
        if not math.isfinite(metric0) or metric0 <= 1e-6:
            raise ValueError('Nominal join tangent is degenerate')
        tangent = np.array([math.cos(self.ego_yaw), math.sin(self.ego_yaw)])
        normal = np.array([-tangent[1], tangent[0]])
        metric_prime = float(np.dot(n1, n2) / metric0)
        self.entry_arc_first = metric0 * tangent
        self.entry_arc_second = metric_prime * tangent + self.entry_curvature_per_m * metric0**2 * normal
        a1 = float(nominal_geometry.parameter_derivative(self.start_progress_m, 1))
        a2 = float(nominal_geometry.parameter_derivative(self.start_progress_m, 2))
        self.entry_first = self.entry_arc_first * a1
        self.entry_second = self.entry_arc_second * a1**2 + self.entry_arc_first * a2
        coefficients = np.zeros((6, 2))
        coefficients[0] = self.ego_xy - p0
        coefficients[1] = (self.entry_arc_first - n1) * self.arc_span_m
        coefficients[2] = .5 * (self.entry_arc_second - n2) * self.arc_span_m**2
        rhs = np.array([-coefficients[:3].sum(axis=0),
                        -coefficients[1] - 2. * coefficients[2], -2. * coefficients[2]])
        coefficients[3:] = np.linalg.solve(np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]]), rhs)
        self.coefficients = coefficients
        self.arc_domain_start_m = self.start_progress_m - 20.
        self.arc_domain_end_m = self.end_progress_m + float(tail_progress_m)
        count = int(math.ceil((self.arc_domain_end_m - self.arc_domain_start_m) / arc_step_m))
        grid = np.unique(np.r_[np.linspace(self.arc_domain_start_m, self.arc_domain_end_m, count + 1),
                               self.start_progress_m, self.end_progress_m])
        nodes, weights = np.polynomial.legendre.leggauss(6)
        # Refine until every tabulated physical interval is <=.5m. This also
        # sets the road/curvature checking density independently of map scale.
        for _ in range(7):
            width = np.diff(grid)
            locations = grid[:-1, None] + .5 * width[:, None] * (nodes + 1.)
            ds = .5 * width * np.sum(self.metric(locations) * weights, axis=1)
            split = ds > .5
            if not np.any(split):
                break
            grid = np.sort(np.r_[grid, .5 * (grid[:-1][split] + grid[1:][split])])
            if len(grid) > 40000:
                raise ValueError('Rejoin physical metric is too large')
        if np.any(ds > .501) or np.any(ds <= 1e-8) or not np.isfinite(ds).all():
            raise ValueError('Invalid rejoin physical arc table')
        self.s = grid
        self.base_ds = np.diff(grid)
        self.ds = ds
        self.arc = np.r_[0., np.cumsum(ds)]
        self.physical_length_m = self.physical_distance(self.start_progress_m, self.end_progress_m)
        self._build_speed_profile()

    def _correction(self, progress, order=0):
        values = np.asarray(progress, float)
        u = (self.nominal_geometry.arc_parameter(values) - self.arc_start_m) / self.arc_span_m
        coefficients = self.coefficients.copy()
        for _ in range(order):
            coefficients = coefficients[1:] * np.arange(1, len(coefficients))[:, None]
        result = np.moveaxis(np.polynomial.polynomial.polyval(u, coefficients), 0, -1) / self.arc_span_m**order
        return np.asarray(result, float)

    def position(self, progress):
        values = np.asarray(progress, float)
        nominal = _position(self.nominal_geometry, values)
        middle = (values >= self.start_progress_m) & (values <= self.end_progress_m)
        result = np.where(middle[..., None], nominal + self._correction(values), nominal)
        delta = self.nominal_geometry.arc_parameter(values) - self.arc_start_m
        before = self.ego_xy + delta[..., None] * self.entry_arc_first + .5 * delta[..., None]**2 * self.entry_arc_second
        return np.where((values < self.start_progress_m)[..., None], before, result)

    def _intrinsic_derivative(self, progress, order=1):
        values = np.asarray(progress, float)
        nominal = np.asarray(self.nominal_geometry.arc_derivative(values, order), float)
        middle = (values >= self.start_progress_m) & (values <= self.end_progress_m)
        result = np.where(middle[..., None], nominal + self._correction(values, order), nominal)
        if order == 1:
            before = (self.entry_arc_first + (self.nominal_geometry.arc_parameter(values) - self.arc_start_m)[..., None] * self.entry_arc_second)
        elif order == 2:
            before = np.broadcast_to(self.entry_arc_second, values.shape + (2,))
        else:
            before = np.zeros(values.shape + (2,))
        return np.where((values < self.start_progress_m)[..., None], before, result)

    def derivative(self, progress, order=1):
        if order not in (1, 2, 3):
            raise ValueError('Finite rejoin derivative order must be1,2or3')
        one = self.nominal_geometry.parameter_derivative(progress, 1)
        first = self._intrinsic_derivative(progress, 1)
        if order == 1:
            return first * one[..., None]
        two = self.nominal_geometry.parameter_derivative(progress, 2)
        second = self._intrinsic_derivative(progress, 2)
        if order == 2:
            return second * one[..., None]**2 + first * two[..., None]
        three = self.nominal_geometry.parameter_derivative(progress, 3)
        return (self._intrinsic_derivative(progress, 3) * one[..., None]**3
                + 3. * second * one[..., None] * two[..., None] + first * three[..., None])

    def metric(self, progress):
        value = np.linalg.norm(self.derivative(progress), axis=-1)
        if np.any(~np.isfinite(value)) or np.any(value <= 1e-7):
            raise ValueError('Rejoin has a zero or invalid tangent')
        return float(value) if np.ndim(value) == 0 else value

    def curvature_at(self, progress):
        first, second = self._intrinsic_derivative(progress), self._intrinsic_derivative(progress, 2)
        value = (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]) / np.linalg.norm(first, axis=-1)**3
        return float(value) if np.ndim(value) == 0 else value

    def _arc_coordinate(self, progress):
        value = np.asarray(progress, float)
        if (not np.isfinite(value).all() or np.any(value < self.s[0] - 1e-8)
                or np.any(value > self.s[-1] + 1e-8)):
            raise ValueError('Arc query outside the finite rejoin domain')
        return np.interp(value, self.s, self.arc)

    def physical_distance(self, start, end):
        return float(self._arc_coordinate(end) - self._arc_coordinate(start))

    def advance_by_distance(self, progress, distance):
        target = self._arc_coordinate(progress) + np.asarray(distance, float)
        if not np.isfinite(target).all() or np.any(target < -1e-8) or np.any(target > self.arc[-1] + 1e-8):
            raise ValueError('Advance outside the finite rejoin arc domain')
        result = np.interp(target, self.arc, self.s)
        return float(result) if np.ndim(result) == 0 else result

    def nominal_offset_m(self, progress):
        values = np.asarray(progress, float)
        ref = _position(self.reference, values)
        tangent = np.asarray(self.reference.derivative(values, 1), float).copy()
        tangent /= np.linalg.norm(tangent, axis=-1)[..., None]
        normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
        result = np.sum((self.position(values) - ref) * normal, axis=-1)
        return float(result) if np.ndim(result) == 0 else result

    def _build_speed_profile(self):
        mask = (self.s >= self.start_progress_m) & (self.s <= self.end_progress_m)
        self.profile_s = self.s[mask]
        kappa = np.abs(self.curvature_at(self.profile_s))
        nodes, _ = np.polynomial.legendre.leggauss(6)
        locations = self.profile_s[:-1, None] + .5 * np.diff(self.profile_s)[:, None] * (nodes + 1.)
        interval = np.maximum(np.max(abs(self.curvature_at(locations)), axis=1),
                              np.maximum(kappa[:-1], kappa[1:]))
        # Each endpoint retains the more restrictive neighboring interval
        # envelope; speed interpolation cannot erase a sharp interior peak.
        kappa = np.maximum(kappa, np.r_[interval, 0.])
        kappa = np.maximum(kappa, np.r_[0., interval])
        squared = np.minimum(self.max_speed_mps**2,
                             self.lateral_accel_mps2 / np.maximum(kappa, 1e-8))
        exit_cap = self._nominal_cap(self.end_progress_m, 0.)
        squared[-1] = min(squared[-1], exit_cap**2)
        profile_arc = self._arc_coordinate(self.profile_s)
        squared = np.minimum.accumulate((squared + 2. * self.brake_decel_mps2 * profile_arc)[::-1])[::-1] - 2. * self.brake_decel_mps2 * profile_arc
        squared = np.minimum.accumulate(squared - 2. * self.acceleration_mps2 * profile_arc) + 2. * self.acceleration_mps2 * profile_arc
        self.profile_speed = np.sqrt(np.maximum(squared, 0.))
        self.speed = self.profile_speed

    def _speed_cap(self, progress):
        if progress < self.start_progress_m:
            return float(self.profile_speed[0])
        if progress <= self.end_progress_m:
            return float(np.interp(progress, self.profile_s, self.profile_speed))
        return self._nominal_cap(progress, 0.)

    def _nominal_cap(self, progress, span):
        # Handover restores the original nominal controller/profile. A global
        # reinterpolation profile must not silently retime the rest of the lap.
        return float(self.nominal_profile.minimum_speed(progress, span))

    def minimum_speed(self, progress, original_progress_span):
        progress, span = float(progress), float(original_progress_span)
        if not math.isfinite(progress + span) or span < 0.:
            raise ValueError('Finite progress and nonnegative original-progress span required')
        end = progress + span
        values = [self._speed_cap(progress), self._speed_cap(end)]
        inside = (self.profile_s >= progress) & (self.profile_s <= end)
        if np.any(inside):
            values.append(float(self.profile_speed[inside].min()))
        if end > self.end_progress_m:
            start = max(progress, self.end_progress_m)
            values.append(self._nominal_cap(start, end - start))
        return min(values)

    def sample(self, progress):
        progress = float(progress)
        first = self.derivative(progress)
        return self.position(progress), math.atan2(first[1], first[0]), self._speed_cap(progress), self.curvature_at(progress)


def _smooth_reference_projection(reference, points, hints):
    """Local normal projection onto the C2 reference, unwrapped near hints."""
    points, hints = np.asarray(points, float), np.asarray(hints, float)
    progress = hints.copy()
    for _ in range(35):
        base = _position(reference, progress)
        first = np.asarray(reference.derivative(progress, 1), float)
        second = np.asarray(reference.derivative(progress, 2), float)
        squared = np.sum(first * first, axis=-1)
        denominator = squared - np.sum((points - base) * second, axis=-1)
        if np.any(denominator <= .15 * squared):
            raise ValueError('Smooth reference projection folds or loses its local normal chart')
        step = np.sum((points - base) * first, axis=-1) / denominator
        progress += np.clip(step, -3., 3.)
        if np.max(abs(step)) < 1e-10:
            break
    else:
        raise ValueError('Smooth reference projection did not converge')
    if np.max(abs(progress - hints)) > 20.:
        raise ValueError('Smooth reference projection leaves the local ordered section')
    return progress


def _legacy_polyline_projection(reference, points, hints, return_frame=False):
    source = getattr(reference, 'source', None)
    if source is None or not all(hasattr(source, x) for x in ('segment', 's', 'ds', 'xy')):
        return None
    points, hints = np.asarray(points, float), np.asarray(hints, float)
    count = len(source.s)
    base = np.searchsorted(source.s, hints % source.length, side='right') - 1
    base += np.floor(hints / source.length).astype(int) * count
    radius = int(math.ceil(90. / float(np.min(source.ds)))) + 2
    extended = base[:, None] + np.arange(-radius, radius + 1)
    indices = extended % count
    segment = source.segment[indices]
    delta = points[:, None, :] - source.xy[indices]
    fraction = np.clip(np.sum(delta * segment, axis=-1) / source.ds[indices]**2, 0., 1.)
    projected = source.xy[indices] + fraction[..., None] * segment
    selected = np.argmin(np.sum((points[:, None, :] - projected)**2, axis=-1), axis=1)
    row = np.arange(len(points))
    progress = (source.s[indices[row, selected]] + (extended[row, selected] // count) * source.length
                + fraction[row, selected] * source.ds[indices[row, selected]])
    if return_frame:
        tangent = segment[row, selected] / source.ds[indices[row, selected], None]
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        lateral = np.sum((points - projected[row, selected]) * normal, axis=1)
        return progress, lateral, normal
    return progress


def _reference_coordinate_discrepancy(reference, points, chart, projection):
    first = np.asarray(reference.derivative(chart, 1))
    tangent = first / np.linalg.norm(first, axis=-1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    lateral = np.sum((points - _position(reference, chart)) * normal, axis=1)
    projected_first = np.asarray(reference.derivative(projection, 1))
    projected_tangent = projected_first / np.linalg.norm(projected_first, axis=-1)[:, None]
    projected_normal = np.column_stack([-projected_tangent[:, 1], projected_tangent[:, 0]])
    projected_lateral = np.sum((points - _position(reference, projection)) * projected_normal, axis=1)
    radius = math.hypot(4.5, 1.05)
    center_error = abs(lateral - projected_lateral)
    normal_error = radius * np.linalg.norm(normal - projected_normal, axis=1)
    longitudinal = float(np.max(abs(projection - chart)))
    raw = _legacy_polyline_projection(reference, points, chart, return_frame=True)
    if raw is not None:
        raw_progress, raw_lateral, raw_normal = raw
        longitudinal = max(longitudinal, float(np.max(abs(raw_progress - chart))))
        center_error = np.maximum(center_error, abs(lateral - raw_lateral))
        normal_error = np.maximum(normal_error, radius * np.linalg.norm(normal - raw_normal, axis=1))
    return {'longitudinal_m': longitudinal, 'lateral_m': float(np.max(center_error + normal_error)),
            'maximum_center_lateral_coordinate_error_m': float(center_error.max()),
            'maximum_body_normal_rotation_reserve_m': float(normal_error.max())}


class ReferenceProgressRejoin:
    """Unchanged physical curve in an endpoint-anchored smooth-reference chart.

    The legacy polygonal projection is nonsmooth and is NOT claimed identical.
    Coordinate residuals are measured densely and added to the ego traffic tube;
    the actual-pose gate and all physical geometry/limits remain unchanged.
    """
    finite_domain = True
    reference_progress_parameterization_enabled = True

    def __init__(self, base, configuration=None):
        self._base = base
        configuration = dict(configuration or {})
        self.start_progress_m, self.end_progress_m = base.start_progress_m, base.end_progress_m
        self.span_m = self.end_progress_m - self.start_progress_m
        count = max(2, int(math.ceil(base.physical_length_m / .1)) + 1)
        u = np.unique(np.r_[base.advance_by_distance(self.start_progress_m,
                           np.linspace(0., base.physical_length_m, count)),
                           self.start_progress_m, self.end_progress_m])
        projected = _smooth_reference_projection(base.reference, base.position(u), u)
        fraction = (u - self.start_progress_m) / self.span_m
        smooth = fraction**3 * (10. - 15. * fraction + 6. * fraction**2)
        anchor0, anchor1 = projected[0] - u[0], projected[-1] - u[-1]
        chart = projected - (anchor0 + (anchor1 - anchor0) * smooth)
        chart[0], chart[-1] = self.start_progress_m, self.end_progress_m
        if np.any(np.diff(projected) <= 1e-8) or np.any(np.diff(chart) <= 1e-8):
            raise ValueError('Reference projection is folded or nonmonotone')
        self._chart = PchipInterpolator(u, chart, extrapolate=False)
        self._inverse_seed = PchipInterpolator(chart, u, extrapolate=False)
        # Check quarter intervals as well as all interpolation knots. This is a
        # measured compatibility qualification, not an analytic polygonal bound.
        check_u = np.unique(np.r_[u, (u[:-1, None] + np.diff(u)[:, None] * np.array([.25, .5, .75])).ravel()])
        check_q = self._to_chart(check_u)
        check_points = base.position(check_u)
        check_projection = _smooth_reference_projection(base.reference, check_points, check_u)
        chart_error = float(np.max(abs(check_projection - check_q)))
        raw = _legacy_polyline_projection(base.reference, check_points, check_u)
        compatibility = chart_error if raw is None else float(np.max(abs(raw - check_q)))
        # An explicit small numerical allowance is added to the measured bound;
        # it does not widen the actual 1.25m pose gate.
        self.chart_vs_projection_residual_bound_m = chart_error + .005
        self.sampled_legacy_projection_compatibility_residual_bound_m = compatibility + .005
        self.reference_progress_residual_bound_m = max(
            self.chart_vs_projection_residual_bound_m,
            self.sampled_legacy_projection_compatibility_residual_bound_m)
        maximum = float(configuration.get('maximum_reference_chart_residual_m', .5))
        if not 0. < maximum <= .625 or self.reference_progress_residual_bound_m > maximum:
            raise ValueError('Reference-chart compatibility residual exceeds the declared bound')
        discrepancy = _reference_coordinate_discrepancy(base.reference, check_points, check_q, check_projection)
        self.reference_lateral_residual_bound_m = discrepancy['lateral_m'] + .005
        merged = np.sort(np.r_[self._to_chart(base.s), chart])
        for anchor in (self.start_progress_m, self.end_progress_m):
            merged[abs(merged - anchor) < 1e-7] = anchor
        self.s = merged[np.r_[True, np.diff(merged) > 1e-7]]
        self.base_ds = np.diff(self.s)
        table_parameters = self._to_parameter(self.s)
        if not np.isfinite(table_parameters).all():
            raise ValueError('Projected chart inverse returned a nonfinite parameter')
        self.arc = base._arc_coordinate(table_parameters)
        self.ds = np.diff(self.arc)
        if np.any(self.ds <= 0.) or np.any(self.base_ds <= 0.):
            raise ValueError('Invalid projected route physical-arc table')
        self.profile_s = self._to_chart(base.profile_s)
        self.profile_speed = base.profile_speed.copy()
        self.speed = self.profile_speed
        self.coordinate_report = {
            'enabled': True, 'coordinate': 'endpoint-anchored smooth-reference normal projection',
            'legacy_polyline_identity_claimed': False,
            'physical_curve_changed': False, 'physical_speed_profile_changed': False,
            'anchor_projection_offsets_m': [float(anchor0), float(anchor1)],
            'chart_vs_projection_residual_bound_m': self.chart_vs_projection_residual_bound_m,
            'sampled_legacy_projection_compatibility_residual_bound_m': self.sampled_legacy_projection_compatibility_residual_bound_m,
            'ego_longitudinal_coordinate_reserve_m': self.reference_progress_residual_bound_m,
            'ego_lateral_coordinate_reserve_m': self.reference_lateral_residual_bound_m,
            'maximum_center_lateral_coordinate_error_m': discrepancy['maximum_center_lateral_coordinate_error_m'],
            'maximum_body_normal_rotation_reserve_m': discrepancy['maximum_body_normal_rotation_reserve_m'],
            'compatibility_is_dense_empirical_qualification': True,
            'verification_sample_count': len(check_u),
            'minimum_projection_progress_step_m': float(np.diff(projected).min()),
            'actual_pose_tolerance_changed': False}

    def traffic_coordinate_reserves(self, start, physical_reach_m):
        count = max(2, int(math.ceil(float(physical_reach_m) / .25)) + 1)
        q = self.advance_by_distance(start, np.linspace(0., float(physical_reach_m), count))
        points = self.position(q)
        projection = _smooth_reference_projection(self.reference, points, q)
        discrepancy = _reference_coordinate_discrepancy(self.reference, points, q, projection)
        longitudinal = max(self.reference_progress_residual_bound_m, discrepancy['longitudinal_m'] + .005)
        lateral = max(self.reference_lateral_residual_bound_m, discrepancy['lateral_m'] + .005)
        return {'longitudinal_m': longitudinal, 'lateral_m': lateral,
                'maximum_center_lateral_coordinate_error_m': discrepancy['maximum_center_lateral_coordinate_error_m'],
                'maximum_body_normal_rotation_reserve_m': discrepancy['maximum_body_normal_rotation_reserve_m'],
                'sample_count': count, 'maximum_physical_spacing_m': float(physical_reach_m) / (count - 1),
                'scope': 'Dense measured coordinate compatibility across the current ego traffic reach, including original nominal tail'}

    def __getattr__(self, name):
        base = self.__dict__.get('_base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def _to_chart(self, parameter):
        values = np.asarray(parameter, float)
        result = values.copy()
        inside = (values >= self.start_progress_m) & (values <= self.end_progress_m)
        result[inside] = self._chart(values[inside])
        return float(result) if result.ndim == 0 else result

    def _to_parameter(self, progress, order=0):
        values = np.asarray(progress, float)
        flat = values.ravel()
        inside = (flat >= self.start_progress_m) & (flat <= self.end_progress_m)
        u = flat.copy()
        u[inside] = np.clip(self._inverse_seed(flat[inside]), self._chart.x[0], self._chart.x[-1])
        for _ in range(7):
            if not np.any(inside):
                break
            first = self._chart(u[inside], 1)
            if np.any(~np.isfinite(first)) or np.any(first <= .05):
                raise ValueError('Projected reference chart has a collapsed inverse')
            delta = (self._chart(u[inside]) - flat[inside]) / first
            u[inside] = np.clip(u[inside] - delta, self._chart.x[0], self._chart.x[-1])
            if np.max(abs(delta)) < 1e-11:
                break
        if np.any(inside) and (not np.isfinite(u[inside]).all() or np.max(abs(self._chart(u[inside]) - flat[inside])) > 1e-7):
            raise ValueError('Projected reference chart inverse did not converge')
        result = u
        if order:
            result = np.ones_like(u) if order == 1 else np.zeros_like(u)
            a = self._chart(u[inside], 1)
            if order == 1:
                result[inside] = 1. / a
            elif order == 2:
                result[inside] = -self._chart(u[inside], 2) / a**3
            elif order == 3:
                b, c = self._chart(u[inside], 2), self._chart(u[inside], 3)
                result[inside] = (3. * b*b - c*a) / a**5
            else:
                raise ValueError('Inverse derivative order must be1,2or3')
        result = result.reshape(values.shape)
        return float(result) if result.ndim == 0 else result

    def position(self, progress):
        return self._base.position(self._to_parameter(progress))

    def derivative(self, progress, order=1):
        u = self._to_parameter(progress)
        one = np.asarray(self._to_parameter(progress, 1))
        first = self._base.derivative(u, 1)
        if order == 1:
            return first * one[..., None]
        two = np.asarray(self._to_parameter(progress, 2))
        second = self._base.derivative(u, 2)
        if order == 2:
            return second * one[..., None]**2 + first * two[..., None]
        if order != 3:
            raise ValueError('Derivative order must be1,2or3')
        three = np.asarray(self._to_parameter(progress, 3))
        return self._base.derivative(u, 3) * one[..., None]**3 + 3. * second * one[..., None] * two[..., None] + first * three[..., None]

    def metric(self, progress):
        result = np.linalg.norm(self.derivative(progress), axis=-1)
        return float(result) if np.ndim(result) == 0 else result

    def curvature_at(self, progress):
        return self._base.curvature_at(self._to_parameter(progress))

    def _arc_coordinate(self, progress):
        return self._base._arc_coordinate(self._to_parameter(progress))

    def physical_distance(self, start, end):
        return self._base.physical_distance(self._to_parameter(start), self._to_parameter(end))

    def advance_by_distance(self, progress, distance):
        return self._to_chart(self._base.advance_by_distance(self._to_parameter(progress), distance))

    def minimum_speed(self, progress, original_progress_span):
        if not math.isfinite(float(progress) + float(original_progress_span)) or original_progress_span < 0.:
            raise ValueError('Finite progress and nonnegative chart span required')
        start = self._to_parameter(progress)
        end = self._to_parameter(float(progress) + float(original_progress_span))
        return self._base.minimum_speed(start, end - start)

    def nominal_offset_m(self, progress):
        values = np.asarray(progress, float)
        first = np.asarray(self.reference.derivative(values, 1), float)
        tangent = first / np.linalg.norm(first, axis=-1)[..., None]
        normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
        result = np.sum((self.position(values) - _position(self.reference, values)) * normal, axis=-1)
        return float(result) if result.ndim == 0 else result

    def sample(self, progress):
        first = self.derivative(progress)
        return self.position(progress), math.atan2(first[1], first[0]), self.minimum_speed(progress, 0.), self.curvature_at(progress)



def _road_report(route, road_rule, start):
    values = route.s[(route.s >= max(start, route.start_progress_m)) & (route.s <= route.end_progress_m)]
    if not len(values):
        values = np.array([min(start, route.end_progress_m)])
    xy = route.position(values)
    derivative = route.derivative(values)
    tangent = derivative / np.linalg.norm(derivative, axis=1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    wheels = np.stack([xy + axle * tangent + side * road_rule.half_track_m * normal
                       for axle, side in ((0., 1.), (0., -1.), (road_rule.wheelbase_m, 1.), (road_rule.wheelbase_m, -1.))], axis=1)
    progress = np.repeat(values, 4)
    raw = road_rule.contains_points(wheels.reshape(-1, 2), progress_m=progress).reshape(-1, 4)
    margin = float(road_rule.planning_patch_margin_m)
    patch = road_rule.contains_points(wheels.reshape(-1, 2), progress_m=progress, margin_m=margin).reshape(-1, 4)
    pair = (patch[:, 0] & patch[:, 2]) | (patch[:, 1] & patch[:, 3])
    return {'clear': bool(np.all(raw.sum(axis=1) >= 2) and np.all(pair)),
            'minimum_raw_wheel_centers': int(raw.sum(axis=1).min()),
            'same_side_patch_pair_violations': int(np.sum(~pair)),
            'planning_patch_margin_m': margin, 'sample_count': len(values),
            'maximum_physical_sample_interval_m': float(np.diff(route._arc_coordinate(values)).max()) if len(values) > 1 else 0.}


_REFERENCE_BOUND_CACHE = {}


def _quadratic_range(a, b, c, low, high):
    left, right = a * low**2 + b * low + c, a * high**2 + b * high + c
    vertex = np.divide(-b, 2. * a, out=np.zeros_like(a), where=abs(a) > 1e-16)
    has_vertex = (abs(a) > 1e-16) & (vertex > low) & (vertex < high)
    value = a * vertex**2 + b * vertex + c
    return np.minimum(np.minimum(left, right), np.where(has_vertex, value, np.inf)), \
        np.maximum(np.maximum(left, right), np.where(has_vertex, value, -np.inf))


def _reference_bounds(reference):
    """Polynomial interval bounds, not endpoints of a propagated mean path."""
    cached = _REFERENCE_BOUND_CACHE.get(id(reference))
    if cached is not None and cached[0] is reference:
        return cached[1]
    if hasattr(reference, '_curve'):
        curve = reference._curve
        starts, ends, segment_ids = [], [], []
        for index, (left, right) in enumerate(zip(curve.x[:-1], curve.x[1:])):
            grid = np.linspace(left, right, max(1, int(math.ceil((right - left) / .25))) + 1)
            starts.extend(grid[:-1]); ends.extend(grid[1:]); segment_ids.extend([index] * (len(grid) - 1))
        starts, ends, ids = np.asarray(starts), np.asarray(ends), np.asarray(segment_ids, int)
        low, high = starts - curve.x[ids], ends - curve.x[ids]
        a, b, c = 3. * curve.c[0, ids], 2. * curve.c[1, ids], curve.c[2, ids]
        xlo, xhi = _quadratic_range(a[:, 0], b[:, 0], c[:, 0], low, high)
        ylo, yhi = _quadratic_range(a[:, 1], b[:, 1], c[:, 1], low, high)
        min_x = np.where((xlo <= 0.) & (xhi >= 0.), 0., np.minimum(abs(xlo), abs(xhi)))
        min_y = np.where((ylo <= 0.) & (yhi >= 0.), 0., np.minimum(abs(ylo), abs(yhi)))
        metric_low = np.sqrt(min_x**2 + min_y**2) * (1. - 1e-10)
        metric_high = np.sqrt(np.maximum(abs(xlo), abs(xhi))**2 + np.maximum(abs(ylo), abs(yhi))**2) * (1. + 1e-10)
        qa = b[:, 0] * a[:, 1] - a[:, 0] * b[:, 1]
        qb = 2. * (c[:, 0] * a[:, 1] - c[:, 1] * a[:, 0])
        qc = c[:, 0] * b[:, 1] - c[:, 1] * b[:, 0]
        cross_low, cross_high = _quadratic_range(qa, qb, qc, low, high)
        denominator_low = np.maximum(metric_low, 1e-12)**3
        denominator_high = np.maximum(metric_high, 1e-12)**3
        ratios = np.stack([cross_low / denominator_low, cross_low / denominator_high,
                           cross_high / denominator_low, cross_high / denominator_high])
        result = {'starts': starts, 'ends': ends, 'metric_low': metric_low,
                  'metric_high': metric_high, 'kappa_low': ratios.min(axis=0) - 1e-10,
                  'kappa_high': ratios.max(axis=0) + 1e-10, 'method': 'CubicSpline polynomial derivative interval bounds'}
    elif getattr(reference, 'intrinsic_arc_parameter', False) and hasattr(reference, 'constant_curvature_per_m'):
        curvature = float(reference.constant_curvature_per_m)
        result = {'starts': np.array([0.]), 'ends': np.array([reference.length]),
                  'metric_low': np.array([1.]), 'metric_high': np.array([1.]),
                  'kappa_low': np.array([curvature]), 'kappa_high': np.array([curvature]),
                  'method': 'Declared analytic intrinsic-arc constant curvature'}
    else:
        raise ValueError('Traffic metric needs a continuous polynomial reference or analytic constant-curvature contract')
    _REFERENCE_BOUND_CACHE[id(reference)] = (reference, result)
    return result


def _parallel_metric_bounds(reference, low, high, lane_low, lane_high):
    table = _reference_bounds(reference)
    if high - low >= reference.length:
        mask = np.ones(len(table['starts']), bool)
    else:
        left = low % reference.length
        right = left + max(0., high - low)
        mask = (table['ends'] >= left) & (table['starts'] <= min(right, reference.length))
        if right > reference.length:
            mask |= table['starts'] <= right - reference.length
    if not np.any(mask):
        raise ValueError('No reference polynomial interval covers traffic state')
    k_low, k_high = table['kappa_low'][mask], table['kappa_high'][mask]
    products = np.stack([lane_low * k_low, lane_low * k_high,
                         lane_high * k_low, lane_high * k_high])
    scale_low, scale_high = 1. - products.max(axis=0), 1. - products.min(axis=0)
    return float(np.min(table['metric_low'][mask] * scale_low)), \
        float(np.max(table['metric_high'][mask] * scale_high)), float(np.min(scale_low))


def measured_motion_bounds(reference, row, *, measurement_ego_progress_m,
                           elapsed_since_measurement_s=0., times_s):
    """Enclose physical tangent velocity on the original-progress coordinate.

    Each numerical step bounds the metric over an invariant reachable progress
    and lateral interval. It never propagates only the two endpoints. A fold
    loses the 1D coordinate contract and yields a full-period uncertainty set;
    it is not converted to an apparently credible clipped metric. Track age_s
    is deliberately unused: rows already represent the observation's KF state.
    """
    times = np.asarray(times_s, float)
    elapsed = float(elapsed_since_measurement_s)
    if (times.ndim != 1 or not len(times) or times[0] < 0. or np.any(np.diff(times) < 0.)
            or not np.isfinite(times).all() or not math.isfinite(elapsed) or elapsed < 0.):
        raise ValueError('Finite ordered future times and nonnegative held-observation elapsed time required')
    distance, velocity = float(row['relative_distance_m']), float(row.get('speed_mps', 0.))
    sigma = float(row.get('speed_std_mps', 2.))
    lateral = float(row['lateral_m'])
    lateral_speed = float(row.get('lateral_velocity_mps', row.get('lateral_speed_mps', 0.)))
    lateral_sigma = float(row.get('lateral_velocity_std_mps', row.get('lateral_speed_std_mps', 0.)))
    position_std = float(row.get('position_std_m', .35))
    lmin = float(row.get('lateral_min_m', lateral - 1.05))
    lmax = float(row.get('lateral_max_m', lateral + 1.05))
    raw = [distance, velocity, sigma, lateral, lateral_speed, lateral_sigma, position_std, lmin, lmax,
           measurement_ego_progress_m]
    if not np.isfinite(raw).all() or min(sigma, lateral_sigma, position_std) < 0. or lmin > lmax:
        raise ValueError('Invalid measured traffic state')
    uncertainty = min(.5, max(.15, position_std))
    lmin = min(lmin, lateral - 1.05) - uncertainty
    lmax = max(lmax, lateral + 1.05) + uncertainty
    sigma = max(1., sigma)
    vlow, vhigh = (velocity - sigma, velocity + sigma) if row.get('velocity_valid', True) else (0., 70.)
    origin = float(measurement_ego_progress_m) + distance
    low = high = mean = origin
    lower, upper, means, lateral_low, lateral_high = [], [], [], [], []
    elapsed_total = 0.
    folded = False
    minimum_metric = math.inf
    for target in elapsed + times:
        while elapsed_total < target - 1e-10 and not folded:
            dt = min(.05, target - elapsed_total)
            ta, tb = elapsed_total, elapsed_total + dt
            lanes = [lmin + (lateral_speed - lateral_sigma) * ta,
                     lmax + (lateral_speed + lateral_sigma) * ta,
                     lmin + (lateral_speed - lateral_sigma) * tb,
                     lmax + (lateral_speed + lateral_sigma) * tb]
            lane_low, lane_high = min(lanes), max(lanes)
            metric_low, metric_high, scale = _parallel_metric_bounds(reference, low, high, lane_low, lane_high)
            pad = max(abs(vlow), abs(vhigh)) * dt / max(metric_low, .05) + 1e-8
            domain_low = low - pad if vlow < 0. else low
            domain_high = high + pad if vhigh > 0. else high
            for _ in range(12):
                mlow, mhigh, scale = _parallel_metric_bounds(reference, domain_low, domain_high, lane_low, lane_high)
                if scale <= .20 or mlow <= 1e-6 or not math.isfinite(mlow + mhigh):
                    folded = True
                    break
                rates = [vlow / mlow, vlow / mhigh, vhigh / mlow, vhigh / mhigh]
                rate_low, rate_high = min(rates), max(rates)
                needed_low = low + min(0., rate_low) * dt
                needed_high = high + max(0., rate_high) * dt
                if needed_low >= domain_low - 1e-8 and needed_high <= domain_high + 1e-8:
                    minimum_metric = min(minimum_metric, mlow)
                    break
                domain_low = min(domain_low, needed_low - 1e-7)
                domain_high = max(domain_high, needed_high + 1e-7)
            else:
                folded = True
            if folded:
                break
            low += rate_low * dt
            high += rate_high * dt
            first = np.asarray(reference.derivative(mean, 1), float)
            second = np.asarray(reference.derivative(mean, 2), float)
            metric = float(np.linalg.norm(first))
            kappa = float((first[0] * second[1] - first[1] * second[0]) / metric**3)
            center_lane = lateral + lateral_speed * elapsed_total
            center_metric = metric * (1. - center_lane * kappa)
            if center_metric <= 0.:
                folded = True
                break
            mean += (.5 * (vlow + vhigh) if not row.get('velocity_valid', True) else velocity) / center_metric * dt
            if not low - 1e-7 <= mean <= high + 1e-7:
                raise ValueError('Traffic interval failed its own mean-enclosure check')
            elapsed_total += dt
        if folded:
            low, high = origin - reference.length, origin + reference.length
            means.append(None)
        else:
            means.append(mean)
        lower.append(low); upper.append(high)
        lateral_low.append(lmin + (lateral_speed - lateral_sigma) * target)
        lateral_high.append(lmax + (lateral_speed + lateral_sigma) * target)
    return {'progress_lower_m': np.array(lower), 'progress_upper_m': np.array(upper),
            'progress_mean_m': means, 'lateral_lower_m': np.array(lateral_low),
            'lateral_upper_m': np.array(lateral_high), 'metric_fold_fully_inflated': folded,
            'minimum_parallel_metric_m_per_original_m': None if not math.isfinite(minimum_metric) else minimum_metric,
            'mean_enclosure_failures': 0,
            'assumption': 'Continuous-reference polynomial interval bounds under measured physical-velocity intervals; unobserved traffic and model errors remain outside this guarantee'}


def _traffic_report(route, start, speed, observation, configuration):
    horizon = float(configuration.get('traffic_horizon_s', 3.))
    step = float(configuration.get('traffic_step_s', .1))
    times = np.linspace(0., horizon, int(math.ceil(horizon / step)) + 1)
    stop_time = min(horizon, speed / route.brake_decel_mps2)
    braking_t = np.minimum(times, stop_time)
    lower_distance = speed * braking_t - .5 * route.brake_decel_mps2 * braking_t**2
    constraint = configuration.get('longitudinal_constraint')
    if constraint not in (None, 'braking_or_coasting'):
        raise ValueError('Unknown rejoin longitudinal constraint')
    # This certificate is conditional on the controller enforcing zero
    # positive longitudinal drive. It covers coasting through maximum declared
    # braking; it does not assume a guaranteed achieved deceleration.
    upper_distance = (speed * times if constraint == 'braking_or_coasting' else
                      speed * times + .5 * route.acceleration_mps2 * times**2)
    coordinate_context = (route.traffic_coordinate_reserves(start, float(upper_distance.max()))
                          if getattr(route, 'reference_progress_parameterization_enabled', False) else None)
    coordinate_reserve = (coordinate_context['longitudinal_m'] if coordinate_context is not None else 0.)
    # Coordinate uncertainty broadens the comparison coordinates, never the
    # physical set of reachable poses used for lateral/body sampling.
    low = route.advance_by_distance(start, lower_distance)
    high = route.advance_by_distance(start, upper_distance)
    comparison_low = low - coordinate_reserve
    comparison_high = high + coordinate_reserve
    # Include all dense route vertices in each progress window as well as the
    # endpoints; a bent rejoin's middle cannot be replaced with its end lane.
    lane_low, lane_high, ego_lateral_half = [], [], []
    for left, right in zip(low, high):
        middle = route.s[(route.s > left) & (route.s < right)]
        locations = np.r_[left, middle, right]
        offsets = route.nominal_offset_m(locations)
        lane_low.append(float(offsets.min())); lane_high.append(float(offsets.max()))
        path_first = route.derivative(locations)
        reference_first = np.asarray(route.reference.derivative(locations, 1), float)
        path_heading = np.arctan2(path_first[:, 1], path_first[:, 0])
        reference_heading = np.arctan2(reference_first[:, 1], reference_first[:, 0])
        delta = path_heading - reference_heading
        half_width = float(configuration.get('ego_half_width_m', 1.05))
        # The vehicle origin is near the rear axle. Include the front/rear
        # body length projected across the map normal during a return turn.
        extent = half_width * abs(np.cos(delta)) + 4.5 * abs(np.sin(delta))
        ego_lateral_half.append(float(extent.max()))
    lane_low, lane_high = np.asarray(lane_low), np.asarray(lane_high)
    half = float(configuration.get('ego_half_width_m', 1.05)) + float(configuration.get('lateral_margin_m', .10))
    half = np.maximum(half, np.asarray(ego_lateral_half) + float(configuration.get('lateral_margin_m', .10)))
    half = half + (coordinate_context['lateral_m'] if coordinate_context is not None else 0.)
    ego_metric, _, ego_parallel_scale = _parallel_metric_bounds(route.reference,
        float(comparison_low.min()), float(comparison_high.max()), float(np.min(lane_low - half)), float(np.max(lane_high + half)))
    measurement_progress = float(observation.get('_measurement_ego_progress_m', start))
    elapsed = float(observation.get('_elapsed_since_measurement_s', 0.))
    rows = [*observation.get('hazard_tracks', []), *observation.get('confirmed_tracks', [])]
    unique = {}
    for index, row in enumerate(rows):
        identity = row.get('track_id', ('anonymous', index))
        key = (type(identity).__name__, str(identity))
        if key not in unique or float(row.get('age_s', 0.)) < float(unique[key].get('age_s', 0.)):
            unique[key] = row
    blockers = []
    folded_rows = []
    minimum_metrics = []
    for row in unique.values():
        motion = measured_motion_bounds(route.reference, row,
            measurement_ego_progress_m=measurement_progress, elapsed_since_measurement_s=elapsed, times_s=times)
        hazard_low, hazard_high = motion['progress_lower_m'], motion['progress_upper_m']
        hazard_left, hazard_right = motion['lateral_upper_m'], motion['lateral_lower_m']
        if motion['metric_fold_fully_inflated']:
            folded_rows.append(row.get('track_id'))
        metric = motion['minimum_parallel_metric_m_per_original_m']
        if metric is not None:
            minimum_metrics.append(metric)
        # A physical full-body longitudinal reserve becomes wider in the
        # original-progress coordinate on a compressed inside parallel curve.
        longitudinal_reserve = (float(route.reference.length) if ego_parallel_scale <= .20 or ego_metric <= 1e-6
                                else 8.5 / max(.05, min(ego_metric, metric if metric is not None else .05)))
        values = np.array([hazard_low - comparison_high - longitudinal_reserve, comparison_low - longitudinal_reserve - hazard_high,
                           hazard_right - lane_high - half, lane_low - half - hazard_left])
        first = _first_overlap(times, values)
        if first is not None:
            blockers.append({'track_id': row.get('track_id'), 'first_swept_conflict_s': first})
    report = {'clear': not blockers, 'blockers': blockers, 'all_provided_track_rows_considered': len(unique),
            'top_n_filter_used': False, 'opponent_ground_truth_used': False,
            'horizon_s': horizon, 'time_step_s': step,
            'measurement_ego_progress_m': measurement_progress, 'elapsed_since_measurement_s': elapsed,
            'folded_metric_fully_inflated_track_ids': folded_rows,
            'minimum_measured_parallel_metric': min(minimum_metrics) if minimum_metrics else None,
            'mean_enclosure_failures': 0,
            'assumption': 'Physical velocity converted through continuous reference/parallel metric with reachable-interval bounds; no unseen-space or arbitrary-model-error guarantee'}

    if coordinate_context is not None:
        report['reference_progress_coordinate_reserves'] = coordinate_context
    if constraint == 'braking_or_coasting':
        report.update(longitudinal_constraint=constraint,
            positive_drive_permitted=False,
            guaranteed_achieved_deceleration_assumed=False,
            physical_longitudinal_distance_envelope='hard_braking_lower_to_constant_speed_coasting_upper')
    return report


def validate_rejoin(route, *, current_progress_m, body_speed_mps, observation,
                    road_rule, ego_xy=None, ego_yaw=None, configuration=None):
    configuration = dict(configuration or {})
    progress, speed = float(current_progress_m), float(body_speed_mps)
    if not math.isfinite(progress + speed) or speed < 0.:
        raise ValueError('Finite progress and nonnegative speed required')
    report = {'version': VERSION, 'clear': False, 'opponent_ground_truth_used': False}
    if progress < route.start_progress_m - .5 or progress > route.end_progress_m + .5:
        return {**report, 'reason': 'outside_active_rejoin_interval'}
    if ego_xy is not None:
        point = np.asarray(ego_xy, float)[:2]
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('Finite actual XY required')
        error = float(np.linalg.norm(point - route.position(progress)))
        if error > float(configuration.get('maximum_tracking_error_m', 1.25)):
            return {**report, 'reason': 'actual_pose_left_cached_rejoin', 'tracking_error_m': error}
    if ego_yaw is not None and abs(_wrap(ego_yaw - route.sample(progress)[1])) > .4:
        return {**report, 'reason': 'actual_heading_left_cached_rejoin'}
    cap = route.minimum_speed(progress, 0.)
    if speed > cap + .05:
        return {**report, 'reason': 'current_speed_exceeds_rejoin_braking_envelope',
                'current_speed_mps': speed, 'route_speed_cap_mps': cap}
    road = _road_report(route, road_rule, progress)
    if not road['clear']:
        return {**report, 'reason': 'rejoin_wheel_sweep_outside_road', 'road': road}
    traffic = _traffic_report(route, progress, speed, observation, configuration)
    if getattr(route, 'reference_progress_parameterization_enabled', False):
        report['reference_progress_parameterization'] = route.coordinate_report
    return {**report, 'clear': traffic['clear'],
            'reason': 'measured_rejoin_clear' if traffic['clear'] else 'measured_rejoin_traffic_conflict',
            'route_speed_cap_mps': cap, 'road': road, 'traffic': traffic}


def build_rejoin(continuous_reference, continuous_nominal_geometry, nominal_speed_profile, *,
                 start_progress_m, ego_xy, ego_yaw, body_speed_mps,
                 entry_curvature_per_m, road_rule, observation, configuration=None):
    configuration = dict(configuration or {})
    report = {'version': VERSION, 'enabled': bool(configuration.get('enabled', False)),
              'accepted': False, 'opponent_ground_truth_used': False, 'attempts': []}
    if not report['enabled']:
        return None, {**report, 'reason': 'disabled'}
    if road_rule is None:
        return None, {**report, 'reason': 'actual_road_geometry_required'}
    start, yaw, speed, entry_curvature = map(float,
        (start_progress_m, ego_yaw, body_speed_mps, entry_curvature_per_m))
    point = np.asarray(ego_xy, float)[:2]
    if point.shape != (2,) or not np.isfinite(np.r_[point, start, yaw, speed, entry_curvature]).all() or speed < 0.:
        raise ValueError('Finite actual pose, curvature and nonnegative body speed required')
    lateral = float(np.linalg.norm(point - _position(continuous_nominal_geometry, start)))
    minimum = float(configuration.get('minimum_join_progress_m', 30.))
    maximum = float(configuration.get('maximum_join_progress_m', 160.))
    if not 10. <= minimum <= maximum <= 200.:
        raise ValueError('Finite join range must stay within10..200original-map metres')
    horizon = float(configuration.get('traffic_horizon_s', 3.))
    step = float(configuration.get('traffic_step_s', .1))
    if not 1. <= horizon <= 4. or not .05 <= step <= .2:
        raise ValueError('Invalid bounded measured-traffic horizon')
    base = max(minimum, 2. * speed, 6. * lateral)
    candidates = sorted(set(min(maximum, base * scale) for scale in (1., 1.5, 2.)))
    lateral_accel = float(getattr(continuous_reference, 'lateral_accel', 8.))
    brake_decel = float(getattr(continuous_reference, 'brake_decel', 8.))
    acceleration = float(getattr(continuous_reference, 'acceleration', 5.))
    max_speed = float(getattr(continuous_reference, 'max_speed', 100.))
    nominal_arc = _arc_geometry(nominal_speed_profile, continuous_nominal_geometry)
    for distance in candidates:
        try:
            end = nominal_arc.snap_source_vertex(start + distance, start + minimum, start + maximum)
            route = RejoinTrajectory(continuous_reference, nominal_arc, nominal_speed_profile,
                start_progress_m=start, end_progress_m=end, ego_xy=point, ego_yaw=yaw,
                entry_curvature_per_m=entry_curvature, max_speed_mps=max_speed,
                lateral_accel_mps2=lateral_accel, brake_decel_mps2=brake_decel,
                acceleration_mps2=acceleration,
                tail_progress_m=max(100., float(nominal_arc.advance_by_distance(
                    end, max_speed * horizon + .5 * acceleration * horizon**2 + 30.)) - end))
            if configuration.get('reference_progress_parameterization_enabled', False):
                route = ReferenceProgressRejoin(route, configuration)
            old_end = nominal_speed_profile.sample(end)
            actual_end = route.sample(end)
            handover = {'position_error_m': float(np.linalg.norm(np.asarray(old_end[0]) - actual_end[0])),
                        'heading_error_rad': abs(_wrap(old_end[1] - actual_end[1])),
                        'curvature_error_per_m': abs(float(old_end[3]) - actual_end[3]),
                        'original_source_vertex': nominal_arc.source_s is not None}
            if (handover['position_error_m'] > .01
                    or handover['heading_error_rad'] > float(configuration.get('maximum_handover_heading_error_rad', .15))
                    or handover['curvature_error_per_m'] > float(configuration.get('maximum_handover_curvature_error_per_m', .05))):
                report['attempts'].append({'join_progress_span_m': end - start, 'reason': 'nominal_handover_mismatch', 'handover': handover})
                continue
            sample_s = route.s[(route.s >= start) & (route.s <= route.end_progress_m)]
            derivative = route.derivative(sample_s)
            ref_first = np.asarray(continuous_reference.derivative(sample_s, 1), float).copy()
            ref_first /= np.linalg.norm(ref_first, axis=1)[:, None]
            forward = np.sum(derivative * ref_first, axis=1)
            if np.any(forward <= .15):
                report['attempts'].append({'join_progress_span_m': distance, 'reason': 'reversed_or_collapsed_original_progress'})
                continue
            along = np.sum((route.position(sample_s) - _position(continuous_reference, sample_s)) * ref_first, axis=1)
            if np.max(np.abs(along)) > float(configuration.get('maximum_tangential_mapping_residual_m', 4.)):
                report['attempts'].append({'join_progress_span_m': distance, 'reason': 'original_progress_mapping_residual_too_large',
                                           'maximum_residual_m': float(np.max(np.abs(along)))})
                continue
            validation = validate_rejoin(route, current_progress_m=start, body_speed_mps=speed,
                observation=observation, road_rule=road_rule, ego_xy=point, ego_yaw=yaw, configuration=configuration)
            report['attempts'].append({'join_progress_span_m': end - start, 'requested_join_progress_span_m': distance,
                                       'handover': handover, **validation})
            if validation['clear']:
                report.update(accepted=True, reason='bounded_measured_C2_rejoin',
                    start_progress_m=start, end_progress_m=route.end_progress_m,
                    physical_join_length_m=route.physical_length_m,
                    maximum_raw_curvature_per_m=float(np.max(abs(route.curvature_at(sample_s)))),
                    entry_speed_cap_mps=route.minimum_speed(start, 0.),
                    curvature_lateral_budget_mps2=lateral_accel, brake_decel_mps2=brake_decel,
                    coefficient_condition='physical-nominal-arc quintic correction matches actual pose/tangent/curvature and becomes zero intrinsic C2 at nominal exit',
                    nominal_topology_inherited=True, finite_original_progress_domain=True,
                    profile_uses_actual_physical_arc=True, original_nominal_profile_restored_after_exit=True,
                    handover=handover, unseen_space_verified=False)
                if getattr(route, 'reference_progress_parameterization_enabled', False):
                    report['reference_progress_parameterization'] = route.coordinate_report
                return route, report
        except ValueError as exc:
            report['attempts'].append({'join_progress_span_m': distance, 'reason': 'invalid_or_untrackable_candidate', 'detail': str(exc)})
    return None, {**report, 'reason': 'no_checked_rejoin_candidate'}
