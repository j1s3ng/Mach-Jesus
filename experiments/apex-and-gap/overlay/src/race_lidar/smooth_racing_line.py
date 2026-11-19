"""Map-only, periodic minimum-curvature line in the original Frenet frame.

The original map remains authoritative for progress, opponent projection and
track limits. This module does not use ground truth, learned predictions or
recorded racing pace. It plans a bounded simulator line, not an exact F1 line.
"""
from __future__ import annotations

import math
import hashlib
import json
import time
import copy
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares, minimize
from scipy.sparse import coo_matrix
from scipy.signal import find_peaks

VERSION = 'periodic-physical-time-two-tyre-line-v3'


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
    # The sampled path's raw curvature is never averaged away to gain speed.
    # Keep the short noise envelope only where it is MORE restrictive.
    smooth = gaussian_filter1d(curvature, 1.2, mode='wrap')
    limiting_curvature = np.maximum(np.abs(curvature), np.abs(smooth))
    speed = np.minimum(max_speed, np.sqrt(lateral_accel / np.maximum(limiting_curvature, 1e-5)))
    # Exact periodic acceleration/braking envelopes. Three copies give every
    # center-lap point a full predecessor and successor lap. Min-plus scans
    # avoid a Python loop for every offline objective evaluation.
    s = np.r_[0., np.cumsum(ds[:-1])]
    length = float(ds.sum())
    extended_s = np.concatenate((s - length, s, s + length))
    squared = np.tile(speed ** 2, 3)
    squared = np.minimum.accumulate(squared - 2. * acceleration * extended_s) + 2. * acceleration * extended_s
    squared = np.minimum.accumulate((squared + 2. * brake_decel * extended_s)[::-1])[::-1] - 2. * brake_decel * extended_s
    speed = np.sqrt(np.maximum(squared[len(ds):2 * len(ds)], 1e-8))
    return speed, ds, heading, curvature


def _lap_time_refine(reference, offset, normal, limits, *, offset_limit,
                     lateral_rate, knot_spacing_m=8., max_iterations=70,
                     time_budget_s=45., lower=None, upper=None, apex_ids=(), apex_targets=()):
    """Finite-budget map-only optimization of driven time, not curvature energy.

    Nonnegative periodic cubic B-spline weights preserve coefficient bounds.
    A lateral-rate penalty discourages untrackable offset reversals. The final
    geometry is projected onto that rate bound and accepted only if its actual
    physical-profile lap estimate improves the input path.
    """
    n = len(reference.s)
    count = max(24, int(math.ceil(reference.length / knot_spacing_m)))
    coordinate = np.arange(n, dtype=float) * count / n
    knot_s = np.arange(count) * reference.length / count
    initial = np.interp(knot_s, np.r_[reference.s, reference.length], np.r_[offset, offset[0]])
    lower = np.full(n, -offset_limit) if lower is None else np.asarray(lower)
    upper = np.full(n, offset_limit) if upper is None else np.asarray(upper)
    coefficient_bounds = []
    for j in range(count):
        distance = np.abs((reference.s-knot_s[j]+reference.length/2) % reference.length-reference.length/2)
        support = distance <= 2.1*reference.length/count
        coefficient_bounds.append((float(lower[support].max()), float(upper[support].min())))
    initial = np.clip(initial, [x[0] for x in coefficient_bounds], [x[1] for x in coefficient_bounds])
    baseline_xy = reference.xy + offset[:, None] * normal
    baseline_profile = _profile(baseline_xy, **limits)
    baseline_time = float(np.sum(baseline_profile[1] / baseline_profile[0]))
    start = time.monotonic()
    best = [math.inf, initial.copy()]
    calls = 0

    def decode(coefficients):
        return map_coordinates(coefficients, [coordinate], order=3,
                               prefilter=False, mode='grid-wrap')

    def objective(coefficients):
        nonlocal calls
        values = decode(coefficients)
        xy = reference.xy + values[:, None] * normal
        speed, ds, _, curvature = _profile(xy, **limits)
        slope = (np.roll(values, -1) - values) / reference.ds
        rate = abs(slope) * np.maximum(speed, np.roll(speed, -1))
        duration = float(np.sum(ds / speed))
        # Penalize rate violations and abrupt map-relative steering changes;
        # both scales remain in the report, rather than hidden speed boosts.
        rate_cost = (0. if lateral_rate is None else .2 *
            float(np.sum(np.maximum(rate - lateral_rate, 0.) ** 2 * ds / speed)))
        bending = np.roll(values, -1) - 2. * values + np.roll(values, 1)
        cost = duration + rate_cost + .015 * float(np.sum(bending ** 2))
        if len(apex_ids):
            cost += 8. * float(np.sum((values[apex_ids]-apex_targets)**2))
        calls += 1
        if cost < best[0]:
            best[:] = [cost, coefficients.copy()]
        return cost

    class BudgetReached(Exception):
        pass

    def callback(_):
        if time.monotonic() - start >= time_budget_s:
            raise BudgetReached()

    terminated_by_budget = False
    result = None
    try:
        result = minimize(objective, initial, method='L-BFGS-B',
            bounds=coefficient_bounds,
            options={'maxiter': max_iterations, 'maxfun': 50000, 'ftol': 2e-8,
                     'gtol': 2e-5, 'maxls': 20, 'finite_diff_rel_step': 1e-5},
            callback=callback)
    except BudgetReached:
        terminated_by_budget = True
    candidate = decode(best[1])
    for _ in range(0 if lateral_rate is None else 5):
        profile = _profile(reference.xy + candidate[:, None] * normal, **limits)
        edge_speed = np.maximum(profile[0], np.roll(profile[0], -1))
        candidate = _rate_project(candidate,
            .98 * lateral_rate * reference.ds / np.maximum(edge_speed, 1.), iterations=180)
        candidate = np.clip(candidate, lower, upper)
    candidate_xy = reference.xy + candidate[:, None] * normal
    candidate_profile = _profile(candidate_xy, **limits)
    candidate_time = float(np.sum(candidate_profile[1] / candidate_profile[0]))
    accepted = candidate_time < baseline_time - 1e-7
    metadata = {'method': 'bounded_periodic_cubic_bspline_physical_lap_time',
        'knot_spacing_target_m': knot_spacing_m, 'coefficient_count': count,
        'max_iterations': max_iterations, 'time_budget_s': time_budget_s,
        'wall_seconds': time.monotonic() - start, 'objective_calls': calls,
        'terminated_by_wall_budget': terminated_by_budget,
        'solver_converged': bool(result.success) if result is not None else False,
        'message': str(result.message) if result is not None else 'finite wall budget reached',
        'input_estimated_lap_s': baseline_time, 'candidate_estimated_lap_s': candidate_time,
        'accepted': bool(accepted), 'global_optimum_claimed': False}
    return (candidate if accepted else offset), metadata


def _residual(offset, xy, normal, base_ds, offset_weight, slope_weight, bend_weight,
              apex_weight=None, apex_target=None):
    path = xy + offset[:, None] * normal
    ds, _, curvature = _geometry(path)
    local_ds = .5 * (ds + np.roll(ds, 1))
    slope = (np.roll(offset, -1) - offset) / base_ds
    bend = (np.roll(offset, -1) - 2. * offset + np.roll(offset, 1)) / np.maximum(base_ds ** 2, .01)
    weight = offset_weight if apex_weight is None else offset_weight + apex_weight
    goal = 0. if apex_weight is None else apex_weight*apex_target/weight
    return np.concatenate((curvature * np.sqrt(local_ds),
        np.sqrt(weight) * (offset-goal) * np.sqrt(base_ds),
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


def _road_pair_mask(xy, reference, road):
    heading = _geometry(xy)[1]
    tangent = np.column_stack((np.cos(heading),np.sin(heading)))
    normal = np.column_stack((-np.sin(heading),np.cos(heading)))
    inside = [road.contains_points(xy+axle*tangent+side*.85*normal,
        progress_m=reference.s,margin_m=road.planning_patch_margin_m)
        for axle,side in ((0.,1.),(0.,-1.),(3.2,1.),(3.2,-1.))]
    return (inside[0]&inside[2])|(inside[1]&inside[3])


def _ordered_projection(reference, points, guesses):
    base=np.searchsorted(reference.s,np.asarray(guesses)%reference.length,side='right')-1
    ids=(base[:,None]+np.arange(-20,21))%len(reference.s)
    segment=reference.segment[ids]
    delta=np.asarray(points)[:,None,:]-reference.xy[ids]
    fraction=np.clip(np.sum(delta*segment,axis=2)/reference.ds[ids]**2,0.,1.)
    nearest=reference.xy[ids]+fraction[:,:,None]*segment
    best=np.argmin(np.sum((nearest-np.asarray(points)[:,None,:])**2,axis=2),axis=1)
    chosen=ids[np.arange(len(points)),best]
    progress=reference.s[chosen]+fraction[np.arange(len(points)),best]*reference.ds[chosen]
    return progress+np.round((np.asarray(guesses)-progress)/reference.length)*reference.length


def _isotonic(values):
    levels=[];weights=[];sizes=[]
    for value in values:
        levels.append(float(value));weights.append(1.);sizes.append(1)
        while len(levels)>1 and levels[-2]>levels[-1]:
            weight=weights[-2]+weights[-1]
            level=(levels[-2]*weights[-2]+levels[-1]*weights[-1])/weight
            size=sizes[-2]+sizes[-1]
            levels[-2:]=[level];weights[-2:]=[weight];sizes[-2:]=[size]
    return np.repeat(levels,sizes)


def _smooth_actual_path(reference, xy, normal, road, lower, upper, limits,
                        smoothing_distances_m=(6.,8.,10.)):
    """Smooth XY in physical arc length, retaining original-progress indexing.

    Original-map normals can fold at noisy tight corners. A path indexed only
    by a scalar normal offset inherits those artificial curvature spikes.
    This stage smooths actual XY and records its tangential mapping shift;
    it never changes the original projection/ranking reference.
    """
    ds=_geometry(xy)[0];s=np.r_[0.,np.cumsum(ds[:-1])];length=float(ds.sum())
    grid=np.arange(0.,length,.5)
    dense=np.column_stack([np.interp(grid,np.r_[s,length],np.r_[xy[:,i],xy[0,i]]) for i in (0,1)])
    candidates=[]
    for smoothing in smoothing_distances_m:
        fit=gaussian_filter1d(dense,float(smoothing)/.5,axis=0,mode='wrap')
        original_guess=np.interp(grid,np.r_[s,length],np.r_[reference.s,reference.length])
        projected=_ordered_projection(reference,fit,original_guess)
        ordered=_isotonic(projected)
        unique,first=np.unique(ordered,return_index=True)
        fit_unique=fit[first]
        extended=np.r_[unique-reference.length,unique,unique+reference.length]
        candidate=np.column_stack([np.interp(reference.s,extended,np.tile(fit_unique[:,i],3)) for i in (0,1)])
        iterations=0
        for iterations in range(100):
            valid=_road_pair_mask(candidate,reference,road)
            if valid.all():break
            toward=reference.xy-candidate
            toward/=np.maximum(np.linalg.norm(toward,axis=1)[:,None],.01)
            correction=gaussian_filter1d((~valid)[:,None]*toward,2.5,axis=0,mode='wrap')
            candidate+=.8*correction
        profile=_profile(candidate,**limits)
        metric=SmoothRacingLine._metrics(candidate,profile)
        actual_progress=_ordered_projection(reference,candidate,reference.s)
        candidates.append((not valid.all(),metric['estimated_flying_lap_s'],candidate,
            profile,{'smoothing_distance_m':float(smoothing),'repair_iterations':iterations,
            'planning_pair_violations_at_map_samples':int((~valid).sum()),
            'maximum_isotonic_progress_adjustment_m':float(np.max(abs(ordered-projected))),
            'maximum_final_progress_projection_residual_m':float(np.max(abs(actual_progress-reference.s))),
            'p99_final_progress_projection_residual_m':float(np.percentile(abs(actual_progress-reference.s),99)),**metric}))
    valid_candidates=[x for x in candidates if not x[0]]
    if not valid_candidates:
        raise ValueError('No smoothed path satisfies planned tyre-pair geometry')
    chosen=min(valid_candidates,key=lambda x:x[1])
    return chosen[2],chosen[3],{'method':'physical_arc_length_xy_smoothing_and_local_tyre_pair_repair',
        'selected_smoothing_distance_m':chosen[4]['smoothing_distance_m'],
        'candidates':[x[4] for x in candidates],
        'original_progress_mapping_retained':True,'original_ranking_reference_changed':False}


class SmoothRacingLine:
    """Periodic line and speed profiles indexed by ORIGINAL reference progress.

    ``nominal_offset_m`` returns an absolute offset in the original map's normal
    direction. A policy may instead request either passing side; call
    ``feasible_speed(..., lateral_offset=d)`` for that absolute constant lane.
    Projection/ranking must continue to use the original reference.
    """
    def __init__(self, reference, *, road=None, offset_limit_m=12., passing_limit_m=12.,
                 optimize=True, max_nfev=300, smoothing_distances_m=(6., 12., 20.),
                 offset_weight=1e-6, slope_weight=.002, bend_weight=.2,
                 nominal_lateral_rate_mps=None, lap_time_refinement=True,
                 refinement_time_budget_s=45., refinement_max_iterations=70,
                 target_geometric_apexes=True):
        self.reference = reference
        self.length = float(reference.length)
        self.s = np.asarray(reference.s, float).copy()
        self.base_ds = np.asarray(reference.ds, float).copy()
        self.offset_limit_m = float(offset_limit_m)
        self.passing_limit_m = float(passing_limit_m)
        self.nominal_lateral_rate_mps = (None if nominal_lateral_rate_mps is None
            else float(nominal_lateral_rate_mps))
        if self.nominal_lateral_rate_mps is not None and not 0. < self.nominal_lateral_rate_mps <= 20.:
            raise ValueError("Invalid map-relative nominal rate budget")
        if (not 0. < self.offset_limit_m <= self.passing_limit_m <= 12.
                or min(offset_weight, slope_weight, bend_weight) < 0.):
            raise ValueError('Invalid line bounds or regularization')
        self.normal = np.column_stack((-np.sin(reference.heading), np.cos(reference.heading)))
        self.path_geometry_representation = 'normal_offset'
        self.road_geometry_sha256 = None if road is None else road.geometry_sha256
        self.road = road
        if road is not None:
            if road._bounds is None:
                road._build_center_bounds()
            self.offset_lower = np.maximum(road._bounds[:, 0], -self.offset_limit_m)
            self.offset_upper = np.minimum(road._bounds[:, 1], self.offset_limit_m)
        else:
            self.offset_lower = np.full(len(self.s), -self.offset_limit_m)
            self.offset_upper = np.full(len(self.s), self.offset_limit_m)
        self.limits = dict(max_speed=float(reference.max_speed),
            lateral_accel=float(reference.lateral_accel), brake_decel=float(reference.brake_decel),
            acceleration=float(reference.acceleration))
        base_profile = _profile(reference.xy, **self.limits)
        self.baseline_profile = base_profile[0]
        self.baseline_metrics = self._metrics(reference.xy, base_profile)
        n = len(self.s)
        solution = np.zeros(n)
        self.apex_ids = np.array([], dtype=int)
        self.apex_targets = np.array([], dtype=float)
        self.flow_anchors = []
        apex_weight, apex_target = np.zeros(n), np.zeros(n)
        if road is not None and target_geometric_apexes:
            peaks, _ = find_peaks(abs(reference.curvature), height=.012,
                prominence=.009, distance=max(1, int(85./np.mean(self.base_ds))))
            signs = np.sign(reference.curvature[peaks])
            targets = np.where(signs>0, self.offset_upper[peaks]-.25,
                self.offset_lower[peaks]+.25)
            self.apex_ids, self.apex_targets = peaks, targets
            apex_weight[peaks] = .4
            apex_target[peaks] = targets
            for k, peak in enumerate(peaks):
                previous = peaks[k-1]
                following = peaks[(k+1) % len(peaks)]
                before = float((self.s[peak]-self.s[previous]) % self.length)
                after = float((self.s[following]-self.s[peak]) % self.length)
                sign = float(signs[k])
                for phase, direction, gap, neighbor in (
                    ('entry', -1., before, previous), ('exit', 1., after, following)):
                    # Linked opposite turns use the preceding/following apex
                    # directly, rather than requesting an extra lane reversal.
                    if gap < 200. and np.sign(reference.curvature[neighbor]) != sign:
                        continue
                    anchor_s = (self.s[peak] + direction*min(80., .42*gap)) % self.length
                    index = int(np.argmin(abs((self.s-anchor_s+self.length/2) % self.length-self.length/2)))
                    target = (.85*self.offset_lower[index] if sign>0
                        else .85*self.offset_upper[index])
                    if apex_weight[index] < .03:
                        apex_weight[index], apex_target[index] = .03, target
                        self.flow_anchors.append({'index':index,'phase':phase,
                            'progress_m':float(self.s[index]),'target_offset_m':float(target)})
        anchor_ids = np.flatnonzero(apex_weight)
        anchor_targets = apex_target[anchor_ids]
        self.plan_artifact_sha256 = None
        self.optimization_executed_this_instance = bool(optimize)
        self.optimization = {'enabled': bool(optimize), 'method': 'bounded_sparse_periodic_true_curvature_least_squares',
            'known_map_only': True, 'max_nfev': int(max_nfev)}
        if optimize:
            result = least_squares(_residual, solution, bounds=(self.offset_lower, self.offset_upper),
                args=(np.asarray(reference.xy, float), self.normal, self.base_ds,
                      offset_weight, slope_weight, bend_weight, apex_weight, apex_target),
                jac_sparsity=_jacobian_sparsity(n), method='trf', tr_solver='lsmr',
                ftol=2e-7, xtol=2e-7, gtol=2e-7, max_nfev=int(max_nfev))
            solution = np.asarray(result.x, float)
            self.optimization.update(nfev=int(result.nfev), success=bool(result.success),
                status=int(result.status), message=str(result.message), cost=float(result.cost),
                solver_converged=bool(result.success), global_optimum_claimed=False)
        candidates = []
        # Smoothing choices are predeclared, map-only. Include the original line
        # as a fallback so a numerical optimization cannot worsen its estimate.
        smoothing_choices = (0., 2., 4.) if len(self.apex_ids) else (None, *smoothing_distances_m)
        for smoothing in smoothing_choices:
            if smoothing is None:
                offset = np.zeros(n)
            else:
                offset = (solution.copy() if smoothing == 0. else
                    gaussian_filter1d(solution, max(.5, float(smoothing) / np.mean(self.base_ds)), mode='wrap'))
                offset = np.clip(offset, self.offset_lower, self.offset_upper)
            xy = reference.xy + offset[:, None] * self.normal
            profile = _profile(xy, **self.limits)
            if smoothing is not None:
                fastest = np.maximum(self.baseline_profile, profile[0])
                for _ in range(4):
                    # Reserve a small margin for profile changes caused by the
                    # projection itself. Original progress is the rate frame.
                    edge_speed = np.maximum(fastest, np.roll(fastest, -1))
                    step = (np.full(n, np.inf) if self.nominal_lateral_rate_mps is None
                        else self.nominal_lateral_rate_mps * self.base_ds / np.maximum(1.05 * edge_speed, 1.))
                    offset = _rate_project(offset, step)
                    offset = np.clip(offset, self.offset_lower, self.offset_upper)
                    xy = reference.xy + offset[:, None] * self.normal
                    profile = _profile(xy, **self.limits)
                    fastest = np.maximum(fastest, profile[0])
            metrics = self._metrics(xy, profile)
            apex_cost = 8.*float(np.sum((offset[anchor_ids]-anchor_targets)**2))
            candidates.append((metrics['estimated_flying_lap_s']+apex_cost, offset, xy, profile, metrics, smoothing))
        _, self.offset, self.xy, profile, self.metrics, selected_smoothing = min(candidates, key=lambda x: x[0])
        if optimize and lap_time_refinement:
            self.offset, refinement = _lap_time_refine(reference, self.offset,
                self.normal, self.limits, offset_limit=self.offset_limit_m,
                lateral_rate=self.nominal_lateral_rate_mps,
                lower=self.offset_lower, upper=self.offset_upper,
                apex_ids=anchor_ids, apex_targets=anchor_targets,
                max_iterations=refinement_max_iterations,
                time_budget_s=refinement_time_budget_s)
            self.xy = reference.xy + self.offset[:, None] * self.normal
            profile = _profile(self.xy, **self.limits)
            self.metrics = self._metrics(self.xy, profile)
            self.optimization['physical_lap_time_refinement'] = refinement
        if road is not None and len(self.apex_ids):
            self.xy,profile,smoothed=_smooth_actual_path(reference,self.xy,
                self.normal,road,self.offset_lower,self.offset_upper,self.limits)
            self.offset=np.sum((self.xy-reference.xy)*self.normal,axis=1)
            self.metrics=self._metrics(self.xy,profile)
            self.path_geometry_representation='smooth_xy_reindexed_by_original_ordered_projection'
            self.optimization['physical_xy_smoothing']=smoothed
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
            offset_lower=self.offset_lower, offset_upper=self.offset_upper,
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
                 'offset_lower', 'offset_upper', 'baseline_profile', 'lane_keys', 'lane_profiles')}
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
        for name in ('original_s', 'offset', 'speed', 'ds', 'heading', 'curvature', 'baseline_profile', 'offset_lower', 'offset_upper'):
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
        self.nominal_lateral_rate_mps = metadata['nominal_lateral_rate_budget_mps']
        self.apex_ids = np.asarray([x['index'] for x in metadata.get('geometric_apexes', [])], int)
        self.apex_targets = np.asarray([x['target_offset_m'] for x in metadata.get('geometric_apexes', [])], float)
        self.flow_anchors = metadata.get('outside_entry_exit_anchors', [])
        self.road_geometry_sha256 = metadata.get('road_geometry_sha256')
        self.path_geometry_representation = metadata.get('path_geometry_representation','normal_offset')
        self.road = None
        self.offset_lower, self.offset_upper = arrays['offset_lower'], arrays['offset_upper']
        self.normal = np.column_stack((-np.sin(reference.heading), np.cos(reference.heading)))
        if (np.max(np.abs(arrays['offset'])) > self.offset_limit_m + 1e-8
                or (self.path_geometry_representation=='normal_offset' and
                    (np.any(arrays['offset'] < self.offset_lower-1e-8)
                     or np.any(arrays['offset'] > self.offset_upper+1e-8)))
                or not np.allclose(np.sum((arrays['xy']-reference.xy)*self.normal,axis=1),arrays['offset'],rtol=0.,atol=1e-9)):
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

    def retimed(self, reference):
        """Recompute physical profiles on frozen geometry; never optimize XY."""
        if self._reference_digest(reference)!=self._reference_digest(self.reference):
            raise ValueError('Retiming requires the exact original ranking geometry')
        result=copy.copy(self)
        result.reference=reference
        result.limits=dict(max_speed=float(reference.max_speed),
            lateral_accel=float(reference.lateral_accel),brake_decel=float(reference.brake_decel),
            acceleration=float(reference.acceleration))
        result.speed,result.ds,result.heading,result.curvature=_profile(self.xy,**result.limits)
        base=_profile(reference.xy,**result.limits)
        result.baseline_profile=base[0]
        result.baseline_metrics=self._metrics(reference.xy,base)
        result.metrics=self._metrics(self.xy,(result.speed,result.ds,result.heading,result.curvature))
        result.estimated_flying_lap_s=result.metrics['estimated_flying_lap_s']
        result._lane_profiles={float(k):_profile(reference.xy+float(k)*self.normal,
            **result.limits)[0] for k in self._lane_keys}
        result.optimization=copy.deepcopy(self.optimization)
        result.optimization['profile_retiming']={'geometry_reoptimized':False,
            'source_plan_sha256':self.plan_artifact_sha256,
            'old_limits':self.limits,'new_limits':result.limits}
        result.optimization_executed_this_instance=False
        result.plan_artifact_sha256=None
        return result

    def center_bounds(self, progress_m, preview_m=0.):
        """Intersection of usable center bounds over an ORIGINAL-progress span."""
        if preview_m < 0 or not math.isfinite(preview_m):
            raise ValueError('Preview must be finite and nonnegative')
        wrapped = float(progress_m) % self.length
        first = int(np.searchsorted(self.s, wrapped, side='right')-1)
        count = min(len(self.s), int(math.ceil(preview_m/np.min(self.base_ds)))+2)
        ids = np.arange(first, first+count) % len(self.s)
        return float(self.offset_lower[ids].max()), float(self.offset_upper[ids].min())

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
        segment_time = 2.*self.ds/np.maximum(self.speed+np.roll(self.speed,-1),1e-8)
        map_offset_rate = np.abs(np.roll(self.offset,-1)-self.offset)/segment_time
        return {'version': VERSION, 'optimization_executed_this_instance': self.optimization_executed_this_instance,
            'plan_artifact_sha256': self.plan_artifact_sha256, 'scope': 'known approximate simulator map; not a surveyed F1 racing line',
            'reference_length_m_for_ranking': self.length, 'original_reference_geometry_changed': False,
            'nominal_offset_limit_m': self.offset_limit_m, 'passing_offset_limit_m': self.passing_limit_m,
            'road_geometry_sha256': self.road_geometry_sha256,
            'path_geometry_representation': self.path_geometry_representation,
            'minimum_derived_center_bound_m': float(self.offset_lower.min()),
            'maximum_derived_center_bound_m': float(self.offset_upper.max()),
            'maximum_nominal_excess_of_constant_heading_center_bounds_m': float(max(
                np.max(self.offset-self.offset_upper),np.max(self.offset_lower-self.offset),0.)),
            'geometric_apexes': [{'index': int(i), 'progress_m': float(self.s[i]),
                'target_offset_m': float(target), 'planned_offset_m': float(self.offset[i]),
                'distance_to_target_m': float(abs(self.offset[i]-target))}
                for i, target in zip(self.apex_ids, self.apex_targets)],
            'outside_entry_exit_anchors': self.flow_anchors,
            'maximum_nominal_offset_m': float(np.max(np.abs(self.offset))),
            'maximum_offset_slope_m_per_m': float(np.max(np.abs(np.diff(np.r_[self.offset, self.offset[0]]) / self.base_ds))),
            'nominal_lateral_rate_budget_mps': self.nominal_lateral_rate_mps,
            'nominal_map_offset_rate_cap_enforced': self.nominal_lateral_rate_mps is not None,
            'map_relative_offset_rate_actual_path_time_mps': dict(zip(
                ['p50','p90','p95','p99','max'],
                [float(x) for x in np.percentile(map_offset_rate,[50,90,95,99,100])])),
            'maximum_profile_lateral_rate_mps': float(np.max(np.abs(np.diff(np.r_[self.offset, self.offset[0]]) / self.base_ds) * np.maximum(self.speed, np.roll(self.speed, -1)))),
            'limits': self.limits, 'baseline': self.baseline_metrics, 'optimized': self.metrics,
            'profile_curvature_definition': 'max(abs(raw geometric curvature), abs(short smoothed curvature)); smoothing never raises the raw-curvature speed limit',
            'maximum_raw_geometric_lateral_accel_mps2': float(np.max(self.speed ** 2 * np.abs(_geometry(self.xy)[2]))),
            'optimization': self.optimization,
            'estimated_lap_time_is_model_only_not_a_simulation_result': True}
