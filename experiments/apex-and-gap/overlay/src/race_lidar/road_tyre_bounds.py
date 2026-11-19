"""Generated-road geometry and the user's custom two-tyre boundary rule.

The white-line outer edges are the original tarmac ribbon edges. Kerbs are
excluded. Auditing counts four wheel contact centers; optional inward margin
is a conservative finite-patch approximation, not a full tire contact model.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


class RoadTyreBounds:
    wheelbase_m = 3.2
    half_track_m = .85

    def __init__(self, center_xy, normal_xy, half_width_m, *, reference=None,
                 geometry_sha256=None, planning_patch_margin_m=.18,
                 maximum_offset_m=12., build_bounds=False):
        self.center = np.asarray(center_xy, float).copy()
        self.normal = np.asarray(normal_xy, float).copy()
        self.width = np.asarray(half_width_m, float).copy()
        if (self.center.ndim != 2 or self.center.shape[1:] != (2,) or len(self.center)<4
                or self.normal.shape != self.center.shape or self.width.shape != (len(self.center),)
                or not all(np.isfinite(x).all() for x in (self.center,self.normal,self.width))
                or np.any(self.width <= 0.)
                or not np.allclose(np.linalg.norm(self.normal,axis=1),1.,atol=1e-5)):
            raise ValueError('Invalid generated-road geometry')
        self.geometry_sha256 = geometry_sha256
        self.planning_patch_margin_m = float(planning_patch_margin_m)
        self.maximum_offset_m = float(maximum_offset_m)
        self.reference = reference
        if not 0. <= self.planning_patch_margin_m < float(self.width.min()):
            raise ValueError('Invalid finite-patch planning margin')
        self._tree = cKDTree(self.center)
        self.left = self.center + self.normal * self.width[:, None]
        self.right = self.center - self.normal * self.width[:, None]
        self._edge_cache = {0.: (self.left,self.right)}
        self._bounds = None
        if reference is not None and build_bounds:
            self._build_center_bounds()

    @classmethod
    def from_npz(cls, path, *, reference=None, expected_sha256=None, **kwargs):
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('Generated road geometry SHA256 differs')
        with np.load(path, allow_pickle=False) as data:
            return cls(data['center_xy'], data['normal_xy'], data['half_width_m'],
                reference=reference, geometry_sha256=digest, **kwargs)

    def _base_indices(self, points, progress_m=None):
        if progress_m is None or self.reference is None:
            return np.asarray(self._tree.query(points)[1], int)
        progress = np.broadcast_to(np.asarray(progress_m), (len(points),))
        # Original and road archives share ordered source packets. This hint
        # restricts membership to the local ordered strip, not another branch.
        return np.floor((progress % self.reference.length) /
            self.reference.length * len(self.center)).astype(int)

    def contains_points(self, points, *, progress_m=None, margin_m=0.):
        points = np.atleast_2d(np.asarray(points, float))
        if points.shape[1:] != (2,) or not np.isfinite(points).all():
            raise ValueError('Road queries need finite XY points')
        base = self._base_indices(points, progress_m)
        offsets = np.arange(-16, 17) if progress_m is not None else np.arange(-4, 5)
        ids = (base[:, None] + offsets) % len(self.center)
        nxt = (ids + 1) % len(self.center)
        margin = float(margin_m)
        if not math.isfinite(margin) or not 0. <= margin < float(self.width.min()):
            raise ValueError('Invalid road query margin')
        if margin not in self._edge_cache:
            width = self.width-margin
            self._edge_cache[margin] = (self.center+self.normal*width[:,None],
                self.center-self.normal*width[:,None])
        left,right = self._edge_cache[margin]
        a, b, c, d = left[ids], right[ids], right[nxt], left[nxt]

        def in_triangle(x, y, z):
            p = points[:, None, :]
            def cross(u, v):
                return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
            s1, s2, s3 = cross(y-x, p-x), cross(z-y, p-y), cross(x-z, p-z)
            return ((s1 >= -1e-9) & (s2 >= -1e-9) & (s3 >= -1e-9)) | \
                   ((s1 <= 1e-9) & (s2 <= 1e-9) & (s3 <= 1e-9))
        return np.any(in_triangle(a, b, c) | in_triangle(a, c, d), axis=1)

    def wheel_contact_positions(self, position_xy, yaw):
        point = np.asarray(position_xy, float)[:2]
        tangent = np.array([math.cos(yaw), math.sin(yaw)])
        normal = np.array([-math.sin(yaw), math.cos(yaw)])
        return np.array([point + side * self.half_track_m * normal + axle * tangent
            for axle in (0., self.wheelbase_m) for side in (1., -1.)])

    def tyre_contact_status(self, position_xy, yaw, progress_m=None):
        wheels = self.wheel_contact_positions(position_xy, yaw)
        raw = self.contains_points(wheels, progress_m=progress_m)
        patch = self.contains_points(wheels, progress_m=progress_m,
            margin_m=self.planning_patch_margin_m)
        # Order: rear-left, rear-right, front-left, front-right.
        pair = bool((patch[0] and patch[2]) or (patch[1] and patch[3]))
        return {'wheel_names': ['rear_left', 'rear_right', 'front_left', 'front_right'],
            'wheel_contact_centers_xy': wheels.tolist(),
            'wheel_centers_on_track': raw.tolist(), 'on_track_count': int(raw.sum()),
            'requested_two_tyre_rule_valid': bool(raw.sum() >= 2),
            'planning_patch_margin_m': self.planning_patch_margin_m,
            'patch_margin_centers_on_track': patch.tolist(),
            'patch_margin_on_track_count': int(patch.sum()),
            'planning_same_side_pair_valid': pair,
            'diagonal_only_two_centers': bool(raw.sum() == 2 and
                not ((raw[0] and raw[2]) or (raw[1] and raw[3]))),
            'kerbs_count_as_track': False, 'geometry_sha256': self.geometry_sha256}

    def footprint_valid(self, position_xy, yaw, progress_m=None):
        return self.tyre_contact_status(position_xy, yaw, progress_m)[
            'requested_two_tyre_rule_valid']

    def _build_center_bounds(self):
        ref = self.reference
        normal = np.column_stack((-np.sin(ref.heading), np.cos(ref.heading)))
        tangent = np.column_stack((np.cos(ref.heading), np.sin(ref.heading)))
        grid = np.arange(-self.maximum_offset_m, self.maximum_offset_m + .125, .25)
        legal = np.zeros((len(grid), len(ref.s)), bool)
        for j, offset in enumerate(grid):
            origin = ref.xy + offset * normal
            inside = []
            for axle, side in ((0., 1.), (0., -1.), (self.wheelbase_m, 1.), (self.wheelbase_m, -1.)):
                pts = origin + axle * tangent + side * self.half_track_m * normal
                inside.append(self.contains_points(pts, progress_m=ref.s,
                    margin_m=self.planning_patch_margin_m))
            legal[j] = (inside[0] & inside[2]) | (inside[1] & inside[3])
        zero = int(np.argmin(abs(grid)))
        if not legal[zero].all():
            raise ValueError('Original reference does not support a tyre pair on local road')
        lower = np.zeros(len(ref.s)); upper = np.zeros(len(ref.s))
        live_low = np.ones(len(ref.s), bool); live_high = live_low.copy()
        for j in range(zero-1, -1, -1):
            live_low &= legal[j]
            lower[live_low] = grid[j]
        for j in range(zero+1, len(grid)):
            live_high &= legal[j]
            upper[live_high] = grid[j]
        # A quarter-grid inward reserve avoids treating a coarse grid sample
        # as the exact legal extreme. No full-body width margin is applied.
        self._bounds = np.column_stack((lower + .125, upper - .125))

    def center_bounds(self, progress_m, preview_m=0.):
        if self._bounds is None:
            if self.reference is None:
                raise ValueError('center_bounds requires the original ranking reference')
            self._build_center_bounds()
        if preview_m < 0 or not math.isfinite(preview_m):
            raise ValueError('Preview must be finite and nonnegative')
        ref = self.reference
        progress = float(progress_m) % ref.length
        first = int(np.searchsorted(ref.s, progress, side='right')-1)
        count = min(len(ref.s), int(math.ceil(preview_m / np.min(ref.ds))) + 2)
        ids = (np.arange(first, first+count)) % len(ref.s)
        return float(self._bounds[ids, 0].max()), float(self._bounds[ids, 1].min())

    def report(self):
        return {'geometry_sha256': self.geometry_sha256,
            'rule': 'at least two of four wheel contact centers on tarmac/white-line strip; kerbs excluded',
            'planning_rule': 'same-side front/rear pair inside strip shrunk by finite-patch margin',
            'planning_patch_margin_m': self.planning_patch_margin_m,
            'wheelbase_m': self.wheelbase_m, 'track_m': 2*self.half_track_m,
            'minimum_derived_bound_m': None if self._bounds is None else float(self._bounds[:, 0].min()),
            'maximum_derived_bound_m': None if self._bounds is None else float(self._bounds[:, 1].max())}
