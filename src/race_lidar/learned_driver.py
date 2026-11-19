"""Portable neural track-pace and traffic policy, with NumPy-only inference.

The model predicts high-level target speed and lateral offset. Physical steering,
torque/braking and feasibility limits remain the conventional controller's job.
No reference-speed label or ground-truth opponent state is an input feature.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CURVATURE_DISTANCES_M = (0., 5., 10., 20., 40., 80., 120.)
PROGRESS_HARMONICS = (1, 2, 4, 8)
FEATURE_NAMES = tuple(
    name for harmonic in PROGRESS_HARMONICS
    for name in (f"progress_sin_{harmonic}", f"progress_cos_{harmonic}")
) + ("body_speed", "heading_error", "cross_track") + tuple(
    f"curvature_{int(distance)}m" for distance in CURVATURE_DISTANCES_M
) + ("front_distance", "front_lateral", "front_speed", "left_clearance", "right_clearance", "last_target_offset")


def _number(state, key, default, lower, upper):
    value = float(state.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f"Policy state {key} must be finite")
    return float(np.clip(value, lower, upper))


def feature_vector(state):
    """Feature contract shared by training and runtime.

    Positive lateral values mean left of the reference. A negative front distance
    represents an associated car that is now alongside/behind during a pass.
    ``reference_speed_mps`` may be present for an external guard, but is ignored.
    """
    progress = float(state["progress_fraction"])
    if not math.isfinite(progress):
        raise ValueError("Policy progress_fraction must be finite")
    progress %= 1.
    vector = []
    for harmonic in PROGRESS_HARMONICS:
        angle = 2 * math.pi * harmonic * progress
        vector.extend((math.sin(angle), math.cos(angle)))
    speed = _number(state, "body_speed_mps", 0., 0., 80.)
    heading = _number(state, "heading_error_rad", 0., -math.pi, math.pi)
    vector.extend((speed / 60., heading / .5,
                   _number(state, "cross_track_m", 0., -8., 8.) / 4.))
    curvature = np.asarray(state["curvature_ahead"], dtype=np.float32)
    if curvature.shape != (len(CURVATURE_DISTANCES_M),) or not np.isfinite(curvature).all():
        raise ValueError("curvature_ahead needs seven finite values at 0,5,10,20,40,80,120 m")
    vector.extend((np.clip(curvature, -.15, .15) / .03).tolist())
    vector.extend((
        _number(state, "front_distance_m", 60., -20., 60.) / 60.,
        _number(state, "front_lateral_m", 0., -10., 10.) / 5.,
        _number(state, "front_speed_mps", speed, 0., 80.) / 60.,
        _number(state, "left_clearance_m", 10., 0., 10.) / 10.,
        _number(state, "right_clearance_m", 10., 0., 10.) / 10.,
        _number(state, "last_target_offset_m", 0., -3.5, 3.5) / 3.5,
    ))
    return np.asarray(vector, dtype=np.float32)


class LearnedDriver:
    def __init__(self, weights, biases, mean, scale, metadata=None, *, checkpoint_sha256=None):
        self.weights = [np.asarray(value, dtype=np.float32) for value in weights]
        self.biases = [np.asarray(value, dtype=np.float32) for value in biases]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.metadata = dict(metadata or {})
        self.checkpoint_sha256 = checkpoint_sha256
        if self.mean.shape != (len(FEATURE_NAMES),) or self.scale.shape != self.mean.shape:
            raise ValueError("Policy normalization does not match the feature contract")
        if not self.weights or len(self.weights) != len(self.biases) or np.any(self.scale <= 0):
            raise ValueError("Invalid policy layers or normalization scales")
        width = len(FEATURE_NAMES)
        for weight, bias in zip(self.weights, self.biases):
            if weight.ndim != 2 or weight.shape[1] != width or bias.shape != (weight.shape[0],):
                raise ValueError("Invalid policy layer dimensions")
            width = weight.shape[0]
        if width != 2:
            raise ValueError("Policy output must contain speed and offset")
        if not all(np.isfinite(array).all() for array in [*self.weights, *self.biases, self.mean, self.scale]):
            raise ValueError("Nonfinite learned policy parameters")

    @classmethod
    def from_npz(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.get("feature_names") != list(FEATURE_NAMES):
                raise ValueError("Saved policy uses a different feature contract")
            count = int(data["layer_count"].item())
            if not 1 <= count <= 8:
                raise ValueError("Unsupported policy layer count")
            return cls([data[f"weight_{i}"] for i in range(count)],
                       [data[f"bias_{i}"] for i in range(count)], data["feature_mean"],
                       data["feature_scale"], metadata,
                       checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def forward_features(self, features):
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1:] != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("Expected finite encoded policy features")
        values = (values - self.mean) / self.scale
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values = values @ weight.T + bias
            if index + 1 < len(self.weights):
                values = np.maximum(values, 0.)
        return values

    def predict(self, state):
        output = self.forward_features(feature_vector(state))
        if not np.isfinite(output).all():
            raise RuntimeError("Learned policy produced nonfinite output")
        return {"target_speed_mps": float(np.clip(output[0] * 60., 0., 60.)),
                "target_offset_m": float(np.clip(output[1] * 3.5, -3.5, 3.5)),
                "model_used": True}


def save_numpy_model(path, weights, biases, mean, scale, metadata):
    """Save only numeric arrays and JSON, never executable Python pickle."""
    path = Path(path)
    metadata = {**metadata, "feature_names": list(FEATURE_NAMES),
                "format": "numpy_mlp_relu_v1", "output_scales": [60., 3.5]}
    arrays = {"layer_count": np.int32(len(weights)), "feature_mean": np.asarray(mean, np.float32),
              "feature_scale": np.asarray(scale, np.float32), "metadata_json": json.dumps(metadata)}
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        arrays[f"weight_{index}"] = np.asarray(weight, np.float32)
        arrays[f"bias_{index}"] = np.asarray(bias, np.float32)
    # Validate the complete artifact before committing it to disk.
    LearnedDriver(weights, biases, mean, scale, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path
