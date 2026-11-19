"""Portable learned racing policy with measured occupancy in all three lanes.

This version retains the original 24 ego/route features and adds long-range
front/rear summaries for each lane. Opponent inputs come exclusively from the
supplied measured hazard rows. Source/reference pace and simulator truth fields
are ignored. The checkpoint contains numeric NumPy arrays and JSON, not pickle.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .learned_driver import FEATURE_NAMES as BASE_FEATURE_NAMES, feature_vector

LANE_OFFSETS_M = (-3.2, 0., 3.2)
LANE_NAMES = ("right", "center", "left")
LANE_FIELDS = ("front_gap", "front_speed", "front_position_std", "front_speed_std",
               "rear_gap", "rear_closing_speed", "rear_position_std", "rear_speed_std")
FEATURE_NAMES = tuple(BASE_FEATURE_NAMES) + tuple(f"{lane}_{field}" for lane in LANE_NAMES for field in LANE_FIELDS)
OUTPUT_NAMES = tuple(f"{lane}_speed_normalized" for lane in LANE_NAMES) + tuple(f"{lane}_logit" for lane in LANE_NAMES)
MODEL_FORMAT = "numpy_rich_mlp_relu_v2"
SUMMARY_MODES = ("nearest", "stopping_critical")


def measured_hazards(state):
    """Read only the explicit measured-opponent contract, never GT fallbacks."""
    if "hazard_tracks" not in state:
        raise ValueError("Rich policy requires hazard_tracks; use an empty list when none were observed")
    hazards = state["hazard_tracks"]
    if not isinstance(hazards, (list, tuple)):
        raise ValueError("hazard_tracks must be a list of measured hazard rows")
    result = []
    for row in hazards:
        if not isinstance(row, dict):
            continue
        try:
            distance = float(row["relative_distance_m"])
            lateral = float(row["lateral_m"])
            speed = float(row.get("speed_mps", 0.))
            position_std = float(row.get("position_std_m", row.get("position_uncertainty_m", .35)))
            speed_std = float(row.get("speed_std_mps", row.get("speed_uncertainty_mps", 4.)))
            lower = float(row.get("lateral_min_m", lateral - 1.05))
            upper = float(row.get("lateral_max_m", lateral + 1.05))
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite([distance, lateral, speed, position_std, speed_std, lower, upper]).all():
            continue
        if not -100. <= distance <= 150. or abs(lateral) > 10.:
            continue
        valid = bool(row.get("velocity_valid", True))
        result.append({"relative_distance_m": distance, "lateral_m": lateral,
            "speed_mps": float(np.clip(speed, -85., 85.)) if valid else 0.,
            "position_std_m": float(np.clip(position_std, 0., 8.)),
            "speed_std_mps": float(np.clip(speed_std, 0., 45.)) if valid else max(15., float(np.clip(speed_std, 0., 45.))),
            # A few visible surface returns do not bound an object's hidden
            # width. Use a conservative generic racing-car planning footprint.
            "lateral_min_m": min(lower, upper, lateral - 1.05),
            "lateral_max_m": max(lower, upper, lateral + 1.05),
            "velocity_valid": valid})
    return result


def _stopping_budget(row):
    """Priority for the front actor that most constrains safe future pace."""
    distance = max(0., row["relative_distance_m"] - 8. - min(2., row["position_std_m"]))
    lead = row["speed_mps"] - min(8., row["speed_std_mps"]) if row["velocity_valid"] else 0.
    if lead >= 0.:
        pace = math.sqrt(2.8 ** 2 + lead ** 2 + 14. * distance) - 2.8
    else:
        approach = -lead
        pace = math.sqrt((2.8 + approach) ** 2 + 14. * max(0., distance - .4 * approach)) - (2.8 + approach)
    return pace, max(0., row["relative_distance_m"])


def lane_summaries(state, *, mode="nearest"):
    if mode not in SUMMARY_MODES:
        raise ValueError(f"Unknown lane summary mode: {mode}")
    hazards = measured_hazards(state)
    ego_speed = float(state.get("body_speed_mps", 0.))
    if not math.isfinite(ego_speed):
        raise ValueError("body_speed_mps must be finite")
    summaries = []
    for lane in LANE_OFFSETS_M:
        occupying = [row for row in hazards if
                     row["lateral_min_m"] - min(1.2, row["position_std_m"]) <= lane + 1.05
                     and row["lateral_max_m"] + min(1.2, row["position_std_m"]) >= lane - 1.05]
        front = [row for row in occupying if row["relative_distance_m"] >= -5.]
        rear = [row for row in occupying if row["relative_distance_m"] < -5.]
        selector = _stopping_budget if mode == "stopping_critical" else lambda row: max(0., row["relative_distance_m"])
        lead = min(front, key=selector) if front else None
        behind = max(rear, key=lambda row: row["relative_distance_m"]) if rear else None
        summaries.append({"lane_offset_m": lane,
            "front_gap_m": min(120., max(0., lead["relative_distance_m"])) if lead else 120.,
            "front_speed_mps": lead["speed_mps"] if lead else max(0., ego_speed),
            "front_position_std_m": lead["position_std_m"] if lead else 0.,
            "front_speed_std_mps": lead["speed_std_mps"] if lead else 0.,
            "rear_gap_m": min(80., abs(behind["relative_distance_m"])) if behind else 80.,
            "rear_closing_speed_mps": max(0., behind["speed_mps"] - ego_speed) if behind else 0.,
            "rear_position_std_m": behind["position_std_m"] if behind else 0.,
            "rear_speed_std_mps": behind["speed_std_mps"] if behind else 0.})
    return summaries


def rich_feature_vector(state, *, mode="nearest"):
    summaries = lane_summaries(state, mode=mode)
    values = list(feature_vector(state))
    for lane in summaries:
        values.extend((lane["front_gap_m"] / 120., np.clip(lane["front_speed_mps"], -85., 85.) / 60.,
                       lane["front_position_std_m"] / 4., lane["front_speed_std_mps"] / 20.,
                       lane["rear_gap_m"] / 80., np.clip(lane["rear_closing_speed_mps"], 0., 85.) / 60.,
                       lane["rear_position_std_m"] / 4., lane["rear_speed_std_mps"] / 20.))
    result = np.asarray(values, np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Expected finite rich policy features")
    return result


class RichDriver:
    def __init__(self, weights, biases, mean, scale, metadata=None, *, checkpoint_sha256=None):
        self.weights = [np.asarray(value, np.float32) for value in weights]
        self.biases = [np.asarray(value, np.float32) for value in biases]
        self.mean, self.scale = np.asarray(mean, np.float32), np.asarray(scale, np.float32)
        self.metadata = dict(metadata or {})
        if self.metadata.get("lane_summary_mode", "nearest") not in SUMMARY_MODES:
            raise ValueError("Unsupported rich checkpoint lane summary mode")
        self.checkpoint_sha256 = checkpoint_sha256
        if self.mean.shape != (len(FEATURE_NAMES),) or self.scale.shape != self.mean.shape or np.any(self.scale <= 0.):
            raise ValueError("Rich policy normalization differs from the feature contract")
        if not self.weights or len(self.weights) != len(self.biases) or len(self.weights) > 8:
            raise ValueError("Invalid rich policy layers")
        width = len(FEATURE_NAMES)
        for weight, bias in zip(self.weights, self.biases):
            if weight.ndim != 2 or weight.shape[1] != width or bias.shape != (weight.shape[0],):
                raise ValueError("Invalid rich policy layer dimensions")
            width = weight.shape[0]
        if width != len(OUTPUT_NAMES):
            raise ValueError("Rich policy requires three speeds and three lane logits")
        if not all(np.isfinite(value).all() for value in [*self.weights, *self.biases, self.mean, self.scale]):
            raise ValueError("Nonfinite rich policy parameters")

    @classmethod
    def from_npz(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.get("format") != MODEL_FORMAT or metadata.get("feature_names") != list(FEATURE_NAMES):
                raise ValueError("Checkpoint is not the rich measured-lane feature contract")
            if metadata.get("output_names") != list(OUTPUT_NAMES):
                raise ValueError("Checkpoint uses different rich policy outputs")
            count = int(data["layer_count"].item())
            if not 1 <= count <= 8:
                raise ValueError("Unsupported rich policy layer count")
            return cls([data[f"weight_{index}"] for index in range(count)],
                       [data[f"bias_{index}"] for index in range(count)], data["feature_mean"], data["feature_scale"],
                       metadata, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def forward_features(self, features):
        values = np.asarray(features, np.float32)
        if values.shape[-1:] != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("Expected finite encoded rich features")
        values = (values - self.mean) / self.scale
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values = values @ weight.T + bias
            if index + 1 < len(self.weights):
                values = np.maximum(values, 0.)
        return values

    def predict(self, state):
        output = self.forward_features(rich_feature_vector(state, mode=self.metadata.get("lane_summary_mode", "nearest")))
        if not np.isfinite(output).all():
            raise RuntimeError("Rich policy produced nonfinite output")
        speeds = np.clip(output[:3] * 60., 0., 60.)
        logits = output[3:]
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        # A numerical tie retains the nearest current lane, without replacing
        # the learned lane scores with a conventional occupancy planner.
        tied = np.flatnonzero(logits >= logits.max() - 1e-7)
        previous = float(state.get("last_target_offset_m", 0.))
        selected = int(min(tied, key=lambda index: abs(LANE_OFFSETS_M[index] - previous)))
        return {"target_speed_mps": float(speeds[selected]), "target_offset_m": LANE_OFFSETS_M[selected],
                "model_used": True, "policy_format": MODEL_FORMAT,
                "lane_target_speeds_mps": speeds.tolist(), "lane_probabilities": probabilities.tolist()}


def save_rich_model(path, weights, biases, mean, scale, metadata):
    path = Path(path)
    metadata = {**metadata, "format": MODEL_FORMAT, "feature_names": list(FEATURE_NAMES),
                "base_feature_names": list(BASE_FEATURE_NAMES), "output_names": list(OUTPUT_NAMES),
                "lane_offsets_m": list(LANE_OFFSETS_M), "reference_speed_is_model_input": False,
                "minimum_planning_obstacle_width_m": 2.1,
                "lane_summary_mode": metadata.get("lane_summary_mode", "nearest"),
                "opponent_inputs": "measured hazard_tracks only", "required_state_keys": ["hazard_tracks"]}
    RichDriver(weights, biases, mean, scale, metadata)
    arrays = {"layer_count": np.int32(len(weights)), "feature_mean": np.asarray(mean, np.float32),
              "feature_scale": np.asarray(scale, np.float32), "metadata_json": json.dumps(metadata)}
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        arrays[f"weight_{index}"] = np.asarray(weight, np.float32)
        arrays[f"bias_{index}"] = np.asarray(bias, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path
