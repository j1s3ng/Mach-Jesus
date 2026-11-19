"""Training-only measured fragment/dropout robustness; frozen runtime untouched."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN005 = ROOT / "runs/improvement-12h/rich-driver-005/source/scripts/train_rich_driver.py"
spec = importlib.util.spec_from_file_location("rich006_frozen005_teacher", FROZEN005)
BASE = importlib.util.module_from_spec(spec); spec.loader.exec_module(BASE)


def teacher(state):
    """Retain a partially executed pass during an uninformative missing return.

    All nonempty-observation labels remain the frozen005 intent/cost teacher.
    Empty rows cannot prove free space: while actual displacement is below2.8m,
    an existing command over0.05m keeps its side among near-equal safe rollouts.
    Reached/clear lanes still release under the base teacher. This finite input
    contract has no last-seen age or object memory, so it cannot fully resolve
    arbitrary prolonged occlusion; physical guards remain necessary.
    """
    result = BASE.intent_teacher(state)
    previous, lateral = float(state["last_target_offset_m"]), float(state["cross_track_m"])
    changed = False
    if not state["hazard_tracks"] and abs(previous) > .05 and abs(lateral) < 2.8:
        preferred = 2 if previous > 0. else 0
        raw = np.asarray(result["raw_lane_costs"], float)
        collision = np.asarray(result["lane_collision_flags"], bool)
        if raw[preferred] <= raw.min() + 6. and (not collision[preferred] or collision.all()):
            eligible = raw <= raw.min() + 6.
            costs = raw - raw[preferred]; costs[eligible & (np.arange(3) != preferred)] += 12.
            result.update(lane_index=preferred, target_offset_m=BASE.LANE_OFFSETS_M[preferred], lane_costs=costs,
                          target_speed_mps=float(result["lane_target_speeds_mps"][preferred]))
            changed = True
    result["partial_pass_missing_return_regularization"] = changed
    return result


def mirror(state):
    result = copy.deepcopy(state)
    for key in ("heading_error_rad", "cross_track_m", "last_target_offset_m", "front_lateral_m"):
        result[key] = -float(result[key])
    result["curvature_ahead"] = [-float(value) for value in result["curvature_ahead"]]
    result["left_clearance_m"], result["right_clearance_m"] = result["right_clearance_m"], result["left_clearance_m"]
    for row in result["hazard_tracks"]:
        row["lateral_m"] = -float(row["lateral_m"])
        low, high = row.get("lateral_min_m"), row.get("lateral_max_m")
        if low is not None and high is not None:
            row["lateral_min_m"], row["lateral_max_m"] = -float(high), -float(low)
    return result


def summarize(state):
    """Regenerate legacy measured summaries, never infer empty space from GT."""
    hazards = state["hazard_tracks"]
    relevant = [row for row in hazards if abs(row["lateral_m"]) <= 6. and -12. <= row["relative_distance_m"] <= 100.]
    ahead = [row for row in relevant if row["relative_distance_m"] >= 0.]
    lead = min(ahead, key=lambda row: row["relative_distance_m"]) if ahead else None
    state.update(front_distance_m=lead["relative_distance_m"] if lead else 60.,
                 front_lateral_m=lead["lateral_m"] if lead else 0.,
                 front_speed_mps=lead.get("speed_mps", 0.) if lead else state["body_speed_mps"])
    for key, lane in (("left_clearance_m", 2.5), ("right_clearance_m", -2.5)):
        state[key] = min(10., min((max(0., abs(row["relative_distance_m"]) - 5.) for row in relevant
                                   if abs(row["lateral_m"] - lane) < 1.7), default=10.))
    return state


def perturb_measured(state, rng, *, add_fragments=True):
    result = copy.deepcopy(state)
    result["body_speed_mps"] = max(0., float(result["body_speed_mps"]) + float(rng.normal(0., .4)))
    result["cross_track_m"] += float(rng.normal(0., .045))
    result["heading_error_rad"] += float(rng.normal(0., .012))
    for row in result["hazard_tracks"]:
        row["relative_distance_m"] += float(rng.normal(0., .35))
        delta = float(rng.normal(0., .045)); row["lateral_m"] += delta
        for key in ("lateral_min_m", "lateral_max_m"):
            if key in row:
                row[key] += delta
        if row.get("velocity_valid", True):
            row["speed_mps"] += float(rng.normal(0., .35))
    central = [row for row in result["hazard_tracks"] if 10. < row["relative_distance_m"] < 110. and abs(row["lateral_m"]) < 1.8]
    if central and add_fragments:
        anchor = min(central, key=lambda row: row["relative_distance_m"])
        for _ in range(int(rng.integers(1, 4))):
            side = float(rng.choice([-1., 1.])); lateral = anchor["lateral_m"] + side * float(rng.uniform(.4, 1.))
            age = float(rng.choice([0., .2, .4])); sigma = float(rng.choice([22.5, 45.]))
            result["hazard_tracks"].append({"relative_distance_m": anchor["relative_distance_m"] + float(rng.uniform(1., 6.)),
                "lateral_m": lateral, "lateral_min_m": lateral - .2, "lateral_max_m": lateral + .2,
                "speed_mps": 0., "velocity_valid": False, "speed_std_mps": sigma,
                "position_std_m": math.hypot(.25, sigma * age), "age_s": age,
                "source": "synthetic_measured_low_support_fragment"})
    return summarize(result)
