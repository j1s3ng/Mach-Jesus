"""Supervised distillation of a short-horizon traffic trajectory-cost teacher.

This is not reinforcement learning. Synthetic inputs describe measured hazard
rows; the teacher sees the same noisy rows as the student, without hidden scene
truth. Entire scene families and a stronger-noise population remain held out.
Deployment uses only the exported NumPy MLP, never the trajectory teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from race_lidar.hotlap_driver import RacingReference
from race_lidar.learned_driver import CURVATURE_DISTANCES_M
from race_lidar.rich_driver import (FEATURE_NAMES, LANE_OFFSETS_M, MODEL_FORMAT, RichDriver,
                                    measured_hazards, rich_feature_vector, save_rich_model)
from scripts.training_sources import POLICY_VERSION, validate_training_source

TRAIN_FAMILIES = ("clear", "single_lead", "blocked_right", "blocked_left", "all_blocked",
                  "rear_pressure", "committed_left", "committed_right", "moving_pack")
HELDOUT_FAMILIES = ("staggered_double_block", "oncoming_escape", "offset_mixed_pack")
EXPANDED_TRAIN_FAMILIES = TRAIN_FAMILIES + HELDOUT_FAMILIES + ("offset_lead", "uncertain_block", "rear_zipper")
FRESH_HELDOUT_FAMILIES = ("wide_obstacle", "braking_wave", "alternating_convoy")
CRITICAL_TRAIN_FAMILIES = EXPANDED_TRAIN_FAMILIES + FRESH_HELDOUT_FAMILIES + ("slow_escape",)
CRITICAL_HELDOUT_FAMILIES = ("lane_edge_pair", "rear_approach_far_stop", "wide_moving_offset")
SPEED_FACTORS = (0., .04, .09, .35, .60, .85, 1.)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nominal_pace(state, max_speed=32.):
    """Generic curvature/braking budget, without any source reference pace."""
    curvature = np.abs(np.asarray(state["curvature_ahead"], float))
    limits = np.sqrt(8. / np.maximum(curvature, 1e-5))
    current = np.sqrt(limits ** 2 + 2. * 7. * np.asarray(CURVATURE_DISTANCES_M))
    return float(np.clip(current.min(), 4., max_speed))


def lateral_rollout(lateral, destinations, velocity, times):
    """Delayed lateral slew bounded by forward speed; zero speed means zero shift."""
    destinations, velocity, times = np.asarray(destinations), np.asarray(velocity), np.asarray(times)
    dt = times[1] - times[0]
    rates = np.minimum(1., np.maximum(0., velocity) / 3.)
    active_dt = np.clip(times[1:] - .35, 0., dt)
    increments = .5 * (rates[:, 1:] + rates[:, :-1]) * active_dt[None, :]
    available = np.column_stack((np.zeros(len(destinations)), np.cumsum(increments, axis=1)))
    difference = destinations - lateral
    path = lateral + np.sign(difference[:, None]) * np.minimum(np.abs(difference[:, None]), available)
    return path, available


def trajectory_costs(state, lanes, speeds, *, horizon_s=4.5, max_speed=32.):
    """Score bounded-acceleration/lane-rate trajectories plus terminal clearance."""
    lanes, speeds = np.asarray(lanes, float), np.asarray(speeds, float)
    times = np.linspace(0., horizon_s, 19)
    dt = times[1] - times[0]
    pace = nominal_pace(state, max_speed)
    body = max(0., float(state["body_speed_mps"]))
    lateral = float(state["cross_track_m"])
    previous = float(state["last_target_offset_m"])
    difference = speeds - body
    acceleration = np.where(difference >= 0., 5., -7.)
    settle = np.abs(difference) / np.abs(acceleration)
    ramp = np.minimum(times[None, :], settle[:, None])
    position = body * ramp + .5 * acceleration[:, None] * ramp ** 2
    position += speeds[:, None] * np.maximum(0., times[None, :] - settle[:, None])
    velocity = body + acceleration[:, None] * ramp
    destination = lanes - lateral
    # Lateral motion requires forward motion. In particular, a stationary car
    # with a zero-speed command cannot slide sideways into a free destination.
    lateral_path, available_lateral = lateral_rollout(lateral, lanes, velocity, times)
    hazards = measured_hazards(state)
    nearby = any(-20. < row["relative_distance_m"] < 80. for row in hazards)
    commitment = .65 if nearby else .08
    cost = -.15 * position[:, -1] + .35 * np.abs(lanes) / 3.2
    cost += commitment * np.abs(lanes - previous) / 3.2 + .05 * np.abs(lanes - lateral)
    cost += .2 * ((speeds - pace) / max(pace, 1.)) ** 2
    collided = np.zeros(len(lanes), bool)
    for row in hazards:
        gap = row["relative_distance_m"]
        sigma_position = min(1.5, row["position_std_m"])
        sigma_speed = min(8., row["speed_std_mps"])
        if row["velocity_valid"]:
            low, high = row["speed_mps"] - sigma_speed, row["speed_mps"] + sigma_speed
        else:
            low, high = 0., 70.
        lower = gap + low * times
        upper = gap + high * times
        distance = np.maximum(0., np.maximum(lower[None, :] - position, position - upper[None, :]))
        overlap = ((lateral_path + 1.05 + sigma_position >= row["lateral_min_m"])
                   & (lateral_path - 1.05 - sigma_position <= row["lateral_max_m"]))
        margin = 7.5 + sigma_position
        penetration = np.maximum(0., 1. - distance / margin)
        collision = overlap & (penetration > 0.)
        collided |= np.any(collision, axis=1)
        cost += 250. * np.sum(penetration ** 2 * overlap, axis=1) * dt
        lateral_distance = np.maximum(0., np.maximum(row["lateral_min_m"] - lateral_path,
                                                    lateral_path - row["lateral_max_m"]))
        cost += .8 * np.sum(np.exp(-distance / 12.) * np.exp(-lateral_distance / .7), axis=1) * dt
        in_destination = ((lanes + 1.05 + sigma_position >= row["lateral_min_m"])
                          & (lanes - 1.05 - sigma_position <= row["lateral_max_m"]))
        at_endpoint = ((lateral_path[:, -1] + 1.05 + sigma_position >= row["lateral_min_m"])
                       & (lateral_path[:, -1] - 1.05 - sigma_position <= row["lateral_max_m"]))
        fraction_reached = np.minimum(1., available_lateral[:, -1] / np.maximum(.01, np.abs(destination)))
        fraction_reached = np.where(np.abs(destination) < .01, 1., fraction_reached)
        terminal_occupancy = fraction_reached * in_destination + (1. - fraction_reached) * at_endpoint
        remaining = gap + low * horizon_s - position[:, -1]
        slow = max(0., pace - max(0., low)) / max(1., pace)
        # Terminal cost values a clear destination even if a crossing from the
        # opposite side cannot finish inside the finite planning horizon.
        blocking = terminal_occupancy * (remaining > -8.) * (remaining < 150.) * (gap > -8.)
        cost += 12. * slow * np.exp(-np.clip(remaining, 0., 200.) / 70.) * blocking
        braking_need = velocity[:, -1] ** 2 / 14. + .3 * velocity[:, -1]
        available = remaining - 8. - sigma_position + max(0., low) ** 2 / 14.
        cost += 20. * np.clip((braking_need - available) / np.maximum(20., braking_need), 0., 3.) * blocking
    cost += 1200. * collided
    return cost, collided


def trajectory_teacher(state, *, horizon_s=4.5, max_speed=32.):
    pace = nominal_pace(state, max_speed)
    lanes = np.repeat(LANE_OFFSETS_M, len(SPEED_FACTORS))
    speeds = np.tile(np.asarray(SPEED_FACTORS) * pace, len(LANE_OFFSETS_M))
    costs, collision = trajectory_costs(state, lanes, speeds, horizon_s=horizon_s, max_speed=max_speed)
    costs = costs.reshape(3, -1)
    index = np.argmin(costs, axis=1)
    lane_costs = costs[np.arange(3), index]
    lane_speeds = pace * np.asarray(SPEED_FACTORS)[index]
    selected = int(np.argmin(lane_costs))
    return {"lane_index": selected, "target_offset_m": LANE_OFFSETS_M[selected],
            "target_speed_mps": float(lane_speeds[selected]), "lane_target_speeds_mps": lane_speeds,
            "lane_costs": lane_costs, "lane_collision_flags": collision.reshape(3, -1)[np.arange(3), index]}


def intent_teacher(state, *, horizon_s=4.5, max_speed=32., tie_margin=6., intent_cost=12.):
    """Regularize near-equal trajectory choices using emerging measured intent.

    This is the next training teacher, not a runtime policy or a replacement of
    the frozen003 teacher. Every candidate still receives the same bounded
    acceleration/lateral trajectory rollout. Intent affects only candidates
    within ``tie_margin`` of its best cost and cannot override a uniquely safer
    candidate. Small actual slew commands (e.g. +0.2m) already signal intent;
    waiting for a fully reached lane makes 5Hz left/right chatter self-defeating.
    """
    if tie_margin < 0. or intent_cost <= tie_margin:
        raise ValueError("Intent penalty must exceed a nonnegative tie margin")
    teacher = trajectory_teacher(state, horizon_s=horizon_s, max_speed=max_speed)
    raw_costs = np.asarray(teacher["lane_costs"], float).copy()
    previous = float(state["last_target_offset_m"])
    actual = float(state["cross_track_m"])
    hazards = measured_hazards(state)
    lookahead = min(120., max(60., 8. + 4. * max(0., float(state["body_speed_mps"]))))
    pace = nominal_pace(state, max_speed)
    central = [row for row in hazards if row["lateral_min_m"] <= 1.35 and row["lateral_max_m"] >= -1.35]
    passing = any(-14. < row["relative_distance_m"] < lookahead
                  and (not row["velocity_valid"] or row["speed_mps"] - min(8., row["speed_std_mps"]) < pace - 2.)
                  for row in central)
    alongside = any(abs(row["relative_distance_m"]) < 14. for row in central)
    if passing or (alongside and abs(previous) > .05):
        side = math.copysign(1., previous) if abs(previous) > .05 else math.copysign(1., actual) if abs(actual) > .6 else 1.
        preferred = 2 if side > 0. else 0
    else:
        preferred = 1
    eligible = raw_costs <= raw_costs.min() + tie_margin
    collision = np.asarray(teacher["lane_collision_flags"], bool)
    if not collision.all():
        eligible &= ~collision
    applied = bool(eligible[preferred])
    if applied:
        # Cost differences for clearly inferior/blocked lanes are retained.
        # Only the near-equal alternatives receive an explicit intent penalty.
        costs = raw_costs - raw_costs[preferred]
        costs[eligible & (np.arange(3) != preferred)] += intent_cost
        teacher.update(lane_index=preferred, target_offset_m=LANE_OFFSETS_M[preferred],
                       target_speed_mps=float(teacher["lane_target_speeds_mps"][preferred]), lane_costs=costs)
    teacher.update(raw_lane_costs=raw_costs, intent_regularization_applied=applied,
                   intent_preferred_lane=LANE_OFFSETS_M[preferred], intent_tie_margin=tie_margin)
    return teacher


def generate_commit_scene(reference, family, rng):
    """Training-only measured covariance/intent families motivated by004 failure."""
    state = generate_scene(reference, "clear", rng, expanded=True)
    state["body_speed_mps"] = float(rng.uniform(4., 33.))
    side = float(rng.choice([-1., 1.]))
    previous = side * float(rng.choice([0., .08, .2, .4, .8, 1.5, 2.5, 3.2]))
    state["last_target_offset_m"] = previous
    state["cross_track_m"] = previous * float(rng.uniform(.02, .8)) + float(rng.normal(0., .045))
    gap = float(rng.uniform(35., 118.))

    def row(distance, lane, speed, *, aged=False):
        item = make_hazard(distance, lane, speed, rng)
        item["age_s"] = float(rng.choice([0., .2, .4])) if aged else 0.
        item["speed_std_mps"] = float(rng.choice([2., 7., 15., 22.5, 45.]))
        item["velocity_valid"] = bool(rng.random() > .25)
        if not item["velocity_valid"]:
            item["speed_mps"] = 0.
            item["speed_std_mps"] = max(15., item["speed_std_mps"])
        item["position_std_m"] = float(math.hypot(.25, item["speed_std_mps"] * item["age_s"]))
        item["source"] = "synthetic_measured_covariance_cycle"
        return item

    if family == "commit_covariance_cycle":
        hazards = [row(gap, rng.normal(0., .12), rng.uniform(0., 8.), aged=bool(rng.integers(2)))]
    elif family == "commit_blocked_side":
        hazards = [row(gap, 0., rng.uniform(0., 8.)), row(gap + rng.uniform(-8., 8.), side * 3.2, rng.uniform(0., 8.))]
    elif family == "commit_duplicate_pack":
        hazards = [row(gap, rng.normal(0., .1), rng.uniform(0., 8.), aged=True)]
        duplicate = dict(hazards[0])
        duplicate.update(relative_distance_m=gap + float(rng.normal(0., 2.)), position_std_m=.25, age_s=0.)
        hazards.append(duplicate)
        if rng.random() < .5:
            hazards.append(row(-rng.uniform(25., 65.), -side * 3.2, rng.uniform(25., 50.)))
    elif family == "commit_clear_release":
        hazards = [row(-rng.uniform(5., 35.), 0., rng.uniform(0., 20.))]
    else:
        raise ValueError(f"Unknown commitment scene family: {family}")
    state["hazard_tracks"] = hazards
    ahead = [item for item in hazards if item["relative_distance_m"] >= 0.]
    lead = min(ahead, key=lambda item: item["relative_distance_m"]) if ahead else max(hazards, key=lambda item: item["relative_distance_m"])
    state.update(front_distance_m=lead["relative_distance_m"], front_lateral_m=lead["lateral_m"], front_speed_mps=lead["speed_mps"])
    for key, lane in (("left_clearance_m", 2.5), ("right_clearance_m", -2.5)):
        state[key] = min(10., min((max(0., abs(item["relative_distance_m"]) - 5.) for item in hazards
                                   if abs(item["lateral_m"] - lane) < 1.7), default=10.))
    return state


def make_hazard(distance, lateral, speed, rng, *, noise=1.):
    position_std = float(rng.uniform(.15, .65) * noise)
    speed_std = float(rng.uniform(.7, 3.) * noise)
    valid = bool(rng.random() > .16)
    distance += rng.normal(0., position_std)
    lateral += rng.normal(0., .12 * noise)
    speed += rng.normal(0., speed_std * .25)
    return {"relative_distance_m": float(distance), "lateral_m": float(lateral),
            "speed_mps": float(speed) if valid else 0., "position_std_m": position_std,
            "speed_std_mps": speed_std if valid else float(rng.uniform(15., 35.)), "velocity_valid": valid,
            "age_s": float(rng.uniform(0., .6)), "source": "synthetic_measured_hazard"}


def generate_scene(reference, family, rng, *, noise=1., expanded=False):
    progress = float(rng.uniform(0., reference.length))
    curvature = [reference.sample(progress + distance)[3] for distance in CURVATURE_DISTANCES_M]
    state = {"progress_fraction": progress / reference.length, "body_speed_mps": float(rng.uniform(0., 35.)),
             "heading_error_rad": float(rng.normal(0., .04)), "cross_track_m": 0.,
             "curvature_ahead": curvature, "last_target_offset_m": 0.,
             "front_distance_m": 60., "front_lateral_m": 0., "front_speed_mps": 25.,
             "left_clearance_m": 10., "right_clearance_m": 10.}
    hazards = []
    add = lambda distance, lane, speed: hazards.append(make_hazard(distance, lane, speed, rng, noise=noise))
    if family == "clear":
        state["last_target_offset_m"] = float(rng.choice(LANE_OFFSETS_M))
    elif family == "single_lead":
        add(rng.uniform(18., 115.), rng.normal(0., .15), rng.choice([0., 5., 15., 25.]))
    elif family in ("blocked_right", "blocked_left"):
        side = -1. if family == "blocked_right" else 1.
        gap = rng.uniform(40., 115.)
        add(gap, 0., rng.uniform(0., 8.))
        add(gap + rng.uniform(-12., 12.), side * 3.2, rng.uniform(0., 8.))
        if rng.random() < .6:
            state["last_target_offset_m"] = side * 3.2
    elif family == "all_blocked":
        gap = rng.uniform(15., 90.)
        for lane in LANE_OFFSETS_M:
            add(gap + rng.uniform(-10., 10.), lane, rng.uniform(0., 15.))
    elif family == "rear_pressure":
        side = float(rng.choice([-1., 1.]))
        add(rng.uniform(25., 90.), 0., rng.uniform(0., 12.))
        add(-rng.uniform(10., 60.), side * 3.2, state["body_speed_mps"] + rng.uniform(10., 35.))
    elif family in ("committed_left", "committed_right"):
        side = 1. if family == "committed_left" else -1.
        state["last_target_offset_m"] = side * 3.2
        add(rng.uniform(-18., 25.), 0., max(0., state["body_speed_mps"] + rng.normal(0., 3.)))
        if rng.random() < .4:
            add(rng.uniform(50., 110.), -side * 3.2, rng.uniform(0., 15.))
    elif family == "moving_pack":
        for lane in LANE_OFFSETS_M:
            add(rng.uniform(20., 120.), lane, rng.uniform(0., 38.))
        add(-rng.uniform(12., 75.), float(rng.choice(LANE_OFFSETS_M)), rng.uniform(15., 55.))
    elif family == "staggered_double_block":
        side = float(rng.choice([-1., 1.]))
        add(rng.uniform(25., 60.), side * rng.uniform(.2, 1.), rng.uniform(0., 15.))
        add(rng.uniform(60., 115.), side * rng.uniform(2.1, 3.5), rng.uniform(0., 15.))
        add(-rng.uniform(20., 70.), -side * 3.2, rng.uniform(20., 50.))
    elif family == "oncoming_escape":
        side = float(rng.choice([-1., 1.]))
        add(rng.uniform(40., 110.), 0., -rng.uniform(5., 25.))
        add(rng.uniform(40., 100.), side * 3.2, rng.uniform(0., 15.))
    elif family == "offset_mixed_pack":
        for _ in range(int(rng.integers(2, 7))):
            add(rng.uniform(-60., 120.), rng.uniform(-3.8, 3.8), rng.uniform(-8., 45.))
    elif family == "offset_lead":
        add(rng.uniform(20., 115.), rng.uniform(-3.8, 3.8), rng.uniform(-20., 35.))
        if rng.random() < .5:
            add(rng.uniform(45., 120.), float(rng.choice(LANE_OFFSETS_M)), rng.uniform(0., 20.))
    elif family == "uncertain_block":
        for lane in rng.choice(LANE_OFFSETS_M, size=2, replace=False):
            add(rng.uniform(30., 115.), lane, rng.uniform(0., 15.))
        for row in hazards:
            row.update(position_std_m=float(rng.uniform(.4, 2.)), speed_std_mps=float(rng.uniform(10., 45.)))
    elif family == "rear_zipper":
        side = float(rng.choice([-1., 1.]))
        add(rng.uniform(30., 90.), 0., rng.uniform(0., 12.))
        add(-rng.uniform(25., 65.), side * 3.2, rng.uniform(30., 60.))
        add(rng.uniform(60., 115.), -side * 3.2, rng.uniform(0., 18.))
    elif family == "wide_obstacle":
        side = float(rng.choice([-1., 1.]))
        add(rng.uniform(35., 110.), side * rng.uniform(.5, 1.8), rng.uniform(0., 12.))
        half_width = float(rng.uniform(1.5, 2.8))
        hazards[-1].update(lateral_min_m=hazards[-1]["lateral_m"] - half_width,
                           lateral_max_m=hazards[-1]["lateral_m"] + half_width)
    elif family == "braking_wave":
        lane = float(rng.choice(LANE_OFFSETS_M))
        add(rng.uniform(18., 40.), lane, rng.uniform(25., 40.))
        add(rng.uniform(65., 115.), lane, rng.uniform(0., 8.))
        add(rng.uniform(45., 100.), -3.2 if lane >= 0. else 3.2, rng.uniform(5., 20.))
    elif family == "alternating_convoy":
        for lane in LANE_OFFSETS_M:
            add(rng.uniform(20., 55.), lane, rng.uniform(5., 35.))
            add(rng.uniform(70., 120.), lane, rng.uniform(0., 30.))
    elif family == "slow_escape":
        side = float(rng.choice([-1., 1.]))
        state.update(body_speed_mps=float(rng.uniform(0., 6.)), last_target_offset_m=side * 3.2)
        gap = rng.uniform(7., 40.)
        add(gap, 0., 0.)
        add(gap + rng.uniform(-2., 2.), side * 3.2, 0.)
    elif family == "lane_edge_pair":
        side = float(rng.choice([-1., 1.]))
        add(rng.uniform(30., 80.), side * rng.uniform(1.3, 2.3), rng.uniform(0., 20.))
        add(rng.uniform(80., 120.), -side * rng.uniform(1.2, 2.6), rng.uniform(5., 30.))
    elif family == "rear_approach_far_stop":
        side = float(rng.choice([-1., 1.]))
        add(rng.uniform(20., 35.), 0., rng.uniform(30., 45.))
        add(rng.uniform(80., 115.), 0., rng.uniform(0., 5.))
        add(-rng.uniform(35., 70.), side * 3.2, rng.uniform(35., 65.))
        add(rng.uniform(55., 100.), -side * 3.2, rng.uniform(10., 25.))
    elif family == "wide_moving_offset":
        add(rng.uniform(40., 105.), rng.uniform(-1., 1.), rng.uniform(5., 20.))
        half_width = float(rng.uniform(1.4, 2.4))
        hazards[-1].update(lateral_min_m=hazards[-1]["lateral_m"] - half_width,
                           lateral_max_m=hazards[-1]["lateral_m"] + half_width)
        add(-rng.uniform(25., 60.), float(rng.choice(LANE_OFFSETS_M)), rng.uniform(30., 55.))
    else:
        raise ValueError(f"Unknown scene family: {family}")
    if expanded:
        for row in hazards:
            row["lateral_m"] += float(rng.uniform(-.55, .55))
            if rng.random() < .35:
                row["speed_std_mps"] = float(rng.uniform(3., 30.))
            if not row["velocity_valid"]:
                row["speed_std_mps"] = float(rng.choice([45., rng.uniform(15., 45.)]))
            if family in ("moving_pack", "offset_mixed_pack") and rng.random() < .25:
                row["speed_mps"] = -float(rng.uniform(2., 20.))
    state["cross_track_m"] = state["last_target_offset_m"] + float(rng.normal(0., .18 * noise))
    if hazards:
        ahead = [row for row in hazards if row["relative_distance_m"] >= 0.]
        nearest = min(ahead, key=lambda row: row["relative_distance_m"]) if ahead else max(hazards, key=lambda row: row["relative_distance_m"])
        state.update(front_distance_m=nearest["relative_distance_m"], front_lateral_m=nearest["lateral_m"], front_speed_mps=nearest["speed_mps"])
        for key, lane in (("left_clearance_m", 2.5), ("right_clearance_m", -2.5)):
            state[key] = min(10., min((max(0., abs(row["relative_distance_m"]) - 5.) for row in hazards
                                      if abs(row["lateral_m"] - lane) < 1.7), default=10.))
    state["hazard_tracks"] = hazards
    return state


def build_dataset(reference, *, seed, samples_per_family, heldout_samples, max_speed, horizon_s, expanded=False,
                  summary_mode="nearest", critical=False):
    rng = np.random.default_rng(seed)
    features, speeds, labels, costs, collisions, families, splits = [], [], [], [], [], [], []
    states = []
    train_families = CRITICAL_TRAIN_FAMILIES if critical else EXPANDED_TRAIN_FAMILIES if expanded else TRAIN_FAMILIES
    heldout_families = CRITICAL_HELDOUT_FAMILIES if critical else FRESH_HELDOUT_FAMILIES if expanded else HELDOUT_FAMILIES
    for family in train_families + heldout_families + ("noise_heldout",):
        count = samples_per_family if family in train_families else heldout_samples
        for _ in range(count):
            underlying = str(rng.choice(train_families + heldout_families)) if family == "noise_heldout" else family
            state = generate_scene(reference, underlying, rng, noise=2.5 if family == "noise_heldout" else 1., expanded=expanded)
            teacher = trajectory_teacher(state, max_speed=max_speed, horizon_s=horizon_s)
            features.append(rich_feature_vector(state, mode=summary_mode))
            speeds.append(teacher["lane_target_speeds_mps"] / 60.)
            labels.append(teacher["lane_index"])
            costs.append(teacher["lane_costs"])
            collisions.append(teacher["lane_collision_flags"])
            families.append(family)
            splits.append("train" if family in train_families else "noise_test" if family == "noise_heldout" else "family_test")
            states.append(state)
        print(f"Generated {family}: {count} measured-scene teacher examples", flush=True)
    features, speeds = np.asarray(features, np.float32), np.asarray(speeds, np.float32)
    labels, families, splits = np.asarray(labels), np.asarray(families), np.asarray(splits)
    fit_indices = np.flatnonzero(splits == "train")
    rng.shuffle(fit_indices)
    splits[fit_indices[:max(1, len(fit_indices) // 10)]] = "validation"
    return {"features": features, "speeds": speeds, "labels": labels, "costs": np.asarray(costs, np.float32),
            "collisions": np.asarray(collisions, bool), "families": families, "splits": splits, "states": states}


def evaluate(output, dataset, mask):
    selected = np.asarray(output)[mask]
    target_speed, labels = dataset["speeds"][mask], dataset["labels"][mask]
    prediction = np.argmax(selected[:, 3:], axis=1)
    index = np.arange(len(prediction))
    costs = dataset["costs"][mask]
    best = costs.min(1)
    return {"samples": len(prediction), "lane_accuracy": float((prediction == labels).mean()),
            "all_lane_speed_mae_mps": float(np.abs(selected[:, :3] - target_speed).mean() * 60.),
            "selected_speed_mae_mps": float(np.abs(selected[index, prediction] - target_speed[index, prediction]).mean() * 60.),
            "discrete_lane_cost_regret_mean": float(np.maximum(0., costs[index, prediction] - best).mean()),
            "avoidable_collision_lane_rate": float((dataset["collisions"][mask][index, prediction]
                                                       & ~dataset["collisions"][mask].all(1)).mean())}


def probe_states(reference):
    rng = np.random.default_rng(187)
    result = {}
    for name, side in (("blocked_right_from_center", -1.), ("blocked_left_from_center", 1.),
                       ("blocked_right_already_right", -1.), ("blocked_left_already_left", 1.)):
        state = generate_scene(reference, "clear", rng)
        state.update(progress_fraction=.05, curvature_ahead=[0.] * 7, body_speed_mps=20.,
                     cross_track_m=0., last_target_offset_m=0., front_distance_m=100., front_speed_mps=0.)
        if "already" in name:
            state.update(cross_track_m=side * 3.2, last_target_offset_m=side * 3.2)
        state["hazard_tracks"] = [{"relative_distance_m": 100., "lateral_m": lane, "speed_mps": 0.,
                                   "position_std_m": .35, "speed_std_mps": 1., "velocity_valid": True}
                                  for lane in (0., side * 3.2)]
        result[name] = state
    return result


def train(seed, dataset, args, reference):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    output = args.output / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=False)
    fit, validation = dataset["splits"] == "train", dataset["splits"] == "validation"
    x = dataset["features"]
    mean, scale = x[fit].mean(0), np.maximum(x[fit].std(0), .08)
    features = torch.from_numpy((x - mean) / scale)
    speeds = torch.from_numpy(dataset["speeds"])
    labels = torch.from_numpy(dataset["labels"].astype(np.int64))
    relative_cost = dataset["costs"] - dataset["costs"].min(1, keepdims=True)
    utility = torch.from_numpy(-np.log1p(relative_cost / 5.).astype(np.float32))
    regret = torch.from_numpy(np.minimum(relative_cost / 10., 50.).astype(np.float32))
    soft_labels = torch.softmax(torch.from_numpy(-relative_cost.astype(np.float32) / args.utility_temperature), dim=1)
    model = torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES), 192), torch.nn.ReLU(),
                                torch.nn.Dropout(.04 if args.cost_distillation else 0.),
                                torch.nn.Linear(192, 192), torch.nn.ReLU(),
                                torch.nn.Dropout(.04 if args.cost_distillation else 0.),
                                torch.nn.Linear(192, 96), torch.nn.ReLU(), torch.nn.Linear(96, 6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0001)
    counts = np.bincount(dataset["labels"][fit], minlength=3)
    weights = torch.from_numpy((counts.sum() / np.maximum(1., counts)) ** .35).float()
    weights /= weights.mean()
    rng = np.random.default_rng(seed)
    indices = np.flatnonzero(fit)
    best, best_epoch, best_weights = math.inf, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        model.train()
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            prediction = model(features[batch])
            if args.cost_distillation:
                probabilities = torch.softmax(prediction[:, 3:], dim=1)
                utility_loss = torch.nn.functional.smooth_l1_loss(prediction[:, 3:], utility[batch])
                soft_loss = -(soft_labels[batch] * torch.log_softmax(prediction[:, 3:], dim=1)).sum(1).mean()
                regret_loss = (probabilities * regret[batch]).sum(1).mean()
                lane_loss = .5 * utility_loss + soft_loss + .7 * regret_loss
            else:
                lane_loss = torch.nn.functional.cross_entropy(prediction[:, 3:], labels[batch], weight=weights)
            speed_loss = torch.nn.functional.smooth_l1_loss(prediction[:, :3], speeds[batch])
            loss = lane_loss + 5. * speed_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            predicted = model(features).numpy()
            val = evaluate(predicted, dataset, validation)
        if args.cost_distillation:
            selection = val["discrete_lane_cost_regret_mean"] / 200. + 2. * val["avoidable_collision_lane_rate"] + .005 * val["all_lane_speed_mae_mps"]
        else:
            selection = 1. - val["lane_accuracy"] + .02 * val["all_lane_speed_mae_mps"]
        history.append({"epoch": epoch, "validation": val, "selection_loss": selection})
        if selection < best:
            best, best_epoch = selection, epoch
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0:
            print(f"seed{seed} epoch{epoch}/{args.epochs}: validation lane={val['lane_accuracy']:.3f} speedMAE={val['all_lane_speed_mae_mps']:.2f}", flush=True)
    model.load_state_dict(best_weights)
    with torch.inference_mode():
        predicted = model(features).numpy()
    metadata = {"method": "supervised_distillation_of_short_horizon_trajectory_cost_teacher",
                "trained": True, "reinforcement_learning": False, "seed": seed,
                "teacher_horizon_s": args.horizon, "teacher_max_speed_mps": args.max_speed,
                "teacher_speed_factors": list(SPEED_FACTORS), "teacher_acceleration_mps2": 5.,
                "teacher_brake_deceleration_mps2": 7., "teacher_lateral_rate_mps": 1.,
                "teacher_uses_hidden_scene_truth": False, "training_device": "cpu",
                "source_reference_sha256": sha256(args.reference), "training_script_sha256": sha256(__file__),
                "training_dataset_sha256": sha256(args.output / "training_examples.npz"), "selected_epoch": best_epoch}
    metadata["training_source_policy"] = POLICY_VERSION
    metadata.update(feature_contract_version="measured_lanes_v2_1", lane_summary_mode=args.lane_summary_mode,
                    teacher_dynamics="speed_coupled_lateral_slew_and_reachable_terminal_occupancy", coverage_version=args.coverage,
                    utility_temperature=args.utility_temperature,
                    lane_training_objective="relative_cost_utilities_soft_targets_expected_regret" if args.cost_distillation else "hard_teacher_lane_crossentropy",
                    feature_normalization={"front_gap_m":120.,"rear_gap_m":80.,"signed_speed_mps":60.,
                                           "position_std_m":4.,"speed_std_mps":20.},
                    heldout_families=list(CRITICAL_HELDOUT_FAMILIES if args.coverage=="critical" else FRESH_HELDOUT_FAMILIES if args.coverage=="expanded" else HELDOUT_FAMILIES))
    layers = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    path = save_rich_model(output / "model.npz", [layer.weight.detach().numpy() for layer in layers],
                          [layer.bias.detach().numpy() for layer in layers], mean, scale, metadata)
    portable = RichDriver.from_npz(path)
    parity = float(np.max(np.abs(portable.forward_features(x[:512]) - predicted[:512])))
    if parity > 1e-4:
        raise RuntimeError(f"NumPy/Torch export mismatch: {parity}")
    probes = {}
    for name, state in probe_states(reference).items():
        teacher = trajectory_teacher(state, max_speed=args.max_speed, horizon_s=args.horizon)
        probes[name] = {"state": state, "teacher_target_offset_m": teacher["target_offset_m"],
                        "teacher_target_speed_mps": teacher["target_speed_mps"], "prediction": portable.predict(state)}
    metrics = {split: evaluate(predicted, dataset, dataset["splits"] == split)
               for split in ("train", "validation", "family_test", "noise_test")}
    families = {family: evaluate(predicted, dataset, dataset["families"] == family)
                for family in np.unique(dataset["families"])}
    report = {"status": "completed", "seed": seed, "checkpoint": str(path), "checkpoint_sha256": sha256(path),
              "model_format": MODEL_FORMAT, "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
              "selected_epoch": best_epoch, "numpy_torch_max_abs_error": parity,
              "reference_speed_is_model_input": False, "reinforcement_learning": False,
              "training_method": metadata["method"], "metrics": metrics, "per_family_metrics": families,
              "probes": probes, "limitations": ["Teacher-supervised learning, not reinforcement learning or observed race reward.",
                "No hidden opponent truth is used: both teacher and student consume the same synthetic measured hazard rows.",
                "Held-out family/noise performance measures imitation of this teacher, not physical collision guarantees.",
                "Shared conventional feasibility, perception and low-level actuator guards remain necessary; physical evaluation is pending."]}
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "epochs.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"seed": seed, "checkpoint": str(path), "metrics": metrics, "probes": {
        name: value["prediction"]["target_offset_m"] for name, value in probes.items()}}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / ".local/f1-reference/reference.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17])
    parser.add_argument("--data-seed", type=int, default=20261114)
    parser.add_argument("--samples-per-family", type=int, default=1800)
    parser.add_argument("--heldout-samples", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-speed", type=float, default=32.)
    parser.add_argument("--horizon", type=float, default=4.5)
    parser.add_argument("--coverage", choices=["basic", "expanded", "critical"], default="basic")
    parser.add_argument("--cost-distillation", action="store_true")
    parser.add_argument("--lane-summary-mode", choices=["nearest", "stopping_critical"], default="nearest")
    parser.add_argument("--utility-temperature", type=float, default=3.)
    args = parser.parse_args()
    source_eligibility = validate_training_source(args.reference)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh training output directory")
    if min(args.samples_per_family, args.heldout_samples, args.epochs, args.batch_size, args.cpu_threads) < 1:
        parser.error("Sample counts, epochs, batch size and threads must be positive")
    if not 3. <= args.horizon <= 5. or not 0. < args.max_speed <= 60. or args.utility_temperature <= 0.:
        parser.error("Teacher horizon must be3–5s; pace and utility temperature must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source_eligibility.json").write_text(json.dumps(source_eligibility, indent=2) + "\n")
    started = time.perf_counter()
    reference = RacingReference.from_npz(args.reference)
    dataset = build_dataset(reference, seed=args.data_seed, samples_per_family=args.samples_per_family,
                            heldout_samples=args.heldout_samples, max_speed=args.max_speed, horizon_s=args.horizon,
                            expanded=args.coverage != "basic", summary_mode=args.lane_summary_mode, critical=args.coverage == "critical")
    np.savez(args.output / "training_examples.npz", **{key: value for key, value in dataset.items() if key != "states"})
    with (args.output / "training_observations.jsonl").open("w") as stream:
        for state in dataset["states"]:
            stream.write(json.dumps(state, allow_nan=False) + "\n")
    reports = [train(seed, dataset, args, reference) for seed in args.seeds]
    report = {"status": "completed", "method": "supervised_trajectory_teacher_distillation", "reinforcement_learning": False,
              "training_device": "cpu", "seeds": args.seeds, "data_seed": args.data_seed,
              "dataset_samples": len(dataset["features"]), "training_families": list(CRITICAL_TRAIN_FAMILIES if args.coverage=="critical" else EXPANDED_TRAIN_FAMILIES if args.coverage=="expanded" else TRAIN_FAMILIES),
              "heldout_scene_families": list(CRITICAL_HELDOUT_FAMILIES if args.coverage=="critical" else FRESH_HELDOUT_FAMILIES if args.coverage=="expanded" else HELDOUT_FAMILIES),
              "coverage_version":args.coverage,"cost_distillation":args.cost_distillation,
              "lane_summary_mode":args.lane_summary_mode,"teacher_dynamics":"speed_coupled_lateral_slew_and_reachable_terminal_occupancy",
              "noise_test": "2.5x position/velocity perturbations; teacher sees the same noisy observations",
              "reference_speed_is_model_input": False, "source_reference_sha256": sha256(args.reference),
              "source_eligibility": source_eligibility,
              "training_seconds": time.perf_counter() - started,
              "models": [{"seed": item["seed"], "checkpoint": item["checkpoint"], "sha256": item["checkpoint_sha256"]} for item in reports]}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
