#!/usr/bin/env python3
"""Train portable V4 driver candidates by supervised trace/traffic imitation.

Prepare measured-pace/synthetic-traffic arrays locally, then train only arrays
on CUDA. This is not reinforcement learning or demonstrated passing behavior.
The existing 24-feature NumPy inference/checkpoint contract is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.training_sources import validate_training_metadata, validate_training_source
from race_lidar.learned_driver import (CURVATURE_DISTANCES_M, FEATURE_NAMES,
                                     LearnedDriver, feature_vector, save_numpy_model)

FAMILIES = ("actual_clear", "perturbed_clear", "following", "stationary_far", "near_stop",
            "oncoming_clipped_alias", "both_sides_blocked", "left_open", "right_open", "side_only",
            "faster_lead", "hold_left", "hold_right", "rear_clear", "no_lead_hold_side", "return_clear",
            "outside_corridor", "dense_sides", "emergency_stationary_dense", "short_gap_cut_in",
            "clearance_boundary", "heading_error_hazard")
HOLDOUT_FAMILIES = (18, 19, 20, 21)
HOLDOUT_SECTORS = (3, 9, 13)
DATA_SEED, FRESH_SEED = 20261114, 9917


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def teacher(state, pace):
    """Stopping/headway labels use only observations available in 24 features.

    Signed oncoming speed cannot survive the current feature encoder: it clips
    to zero. Therefore negative-speed and stopped examples have identical
    conservative labels after encoding, rather than contradictory supervision.
    """
    speed = float(np.clip(state["body_speed_mps"], 0, 80))
    distance = float(np.clip(state["front_distance_m"], -20, 60))
    lateral = float(np.clip(state["front_lateral_m"], -10, 10))
    lead = float(np.clip(state["front_speed_mps"], 0, 80))
    previous = float(np.clip(state["last_target_offset_m"], -3.5, 3.5))
    current = float(np.clip(state["cross_track_m"], -8, 8))
    left, right = (float(np.clip(state[key], 0, 10)) for key in ("left_clearance_m", "right_clearance_m"))
    pace = float(np.clip(pace, 0, 60))
    committed = abs(previous) >= 1.2
    hold = committed and ((-15.5 < distance < 55.) or (distance >= 55. and min(left, right) < 8.5))
    lane = math.copysign(3.2, previous) if hold else 0.
    relevant = 0 <= distance < 55. and abs(lateral-current) < 2.6
    if relevant and not hold and lead < pace-2. and distance >= 14.:
        choices = [(left, 1.), (right, -1.)]
        choices = [(clearance, sign) for clearance, sign in choices
                   if clearance >= 8. and abs(sign*3.2-lateral) >= 2.6]
        if choices:
            _, sign = max(choices, key=lambda pair: (pair[0], pair[1]))
            lane = sign*3.2
    # Lane motion takes time. Brake while the measured body still overlaps the
    # lead's corridor, even when the requested destination is an open lane.
    target = pace
    if relevant:
        braking, reaction, standstill = 7., .7, 7.
        free = max(0., distance-standstill)
        stop_cap = max(0., -braking*reaction + math.sqrt((braking*reaction)**2 + lead*lead + 2*braking*free))
        headway_cap = max(0., lead + (distance-standstill-.85*speed)/1.4)
        target = min(target, stop_cap, headway_cap)
        if distance < 5.:
            target = 0.
    # Heading error is visible ego state; do not teach high pace while sideways.
    heading = abs(float(state["heading_error_rad"]))
    if heading > .2:
        target = min(target, pace*max(.3, 1.-1.5*(heading-.2)))
    return float(target), float(lane), bool(relevant and target < pace-1.), bool(hold)


def augment(state, family, rng, pace):
    value = dict(state)
    if family == 0:
        return value
    value.update(body_speed_mps=float(rng.uniform(0, min(75., max(15., pace+15.)))),
                 heading_error_rad=float(rng.normal(0, .055)), cross_track_m=float(rng.normal(0, .28)),
                 front_distance_m=float(rng.uniform(14, 54)), front_lateral_m=float(rng.uniform(-1, 1)),
                 front_speed_mps=float(rng.uniform(0, max(1., pace-3))),
                 left_clearance_m=10., right_clearance_m=10., last_target_offset_m=0.)
    if family == 1:
        value.update(front_distance_m=60., front_speed_mps=value["body_speed_mps"])
    elif family == 3:
        value.update(front_speed_mps=0., front_distance_m=float(rng.uniform(18, 54)))
    elif family == 4:
        value.update(front_speed_mps=float(rng.uniform(0, 2)), front_distance_m=float(rng.uniform(2, 14)))
    elif family == 5:
        # Negative signed velocity is intentionally aliased by feature_vector.
        value.update(front_speed_mps=float(rng.uniform(-45, 0)), front_distance_m=float(rng.uniform(5, 54)),
                     left_clearance_m=float(rng.uniform(0, 10)), right_clearance_m=float(rng.uniform(0, 10)))
    elif family == 6:
        value.update(left_clearance_m=float(rng.uniform(0, 5)), right_clearance_m=float(rng.uniform(0, 5)))
    elif family == 7:
        value.update(left_clearance_m=float(rng.uniform(8.3, 10)), right_clearance_m=float(rng.uniform(0, 4)),
                     front_lateral_m=float(rng.uniform(-1.8, .3)))
    elif family == 8:
        value.update(left_clearance_m=float(rng.uniform(0, 4)), right_clearance_m=float(rng.uniform(8.3, 10)),
                     front_lateral_m=float(rng.uniform(-.3, 1.8)))
    elif family == 9:
        value.update(front_distance_m=60., front_speed_mps=value["body_speed_mps"])
        value[rng.choice(["left_clearance_m", "right_clearance_m"])] = float(rng.uniform(0, 6))
    elif family == 10:
        value["front_speed_mps"] = float(min(80., pace+rng.uniform(3, 15)))
    elif family in (11, 12):
        sign = 1. if family == 11 else -1.
        value.update(front_distance_m=float(rng.uniform(-11.8, 40)), last_target_offset_m=sign*float(rng.uniform(1.3, 3.2)),
                     cross_track_m=sign*float(rng.uniform(1.3, 3.3)), front_lateral_m=-sign*float(rng.uniform(0, 2.5)))
        value["right_clearance_m" if sign > 0 else "left_clearance_m"] = float(rng.uniform(0, 8))
    elif family == 13:
        sign = float(rng.choice([-1., 1.]))
        value.update(front_distance_m=float(rng.uniform(-20, -16)), last_target_offset_m=sign*3.2,
                     cross_track_m=sign*float(rng.uniform(2.6, 3.4)), front_lateral_m=-sign*2.)
    elif family in (14, 15, 20):
        sign = float(rng.choice([-1., 1.]))
        value.update(front_distance_m=60., front_speed_mps=value["body_speed_mps"],
                     last_target_offset_m=sign*3.2, cross_track_m=sign*float(rng.uniform(2.5, 3.4)))
        if family in (14, 20):
            value["right_clearance_m" if sign > 0 else "left_clearance_m"] = float(rng.uniform(0, 8.2) if family == 14 else rng.uniform(7.5, 9.5))
    elif family == 16:
        value["front_lateral_m"] = float(rng.choice([-1., 1.])*rng.uniform(5.5, 8))
    elif family == 17:
        value.update(left_clearance_m=float(rng.uniform(0, 2)), right_clearance_m=float(rng.uniform(0, 2)),
                     front_distance_m=float(rng.uniform(8, 40)), front_speed_mps=float(rng.uniform(0, 15)))
    elif family == 18:
        value.update(body_speed_mps=float(rng.uniform(25, 65)), front_distance_m=float(rng.uniform(2, 8)), front_speed_mps=0.,
                     left_clearance_m=float(rng.uniform(0, 2)), right_clearance_m=float(rng.uniform(0, 2)))
    elif family == 19:
        value.update(body_speed_mps=float(rng.uniform(25, 65)), front_distance_m=float(rng.uniform(8, 18)),
                     front_speed_mps=float(rng.uniform(0, 8)))
        value[rng.choice(["left_clearance_m", "right_clearance_m"])] = float(rng.uniform(0, 3))
    elif family == 21:
        value.update(body_speed_mps=float(rng.uniform(20, 65)), heading_error_rad=float(rng.choice([-1, 1])*rng.uniform(.2, .5)),
                     cross_track_m=float(rng.uniform(-2, 2)), front_distance_m=float(rng.uniform(8, 25)))
    return value


def prepare(trace, reference_path, output, stride=5):
    from race_lidar.hotlap_driver import RacingReference, wrap_angle
    source_eligibility = [validate_training_source(path) for path in (trace, reference_path, trace.parent/"lap_summary.json")]
    summary = json.loads((trace.parent/"lap_summary.json").read_text())
    if not summary.get("valid_lap") or summary.get("position_resets_after_spawn") != 0:
        raise ValueError("Need a completed valid physical teacher lap with zero pose resets")
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["time_s"] <= summary["lap_time_s"]][::stride]
    reference = RacingReference.from_npz(reference_path)
    bases = []
    for row in rows:
        progress = float(row["progress_m"]) % reference.length
        heading = reference.sample(progress)[1]
        state = {"progress_fraction": progress/reference.length, "body_speed_mps": row["speed_mps"],
                 "heading_error_rad": wrap_angle(heading-row["yaw_rad"]), "cross_track_m": row["cross_track_error_m"],
                 "curvature_ahead": [reference.sample(progress+d)[3] for d in CURVATURE_DISTANCES_M],
                 "front_distance_m": 60., "front_lateral_m": 0., "front_speed_mps": row["speed_mps"],
                 "left_clearance_m": 10., "right_clearance_m": 10., "last_target_offset_m": 0.}
        # Recorded expert action is a label, never an input to the model.
        bases.append((state, float(row["target_speed_mps"]), progress))
    datasets = {}
    for prefix, seed in (("train", DATA_SEED), ("fresh", FRESH_SEED)):
        rng = np.random.default_rng(seed)
        features, targets, sectors, families, progresses, weights, hazards, holds = ([] for _ in range(8))
        for state, pace, progress in bases:
            for family in range(len(FAMILIES)):
                observed = augment(state, family, rng, pace)
                target_speed, offset, hazard, hold = teacher(observed, pace)
                features.append(feature_vector(observed)); targets.append([target_speed/60., offset/3.5])
                sectors.append(min(15, int(progress/reference.length*16))); families.append(family); progresses.append(progress)
                hazards.append(hazard); holds.append(hold)
                weights.append(3. if target_speed < .5 and hazard else 1.8 if hold else 1.5 if family == 0 else 1.)
        for name, value, dtype in (("x", features, np.float32), ("y", targets, np.float32),
                                   ("sectors", sectors, np.int16), ("families", families, np.int16),
                                   ("progress", progresses, np.float32), ("weights", weights, np.float32),
                                   ("hazards", hazards, bool), ("holds", holds, bool)):
            datasets[prefix+"_"+name] = np.asarray(value, dtype=dtype)
    metadata = {"method": "supervised_valid_trace_and_stopping_headway_synthetic_traffic_v4", "reinforcement_learning": False,
                "source_eligibility": source_eligibility,
                "feature_names": list(FEATURE_NAMES), "reference_speed_is_model_input": False,
                "ground_truth_opponent_state_is_model_input": False, "pace_label": "recorded valid expert target_speed_mps",
                "source_trace": str(trace.resolve()), "source_trace_sha256": sha256(trace),
                "source_reference_sha256": sha256(reference_path), "source_valid_lap_time_s": summary["lap_time_s"],
                "teacher_rows": len(rows), "reference_length_m": reference.length, "families": list(FAMILIES),
                "data_seed": DATA_SEED, "fresh_seed": FRESH_SEED, "heldout_sectors": list(HOLDOUT_SECTORS),
                "heldout_families": [FAMILIES[i] for i in HOLDOUT_FAMILIES], "guard_m": 20.,
                "oncoming_limit": "Current feature encoder clips signed front speed to zero; labels treat it as stopped/unknown.",
                "teacher_stopping": {"reaction_seconds": .7, "standstill_gap_m": 7., "braking_mps2": 7., "headway_seconds": .85},
                "limitations": ["One expert physical lap; fresh noise is not an independent track or driving demonstration.",
                                "Traffic labels are synthetic rules, not successful observed overtakes.",
                                "Deployment refits all sectors/families; held-out diagnostics belong to the separate diagnostic fit.",
                                "Physical steering, braking and common safety guards remain conventional."]}
    output.mkdir(parents=True, exist_ok=True)
    datasets["metadata_json"] = json.dumps(metadata)
    np.savez_compressed(output/"training_bundle.npz", **datasets)
    (output/"dataset_report.json").write_text(json.dumps(metadata, indent=2))
    return output/"training_bundle.npz"


def masks(data, prefix, diagnostic):
    length = data["metadata"]["reference_length_m"]
    sector = np.isin(data[prefix+"_sectors"], HOLDOUT_SECTORS)
    family = np.isin(data[prefix+"_families"], HOLDOUT_FAMILIES)
    guard = np.zeros(len(sector), bool)
    for held in HOLDOUT_SECTORS:
        for edge in (held*length/16., (held+1)*length/16.):
            delta = np.abs(data[prefix+"_progress"]-edge)
            guard |= np.minimum(delta, length-delta) <= 20.
    if diagnostic:
        return ~(sector | family | guard)
    return np.ones(len(sector), bool)


def evaluate(prediction, target, families, hazards, holds):
    error = np.abs((prediction-target)*np.array([60., 3.5]))
    overspeed = np.maximum(0, (prediction[:, 0]-target[:, 0])*60.)
    stop = hazards & (target[:, 0] < .5/60.)
    hold = holds & (np.abs(target[:, 1]) > .4)
    actual = families == 0
    return {"samples": len(target), "speed_mae_mps": float(error[:, 0].mean()), "offset_mae_m": float(error[:, 1].mean()),
            "actual_pace_mae_mps": float(error[actual, 0].mean()) if actual.any() else None,
            "hazard_overspeed_p95_mps": float(np.percentile(overspeed[hazards], 95)) if hazards.any() else None,
            "stop_target_fraction_over_2mps": float((prediction[stop, 0]*60. > 2).mean()) if stop.any() else None,
            "hold_direction_accuracy": float((np.sign(prediction[hold, 1]) == np.sign(target[hold, 1])).mean()) if hold.any() else None,
            "hold_offset_mae_m": float(error[hold, 1].mean()) if hold.any() else None}


def fit(data, fit_mask, val_mask, epochs, seed, width, device, output, label):
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    mean = data["train_x"][fit_mask].mean(0)
    scale = np.maximum(data["train_x"][fit_mask].std(0), .08)
    x = torch.as_tensor((data["train_x"]-mean)/scale, device=device)
    y = torch.as_tensor(data["train_y"], device=device)
    weights = torch.as_tensor(data["train_weights"], device=device)
    hazards = torch.as_tensor(data["train_hazards"], device=device)
    vx = torch.as_tensor((data["fresh_x"][val_mask]-mean)/scale, device=device)
    vy = torch.as_tensor(data["fresh_y"][val_mask], device=device)
    vw = torch.as_tensor(data["fresh_weights"][val_mask], device=device)
    vh = torch.as_tensor(data["fresh_hazards"][val_mask], device=device)
    model = torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES), width), torch.nn.ReLU(),
                                torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=1e-5)
    indices = np.flatnonzero(fit_mask)
    def loss(predicted, target, weight, hazard):
        squared = (predicted-target)**2
        value = 2.*squared[:, 0]+squared[:, 1]
        value += 4.*hazard*torch.relu(predicted[:, 0]-target[:, 0])**2
        hold = target[:, 1].abs() > .4
        value += 1.5*hold*torch.relu(-predicted[:, 1]*target[:, 1].sign())**2
        return (value*weight).mean()
    best, selected, saved = math.inf, 0, None
    log = []
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train();rng.shuffle(indices)
        for start in range(0, len(indices), 1024):
            ids = torch.as_tensor(indices[start:start+1024], device=device)
            prediction = model(x[ids]);value = loss(prediction, y[ids], weights[ids], hazards[ids])
            optimizer.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val = float(loss(model(vx), vy, vw, vh))
        log.append({"epoch": epoch, "fresh_seed_selection_loss": val})
        if val < best:
            best, selected = val, epoch
            saved = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0:
            print(f"seed{seed} {label} epoch{epoch}/{epochs} selection={val:.6f}", flush=True)
    model.load_state_dict(saved)
    with torch.inference_mode():
        train_prediction = model(x).cpu().numpy()
        fresh_prediction = model(torch.as_tensor((data["fresh_x"]-mean)/scale, device=device)).cpu().numpy()
    metadata = {**data["metadata"], "trained": True, "seed": seed, "width": width, "selected_epoch": selected,
                "fit": "sector_and_family_excluded_diagnostic" if label == "diagnostic" else "all_sector_family_deployment_refit",
                "training_device": str(device), "independent_physical_evaluation_completed": False}
    layers = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    path = output/("diagnostic_model.npz" if label == "diagnostic" else "model.npz")
    save_numpy_model(path, [layer.weight.detach().cpu().numpy() for layer in layers],
                     [layer.bias.detach().cpu().numpy() for layer in layers], mean, scale, metadata)
    portable = LearnedDriver.from_npz(path)
    parity = float(np.max(np.abs(portable.forward_features(data["fresh_x"][:512])-fresh_prediction[:512])))
    if parity > 3e-5:
        raise RuntimeError(f"CUDA/NumPy parity failed: {parity}")
    (output/(label+"_epochs.json")).write_text(json.dumps(log, indent=2))
    return {"path": str(path), "sha256": sha256(path), "selected_epoch": selected, "numpy_cuda_max_abs_error": parity,
            "training_seconds": time.perf_counter()-started, "fresh_prediction": fresh_prediction,
            "train_metrics": evaluate(train_prediction[fit_mask], data["train_y"][fit_mask],
                                      data["train_families"][fit_mask], data["train_hazards"][fit_mask], data["train_holds"][fit_mask])}


def train(bundle, output, seeds, epochs, width, device):
    validate_training_source(bundle)
    with np.load(bundle, allow_pickle=False) as loaded:
        metadata = json.loads(str(loaded["metadata_json"].item()))
        validate_training_metadata(metadata, context="training_bundle.metadata")
        # A renamed bundle cannot conceal the original reserved run path.
        if "source_trace" in metadata:
            validate_training_source(metadata["source_trace"])
        # Reject sealed provenance before optional Torch or feature-array loading.
        import torch
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA training device is unavailable")
        data = {key: loaded[key] for key in loaded.files if key != "metadata_json"}
        data["metadata"] = metadata
    if data["metadata"]["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Training bundle does not match the runtime feature contract")
    report = {"bundle_sha256": sha256(bundle), "metadata": data["metadata"], "device": str(device),
              "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__, "candidates": []}
    output.mkdir(parents=True, exist_ok=True)
    fit_mask, val_mask = masks(data, "train", True), masks(data, "fresh", True)
    sectors = np.isin(data["fresh_sectors"], HOLDOUT_SECTORS)
    families = np.isin(data["fresh_families"], HOLDOUT_FAMILIES)
    groups = {"fresh_seed_seen_domain": ~sectors & ~families, "heldout_sectors": sectors & ~families,
              "heldout_families": ~sectors & families, "heldout_sector_family_combination": sectors & families}
    for seed in seeds:
        destination = output/f"v4-seed{seed}"
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Choose fresh candidate output: {destination}")
        destination.mkdir(parents=True)
        diagnostic = fit(data, fit_mask, val_mask, epochs, seed, width, device, destination, "diagnostic")
        prediction = diagnostic.pop("fresh_prediction")
        diagnostic["metrics"] = {name: evaluate(prediction[mask], data["fresh_y"][mask], data["fresh_families"][mask],
                                                  data["fresh_hazards"][mask], data["fresh_holds"][mask]) for name, mask in groups.items()}
        deployment = fit(data, np.ones(len(data["train_x"]), bool), np.ones(len(data["fresh_x"]), bool),
                         max(100, diagnostic["selected_epoch"]), seed, width, device, destination, "deployment")
        prediction = deployment.pop("fresh_prediction")
        deployment["fresh_seed_metrics"] = evaluate(prediction, data["fresh_y"], data["fresh_families"], data["fresh_hazards"], data["fresh_holds"])
        candidate = {"seed": seed, "width": width, "diagnostic": diagnostic, "deployment": deployment,
                     "fit_samples": int(fit_mask.sum()), "selection_samples": int(val_mask.sum()),
                     "physical_evaluation_pending": True}
        (destination/"training_report.json").write_text(json.dumps(candidate, indent=2))
        report["candidates"].append(candidate)
        (output/"training_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"candidate": str(destination), "deployment": deployment}), flush=True)
    return report


def self_test():
    state = {"progress_fraction": .2, "body_speed_mps": 40., "heading_error_rad": 0., "cross_track_m": 0.,
             "curvature_ahead": [0.]*7, "front_distance_m": 8., "front_lateral_m": 0., "front_speed_mps": 0.,
             "left_clearance_m": 1., "right_clearance_m": 1., "last_target_offset_m": 0.}
    assert teacher(state, 50.)[0] == 0.
    assert teacher({**state, "front_speed_mps": -25.}, 50.) == teacher(state, 50.)
    np.testing.assert_array_equal(feature_vector(state), feature_vector({**state, "front_speed_mps": -25.}))
    hold = {**state, "front_distance_m": -8., "last_target_offset_m": 3.2, "cross_track_m": 3.}
    assert teacher(hold, 40.)[1] == 3.2
    assert teacher({**hold, "front_distance_m": -18.}, 40.)[1] == 0.
    assert teacher({**state, "front_distance_m": 60., "left_clearance_m": 10., "right_clearance_m": 10.}, 42.)[:2] == (42., 0.)
    np.testing.assert_array_equal(feature_vector(state), feature_vector({**state, "reference_speed_mps": 999., "gt_boxes": [1]}))
    print("Teacher safety, passing hysteresis, clipped oncoming alias, and feature-exclusion checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT/"runs/hotlap-physics-pilot-5/control_trace.jsonl")
    parser.add_argument("--reference", type=Path, default=ROOT/".local/f1-reference/reference.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-bundle", type=Path)
    parser.add_argument("--seeds", default="7,17")
    parser.add_argument("--epochs", type=int, default=225)
    parser.add_argument("--width", type=int, choices=(128, 192), default=192)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    self_test()
    bundle = args.train_bundle or prepare(args.trace, args.reference, args.output)
    if args.prepare_only:
        print(json.dumps({"bundle": str(bundle), "sha256": sha256(bundle)}), flush=True)
        return
    train(bundle, args.output, [int(seed) for seed in args.seeds.split(",")], args.epochs, args.width, args.device)


if __name__ == "__main__":
    main()
