"""Offline engineering-state aggregation with Reference and retained MPC labels.

This is DAgger-style supervised dataset aggregation, not interactive DAgger or
reinforcement learning. Only field4-r0 supplies measured fitting observations.
Field20-r0 was examined in pre-fit design diagnostics: it is a design-informed
whole-run evaluation excluded from fitting and checkpoint selection, not a
pristine unopened test. Formal/bootstrap/holdout/confirmation sources are sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from race_lidar.rich_driver import FEATURE_NAMES, LANE_OFFSETS_M, RichDriver, rich_feature_vector, save_rich_model
from race_lidar.hotlap_driver import RacingReference
from scripts.rich_measured_data import load_run, reference_counterfactual, validate_unsealed_source
from scripts.analyze_rich_measured import guarded
from scripts.training_sources import validate_training_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def renormalize_first_layer(weights, biases, old_mean, old_scale, new_mean, new_scale):
    """Change input normalization without changing the initialized function."""
    result_weights = [np.asarray(value, np.float32).copy() for value in weights]
    result_biases = [np.asarray(value, np.float32).copy() for value in biases]
    first = result_weights[0].copy()
    result_weights[0] = first * (new_scale / old_scale)[None, :]
    result_biases[0] += first @ ((new_mean - old_mean) / old_scale)
    return result_weights, result_biases


def measured_label(state, feasible_speed):
    action = reference_counterfactual(state, feasible_speed)
    distances = np.abs(np.asarray(LANE_OFFSETS_M) - action["target_offset_m"])
    lane = int(np.argmin(distances))
    costs = ((np.asarray(LANE_OFFSETS_M) - action["target_offset_m"]) / 3.2) ** 2 * 15.
    costs -= costs.min()
    return action, lane, costs.astype(np.float32), bool(distances[lane] <= .35)


def actual_split(sample):
    phase = sample["simulation_time_s"] % 50.
    if phase >= 40.:
        return "actual_validation"
    if 38. <= phase < 40. or phase < 2.:
        return "actual_gap"
    return "actual_train"


def fresh_synthetic_cases(args):
    spec = importlib.util.spec_from_file_location("frozen003_mpc", args.mpc_teacher)
    teacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    reference = RacingReference.from_npz(args.reference)
    rng = np.random.default_rng(20261118)
    states, labels, speeds, costs, collisions, families = [], [], [], [], [], []
    for family in ("uncertain_lane_pair", "partial_lane_return", "counterflow_followup"):
        for _ in range(args.fresh_samples_per_family):
            state = teacher.generate_scene(reference, "clear", rng, expanded=True)
            state.update(body_speed_mps=float(rng.uniform(0., 22.)), cross_track_m=0., last_target_offset_m=0.)
            side = float(rng.choice([-1., 1.]))
            def hazard(distance, lateral, speed, unknown=False):
                return {"relative_distance_m": float(distance), "lateral_m": float(lateral), "speed_mps": float(speed),
                        "position_std_m": float(rng.uniform(.25, .8)), "speed_std_mps": 30. if unknown else 3.,
                        "velocity_valid": not unknown, "source": "fresh_synthetic_measured_hazard"}
            if family == "uncertain_lane_pair":
                hazards = [hazard(rng.uniform(35., 95.), 0., 0., True),
                           hazard(rng.uniform(65., 110.), side * 3.2, rng.uniform(0., 15.)),
                           hazard(-rng.uniform(30., 70.), -side * 3.2, rng.uniform(25., 45.))]
            elif family == "partial_lane_return":
                state["last_target_offset_m"] = side * float(rng.uniform(.8, 2.9))
                state["cross_track_m"] = state["last_target_offset_m"] + float(rng.normal(0., .1))
                hazards = [hazard(rng.uniform(-25., 10.), 0., state["body_speed_mps"] + 1.),
                           hazard(rng.uniform(65., 110.), -side * 3.2, rng.uniform(0., 15.))]
            else:
                hazards = [hazard(rng.uniform(20., 35.), 0., rng.uniform(25., 40.)),
                           hazard(rng.uniform(75., 115.), 0., -rng.uniform(5., 15.)),
                           hazard(rng.uniform(55., 90.), side * 3.2, rng.uniform(5., 20.))]
            state["hazard_tracks"] = hazards
            ahead = [row for row in hazards if row["relative_distance_m"] >= 0.]
            lead = min(ahead, key=lambda row: row["relative_distance_m"])
            state.update(front_distance_m=lead["relative_distance_m"], front_lateral_m=lead["lateral_m"], front_speed_mps=lead["speed_mps"])
            for key, lane in (("left_clearance_m", 2.5), ("right_clearance_m", -2.5)):
                state[key] = min(10., min((max(0., abs(row["relative_distance_m"]) - 5.) for row in hazards
                                          if abs(row["lateral_m"] - lane) < 1.7), default=10.))
            label = teacher.trajectory_teacher(state)
            states.append(state)
            labels.append(label["lane_index"])
            speeds.append(label["lane_target_speeds_mps"] / 60.)
            costs.append(label["lane_costs"])
            collisions.append(label["lane_collision_flags"])
            families.append(family)
    data = {"features": np.stack([rich_feature_vector(state, mode="stopping_critical") for state in states]),
            "speeds": np.asarray(speeds, np.float32), "labels": np.asarray(labels), "costs": np.asarray(costs, np.float32),
            "collisions": np.asarray(collisions), "families": np.asarray(families),
            "splits": np.full(len(states), "family_test")}
    folder = args.output / "fresh_synthetic"
    folder.mkdir()
    np.savez(folder / "training_examples.npz", **data)
    (folder / "training_observations.jsonl").write_text("".join(json.dumps(state) + "\n" for state in states))
    return teacher, data, states


def evaluate_fresh(model, data, states, teacher):
    output = model.forward_features(data["features"])
    lanes = np.argmax(output[:, 3:], axis=1)
    regrets, avoidable = [], []
    for index, state in enumerate(states):
        speed = min(20., float(np.clip(output[index, lanes[index]] * 60., 0., 60.)))
        cost, collision = teacher.trajectory_costs(state, [LANE_OFFSETS_M[lanes[index]]], [speed])
        regrets.append(max(0., float(cost[0] - data["costs"][index].min())))
        if not data["collisions"][index].all():
            avoidable.append(bool(collision[0]))
    return {"samples": len(states), "lane_accuracy": float((lanes == data["labels"]).mean()),
            "mean_continuous_action_cost_regret": float(np.mean(regrets)),
            "avoidable_teacher_sweep_violation_rate": float(np.mean(avoidable)) if avoidable else None,
            "physical_collision_metric": False, "used_for_checkpoint_selection": False}


def build_fit_data(args):
    source = args.synthetic
    validate_training_source(source / "training_examples.npz")
    with np.load(source / "training_examples.npz", allow_pickle=False) as saved:
        synthetic = {key: saved[key] for key in saved.files}
    # The original003 family/noise tests remain excluded from fitting.
    keep = np.isin(synthetic["splits"], ["train", "validation"])
    indices = np.flatnonzero(keep)
    data = {"features": synthetic["features"][keep].copy(), "speeds": synthetic["speeds"][keep].copy(),
            "labels": synthetic["labels"][keep].copy(), "costs": synthetic["costs"][keep].copy(),
            "collisions": synthetic["collisions"][keep].copy(),
            "splits": np.where(synthetic["splits"][keep] == "train", "synthetic_train", "synthetic_validation"),
            "kind": np.full(len(indices), "synthetic_mpc"), "lane_mask": np.ones(len(indices), np.float32),
            "speed_mask": np.ones((len(indices), 3), np.float32)}
    actual, manifest = load_run(args.engineering / "rich001-field4-r0")
    records = []
    fields = {key: [] for key in data}
    for sample in actual:
        state = sample["state"]
        action, lane, costs, lane_valid = measured_label(state, sample["label_feasible_speed_mps"])
        speeds = np.zeros(3, np.float32)
        speeds[lane] = action["target_speed_mps"] / 60.
        speed_mask = np.zeros(3, np.float32)
        speed_mask[lane] = 1.
        fields["features"].append(rich_feature_vector(state, mode="stopping_critical"))
        fields["speeds"].append(speeds)
        fields["labels"].append(lane)
        fields["costs"].append(costs)
        fields["collisions"].append(np.zeros(3, bool))
        fields["splits"].append(actual_split(sample))
        fields["kind"].append("measured_reference")
        fields["lane_mask"].append(float(lane_valid))
        fields["speed_mask"].append(speed_mask)
        records.append({"run_id": sample["run_id"], "car_id": sample["car_id"],
            "simulation_time_s": sample["simulation_time_s"], "split": actual_split(sample),
            "observed_state": state, "reference_counterfactual": action,
            "discrete_lane_label": LANE_OFFSETS_M[lane], "lane_label_unambiguous": lane_valid})
    for key in data:
        data[key] = np.concatenate((data[key], np.asarray(fields[key])), axis=0)
    return data, records, manifest


def metrics(prediction, data, split):
    mask = data["splits"] == split
    output = prediction[mask]
    target = data["speeds"][mask]
    lanes = np.argmax(output[:, 3:], axis=1)
    row = np.arange(len(lanes))
    valid = data["lane_mask"][mask] > 0.
    costs = data["costs"][mask]
    value = {"samples": len(lanes), "lane_labels_unambiguous": int(valid.sum()),
             "lane_accuracy": float((lanes[valid] == data["labels"][mask][valid]).mean()) if valid.any() else None,
             "lane_cost_regret": float((costs[row, lanes] - costs.min(1)).mean())}
    if split.startswith("actual"):
        teacher_lane = data["labels"][mask]
        reference = target[row, teacher_lane] * 60.
        commanded = np.clip(output[row, lanes] * 60., 0., 60.)
        shortfall = np.maximum(0., np.minimum(reference, 20.) - np.minimum(commanded, 20.))
        value.update(protocol_capped_speed_shortfall_mps=float(shortfall.mean()),
                     commanded_stop_when_reference_above10_rate=float(((commanded < 2.) & (reference > 10.)).mean()))
    else:
        value.update(all_lane_speed_mae_mps=float(np.abs(output[:, :3] - target).mean() * 60.),
            avoidable_collision_lane_proxy=float((data["collisions"][mask][row, lanes]
                                                  & ~data["collisions"][mask].all(1)).mean()))
    return value


def fit(seed, args, data):
    import torch
    torch.manual_seed(seed)
    initial_path = args.initial / f"seed-{seed}/model.npz"
    initial = RichDriver.from_npz(initial_path)
    x = data["features"].astype(np.float32)
    actual_fit = np.flatnonzero(data["splits"] == "actual_train")
    synthetic_fit = np.flatnonzero(data["splits"] == "synthetic_train")
    indices = np.r_[synthetic_fit, np.tile(actual_fit, args.actual_repeat)]
    mean, scale = x[indices].mean(0), np.maximum(x[indices].std(0), .08)
    weights, biases = renormalize_first_layer(initial.weights, initial.biases, initial.mean, initial.scale, mean, scale)
    layers = []
    linears = []
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(weight))
            layer.bias.copy_(torch.from_numpy(bias))
        layers.append(layer)
        linears.append(layer)
        if index + 1 < len(weights):
            layers.extend((torch.nn.ReLU(), torch.nn.Dropout(.03)))
    model = torch.nn.Sequential(*layers)
    features = torch.from_numpy((x - mean) / scale)
    speeds = torch.from_numpy(data["speeds"].astype(np.float32))
    speed_mask = torch.from_numpy(data["speed_mask"].astype(np.float32))
    lane_mask = torch.from_numpy(data["lane_mask"].astype(np.float32))
    relative = np.maximum(0., data["costs"] - data["costs"].min(1, keepdims=True))
    utilities = torch.from_numpy(-np.log1p(relative / 5.).astype(np.float32))
    regret = torch.from_numpy(np.minimum(relative / 10., 50.).astype(np.float32))
    soft = torch.softmax(torch.from_numpy(-relative.astype(np.float32) / 3.), dim=1)
    model.eval()
    with torch.inference_mode():
        before = model(features[:512]).numpy()
    reparameterization_error = float(np.max(np.abs(before - initial.forward_features(x[:512]))))
    if reparameterization_error > 1e-4:
        raise RuntimeError("Warm-start normalization changed the initialized policy")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate / 8.)
    rng = np.random.default_rng(seed)
    best, best_epoch, best_parameters = math.inf, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        model.train()
        for start in range(0, len(indices), 512):
            batch = indices[start:start + 512]
            output = model(features[batch])
            probability = torch.softmax(output[:, 3:], dim=1)
            lane_loss = .5 * torch.nn.functional.smooth_l1_loss(output[:, 3:], utilities[batch], reduction="none").mean(1)
            lane_loss += -(soft[batch] * torch.log_softmax(output[:, 3:], dim=1)).sum(1)
            lane_loss += .7 * (probability * regret[batch]).sum(1)
            lane_loss = (lane_loss * lane_mask[batch]).sum() / lane_mask[batch].sum().clamp_min(1.)
            speed_loss = torch.nn.functional.smooth_l1_loss(output[:, :3], speeds[batch], reduction="none")
            speed_loss = (speed_loss * speed_mask[batch]).sum() / speed_mask[batch].sum().clamp_min(1.)
            loss = lane_loss + 6. * speed_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(features).numpy()
        actual = metrics(prediction, data, "actual_validation")
        synthetic = metrics(prediction, data, "synthetic_validation")
        selection = .7 * (1. - actual["lane_accuracy"]) + .05 * actual["protocol_capped_speed_shortfall_mps"]
        selection += synthetic["lane_cost_regret"] / 250. + 2. * synthetic["avoidable_collision_lane_proxy"]
        history.append({"epoch": epoch, "actual_validation": actual, "synthetic_validation": synthetic, "selection": selection})
        if selection < best:
            best, best_epoch = selection, epoch
            best_parameters = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0:
            print(f"004 seed{seed} epoch{epoch}: actual lane={actual['lane_accuracy']:.3f}, shortfall={actual['protocol_capped_speed_shortfall_mps']:.2f}; MPC regret={synthetic['lane_cost_regret']:.2f}", flush=True)
    model.load_state_dict(best_parameters)
    model.eval()
    with torch.inference_mode():
        prediction = model(features).numpy()
    output = args.output / f"seed-{seed}"
    output.mkdir(exist_ok=False)
    metadata = {**initial.metadata, "method": "offline_dagger_style_reference_counterfactual_and_mpc_aggregation",
        "reinforcement_learning": False, "interactive_dagger": False, "seed": seed,
        "warm_start_checkpoint_sha256": initial.checkpoint_sha256, "lane_summary_mode": "stopping_critical",
        "actual_fit_run": "rich001-field4-r0", "actual_heldout_run": "rich001-field20-r0",
        "actual_heldout_used_for_gradients_or_selection": False, "reference_cap_used_only_as_label": True,
        "formal_sources_used": False, "selected_epoch": best_epoch,
        "training_script_sha256": sha(__file__), "training_dataset_sha256": sha(args.output / "fitting_examples.npz")}
    path = save_rich_model(output / "model.npz", [layer.weight.detach().numpy() for layer in linears],
        [layer.bias.detach().numpy() for layer in linears], mean, scale, metadata)
    portable = RichDriver.from_npz(path)
    parity = float(np.max(np.abs(portable.forward_features(x[:512]) - prediction[:512])))
    if parity > 1e-4:
        raise RuntimeError("Torch/NumPy DAgger export differs")
    report = {"checkpoint": str(path), "sha256": portable.checkpoint_sha256, "seed": seed,
        "selected_epoch": best_epoch, "warm_start_normalization_error": reparameterization_error,
        "numpy_torch_error": parity, "metrics": {split: metrics(prediction, data, split)
            for split in ("actual_train", "actual_validation", "synthetic_train", "synthetic_validation")}}
    (output / "fit_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "epochs.json").write_text(json.dumps(history, indent=2) + "\n")
    return report


def evaluate_actual(model, samples, visible_distance):
    values = []
    for sample in samples:
        state = sample["state"]
        expert, lane, _, lane_valid = measured_label(state, sample["label_feasible_speed_mps"])
        prediction = model.predict(state)
        own = guarded(prediction, state, sample["label_feasible_speed_mps"], visible_distance)
        reference = guarded(expert, state, sample["label_feasible_speed_mps"], visible_distance)
        values.append({"lane_match": abs(prediction["target_offset_m"] - LANE_OFFSETS_M[lane]) < .1,
            "lane_valid": lane_valid,
            "guarded_shortfall": max(0., reference["target_speed_mps"] - own["target_speed_mps"]),
            "stop_proxy": own["target_speed_mps"] < 2. and reference["target_speed_mps"] > 10.})
    return {"samples": len(values), "unambiguous_lane_accuracy": float(np.mean([row["lane_match"] for row in values if row["lane_valid"]])),
            "mean_one_step_guard_speed_shortfall_mps": float(np.mean([row["guarded_shortfall"] for row in values])),
            "stop_below2_while_expert_guard_above10_rate": float(np.mean([row["stop_proxy"] for row in values]))}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering", type=Path, default=ROOT / "runs/improvement-12h/engineering")
    parser.add_argument("--synthetic", type=Path, default=ROOT / "runs/improvement-12h/rich-driver-003")
    parser.add_argument("--initial", type=Path, default=ROOT / "runs/improvement-12h/rich-driver-003")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--actual-repeat", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.0005)
    parser.add_argument("--visible-distance", type=float, default=45.)
    parser.add_argument("--reference", type=Path, default=ROOT / ".local/f1-reference/reference.npz")
    parser.add_argument("--mpc-teacher", type=Path, default=ROOT / "runs/improvement-12h/rich-driver-003-source/scripts/train_rich_driver.py")
    parser.add_argument("--fresh-samples-per-family", type=int, default=600)
    args = parser.parse_args()
    for path in (args.engineering, args.synthetic, args.initial, args.reference, args.mpc_teacher):
        validate_unsealed_source(path)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh004 output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    source_eligibility = [validate_training_source(path) for path in
                          (args.engineering, args.synthetic, args.initial, args.reference, args.mpc_teacher)]
    (args.output / "source_eligibility.json").write_text(json.dumps(source_eligibility, indent=2) + "\n")
    source_names = ("scripts/train_rich_dagger.py", "scripts/rich_measured_data.py", "scripts/analyze_rich_measured.py",
                    "scripts/training_sources.py",
                    "src/race_lidar/race_policy.py", "src/race_lidar/rich_driver.py", "src/race_lidar/race_control.py")
    sources = {}
    for name in source_names:
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        sources[name] = sha(destination)
    (args.output / "source_sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    data, observations, manifest = build_fit_data(args)
    teacher, fresh, fresh_states = fresh_synthetic_cases(args)
    np.savez(args.output / "fitting_examples.npz", **data)
    with (args.output / "actual_fit_observations.jsonl").open("w") as stream:
        for row in observations:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (args.output / "actual_fit_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    reports = [fit(seed, args, data) for seed in args.seeds]
    # This fitter process opens the20-car source after checkpoint selection.
    # Earlier design diagnostics already examined it; it is not an unseen test.
    heldout, heldout_manifest = load_run(args.engineering / "rich001-field20-r0")
    evaluations = []
    for report in reports:
        model = RichDriver.from_npz(report["checkpoint"])
        evaluations.append({"seed": report["seed"], "checkpoint": report["checkpoint"],
                            "actual_run_holdout": evaluate_actual(model, heldout, args.visible_distance),
                            "fresh_synthetic": evaluate_fresh(model, fresh, fresh_states, teacher)})
    report = {"method": "offline_dagger_style_supervised_dataset_aggregation", "reinforcement_learning": False,
        "interactive_dagger": False, "actual_train_run": "rich001-field4-r0", "entire_actual_heldout_run": "rich001-field20-r0",
        "actual_heldout_used_for_gradients_or_selection": False, "forbidden_sources_opened": [],
        "actual_evaluation_scope": "design-informed whole-run evaluation, excluded from fitting and checkpoint selection",
        "actual_evaluation_pristine_or_unopened": False,
        "reference_speed_is_model_input": False, "feasible_speed_cap_label_only": True,
        "actual_reference_partial_lane_labels_masked": True, "retained_synthetic_source": str(args.synthetic),
        "new_synthetic_families": ["uncertain_lane_pair", "partial_lane_return", "counterflow_followup"],
        "new_synthetic_used_for_gradients_or_selection": False,
        "retained_synthetic_splits": ["train", "validation"], "data_counts": {split: int((data["splits"] == split).sum()) for split in np.unique(data["splits"])},
        "actual_fit_sources": manifest, "actual_holdout_sources": heldout_manifest,
        "source_sha256": sources,
        "source_eligibility": source_eligibility,
        "models": reports, "measured_holdout_evaluations": evaluations, "training_seconds": time.perf_counter() - started,
        "limitations": ["Counterfactual expert labels were not executed from these learner states.",
          "Measured holdout metrics are one-step matched-guard agreement, not lap-time or contact outcomes.",
          "Formal/bootstrap/holdout/confirmation sources were not read; physical gates alone can promote a candidate."]}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"models": reports, "heldout": evaluations, "seconds": report["training_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
