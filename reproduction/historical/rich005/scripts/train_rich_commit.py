"""Rich005 supervised adaptation to measured early-pass intent/covariance cycles.

The failure approach is engineering development data, not an evaluation set.
An entire separately named engineering run is reserved from fitting and epoch
selection. Runtime remains the existing portable48-input/6-output NumPy MLP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from race_lidar.hotlap_driver import RacingReference
from race_lidar.rich_driver import FEATURE_NAMES, RichDriver, rich_feature_vector, save_rich_model
from scripts.rich_measured_data import load_engineering_run
from scripts.train_rich_dagger import evaluate_actual, renormalize_first_layer
from scripts.train_rich_driver import generate_commit_scene, intent_teacher
from scripts.training_sources import validate_training_source

FAMILIES = ("commit_covariance_cycle", "commit_blocked_side", "commit_duplicate_pack", "commit_clear_release")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_teacher_state(fields, state, *, kind, split, cap=60.):
    teacher = intent_teacher(state)
    fields["features"].append(rich_feature_vector(state, mode="stopping_critical"))
    fields["speeds"].append(np.minimum(cap, teacher["lane_target_speeds_mps"]) / 60.)
    fields["labels"].append(teacher["lane_index"])
    fields["costs"].append(teacher["lane_costs"])
    fields["collisions"].append(teacher["lane_collision_flags"])
    fields["splits"].append(split)
    fields["kind"].append(kind)
    fields["lane_mask"].append(1.)
    fields["speed_mask"].append([1., 1., 1.])
    return teacher


def build(args):
    source = args.retained / "fitting_examples.npz"
    validate_training_source(source)
    with np.load(source, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    required = {"features", "speeds", "labels", "costs", "collisions", "splits", "kind", "lane_mask", "speed_mask"}
    if set(data) != required:
        raise ValueError("Retained004 fitting bundle has a different training contract")
    if not set(np.unique(data["splits"])) <= {"synthetic_train", "synthetic_validation", "actual_train", "actual_validation", "actual_gap"}:
        raise ValueError("Retained bundle contains an unexpected evaluation split")
    reference = RacingReference.from_npz(args.reference)
    rng = np.random.default_rng(args.data_seed)
    fields = {key: [] for key in data}
    observations, intent_counts = [], {"synthetic": 0, "failure": 0}
    for family in FAMILIES:
        for index in range(args.samples_per_family):
            state = generate_commit_scene(reference, family, rng)
            split = "commit_validation" if index % 10 == 0 else "commit_train"
            teacher = add_teacher_state(fields, state, kind=family, split=split)
            intent_counts["synthetic"] += int(teacher["intent_regularization_applied"])
            observations.append({"state": state, "kind": family, "split": split})
        print(f"Generated {family}: {args.samples_per_family}", flush=True)
    samples, manifest = load_engineering_run(args.failure_run)
    observed_end = args.failure_end_time
    for sample in samples:
        state = sample["state"]
        if sample["simulation_time_s"] > observed_end or not state["hazard_tracks"]:
            continue
        teacher = add_teacher_state(fields, state, kind="measured_failure_approach", split="failure_train",
                                    cap=sample["label_feasible_speed_mps"])
        intent_counts["failure"] += int(teacher["intent_regularization_applied"])
        observations.append({"state": state, "kind": "measured_failure_approach", "split": "failure_train",
                             "simulation_time_s": sample["simulation_time_s"], "car_id": sample["car_id"]})
    for key in data:
        data[key] = np.concatenate((data[key], np.asarray(fields[key], dtype=data[key].dtype if key not in ("kind", "splits") else None)))
    # Retained string arrays must expand rather than truncate the new labels.
    if not np.any(data["splits"] == "failure_train"):
        raise ValueError("Failure approach contributed no fitting observations")
    return data, observations, manifest, intent_counts


def metrics(prediction, data, mask):
    rows = np.flatnonzero(mask)
    if not len(rows):
        return {"samples": 0}
    output, costs = prediction[rows], data["costs"][rows]
    lane = np.argmax(output[:, 3:], axis=1)
    index = np.arange(len(rows))
    lane_mask = data["lane_mask"][rows] > 0.
    safe_possible = ~data["collisions"][rows].all(1)
    return {"samples": len(rows), "lane_accuracy": float((lane[lane_mask] == data["labels"][rows][lane_mask]).mean()),
            "mean_lane_cost_regret": float(np.maximum(0., costs[index, lane] - costs.min(1)).mean()),
            "avoidable_teacher_lane_collision_rate": float(data["collisions"][rows][index, lane][safe_possible].mean()) if safe_possible.any() else None,
            "all_branch_speed_mae_mps": float((np.abs(output[:, :3] - data["speeds"][rows]) * data["speed_mask"][rows]).sum()
                / max(1., data["speed_mask"][rows].sum()) * 60.)}


def fit(seed, args, data):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    folder = args.output / f"seed-{seed}"
    folder.mkdir()
    initial_path = args.initial / f"seed-{seed}/model.npz"
    validate_training_source(initial_path)
    initial = RichDriver.from_npz(initial_path)
    fit_mask = np.isin(data["splits"], ["synthetic_train", "actual_train", "commit_train", "failure_train"])
    x = np.asarray(data["features"], np.float32)
    mean, scale = x[fit_mask].mean(0), np.maximum(x[fit_mask].std(0), .08)
    weights, biases = renormalize_first_layer(initial.weights, initial.biases, initial.mean, initial.scale, mean, scale)
    layers = []
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(torch.from_numpy(weight)); linear.bias.data.copy_(torch.from_numpy(bias))
        layers.append(linear)
        if index + 1 < len(weights):
            layers.extend([torch.nn.ReLU(), torch.nn.Dropout(.025)])
    model = torch.nn.Sequential(*layers)
    model.eval()
    features = torch.from_numpy((x - mean) / scale)
    with torch.inference_mode():
        initialized = model(features[:512]).numpy()
    warm_parity = float(np.max(np.abs(initialized - initial.forward_features(x[:512]))))
    if warm_parity > 1e-4:
        raise RuntimeError("Warm normalization changed the initialized function")
    start_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    relative = data["costs"] - data["costs"].min(1, keepdims=True)
    utility = torch.from_numpy(-np.log1p(relative / 5.).astype(np.float32))
    soft = torch.softmax(torch.from_numpy(-relative.astype(np.float32) / 3.), dim=1)
    regret = torch.from_numpy(np.minimum(relative / 10., 50.).astype(np.float32))
    speed = torch.from_numpy(data["speeds"].astype(np.float32))
    speed_mask = torch.from_numpy(data["speed_mask"].astype(np.float32))
    lane_mask = torch.from_numpy(data["lane_mask"].astype(np.float32))
    labels = torch.from_numpy(data["labels"].astype(np.int64))
    sample_indices = np.concatenate([np.flatnonzero(data["splits"] == name).repeat(repeat) for name, repeat in
                                    (("synthetic_train", 1), ("actual_train", 4), ("commit_train", 2), ("failure_train", 30))])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.00006)
    best, best_epoch, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(sample_indices)
        model.train()
        for start in range(0, len(sample_indices), args.batch_size):
            batch = sample_indices[start:start + args.batch_size]
            output = model(features[batch])
            probabilities = torch.softmax(output[:, 3:], dim=1)
            utility_loss = torch.nn.functional.smooth_l1_loss(output[:, 3:], utility[batch], reduction="none").mean(1)
            soft_loss = -(soft[batch] * torch.log_softmax(output[:, 3:], dim=1)).sum(1)
            expected_regret = (probabilities * regret[batch]).sum(1)
            hard = torch.nn.functional.cross_entropy(output[:, 3:], labels[batch], reduction="none")
            lane_loss = ((.5 * utility_loss + soft_loss + .7 * expected_regret + .6 * hard) * lane_mask[batch]).mean()
            speed_loss = (torch.nn.functional.smooth_l1_loss(output[:, :3], speed[batch], reduction="none") * speed_mask[batch]).sum() / speed_mask[batch].sum().clamp(min=1.)
            loss = lane_loss + 5. * speed_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        scheduler.step(); model.eval()
        with torch.inference_mode():
            predicted = model(features).numpy()
        validation = {name: metrics(predicted, data, data["splits"] == name) for name in
                      ("synthetic_validation", "actual_validation", "commit_validation")}
        old, actual, new = (validation[key] for key in ("synthetic_validation", "actual_validation", "commit_validation"))
        selection = (old["mean_lane_cost_regret"] / 200. + 2. * (old["avoidable_teacher_lane_collision_rate"] or 0.)
                     + new["mean_lane_cost_regret"] / 35. + 2. * (new["avoidable_teacher_lane_collision_rate"] or 0.)
                     + .2 * (1. - actual["lane_accuracy"]) + .003 * actual["all_branch_speed_mae_mps"])
        history.append({"epoch": epoch, "selection_loss": selection, "validation": validation})
        if selection < best:
            best, best_epoch = selection, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0:
            print(f"seed{seed} epoch{epoch}/{args.epochs} selection={selection:.4f} commitLane={new['lane_accuracy']:.3f}", flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        predicted = model(features).numpy()
    delta = max(float(torch.max(torch.abs(parameter - before)).item()) for parameter, before in zip(model.parameters(), start_parameters))
    if delta <= 1e-8:
        raise RuntimeError("No learned parameter update")
    metadata = {"method": "supervised_trajectory_cost_distillation_with_early_intent_tie_regularization",
                "trained": True, "reinforcement_learning": False, "seed": seed,
                "lane_summary_mode": "stopping_critical", "feature_contract_version": "measured_lanes_v2_1",
                "initial_checkpoint_sha256": sha(initial_path), "training_dataset_sha256": sha(args.output / "fitting_examples.npz"),
                "training_script_sha256": sha(__file__), "teacher_script_sha256": sha(ROOT / "scripts/train_rich_driver.py"),
                "teacher_horizon_s": 4.5, "teacher_max_speed_mps": 32., "intent_tie_margin": 6., "intent_cost": 12.,
                "intent_uses_emerging_measured_slew_m": .05, "reference_speed_is_model_input": False,
                "teacher_uses_hidden_scene_truth": False, "training_device": "cpu", "selected_epoch": best_epoch,
                "engineering_failure_fit_source": str(args.failure_run.resolve()),
                "whole_engineering_evaluation_excluded_from_gradients_and_epoch_selection": str(args.evaluation_run.resolve())}
    linear = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    path = save_rich_model(folder / "model.npz", [layer.weight.detach().numpy() for layer in linear],
                          [layer.bias.detach().numpy() for layer in linear], mean, scale, metadata)
    portable = RichDriver.from_npz(path)
    parity = float(np.max(np.abs(portable.forward_features(x[:512]) - predicted[:512])))
    if parity > 1e-4:
        raise RuntimeError("Portable export mismatch")
    summary = {"seed": seed, "checkpoint": str(path), "checkpoint_sha256": sha(path), "selected_epoch": best_epoch,
               "warm_normalization_max_abs_error": warm_parity, "numpy_torch_max_abs_error": parity,
               "maximum_learned_parameter_change": delta, "selection_loss": best,
               "metrics": {name: metrics(predicted, data, data["splits"] == name) for name in np.unique(data["splits"])}}
    (folder / "training_report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (folder / "epochs.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained", type=Path, default=ROOT / "runs/improvement-12h/rich-driver-004")
    parser.add_argument("--initial", type=Path, default=ROOT / "runs/improvement-12h/rich-driver-004")
    parser.add_argument("--failure-run", type=Path, default=ROOT / "runs/improvement-12h/engineering/rich004-s17-visibility-entry")
    parser.add_argument("--evaluation-run", type=Path, default=ROOT / "runs/improvement-12h/engineering/rich004-s17-field4-diagnostic")
    parser.add_argument("--reference", type=Path, default=ROOT / ".local/f1-reference/reference.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 7])
    parser.add_argument("--data-seed", type=int, default=20261121)
    parser.add_argument("--samples-per-family", type=int, default=5000)
    parser.add_argument("--failure-end-time", type=float, default=25.)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.0004)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh005 output directory")
    if args.failure_run.resolve() == args.evaluation_run.resolve():
        parser.error("Entire engineering evaluation run must differ from fitting failure")
    checks = [validate_training_source(path) for path in (args.retained, args.initial, args.failure_run, args.reference)]
    args.output.mkdir(parents=True, exist_ok=True)
    sources = ("scripts/train_rich_commit.py", "scripts/train_rich_driver.py", "scripts/train_rich_dagger.py",
               "scripts/rich_measured_data.py", "scripts/training_sources.py", "src/race_lidar/rich_driver.py")
    for name in sources:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    source_hashes = {name: sha(args.output / "source" / name) for name in sources}
    import torch
    torch.set_num_threads(args.cpu_threads); torch.set_num_interop_threads(1)
    started = time.perf_counter()
    data, observations, manifest, counts = build(args)
    np.savez(args.output / "fitting_examples.npz", **data)
    with (args.output / "new_observations.jsonl").open("w") as stream:
        for row in observations:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    reports = [fit(seed, args, data) for seed in args.seeds]
    # The complete explicitly named engineering evaluation is opened only after
    # all checkpoint choices. It is never a fitting or epoch-selection source.
    evaluation_manifest, evaluation_reports = [], []
    evaluation_status = "pending_closed_engineering_run"
    if (args.evaluation_run / "heat_results.json").is_file():
        evaluation, evaluation_manifest = load_engineering_run(args.evaluation_run)
        config = json.loads((args.evaluation_run / "config.json").read_text())
        visible = config.get("visible_distance_m", config["rules"]["visible_distance_m"])
        evaluation_reports = [{"seed": item["seed"], **evaluate_actual(RichDriver.from_npz(item["checkpoint"]), evaluation, visible)}
                              for item in reports]
        evaluation_status = "closed_whole_run_counterfactual_completed"
    report = {"method": "supervised_early_intent_and_covariance_cycle_adaptation", "reinforcement_learning": False,
              "feature_count": len(FEATURE_NAMES), "runtime_changed": False, "source_eligibility": checks,
              "source_sha256": source_hashes, "retained_bundle_sha256": sha(args.retained / "fitting_examples.npz"),
              "new_fitting_sources": manifest, "failure_rows_end_time_s": args.failure_end_time,
              "new_teacher_regularization_counts": counts,
              "split_counts": {name: int((data["splits"] == name).sum()) for name in np.unique(data["splits"])},
              "whole_engineering_evaluation_sources": evaluation_manifest,
              "whole_engineering_evaluation_status": evaluation_status,
              "whole_engineering_evaluation_used_for_gradients_or_checkpoint_selection": False,
              "engineering_evaluation_design_access": "Root may observe aggregate development outcomes while fitting; worker opens measured rows after checkpoint choices.",
              "formal_sources_opened": [], "models": reports, "engineering_evaluation": evaluation_reports,
              "training_seconds": time.perf_counter() - started,
              "limitations": ["The physical failure approach is fitted development data, not a generalization result.",
                              "Intent regularization changes only near-equal trajectory preferences; shared guard is fixed.",
                              "Engineering replay is a one-step counterfactual, not a demonstrated closed-loop improvement.",
                              "Only fresh physical engineering/promotion gates establish candidate fitness."]}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"models": reports, "engineering_evaluation": evaluation_reports,
                      "training_seconds": report["training_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
