"""Fit Rich006 on eligible measured CV-shift failures and fragment/dropout noise.

No bound CEM, guard, source eligibility, or portable runtime module is edited.
The unchanged005 fitter is archived/reused; stage weights and final metadata
exports are both preserved. A separate entire combined-detector engineering run
is opened only after checkpoint selection and never supplies fitting rows.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")
import numpy as np
from race_lidar.hotlap_driver import RacingReference
from race_lidar.rich_driver import RichDriver, rich_feature_vector, save_rich_model
from scripts.rich_measured_data import load_engineering_run
from scripts.rich006_teacher import BASE, FROZEN005, mirror, perturb_measured, summarize, teacher
from scripts.training_sources import validate_training_source
from scripts.train_rich_dagger import evaluate_actual


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", type=Path, default=ROOT / "runs/improvement-12h/rich-driver-005")
    parser.add_argument("--failure-run", type=Path, default=ROOT / "runs/improvement-12h/engineering/cv425-nn00517-left")
    parser.add_argument("--positive-run", type=Path, default=ROOT / "runs/improvement-12h/engineering/rich005-s17-left")
    parser.add_argument("--evaluation-run", type=Path, default=ROOT / "runs/improvement-12h/engineering/cv425-nn00517-right")
    parser.add_argument("--reference", type=Path, default=ROOT / ".local/f1-reference/reference.npz")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--measured-augmentations", type=int, default=12000)
    parser.add_argument("--synthetic-per-family", type=int, default=4000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh006 output directory")
    if args.evaluation_run.resolve() in {args.failure_run.resolve(), args.positive_run.resolve()}:
        parser.error("Entire engineering evaluation must be excluded from fitting")
    checks = [validate_training_source(path) for path in (args.initial, args.failure_run, args.positive_run, args.reference)]
    args.output.mkdir(parents=True)
    source_names = ("scripts/train_rich_robust.py", "scripts/rich006_teacher.py")
    for name in source_names:
        target = args.output / "source" / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    frozen_fitter_path = ROOT / "runs/improvement-12h/rich-driver-005/source/scripts/train_rich_commit.py"
    for name, source in (("frozen005_teacher.py", FROZEN005), ("frozen005_fitter.py", frozen_fitter_path)):
        shutil.copyfile(source, args.output / "source" / name)
    with np.load(args.initial / "fitting_examples.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    fields = {key: [] for key in data}; observations, manifests, base_states = [], [], []
    counts = {"actual_failure": 0, "actual_positive": 0, "measured_augmentations": 0, "synthetic": 0, "partial_dropout_labels": 0}
    def add(state, kind, split, cap=60.):
        label = teacher(state)
        fields["features"].append(rich_feature_vector(state, mode="stopping_critical"))
        fields["speeds"].append(np.minimum(cap, label["lane_target_speeds_mps"]) / 60.)
        fields["labels"].append(label["lane_index"]); fields["costs"].append(label["lane_costs"])
        fields["collisions"].append(label["lane_collision_flags"]); fields["splits"].append(split)
        fields["kind"].append(kind); fields["lane_mask"].append(1.); fields["speed_mask"].append([1., 1., 1.])
        counts["partial_dropout_labels"] += int(label["partial_pass_missing_return_regularization"])
        observations.append({"state": state, "kind": kind, "split": split})
    for name, run, end in (("actual_failure", args.failure_run, 16.), ("actual_positive", args.positive_run, 24.)):
        if not (run / "heat_results.json").exists():
            raise ValueError("Measured fitting run must be closed")
        samples, manifest = load_engineering_run(run); manifests.extend(manifest)
        for sample in samples:
            if sample["simulation_time_s"] > end:
                continue
            state = sample["state"]; add(state, name, "failure_train", sample["label_feasible_speed_mps"])
            counts[name] += 1
            if state["hazard_tracks"] and sample["simulation_time_s"] > 4.5:
                base_states.append(state)
    rng = np.random.default_rng(20261127)
    for index in range(args.measured_augmentations):
        state = base_states[int(rng.integers(len(base_states)))]; state = mirror(state) if index % 2 else state
        add(perturb_measured(state, rng), "measured_fragment_augmentation", "commit_train")
        counts["measured_augmentations"] += 1
    reference = RacingReference.from_npz(args.reference)
    for family in ("low_support_pack", "partial_missing_return", "completed_clear_release"):
        for index in range(args.synthetic_per_family):
            if family == "low_support_pack":
                state = BASE.generate_commit_scene(reference, "commit_blocked_side", rng)
                state = perturb_measured(state, rng)
            else:
                state = BASE.generate_scene(reference, "clear", rng, expanded=True)
                side = float(rng.choice([-1., 1.])); state["last_target_offset_m"] = side * float(rng.uniform(.08, 3.2))
                state["cross_track_m"] = side * float(rng.uniform(.01, 2.7) if family == "partial_missing_return" else rng.uniform(2.85, 3.4))
                state["hazard_tracks"] = []; summarize(state)
            # Independently generated synthetic validation; measured-derived
            # perturbations never cross into validation for the same base row.
            add(state, family, "commit_validation" if index % 10 == 0 else "commit_train")
            counts["synthetic"] += 1
    for key in data:
        data[key] = np.concatenate((data[key], np.asarray(fields[key], dtype=data[key].dtype if key not in ("kind", "splits") else None)))
    np.savez(args.output / "fitting_examples.npz", **data)
    with (args.output / "new_observations.jsonl").open("w") as stream:
        for row in observations: stream.write(json.dumps(row, allow_nan=False) + "\n")
    spec = importlib.util.spec_from_file_location("rich006_frozen005_fitter", frozen_fitter_path)
    fitter = importlib.util.module_from_spec(spec); spec.loader.exec_module(fitter)
    import torch
    torch.set_num_threads(args.cpu_threads); torch.set_num_interop_threads(1)
    stage = args.output / "fit-stage"; stage.mkdir(); shutil.copyfile(args.output / "fitting_examples.npz", stage / "fitting_examples.npz")
    fit_args = argparse.Namespace(**vars(args)); fit_args.output = stage
    started = time.perf_counter(); reports = []
    for seed in args.seeds:
        report = fitter.fit(seed, fit_args, data)
        model = RichDriver.from_npz(report["checkpoint"])
        metadata = {**model.metadata, "method": "supervised_measured_fragment_and_partial_dropout_robustness_adaptation",
            "coverage_version": "rich006", "reinforcement_learning": False, "teacher_script_sha256": sha(ROOT / "scripts/rich006_teacher.py"),
            "adaptation_script_sha256": sha(__file__), "fitter_script_sha256": sha(frozen_fitter_path),
            "retained_source_sha256": sha(args.initial / "fitting_examples.npz"),
            "partial_missing_return_does_not_prove_clear_space": True, "runtime_contract_changed": False,
            "new_fitting_runs": [str(args.failure_run.resolve()), str(args.positive_run.resolve())],
            "entire_engineering_evaluation_run_excluded_from_fitting_and_checkpoint_selection": str(args.evaluation_run.resolve())}
        folder = args.output / f"seed-{seed}"; folder.mkdir()
        exported = save_rich_model(folder / "model.npz", model.weights, model.biases, model.mean, model.scale, metadata)
        portable = RichDriver.from_npz(exported)
        parity = float(np.max(np.abs(portable.forward_features(data["features"][:512]) - model.forward_features(data["features"][:512]))))
        if parity != 0.:
            raise RuntimeError("Metadata export changed learned weights")
        report.update(checkpoint=str(exported), checkpoint_sha256=sha(exported), stage_to_final_weight_parity=parity)
        (folder / "training_report.json").write_text(json.dumps(report, indent=2) + "\n"); reports.append(report)
    evaluations = []; evaluation_sources = []
    if (args.evaluation_run / "heat_results.json").is_file():
        held, evaluation_sources = load_engineering_run(args.evaluation_run)
        for report in reports:
            evaluations.append({"seed": report["seed"], **evaluate_actual(RichDriver.from_npz(report["checkpoint"]), held, 70.)})
    result = {"method": "supervised_measured_observation_shift_robustness", "reinforcement_learning": False,
        "runtime_unchanged": True, "bound_cem_modules_edited": False, "source_eligibility": checks,
        "training_seconds": time.perf_counter() - started, "source_counts": counts,
        "fitting_source_files": manifests, "evaluation_source_files": evaluation_sources,
        "evaluation_run_used_for_fitting_or_checkpoint_selection": False, "formal_reserved_training_sources": [],
        "models": reports, "whole_engineering_run_counterfactual": evaluations,
        "limitations": ["New actual failure and mirrored/noisy views are fitting data, not generalization evidence.",
            "Partial dropout retention is a limited observation-contract mitigation, not a complete memory/occlusion solution.",
            "The48-feature summaries can conceal noncritical actors; runtime remains fixed and physical gating is required.",
            "Counterfactual teacher/replay results cannot establish combined detector+driver racing outcomes."]}
    (args.output / "training_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"models": reports, "whole_engineering_evaluation": evaluations, "seconds": result["training_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
