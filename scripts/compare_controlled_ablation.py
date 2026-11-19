#!/usr/bin/env python3
"""CPU-only descriptive matched-arm analysis on the frozen development export.

No fitting, qualification access, checkpoint selection or live service calls.
Incomplete and failed attempts remain explicit; no automatic winner is chosen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.controlled_ablation_paths import CODE, DATASET, PILOT, TRAINING

EXPECTED_PLAN_SHA = "d234ec2802d0b713829a3ef8c28645a0af4df85e44347b256bed438c99f515fe"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def frame_key(row):
    return row["scene_id"], row["frame_id"]


def matched_errors(predictions, truth, threshold=.35):
    """Score-ranked nearest unmatched GT, XY distance<=2m, keyed by GT index."""
    annotated = {frame_key(row): row["gt_boxes"] for row in truth}
    if len(annotated) != len(truth) or len({frame_key(row) for row in predictions}) != len(predictions):
        raise ValueError("Duplicate frame identity")
    if {frame_key(row) for row in predictions} != set(annotated):
        raise ValueError("Prediction/development frame identity mismatch")
    errors = {}
    fp = empty_fp = 0
    for prediction in predictions:
        key = frame_key(prediction)
        target = annotated[key]
        used = set()
        for box in sorted(prediction["boxes"], key=lambda item: -item["score"]):
            if not np.isfinite(box["score"]) or not np.all(np.isfinite(box["center"])):
                raise ValueError("Nonfinite prediction")
            if box["score"] < threshold:
                continue
            choices = [(float(np.linalg.norm(np.asarray(box["center"])[:2] - np.asarray(gt["center"])[:2])), i)
                       for i, gt in enumerate(target) if i not in used]
            distance, selected = min(choices, default=(float("inf"), -1))
            if distance <= 2.:
                used.add(selected)
                errors[(*key, selected)] = distance
            else:
                fp += 1
                empty_fp += int(not target)
    target_count = sum(len(boxes) for boxes in annotated.values())
    empty_count = sum(not boxes for boxes in annotated.values())
    far_targets = {(*frame_key(row), i) for row in truth for i, box in enumerate(row["gt_boxes"])
                   if 30. <= np.linalg.norm(np.asarray(box["center"])[:2]) <= 50.}
    tp = len(errors)
    return errors, {"score_threshold": threshold, "frames": len(truth), "targets": target_count,
        "true_positives": tp, "false_positives": fp, "false_negatives": target_count - tp,
        "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / target_count if target_count else 0.,
        "false_positives_per_frame": fp / len(truth), "empty_frames": empty_count,
        "empty_frame_false_positives": empty_fp,
        "false_positives_per_empty_frame": empty_fp / empty_count if empty_count else None,
        "matched_xy_center_mae_m": float(np.mean(list(errors.values()))) if errors else None,
        "far_30_50m": {"targets": len(far_targets), "matched": len(far_targets & errors.keys()),
                        "recall": len(far_targets & errors.keys()) / len(far_targets) if far_targets else None}}


def paired_errors(first, second):
    keys = sorted(first.keys() & second.keys())
    a = np.array([first[key] for key in keys])
    b = np.array([second[key] for key in keys])
    return {"common_gt_matches": len(keys), "first_xy_center_mae_m": float(a.mean()) if keys else None,
            "second_xy_center_mae_m": float(b.mean()) if keys else None,
            "paired_second_minus_first_mean_m": float((b - a).mean()) if keys else None,
            "scope": "Identical GT IDs matched by both at the stated threshold; descriptive only."}


def cli_values(argv):
    result = {}
    index = 2
    while index < len(argv):
        key = argv[index]
        value = argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith("--") else True
        result[key] = value
        index += 1 if value is True else 2
    return result


def analyze(plan_path, local_root=ROOT):
    plan_path = Path(plan_path)
    if sha(plan_path) != EXPECTED_PLAN_SHA:
        raise ValueError("The immutable predeclared v2 plan changed")
    plan = json.loads(plan_path.read_text())
    if plan["plan_id"] != "controlled-perception-ablation-002":
        raise ValueError("Expected the predeclared four-arm v2 plan")
    root = Path(local_root)
    dataset = root / DATASET
    for name in ("dataset.json", "train.jsonl", "val.jsonl", "training_sources.json"):
        if sha(dataset / name) != plan["pinned_input_sha256"][name]:
            raise ValueError(f"Frozen development/training identity changed:{name}")
    # Only this frozen development manifest is used. No reserved paths inferred.
    truth = read_rows(dataset / "val.jsonl")
    baseline_path = root / PILOT / "pretrained_predictions.jsonl"
    if sha(baseline_path) != plan["pinned_input_sha256"]["baseline_predictions"]:
        raise ValueError("Frozen baseline predictions changed")
    baseline_errors, baseline_metrics = matched_errors(read_rows(baseline_path), truth)
    expected_batches = json.loads((root / TRAINING / "controlled_ablation_sampling_seed7.json").read_text())
    expected_mining = read_rows(root / PILOT / "training_hard_negative_mining.jsonl")
    mining_identity = lambda row: tuple(row[key] for key in ("training_index", "scene_id", "frame_id", "points_sha256", "sampling_weight"))
    reference_mining = [mining_identity(row) for row in expected_mining]
    records = []
    error_maps = {"original_pretrained": baseline_errors}
    for arm in [a for a in plan["arms"] if a["required"]]:
        output = root / Path(arm["output"]).relative_to(CODE)
        record = {"arm_id": arm["arm_id"], "output": str(output), "evidence_issues": [], "attempt_status": "not_mirrored"}
        issues = record["evidence_issues"]
        status_path = output / "status.json"
        if not status_path.exists():
            records.append(record)
            continue
        status = json.loads(status_path.read_text())
        record.update(attempt_status=status.get("status"), mode=arm["mode"], head_lr=arm["head_lr"],
                      backbone_lr=arm["backbone_lr"], seed=arm["seed"], error=status.get("error"),
                      started_utc=status.get("started_utc"), finished_utc=status.get("finished_utc"),
                      timer_training_seconds_includes_adapted_eval=status.get("training_seconds"),
                      training_steps=status.get("training_steps"), gpu=status.get("gpu"),
                      candidate_checkpoint_sha256=status.get("candidate_checkpoint_sha256"))
        record["peak_torch_allocated_bytes"] = status.get("peak_torch_allocated_bytes")
        record["peak_torch_reserved_bytes"] = status.get("peak_torch_reserved_bytes")
        if status.get("started_utc") and status.get("finished_utc"):
            record["wall_seconds_from_status_timestamps"] = (
                datetime.fromisoformat(status["finished_utc"]) - datetime.fromisoformat(status["started_utc"])).total_seconds()
        identities = {"source_dataset_sha256": plan["pinned_input_sha256"]["dataset.json"],
            "source_checkpoint_sha256": plan["pinned_input_sha256"]["original_checkpoint"],
            "source_config_sha256": plan["pinned_input_sha256"]["config"],
            "trainer_sha256": plan["source_sha256"]["scripts/train_sim_detector.py"],
            "mode": arm["mode"], "seed": arm["seed"], "batch_size": 2,
            "learning_rate": arm["head_lr"], "freeze_batchnorm_running_statistics": True,
            "empty_batch_loss_weight": 3., "homogeneous_negative_batch_fraction": .5}
        for key, expected in identities.items():
            if status.get(key) != expected:
                issues.append(f"status_identity:{key}")
        wanted_backbone = cli_values(arm["argv"]).get("--backbone-lr")
        if status.get("backbone_learning_rate") != (float(wanted_backbone) if wanted_backbone else None):
            issues.append("status_identity:backbone_learning_rate")
        for name in ("status.json", "trainer_source.py", "effective_config.json", "losses.jsonl", "training_hard_negative_mining.jsonl", "trainable_parameters.json", "gradient_smoke.json", "adapted_predictions.jsonl", "adapted_metrics.json"):
            if (output / name).exists():
                record.setdefault("file_sha256", {})[name] = sha(output / name)
            elif status.get("status") == "completed":
                issues.append("missing_completed_artifact:" + name)
        if status.get("status") != "completed":
            records.append(record)
            continue
        if any(issue.startswith("missing_completed_artifact:") for issue in issues):
            record["evidence_verified"] = False
            records.append(record)
            continue
        checkpoint = output / "candidate_step_1200.pth"
        record["checkpoint_bytes_verified_locally"] = checkpoint.exists() and sha(checkpoint) == status.get("candidate_checkpoint_sha256")
        if not record["checkpoint_bytes_verified_locally"]:
            issues.append("missing_or_changed_local_checkpoint")
        if status.get("training_steps") != 1200:
            issues.append("gradient_budget_mismatch")
        if status.get("reload_verified_tensors") != 542:
            issues.append("checkpoint_reload_tensor_count_mismatch")
        if sha(output / "trainer_source.py") != identities["trainer_sha256"]:
            issues.append("captured_trainer_source_mismatch")
        if sha(output / "effective_config.json") != sha(root / PILOT / "effective_config.json"):
            issues.append("effective_config_differs_from_frozen_pilot")
        loss_rows = read_rows(output / "losses.jsonl")
        if len(loss_rows) != 1200 or [row["step"] for row in loss_rows] != list(range(1, 1201)):
            issues.append("loss_step_budget_mismatch")
        if not loss_rows:
            record["evidence_verified"] = False
            records.append(record)
            continue
        finite = all(np.isfinite(row[key]) for row in loss_rows for key in ("loss", "unweighted_loss", "gradient_norm"))
        if not finite:
            issues.append("nonfinite_loss_or_gradient")
        wrong_weights = sum(row["loss_multiplier"] != (3. if row["empty_batch"] else 1.) for row in loss_rows)
        if wrong_weights:
            issues.append("empty_loss_multiplier_mismatch")
        sample_mismatch = sum(a["training_frame_ids"] != b["training_frame_ids"] or a["empty_batch"] != b["empty_batch"]
                              for a, b in zip(loss_rows, expected_batches)) + abs(len(loss_rows) - len(expected_batches))
        mined = read_rows(output / "training_hard_negative_mining.jsonl")
        mining_status = status.get("hard_negative_mining", {})
        mining_expected = {"checkpoint_sha256": plan["pinned_input_sha256"]["miner_checkpoint"],
            "training_frames": 10233, "all_training_frames": True,
            "score_threshold": .2, "original_weights_restored_before_fit": True, "validation_mined": False}
        for key, expected in mining_expected.items():
            if mining_status.get(key) != expected:
                issues.append("mining_recipe:" + key)
        mining_mismatch = sum(mining_identity(a) != mining_identity(b) for a, b in zip(mined, expected_mining)) + abs(len(mined) - len(reference_mining))
        record.update(sampling_mismatch_batches_vs_predeclared=sample_mismatch,
            mining_weight_or_identity_mismatch_rows=mining_mismatch,
            sampling_confound=bool(sample_mismatch or mining_mismatch),
            positive_batches=sum(not row["empty_batch"] for row in loss_rows),
            empty_batches=sum(row["empty_batch"] for row in loss_rows),
            sampled_frame_draws=sum(len(row["training_frame_ids"]) for row in loss_rows),
            unique_gradient_frame_ids=len({fid for row in loss_rows for fid in row["training_frame_ids"]}),
            loss_statistics={"finite": finite, "first": loss_rows[0]["loss"], "last": loss_rows[-1]["loss"],
                             "mean_first100": float(np.mean([row["loss"] for row in loss_rows[:100]])),
                             "mean_last100": float(np.mean([row["loss"] for row in loss_rows[-100:]]))})
        trainable = json.loads((output / "trainable_parameters.json").read_text())
        gradient = json.loads((output / "gradient_smoke.json").read_text())
        record.update(optimizer_groups=trainable["optimizer_groups"], trainable_parameter_tensors=len(trainable["names"]), first_step_gradient_audit=gradient)
        if not gradient.get("finite_gradients") or not gradient.get("optimizer_changed_parameter"):
            issues.append("gradient_audit_missing_nonzero_update")
        if arm["mode"] == "head" and any(not name.startswith("dense_head.") for name in trainable["names"]):
            issues.append("head_mode_unexpected_trainable_backbone")
        if arm["mode"] == "full" and not gradient.get("backbone", {}).get("optimizer_changed_parameter"):
            issues.append("full_mode_missing_backbone_update")
        errors, metrics = matched_errors(read_rows(output / "adapted_predictions.jsonl"), truth)
        record["development_score_0p35"] = metrics
        saved = json.loads((output / "adapted_metrics.json").read_text())
        fixed = saved["operating_points"]["0.35"]
        if any(metrics[key] != fixed[key] for key in ("true_positives", "false_positives", "false_negatives")):
            issues.append("independent_matching_disagrees_with_trainer")
        record["all_trainer_operating_points"] = saved["operating_points"]
        record["far_recall_at_0p25"] = saved["visible_target_recall_by_range_at_score_0p25"]["30_50m"]
        record["common_gt_vs_original_at_0p35"] = paired_errors(baseline_errors, errors)
        record["evidence_verified"] = not issues
        if not issues:
            error_maps[arm["arm_id"]] = errors
        records.append(record)
    common = set.intersection(*(set(value) for value in error_maps.values()))
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(plan_path), "analysis_source_sha256": sha(__file__),
        "scope": "Four predeclared seed7 attempts; descriptive whole-run development analysis, no independent qualification, no fitting/selection, no live requests.",
        "statistical_limit": "One development physical run with correlated frames and one matched seed; no independent optimizer-superiority significance claim.",
        "original_pretrained_at_0p35": baseline_metrics, "attempts": records,
        "all_four_completed_and_verified": len(records) == 4 and all(row.get("evidence_verified") for row in records),
        "all_sampling_schedules_matched": len(records) == 4 and all(row.get("sampling_confound") is False for row in records),
        "common_gt_across_verified_available_arms": {"models": list(error_maps), "matches": len(common),
            "xy_center_mae_m": {name: float(np.mean([errors[key] for key in common])) if common else None for name, errors in error_maps.items()}},
        "selection_performed": False, "reserved_inputs_opened": False}
    report["matched_sampling_comparison_valid"] = report["all_four_completed_and_verified"] and report["all_sampling_schedules_matched"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / TRAINING / "controlled_ablation_plan_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior reports; choose a fresh output path")
    report = analyze(args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_four_completed_and_verified": report["all_four_completed_and_verified"],
                      "attempt_statuses": {row["arm_id"]: row["attempt_status"] for row in report["attempts"]},
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
