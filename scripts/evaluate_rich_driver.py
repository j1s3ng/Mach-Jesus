"""Evaluate continuous learned actions against a frozen synthetic cost teacher.

These are teacher-model diagnostics, not measured physical collision rates.
Original training examples and held-out observations are read without changes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from race_lidar.rich_driver import LANE_OFFSETS_M, RichDriver, rich_feature_vector


def summarize(rows):
    if not rows:
        return {"samples": 0, "mean_action_cost_regret": None, "median_action_cost_regret": None,
                "p90_action_cost_regret": None, "regret_over_50_rate": None, "regret_over_500_rate": None,
                "teacher_proxy_violation_rate": None, "teacher_has_noncolliding_candidate_samples": 0,
                "avoidable_teacher_proxy_violation_rate": None}
    regret = np.asarray([row["action_cost_regret"] for row in rows])
    possible = [row for row in rows if row["teacher_has_noncolliding_candidate"]]
    return {"samples": len(rows), "mean_action_cost_regret": float(regret.mean()),
            "median_action_cost_regret": float(np.median(regret)), "p90_action_cost_regret": float(np.percentile(regret, 90)),
            "regret_over_50_rate": float((regret > 50.).mean()), "regret_over_500_rate": float((regret > 500.).mean()),
            "teacher_proxy_violation_rate": sum(row["predicted_teacher_collision"] for row in rows) / len(rows),
            "teacher_has_noncolliding_candidate_samples": len(possible),
            "avoidable_teacher_proxy_violation_rate": sum(row["predicted_teacher_collision"] for row in possible) / len(possible) if possible else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "family_test", "noise_test"])
    parser.add_argument("--speed-cap", type=float, help="Optional shared-protocol speed cap before teacher scoring")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("frozen_cost_teacher", args.teacher_script)
    teacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    with np.load(args.dataset / "training_examples.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    states = [json.loads(line) for line in (args.dataset / "training_observations.jsonl").read_text().splitlines()]
    selected = np.flatnonzero(np.isin(data["splits"], args.splits))
    results = []
    for path in args.checkpoints:
        model = RichDriver.from_npz(path)
        mode = model.metadata.get("lane_summary_mode", "nearest")
        # Comparing checkpoints with different summary modes requires encoding
        # the same observations through each checkpoint's own feature contract.
        encoded = np.stack([rich_feature_vector(states[index], mode=mode) for index in selected])
        predicted = model.forward_features(encoded)
        rows = []
        for position, index in enumerate(selected):
            state = states[index]
            lane = int(np.argmax(predicted[position, 3:]))
            raw_speed = float(np.clip(predicted[position, lane] * 60., 0., 60.))
            speed = min(raw_speed, args.speed_cap) if args.speed_cap is not None else raw_speed
            cost, collided = teacher.trajectory_costs(state, [LANE_OFFSETS_M[lane]], [speed],
                horizon_s=model.metadata["teacher_horizon_s"], max_speed=model.metadata["teacher_max_speed_mps"])
            rows.append({"index": int(index), "family": str(data["families"][index]), "split": str(data["splits"][index]),
                "predicted_lane_m": LANE_OFFSETS_M[lane], "predicted_speed_mps": speed,
                "raw_model_speed_mps": raw_speed, "body_speed_mps": state["body_speed_mps"],
                "teacher_best_lane_m": LANE_OFFSETS_M[int(data["labels"][index])],
                "teacher_best_cost": float(data["costs"][index].min()), "predicted_action_cost": float(cost[0]),
                "action_cost_regret": max(0., float(cost[0] - data["costs"][index].min())),
                "predicted_teacher_collision": bool(collided[0]),
                "teacher_has_noncolliding_candidate": not bool(data["collisions"][index].all())})
        worst = sorted(rows, key=lambda row: row["action_cost_regret"], reverse=True)[:40]
        result = {"checkpoint": str(path), "sha256": model.checkpoint_sha256, "lane_summary_mode": mode,
            "feature_encoding": "recomputed from unchanged observations using each checkpoint's metadata",
            "aggregate": summarize(rows),
            "by_split": {split: summarize([row for row in rows if row["split"] == split]) for split in args.splits},
            "by_family": {family: summarize([row for row in rows if row["family"] == family]) for family in sorted({row["family"] for row in rows})},
            "by_initial_speed": {"body_speed_up_to_22_mps": summarize([row for row in rows if row["body_speed_mps"] <= 22.]),
                                 "body_speed_over_22_mps": summarize([row for row in rows if row["body_speed_mps"] > 22.])},
            "worst_examples": [{**row, "observed_state": states[row["index"]]} for row in worst]}
        results.append(result)
        print(json.dumps({"checkpoint": str(path), "metrics": result["by_split"]}, indent=2), flush=True)
    output = {"method": "continuous_predicted_action_evaluation_against_frozen_trajectory_teacher",
        "physical_collision_metric": False, "hidden_scene_truth_used": False, "shared_speed_cap_mps": args.speed_cap,
        "dataset_sha256": hashlib.sha256((args.dataset / "training_examples.npz").read_bytes()).hexdigest(),
        "teacher_script_sha256": hashlib.sha256(args.teacher_script.read_bytes()).hexdigest(),
        "limitations": ["Teacher swept-envelope uncertainty and dynamics are approximations, not observed Webots outcomes.",
                        "Cost regret compares the continuous student action with the teacher's discrete lane/speed candidates; a student may interpolate to a lower cost.",
                        "These held-out examples are read unchanged; they are not used for model updates by this evaluator."], "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
