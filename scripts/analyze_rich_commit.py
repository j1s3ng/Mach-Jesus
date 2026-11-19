"""Diagnose an eligible engineering approach using measured policy inputs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from race_lidar.rich_driver import RichDriver
from scripts.rich_measured_data import load_engineering_run, reference_counterfactual
from scripts.analyze_rich_measured import guarded
from scripts.training_sources import validate_training_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-time", type=float, default=30.)
    args = parser.parse_args()
    samples, sources = load_engineering_run(args.run)
    config = json.loads((args.run / "config.json").read_text())
    visible = config.get("visible_distance_m", config["rules"]["visible_distance_m"])
    traces = {}
    for car in sorted({row["car_id"] for row in samples}):
        path = args.run / "cars" / car / "control_trace.jsonl"
        validate_training_source(path)
        traces[car] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "role": "ego/control diagnostic only, not model input"})
    models = {}
    for path in args.checkpoints:
        validate_training_source(path)
        models[f"{path.parent.parent.name}/{path.parent.name}"] = RichDriver.from_npz(path)
    rows = []
    for sample in samples:
        if sample["simulation_time_s"] > args.end_time:
            continue
        state, time_s = sample["state"], sample["simulation_time_s"]
        hazards = state["hazard_tracks"]
        expert = reference_counterfactual(state, sample["label_feasible_speed_mps"])
        controls = [r for r in traces[sample["car_id"]] if time_s - 1e-7 <= r["simulation_time_s"] <= time_s + .10001]
        control = controls[-1] if controls else None
        comparisons = {}
        for name, model in models.items():
            action = model.predict(state)
            protected = guarded(action, state, sample["label_feasible_speed_mps"], visible)
            comparisons[name] = {"raw_action": action, "one_step_guard": protected}
        rows.append({"car_id": sample["car_id"], "simulation_time_s": time_s,
                     "state": state, "nearest_forward_measured_gap_m": min((h["relative_distance_m"] for h in hazards
                       if h["relative_distance_m"] >= 0.), default=None),
                     "recorded_control": control, "reference_row_local_counterfactual": expert,
                     "models": comparisons})
    summaries = {}
    for name in models:
        selected = [r for r in rows if r["nearest_forward_measured_gap_m"] is not None
                    and r["simulation_time_s"] < 21.]
        lanes = [r["models"][name]["raw_action"]["target_offset_m"] for r in selected]
        sign_changes = sum(a * b < 0. for a, b in zip(lanes, lanes[1:]))
        changes = sum(a != b for a, b in zip(lanes, lanes[1:]))
        first = next((r for r in selected if abs(r["models"][name]["raw_action"]["target_offset_m"]) > 1.6), None)
        summaries[name] = {"samples": len(selected), "raw_lane_changes": changes,
                           "opposite_side_reversals": sign_changes,
                           "first_noncentral_lane_time_s": first["simulation_time_s"] if first else None,
                           "first_noncentral_gap_m": first["nearest_forward_measured_gap_m"] if first else None,
                           "checkpoint_sha256": models[name].checkpoint_sha256}
    report = {"method": "measured_engineering_approach_reconstruction_and_one_step_counterfactual",
              "run": str(args.run.resolve()), "source_files": sources, "visible_distance_m": visible,
              "hidden_opponent_truth_opened": False, "formal_sources_opened": False,
              "closed_loop_counterfactual": False, "summary": summaries, "rows": rows,
              "limitations": ["Frozen models are replayed on the actual004 ego states, not their own closed-loop trajectories.",
                               "A side change can be reasonable under changed occupancy; counts alone do not prove error.",
                               "Only the recorded learner's requested/achieved trajectory establishes the physical failure."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
