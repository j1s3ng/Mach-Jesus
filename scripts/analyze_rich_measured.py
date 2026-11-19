"""Matched-state intention/guard comparison; no closed-loop outcome estimate."""
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
from race_lidar.race_control import predictive_guard
from race_lidar.rich_driver import RichDriver
from scripts.rich_measured_data import load_run, reference_counterfactual


def guarded(action, state, cap, visible_distance):
    result = predictive_guard(action, {"hazard_tracks": state["hazard_tracks"]}, body_speed=state["body_speed_mps"],
        current_lane=state["last_target_offset_m"], actual_lateral=state["cross_track_m"], feasible_speed=cap,
        dt=.02, visible_distance=visible_distance)
    result["target_speed_mps"] *= float(np.clip(1. - .06 * abs(state["cross_track_m"] - result["target_offset_m"]), .55, 1.))
    if abs(state["heading_error_rad"]) > .4:
        result["target_speed_mps"] = min(result["target_speed_mps"], 12.)
    return result


def summarize(rows):
    def mean(key):
        return float(np.mean([row[key] for row in rows])) if rows else None
    return {"samples": len(rows), "mean_model_raw_speed_mps": mean("model_speed"),
        "mean_reference_label_speed_mps": mean("reference_speed"),
        "mean_model_one_step_guarded_speed_mps": mean("model_guarded_speed"),
        "mean_reference_one_step_guarded_speed_mps": mean("reference_guarded_speed"),
        "mean_guarded_speed_shortfall_mps": mean("guarded_speed_shortfall"),
        "guarded_speed_shortfall_over_2mps_rate": mean("slow_after_same_guard"),
        "model_stop_below2_while_reference_guard_above10_rate": mean("needless_stop_proxy"),
        "desired_lane_disagreement_over1_6m_rate": mean("lane_disagreement"),
        "model_requests_change_while_reference_holds_rate": mean("change_while_expert_holds"),
        "reference_partial_lane_label_rate": mean("partial_reference_label")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering", type=Path, default=ROOT / "runs/improvement-12h/engineering")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--visible-distance", type=float, default=45.)
    args = parser.parse_args()
    reports, manifests = [], []
    for run_id in ("rich001-field4-r0", "rich001-field20-r0"):
        samples, manifest = load_run(args.engineering / run_id, stride=args.stride)
        manifests.extend(manifest)
        reference = [reference_counterfactual(sample["state"], sample["label_feasible_speed_mps"]) for sample in samples]
        guarded_reference = [guarded(action, sample["state"], sample["label_feasible_speed_mps"], args.visible_distance)
                             for action, sample in zip(reference, samples)]
        for path in args.checkpoints:
            model = RichDriver.from_npz(path)
            rows = []
            examples = []
            for sample, expert, expert_guarded in zip(samples, reference, guarded_reference):
                state = sample["state"]
                action = model.predict(state)
                common_guard = guarded(action, state, sample["label_feasible_speed_mps"], args.visible_distance)
                hazards = state["hazard_tracks"]
                forward = any(0. < row["relative_distance_m"] < 95.
                              and abs(row["lateral_m"] - state["cross_track_m"]) < 2.5 for row in hazards)
                uncertain = any(float(row.get("speed_std_mps", row.get("speed_uncertainty_mps", 0.))) >= 10.
                                or float(row.get("position_std_m", row.get("position_uncertainty_m", 0.))) >= 1.
                                for row in hazards)
                partial = min(abs(expert["target_offset_m"] - lane) for lane in (-3.2, 0., 3.2)) > .35
                shortfall = max(0., expert_guarded["target_speed_mps"] - common_guard["target_speed_mps"])
                value = {"car_id": sample["car_id"], "time_s": sample["simulation_time_s"] - 1.,
                    "model_speed": action["target_speed_mps"], "reference_speed": expert["target_speed_mps"],
                    "model_guarded_speed": common_guard["target_speed_mps"],
                    "reference_guarded_speed": expert_guarded["target_speed_mps"],
                    "guarded_speed_shortfall": shortfall, "slow_after_same_guard": shortfall > 2.,
                    "needless_stop_proxy": common_guard["target_speed_mps"] < 2. and expert_guarded["target_speed_mps"] > 10.,
                    "lane_disagreement": abs(action["target_offset_m"] - expert["target_offset_m"]) > 1.6,
                    "change_while_expert_holds": abs(expert["target_offset_m"] - state["last_target_offset_m"]) < .35
                        and abs(action["target_offset_m"] - state["last_target_offset_m"]) > 1.6,
                    "partial_reference_label": partial, "forward_hazard": forward, "high_uncertainty": uncertain}
                rows.append(value)
                if value["needless_stop_proxy"] and len(examples) < 12:
                    examples.append({**value, "measured_state": state, "reference_action": expert,
                                     "model_action": action, "model_guard_flags": common_guard["guards"]})
            report = {"run_id": run_id, "checkpoint": str(path), "sha256": model.checkpoint_sha256,
                "aggregate": summarize(rows), "no_forward_hazard": summarize([row for row in rows if not row["forward_hazard"]]),
                "forward_hazard": summarize([row for row in rows if row["forward_hazard"]]),
                "high_uncertainty": summarize([row for row in rows if row["high_uncertainty"]]),
                "by_car": {car: summarize([row for row in rows if row["car_id"] == car]) for car in sorted({row["car_id"] for row in rows})},
                "stop_difference_examples": examples}
            reports.append(report)
            print(json.dumps({"run": run_id, "checkpoint": str(path), "metrics": report["aggregate"]}, indent=2), flush=True)
    result = {"method": "matched_measured_state_policy_and_one_step_guard_counterfactual", "closed_loop_trial": False,
        "real_outcome_claim": False, "inputs": "authorized engineering policy_actions inputs+hazard_tracks only",
        "feasible_speed_cap_is_label_only": True, "visible_distance_m_for_guard_replay": args.visible_distance,
        "forbidden_sources_opened": [], "source_files": manifests,
        "teacher_source_sha256": hashlib.sha256((ROOT / "src/race_lidar/race_policy.py").read_bytes()).hexdigest(),
        "guard_source_sha256": hashlib.sha256((ROOT / "src/race_lidar/race_control.py").read_bytes()).hexdigest(),
        "limitations": ["Guard replay is one step on recorded ego states; applying a different policy would change future states.",
          "No lap time, overtake, contact or safety outcome is inferred from this counterfactual agreement.",
          "Continuous reference offsets are flagged when far from the student's three discrete lanes."], "comparisons": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
