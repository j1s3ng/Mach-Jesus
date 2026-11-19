#!/usr/bin/env python3
"""Classify one historical mid-run service interruption without changing history."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_improvement_attempt import evaluate_attempt
from scripts.improvement_ledger import environment_rules_hash, OFFICIAL_DETECTOR_SHA256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Keep previous evidence; choose a new output path")
    campaign = ROOT / "runs/improvement-12h"
    run = campaign / "engineering/rich005-s7-field20"
    config = json.loads((run / "config.json").read_text())
    heat = json.loads((run / "heat_results.json").read_text())
    audit = evaluate_attempt(run, snapshot_path=config["source_root"],
        expected_snapshot_sha256="d610f5bc7478223b3b3c2e620057357780729ea40e3d2532cd834638e418953e",
        expected_rules_sha256=environment_rules_hash(config["rules"]),
        expected_policy="neural_rich005_seed7",
        expected_model_sha256="89bafe77e99d160aa97aab5510f587112a6c461e2cd7383a381a4af429c94989",
        expected_detector_checkpoint_sha256=OFFICIAL_DETECTOR_SHA256,
        expected_control_parameters={"visible_distance_m": 70.})
    records, hashes = [], {}
    for entry in config["entrants"]:
        folder = run / "cars" / entry["car_id"]
        result = json.loads((folder / "result.json").read_text())
        timelines = {}
        for name in ("predictions.jsonl", "policy_actions.jsonl", "control_trace.jsonl"):
            data = rows(folder / name)
            timelines[name] = {"rows": len(data), "first_simulation_time_s": data[0]["simulation_time_s"],
                "last_simulation_time_s": data[-1]["simulation_time_s"],
                "last_timestamp_ns": data[-1]["timestamp_ns"]}
        records.append({"car_id": entry["car_id"], **{key: result[key] for key in
            ("status", "reason", "failure", "simulation_time_s", "sensor_frames", "policy_decisions", "vision_url",
             "progress_m", "track_violation_steps", "position_resets_after_spawn", "model_sha256")},
            "authoritative_overlap_samples": heat["collisions_by_car"][entry["car_id"]], "timelines": timelines})
        for name in ("result.json", "controller_summary.json", "predictions.jsonl", "policy_actions.jsonl", "control_trace.jsonl"):
            hashes[str((folder / name).relative_to(ROOT))] = sha(folder / name)
    interrupted = audit["audit_verified"] and len(records) == 20 and all(
        r["failure"] == "RemoteDisconnected: Remote end closed connection without response"
        and r["status"] == "failed" and r["simulation_time_s"] == 108.
        and r["sensor_frames"] == r["policy_decisions"] == 535
        and r["timelines"]["predictions.jsonl"]["last_simulation_time_s"] == 107.8
        and r["timelines"]["policy_actions.jsonl"]["last_simulation_time_s"] == 107.8
        and r["authoritative_overlap_samples"] == r["track_violation_steps"] == r["position_resets_after_spawn"] == 0
        for r in records)
    events = rows(campaign / "ledger.jsonl")
    timing = [{"event": r["event_type"], "utc": r["utc"], "attempt_id": r["payload"]["attempt_id"]}
        for r in events if r["event_type"] in ("attempt_started", "attempt_finished")
        and r["payload"].get("attempt_id") in ("rich005-s7-field20", "search90-field4")]
    old_job = next(j for j in json.loads((campaign / "queue.json").read_text()) if j["job_id"] == "rich005-s7-field20")
    prerequisites = {}
    for requirement in old_job["requires"]:
        name = requirement["job_id"]
        outcome = json.loads((campaign / "engineering" / name / "ledger_outcome.json").read_text())
        prerequisites[name] = {key: outcome.get(key) for key in ("audit_verified", "hazard_pass", "all_clean_finishes")}
    for name in ("config.json", "heat_results.json", "ledger_outcome.json"):
        hashes[str((run / name).relative_to(ROOT))] = sha(run / name)
    log = campaign / "logs/rich005-s7-field20.log"
    hashes[str(log.relative_to(ROOT))] = sha(log)
    hashes[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "INFRASTRUCTURE_INTERRUPTED_AFTER_INITIALIZATION" if interrupted else "INCONCLUSIVE",
        "classification_verified": interrupted, "original_attempt": "rich005-s7-field20",
        "not_uninitialized": "All20 actors generated535 measured sensor/prediction/action rows each before the same remote-disconnect failure.",
        "not_full_laps": "All original failed/DNF results remain unchanged; partial clean trajectories establish neither completion nor driver superiority.",
        "relationship_to_outage": "The two detector endpoints8770/8771 disconnected at the same simulator clock, then the immediately following job failed initialization. This is direct evidence of a shared service/transport interruption; the exact process-exit cause and precise failure UTC are not recoverable from simulation-only rows.",
        "logical_onset": "The service incident began during this run, before the60 previously catalogued wholly uninitialized attempts. Preserve the earlier immutable report and this addendum rather than rewriting it.",
        "recovery_recommendation": "One explicitly linked fresh20-car same-treatment retry can finish the untested completion objective after service preflight. Keep the old attempt/DNFs and all535-frame partial data; do not reclassify it as a successful run or delete it from denominators. Existing four hazards and clean4/8 prerequisite results are verified and need not be invented or rerun as new evidence.",
        "same_treatment_job": old_job, "retry_caveat": "Old allocation was12official-CV PRO6000 plus8official-CV3090. Preserve args including--secondary-cars8 for an exact same-treatment retry; schedule after controlled3090fits.",
        "prerequisites": prerequisites, "ledger_timeline": timing,
        "first_last_successful_sensor_clock_s": [1., 107.8], "failure_simulation_time_s": 108.,
        "successful_sensor_frames_all_cars": sum(r["sensor_frames"] for r in records),
        "zero_authoritative_incidents_before_interruption": all(r["authoritative_overlap_samples"] == r["track_violation_steps"] == r["position_resets_after_spawn"] == 0 for r in records),
        "valid_laps": sum(row["valid_finish"] for row in audit["entrants"]),
        "fresh_audit": audit, "records": records, "file_sha256": hashes,
        "history_mutated": False, "retry_enqueued": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("classification", "classification_verified", "successful_sensor_frames_all_cars", "valid_laps")}))


if __name__ == "__main__":
    main()
