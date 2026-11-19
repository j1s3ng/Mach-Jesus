#!/usr/bin/env python3
"""Bind a fixed or independently TRAIN-calibrated score into full-fit weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT/"src"))
from race_lidar.pipeline_artifact import tensor_state_sha256


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument("--score-authority", type=Path)
    selection_group.add_argument("--calibration-selection", type=Path)
    parser.add_argument("--calibration-protocol", type=Path)
    parser.add_argument("--qualification-reservation", type=Path)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists(): parser.error("Use a fresh artifact path")
    status = json.loads(args.status.read_text())
    if status["candidate_checkpoint_sha256"] != sha(args.parent) or status["mode"] != "full" or not status["gradient_smoke"]["backbone"]["optimizer_changed_parameter"]:
        raise ValueError("Require the verified actual full-backbone parent checkpoint")
    payload = torch.load(args.parent, map_location="cpu", weights_only=True)
    if "race_detector_pipeline" in payload: raise ValueError("Parent already contains a scored pipeline")
    calibration_hash = selection_hash = reservation_hash = None
    if args.calibration_selection:
        if not args.calibration_protocol or not args.qualification_reservation: parser.error("Calibration requires the exact protocol and new qualification reservation")
        selected = json.loads(args.calibration_selection.read_text())
        protocol = json.loads(args.calibration_protocol.read_text())
        report_path = args.calibration_selection.parent/"calibration_report.json"
        report = json.loads(report_path.read_text())
        if (selected["fresh_inputs_opened"] or selected["gradients_executed"] or
            selected["weights_parent_checkpoint_sha256"] != sha(args.parent) or
            selected["training_manifest_sha256"] != payload["dataset_manifest_sha256"]["train"] or
            selected["protocol_sha256"] != sha(args.calibration_protocol) or
            selected["calibration_report_sha256"] != sha(report_path) or
            protocol["reserved_qualification_manifest_sha256"] != sha(args.qualification_reservation) or
            not report["new_functional_candidate"]):
            raise ValueError("Calibration selection/protocol/source lineage differs or has no functional change")
        threshold = selected["effective_car_score_threshold"]
        if threshold == .35 or threshold != report["selected"]["effective_raw_score_threshold"] or not report["selected"]["eligible"]:
            raise ValueError("Do not package an unchanged or unselected operating point")
        calibration_hash, selection_hash, reservation_hash = sha(report_path), sha(args.calibration_selection), sha(args.qualification_reservation)
        authority_hash = sha(args.calibration_protocol)
        method = "TRAIN-only predeclared grid; highest recall withFP/frame<=0.01 and recall>=official0.35;tieslowest rawthreshold"
    else:
        authority = json.loads(args.score_authority.read_text());threshold = authority["score_threshold"]
        authority_hash = sha(args.score_authority);method = "unchanged predeclared0.35 operating point; no threshold calibration"
    state_hash = tensor_state_sha256(payload["model_state"])
    pipeline = {"version": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "effective_car_score_threshold": threshold,
                "threshold_selection": method,
                "score_calibration_sha256": calibration_hash, "score_authority_sha256": authority_hash,
                "score_calibration_selection_sha256": selection_hash, "qualification_reservation_sha256": reservation_hash,
                "weights_parent_checkpoint_sha256": sha(args.parent), "weights_state_sha256": state_hash,
                "training_source_audit_sha256": sha(args.source_audit),
                "source_model_config_sha256": status["source_config_sha256"],
                "source_server_sha256": sha(ROOT/"scripts/serve_leased_candidate.py"),
                "source_postprocess_sha256": sha(ROOT/"src/race_lidar/scored_pipeline.py"),
                "training_manifest_sha256": payload["dataset_manifest_sha256"]["train"],
                "fresh_qualification_model_scores_observed": False,
                "qualification_inputs_exported_before_freeze": not bool(args.calibration_selection),
                "qualification_input_export_authority": "new reserved captures unopened" if args.calibration_selection else "parent explicitly requested export without model scoring"}
    payload["race_detector_pipeline"] = pipeline
    args.output.parent.mkdir(parents=True, exist_ok=True);torch.save(payload, args.output)
    check = torch.load(args.output, map_location="cpu", weights_only=True)
    if tensor_state_sha256(check["model_state"]) != state_hash: raise RuntimeError("Packaging changed learned tensors")
    report = {"artifact": str(args.output.resolve()), "artifact_sha256": sha(args.output), "pipeline": pipeline,
              "weights_identical_to_full_parent": True, "formal_promotion": False, "qualification_scored": False,
              "physical_trial_eligible": False, "port8772_started": False}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n");print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
