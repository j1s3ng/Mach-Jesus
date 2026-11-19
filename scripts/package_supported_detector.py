#!/usr/bin/env python3
"""Create a new pipeline artifact; inherited detector tensors remain identical."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT/"src"))
from race_lidar.point_support import config_sha256, validate_config
from race_lidar.pipeline_artifact import tensor_state_sha256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(parent, selection_path, output):
    import torch
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Use a new artifact and sidecar path")
    selection = json.loads(selection_path.read_text())
    if selection["fresh_inputs_opened"] or sha(parent) != selection["parent_checkpoint_sha256"]:
        raise ValueError("Select on training only and preserve the declared parent identity")
    config = validate_config(selection["postprocess"])
    if config_sha256(config) != selection["postprocess_sha256"]:
        raise ValueError("Selected support configuration changed")
    calibration = json.loads((selection_path.parent/"status.json").read_text())
    if calibration["status"] != "completed" or calibration["parent_checkpoint_sha256"] != sha(parent) or calibration["protocol_sha256"] != selection["protocol_sha256"]:
        raise ValueError("Calibration source lineage differs from selected configuration")
    if sha(selection_path.parent/"calibration_report.json") != selection["calibration_report_sha256"]:
        raise ValueError("Calibration report changed after selection")
    if calibration["source_postprocess_sha256"] != sha(ROOT/"src/race_lidar/point_support.py"):
        raise ValueError("Point-support implementation changed after calibration")
    checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
    if "sensor_support_pipeline" in checkpoint:
        raise ValueError("Parent must be the original unfiltered600-step candidate")
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    state_hash = tensor_state_sha256(state)
    if checkpoint.get("dataset_manifest_sha256", {}).get("train") != selection["training_manifest_sha256"]:
        raise ValueError("Selected calibration and weights training lineage differ")
    metadata = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "parent_checkpoint_sha256": sha(parent), "model_state_sha256": state_hash,
                "weights_unchanged": True, "postprocess_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
                "postprocess_sha256": config_sha256(config), "calibration_selection_sha256": sha(selection_path),
                "calibration_protocol_sha256": selection["protocol_sha256"],
                "calibration_report_sha256": selection["calibration_report_sha256"],
                "training_manifest_sha256": selection["training_manifest_sha256"],
                "source_config_sha256": calibration["model"]["config_sha256"],
                "source_postprocess_sha256": sha(ROOT/"src/race_lidar/point_support.py"),
                "role": "new_pipeline_candidate_not_retroactive_pass_for_rejected_parent",
                "fresh_qualification_inputs_opened_before_freeze": False}
    checkpoint["sensor_support_pipeline"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    reloaded = torch.load(output, map_location="cpu", weights_only=True)
    if tensor_state_sha256(reloaded.get("model_state", reloaded.get("state_dict"))) != state_hash:
        raise RuntimeError("Pipeline packaging changed detector tensors")
    report = {"checkpoint": str(output.resolve()), "checkpoint_sha256": sha(output),
              "pipeline": metadata, "postprocess": config,
              "parent_checkpoint_sha256": sha(parent), "tensor_state_unchanged_verified": True,
              "training_executed": False, "formal_promotion": False, "physical_trial_eligible": False,
              "port8772_started": False}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.parent_checkpoint, args.selection, args.output), indent=2))


if __name__ == "__main__":
    main()
