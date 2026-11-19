#!/usr/bin/env python3
"""Serve a separately hash-pinned sensor-supported detector pipeline artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT/"src"))
from race_lidar.batched_remote import load_batched_predictor, serve
from race_lidar.point_support import config_sha256, validate_config, wrap_predictor
from race_lidar.pipeline_artifact import tensor_state_sha256
from scripts.watch_owned_gpu_services import DEADLINE, CUTOFF, lease_check, register


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_supported_predictor(pcdet_root, config, checkpoint, expected_sha256):
    import torch
    artifact_hash = sha(checkpoint)
    if artifact_hash != expected_sha256:
        raise ValueError("Pipeline artifact differs from expected checkpoint SHA256")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["sensor_support_pipeline"]
    postprocess = validate_config(json.loads(metadata["postprocess_json"]))
    if metadata["version"] != 1 or not metadata["weights_unchanged"]:
        raise ValueError("Unsupported sensor-support artifact")
    if sha(config) != metadata["source_config_sha256"] or config_sha256(postprocess) != metadata["postprocess_sha256"]:
        raise ValueError("Pipeline configuration changed")
    if sha(ROOT/"src/race_lidar/point_support.py") != metadata["source_postprocess_sha256"]:
        raise ValueError("Point-support implementation changed after artifact freeze")
    state_hash = tensor_state_sha256(payload.get("model_state", payload.get("state_dict")))
    if state_hash != metadata["model_state_sha256"]:
        raise ValueError("Inherited detector tensors differ from the declared state identity")
    del payload
    predictor = load_batched_predictor(pcdet_root, config, checkpoint)
    wrapped = wrap_predictor(predictor, postprocess, artifact_sha256=artifact_hash,
                             parent_checkpoint_sha256=metadata["parent_checkpoint_sha256"], state_sha256=state_hash)
    wrapped.metadata.update(postprocess_source_sha256=metadata["source_postprocess_sha256"],
                            pipeline_role=metadata["role"],
                            calibration_protocol_sha256=metadata["calibration_protocol_sha256"])
    return wrapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcdet-root", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--lease-registry", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--batch-wait-ms", type=float, default=5.)
    parser.add_argument("--max-pending", type=int, default=64)
    args = parser.parse_args()
    if args.port != 8772: parser.error("Candidate campaign service permits only8772")
    lease_check(args.lease_registry)
    register(args.lease_registry, os.getpid(), args.port, Path(__file__).resolve())
    predictor = load_supported_predictor(args.pcdet_root, args.config, args.checkpoint, args.expected_sha256)
    predictor.metadata["campaign_lease"] = {"deadline_utc": DEADLINE.isoformat(), "launch_cutoff_utc": CUTOFF.isoformat()}
    serve(predictor, args.port, args.max_batch, args.batch_wait_ms, args.max_pending)


if __name__ == "__main__":
    main()
