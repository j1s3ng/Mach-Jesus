#!/usr/bin/env python3
"""Hash-pinned detector candidate with a finite owned-process campaign lease."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT/"src"))
from race_lidar.batched_remote import load_batched_predictor, serve
from race_lidar.pipeline_artifact import tensor_state_sha256
from race_lidar.scored_pipeline import wrap_scored_predictor
from scripts.watch_owned_gpu_services import DEADLINE, CUTOFF, lease_check, register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcdet-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--lease-registry", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--max-batch", type=int, default=16)
    args = parser.parse_args()
    if args.port != 8772: parser.error("The candidate campaign lease permits only port8772")
    lease_check(args.lease_registry)
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.expected_sha256:
        raise ValueError("Candidate checkpoint differs from expected artifact SHA256")
    import torch
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    pipeline = payload["race_detector_pipeline"]
    if pipeline["weights_state_sha256"] != tensor_state_sha256(payload["model_state"]):
        raise ValueError("Pipeline learned tensor identity differs")
    for file, key in ((Path(__file__).resolve(), "source_server_sha256"), (ROOT/"src/race_lidar/scored_pipeline.py", "source_postprocess_sha256"), (Path(args.config), "source_model_config_sha256")):
        if hashlib.sha256(file.read_bytes()).hexdigest() != pipeline[key]: raise ValueError("Frozen pipeline source/config differs")
    del payload
    lease_check(args.lease_registry)
    register(args.lease_registry, os.getpid(), args.port, Path(__file__).resolve())
    predictor = load_batched_predictor(args.pcdet_root, args.config, args.checkpoint)
    predictor = wrap_scored_predictor(predictor, pipeline)
    predictor.metadata["campaign_lease"] = {"deadline_utc": DEADLINE.isoformat(), "launch_cutoff_utc": CUTOFF.isoformat(),
                                           "registry": str(args.lease_registry), "pid": os.getpid()}
    serve(predictor, args.port, args.max_batch)


if __name__ == "__main__": main()
