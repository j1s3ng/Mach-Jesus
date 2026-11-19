#!/usr/bin/env python3
"""Portable reproduction wrapper; no campaign host, registry or dated lease.

This new wrapper reuses the strictly loaded model and score module. Its source
hash is separately exposed; it is not the historical qualified runtime source.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcdet-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--max-batch", type=int, default=16)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.expected_sha256:
        raise ValueError("Checkpoint differs from the explicit download manifest SHA256")
    import torch
    from race_lidar.batched_remote import load_batched_predictor, serve
    from race_lidar.pipeline_artifact import tensor_state_sha256
    from race_lidar.scored_pipeline import wrap_scored_predictor
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    pipeline = payload.get("race_detector_pipeline")
    if pipeline is not None:
        if tensor_state_sha256(payload["model_state"]) != pipeline["weights_state_sha256"]:
            raise ValueError("Learned tensor state differs from pipeline metadata")
        # Preserve evidence of the original package's source binding while also
        # recording this separately identified portable reproduction wrapper.
        expected = [(ROOT / "scripts/serve_leased_candidate.py", "source_server_sha256"),
                    (ROOT / "src/race_lidar/scored_pipeline.py", "source_postprocess_sha256"),
                    (args.config, "source_model_config_sha256")]
        for path, key in expected:
            if hashlib.sha256(path.read_bytes()).hexdigest() != pipeline[key]:
                raise ValueError("Packaged historical source/config identity differs: " + key)
    del payload
    predictor = load_batched_predictor(args.pcdet_root, args.config, checkpoint)
    if pipeline is not None:
        predictor = wrap_scored_predictor(predictor, pipeline)
    predictor.metadata.update(reproduction_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              reproduction_scope="portable wrapper, not historical campaign runtime attestation")
    serve(predictor, args.port, args.max_batch)


if __name__ == "__main__": main()
