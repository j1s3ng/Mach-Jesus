#!/usr/bin/env python3
"""Serve independent racers' LiDAR with one strictly loaded CUDA model."""
import argparse

from race_lidar.batched_remote import load_batched_predictor, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcdet-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    parser.add_argument("--max-pending", type=int, default=64)
    args = parser.parse_args()
    predictor = load_batched_predictor(args.pcdet_root, args.config, args.checkpoint)
    serve(predictor, args.port, args.max_batch, args.batch_wait_ms, args.max_pending)


if __name__ == "__main__":
    main()
