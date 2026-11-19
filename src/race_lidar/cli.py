"""Small, explicit command line entry points; no implicit long training jobs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Race LiDAR: Mac simulation and NVIDIA VoxelNeXt")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate synthetic fixtures and replay the cluster detector")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument("--frames", type=int, default=120)
    demo.add_argument("--seed", type=int, default=7)
    rep = commands.add_parser("replay", help="Replay manifest.jsonl sensor recordings")
    rep.add_argument("input", type=Path)
    rep.add_argument("--output", type=Path, required=True)
    rep.add_argument("--max-frames", type=int)
    rep.add_argument("--backend",choices=["cluster","remote"],default="cluster")
    rep.add_argument("--detector-url",default="http://127.0.0.1:8765")
    commands.add_parser("doctor", help="Read-only local compute and dependency report")
    serve = commands.add_parser("serve", help="Serve detector on loopback (SSH tunnel for LAN use)")
    serve.add_argument("--backend", choices=["cluster", "voxelnext"], default="cluster")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--pcdet-root", type=Path)
    serve.add_argument("--config", type=Path)
    serve.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = {"platform": platform.platform(), "machine": platform.machine(),
                  "python": sys.executable, "cuda_required_for_voxelnext": True}
        for name in ["numpy", "scipy", "cv2", "PIL", "torch", "huggingface_hub"]:
            report[name] = bool(importlib.util.find_spec(name))
        if report["torch"]:
            import torch
            report.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                          mps_available=torch.backends.mps.is_available())
        print(json.dumps(report, indent=2))
    elif args.command == "demo":
        from .demo import generate_demo
        from .replay import replay
        generate_demo(args.output / "recording", frames=args.frames, seed=args.seed)
        print(json.dumps(replay(args.output / "recording", args.output / "replay"), indent=2))
    elif args.command == "replay":
        from .replay import replay
        print(json.dumps(replay(args.input, args.output, max_frames=args.max_frames,
                               backend=args.backend,detector_url=args.detector_url), indent=2))
    else:
        from .remote import serve_detector
        serve_detector(args.backend, args.port, args.pcdet_root, args.config, args.checkpoint)


if __name__ == "__main__":
    main()
