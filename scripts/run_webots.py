"""Launch the project-local Webots app without changing user preferences."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "runs/webots")
    p.add_argument("--frames", type=int, default=400)
    p.add_argument("--batch", action="store_true", help="Minimize main view; sensor rendering stays enabled")
    p.add_argument("--detector-url", default="", help="E.g. http://127.0.0.1:8765 over an SSH tunnel")
    p.add_argument("--webots", type=Path, default=ROOT / ".local/Webots.app/Contents/MacOS/webots")
    args = p.parse_args()
    if args.frames < 1:
        p.error("--frames must be positive")
    if not args.webots.exists():
        p.error("Webots missing: run scripts/install_webots_macos.sh or supply --webots")
    if (args.output / "recording/manifest.jsonl").exists():
        p.error("Output already contains a recording; choose a fresh --output")
    runtime = ROOT / "webots/controllers/race_recorder/runtime.ini"
    runtime.write_text(f"[python]\nCOMMAND = {sys.executable}\n")
    env = os.environ.copy()
    env.update(RACE_RECORDING=str((args.output / "recording").resolve()), RACE_FRAMES=str(args.frames),
               RACE_DETECTOR_URL=args.detector_url)
    command = [str(args.webots.resolve()), "--stdout", "--stderr", "--mode=fast" if args.batch else "--mode=realtime"]
    if args.batch:
        command += ["--batch", "--minimize", "--no-rendering"]
    command.append(str(ROOT / "webots/worlds/race_loop.wbt"))
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
