"""Run a complete closed-loop physics lap with the available Webots vehicle."""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=ROOT/".local/f1-reference/reference.npz")
    parser.add_argument("--output",type=Path,default=ROOT/"runs/hotlap-pro6000")
    parser.add_argument("--world",type=Path)
    parser.add_argument("--driver-model",type=Path,help="Trained neural driver checkpoint (.npz)")
    parser.add_argument("--backmarkers",type=int,default=0,help="Number of physical slower cars (0 to 3)")
    parser.add_argument("--webots",type=Path,default=ROOT/".local/Webots.app/Contents/MacOS/webots")
    parser.add_argument("--detector-url",default="http://127.0.0.1:8765")
    parser.add_argument("--batch",action="store_true")
    parser.add_argument("--physics-only",action="store_true",help="Validate the complete driving lap without sensor rendering/inference")
    parser.add_argument("--max-seconds",type=float,default=360.)
    parser.add_argument("--max-speed",type=float,default=60.,help="m/s upper bound; corners also obey the grip/braking profile")
    parser.add_argument("--lateral-accel",type=float,default=8.,help="m/s^2 planning limit")
    parser.add_argument("--brake-decel",type=float,default=8.)
    parser.add_argument("--acceleration",type=float,default=5.)
    parser.add_argument("--sensor-period-ms",type=int,default=100)
    args = parser.parse_args()
    if not 0 <= args.backmarkers <= 3:
        parser.error("--backmarkers must be between 0 and 3")
    generate_world = args.world is None
    if args.world is None:
        args.world=ROOT/"webots/worlds/race_learned.wbt" if args.backmarkers else ROOT/"webots/worlds/hotlap.wbt"
    for name in ("reference","webots"):
        if not getattr(args,name).is_file():parser.error(f"Missing {name}: {getattr(args,name)}")
    if args.driver_model is not None and not args.driver_model.is_file():
        parser.error(f"Missing trained driver: {args.driver_model}")
    if not generate_world and not args.world.is_file():
        parser.error(f"Missing world: {args.world}")
    if any(not math.isfinite(value) or value<=0 for value in
           (args.max_seconds,args.max_speed,args.lateral_accel,args.brake_decel,args.acceleration)):
        parser.error("Time, speed and acceleration limits must be finite and positive")
    if args.sensor_period_ms<20 or args.sensor_period_ms%20:
        parser.error("--sensor-period-ms must be a positive multiple of the 20 ms physics step")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output exists; choose a new directory to preserve lap evidence")
    args.output.mkdir(parents=True,exist_ok=True)
    if generate_world:
        subprocess.run([sys.executable,str(ROOT/"scripts/generate_hotlap_world.py"),"--reference",str(args.reference),
                        "--output",str(args.world),"--backmarkers",str(args.backmarkers)],check=True)
    if not args.world.is_file():
        parser.error(f"Missing generated world: {args.world}")
    for name in ("hotlap_recorder","backmarker_driver"):
        controller=ROOT/"webots/controllers"/name
        controller.mkdir(parents=True,exist_ok=True)
        (controller/"runtime.ini").write_text(f"[python]\nCOMMAND = {sys.executable}\n")
    env=os.environ.copy()
    env.update(RACE_REFERENCE=str(args.reference.resolve()),RACE_RECORDING=str((args.output/"recording").resolve()),
               RACE_DETECTOR_URL=args.detector_url,RACE_HOTLAP_MAX_SECONDS=str(args.max_seconds),
               RACE_HOTLAP_TARGET_LAPS="1",RACE_CAPTURE="0" if args.physics_only else "1",
               RACE_MAX_SPEED_MPS=str(args.max_speed),RACE_LATERAL_ACCEL_MPS2=str(args.lateral_accel),
               RACE_BRAKE_DECEL_MPS2=str(args.brake_decel),RACE_ACCEL_MPS2=str(args.acceleration),
               RACE_SENSOR_PERIOD_MS=str(args.sensor_period_ms),
               RACE_DRIVER_MODEL=str(args.driver_model.resolve()) if args.driver_model else "",
               RACE_BACKMARKERS=str(args.backmarkers))
    command=[str(args.webots.resolve()),"--stdout","--stderr","--mode=fast" if args.batch else "--mode=realtime"]
    if args.batch:command += ["--batch","--minimize","--no-rendering"]
    command.append(str(args.world.resolve()))
    raise SystemExit(subprocess.call(command,env=env))


if __name__=="__main__":main()
