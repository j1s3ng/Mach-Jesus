from __future__ import annotations

import argparse
from pathlib import Path

from race_lidar.projection import export_overlay

parser = argparse.ArgumentParser(description="Export RGB/LiDAR projection diagnostics for the Webots camera")
parser.add_argument("recording",type=Path)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--count",type=int,default=6)
args = parser.parse_args()
if args.count <= 0:
    parser.error("--count must be positive")
print(export_overlay(args.recording,args.output,args.count))
