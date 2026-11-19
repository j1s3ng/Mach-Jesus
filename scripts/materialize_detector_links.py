#!/usr/bin/env python3
"""Complete a qualification-only export using exact existing training clouds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cv_training_sources import validate_export_training_sources


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-export", type=Path, required=True)
    parser.add_argument("--training-releases", type=Path, required=True)
    args = parser.parse_args()
    validate_export_training_sources(args.training_export, args.training_releases)
    validate_export_training_sources(args.dataset, args.training_releases)
    metadata = json.loads((args.dataset/"dataset.json").read_text())
    if metadata["validation_purpose"] != "qualification_only_no_fitting_or_development_selection": raise ValueError("Require a qualification-only export")
    if (metadata["reused_training_export"]["dataset_metadata_sha256"] != sha(args.training_export/"dataset.json") or
            (args.dataset/"train.jsonl").read_bytes() != (args.training_export/"train.jsonl").read_bytes()):
        raise ValueError("Frozen training-export identity differs")
    created = 0
    for line in (args.dataset/"train.jsonl").read_text().splitlines():
        row = json.loads(line);source = (args.training_export/row["points_file"]).resolve();target = (args.dataset/row["points_file"]).resolve()
        if not source.is_relative_to(args.training_export.resolve()) or not target.is_relative_to(args.dataset.resolve()) or sha(source) != row["points_sha256"]:
            raise ValueError("Training cloud changed or escapes export")
        if target.exists():
            if sha(target) != row["points_sha256"]: raise ValueError("Existing destination differs")
        else: os.link(source, target);created += 1
    print(json.dumps({"training_cloud_hard_links_created": created, "duplicate_training_point_bytes_copied": 0}))


if __name__ == "__main__": main()
