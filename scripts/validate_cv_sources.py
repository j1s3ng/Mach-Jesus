#!/usr/bin/env python3
"""Archive or verify strict portable training-source eligibility for a CV export."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cv_training_sources import archive_training_sources, validate_export_training_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-releases", type=Path)
    parser.add_argument("--archive", action="store_true", help="Locally validate live source lineage and archive exact configs first")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists(): parser.error("Existing source audit result is immutable")
    if args.archive: archive_training_sources(args.dataset, args.training_releases)
    result = validate_export_training_sources(args.dataset, args.training_releases)
    result["validated_utc"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(result, indent=2)+"\n"
    if args.output: args.output.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
