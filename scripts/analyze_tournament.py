#!/usr/bin/env python3
"""Validate race results and write JSON, CSV, and a human-readable report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from race_lidar.race_results import analyze_directory, write_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Tournament or heat artifact directory")
    parser.add_argument("--output", type=Path, help="Defaults to the input directory")
    args = parser.parse_args()
    summary = analyze_directory(args.directory)
    write_analysis(args.output or args.directory, summary)
    print(json.dumps({key: summary[key] for key in (
        "heat_count", "policy_count", "best_policy_supported", "best_policies_among_tested",
        "observed_leading_policies", "qualification_issues")}, indent=2))


if __name__ == "__main__":
    main()
