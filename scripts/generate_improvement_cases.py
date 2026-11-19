#!/usr/bin/env python3
"""Freeze CLI-compatible homogeneous-fleet and stopped-hazard evaluation cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.improvement_ledger import digest, scenario_identity
except ModuleNotFoundError:
    from improvement_ledger import digest, scenario_identity


def formal_cases(split="holdout"):
    """Reserved evaluation starts; their outputs must never enter training."""
    bases = {"bootstrap": 310001, "holdout": 510001, "confirmation": 810001}
    if split not in bases:
        raise ValueError("split must be bootstrap, holdout, or confirmation")
    base = bases[split]
    cases = []
    definitions = []
    for cars in (4, 8, 20):
        for index, spacing in enumerate((12., 14., 16., 18.)):
            definitions.append((f"race-{cars}-spacing-{spacing:g}", {
                "scenario": "race", "cars": cars, "rotation": 0,
                "seed": base+cars*100+37*index, "spacing": spacing,
                "seconds": 480., "grid_jitter": .3}))
    for index, (name, target, ego, blocked, cars) in enumerate((
        ("flying-visibility-entry", 300., 0., "right", 2),
        ("stopped-right-block", 1000., 900., "right", 3),
        ("stopped-left-block", 3000., 2900., "left", 3),
    )):
        definitions.append((name, {"scenario": "stopped", "cars": cars, "rotation": 0,
            "seed": base+9001+37*index, "spacing": 12., "seconds": 120., "grid_jitter": 0.,
            "obstacle_progress": target, "ego_progress": ego, "ego_lane": 0., "blocked_side": blocked}))
    for name, definition in definitions:
        scenario = scenario_identity(definition)
        sha = digest(scenario)
        hazard = scenario["scenario"] == "stopped"
        expected = 1 if hazard else scenario["cars"]
        argv = [item for key, value in scenario.items()
                for item in ("--"+key.replace("_", "-"), str(value))]
        cases.append({"schema_version": 1, "case_id": f"{split}-{name}", "split": split,
            "scenario_id": sha[:20], "scenario_sha256": sha, "comparison_key": sha,
            "kind": "hazard" if hazard else "race", "is_hazard": hazard,
            "tier": str(scenario["cars"]), "cars": scenario["cars"],
            "expected_entrants": expected, "evaluated_entrants": expected,
            "excluded_from_training": True, "scenario": scenario, "cli_args": argv})
    return cases


def verify_case(case):
    """Reject stale hashes or any policy/model options masquerading as a world."""
    scenario = scenario_identity(case["scenario"])
    if set(case["scenario"]) != set(scenario):
        raise ValueError("Scenario contains unsupported treatment or recording fields")
    sha = digest(scenario)
    if case.get("scenario_sha256") != sha or case.get("scenario_id") != sha[:20] or case.get("comparison_key") != sha:
        raise ValueError("Case identity does not match the exact launcher scenario")
    return scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["bootstrap", "holdout", "confirmation"], default="holdout")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-directory", type=Path, help="Also emit one definition JSON per case")
    args = parser.parse_args()
    cases = formal_cases(args.split)
    content = json.dumps(cases, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end="")
    if args.case_directory:
        args.case_directory.mkdir(parents=True, exist_ok=True)
        for case in cases:
            (args.case_directory/(case["case_id"]+".json")).write_text(json.dumps(case, indent=2)+"\n")


if __name__ == "__main__":
    main()
