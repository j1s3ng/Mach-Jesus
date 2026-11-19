"""Freeze the best completed development result without inspecting held-out data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import sha256


def select(run_paths, output):
    candidates = []
    for directory in run_paths:
        status = json.loads((directory / "status.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        if status.get("status") != "completed" or status.get("phase") != "evaluate" or metrics.get("split") != "development":
            raise ValueError(f"Only completed development evaluations may select a model: {directory}")
        if metrics.get("ap_101") is None:
            raise ValueError(f"Development AP is undefined: {directory}")
        candidates.append((float(metrics["ap_101"]), str(directory), status))
    if not candidates:
        raise ValueError("No development evaluations supplied")
    plan_hashes = {item[2]["plan_sha256"] for item in candidates}
    if len(plan_hashes) != 1:
        raise ValueError("Cannot compare results from different frozen study plans")
    _, winner, status = max(candidates, key=lambda item: (item[0], item[1]))
    checkpoint = Path(status["checkpoint_path"])
    if sha256(checkpoint) != status["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after evaluation")
    result = {"experiment": status["experiment"]["name"], "checkpoint": str(checkpoint),
              "checkpoint_sha256": status["checkpoint_sha256"], "plan_sha256": status["plan_sha256"],
              "selection_metric": "custom development XY-center AP101 at 2m, <=50m; not official A2RL AP",
              "development_result": winner, "candidates": [item[1] for item in candidates],
              "tie_breaker": "lexicographic run-directory name"}
    if output.exists():
        raise ValueError(f"Refusing to overwrite an existing frozen selection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = select(args.runs, args.output)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
