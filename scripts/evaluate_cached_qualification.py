#!/usr/bin/env python3
"""Reproduce archived metrics on every saved frame; no models, fitting or GPU."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.qualify_sim_detector import matches


def summarize(rows):
    truth = {(r["scene_id"], r["frame_id"]): r for r in rows}
    assert len(truth) == len(rows), "Duplicate cached frame"
    before = [{"scene_id": r["scene_id"], "frame_id": r["frame_id"], "boxes": r["original_boxes"]} for r in rows]
    after = [{"scene_id": r["scene_id"], "frame_id": r["frame_id"], "boxes": r["adapted_boxes"]} for r in rows]
    old, old_fp = matches(before, truth, .35, 2.)
    new, new_fp = matches(after, truth, .425, 2.)
    targets = sum(len(r["gt_boxes"]) for r in rows)
    common = set(old) & set(new)
    far = {(*key, i) for key, row in truth.items() for i, box in enumerate(row["gt_boxes"])
           if 30. <= np.linalg.norm(np.asarray(box["center"])[:2]) <= 50.}
    return {"frames": len(rows), "visible_targets": targets, "common_matched_targets": len(common),
            "pretrained": {"recall": len(old)/targets, "false_positives": old_fp,
                           "false_positives_per_frame": old_fp/len(rows),
                           "common_matched_center_mean_m": float(np.mean([old[k] for k in common]))},
            "adapted": {"recall": len(new)/targets, "false_positives": new_fp,
                        "false_positives_per_frame": new_fp/len(rows),
                        "common_matched_center_mean_m": float(np.mean([new[k] for k in common]))},
            "far_targets": len(far), "pretrained_far_recall": len(set(old)&far)/len(far),
            "adapted_far_recall": len(set(new)&far)/len(far)}


def compare(actual, expected):
    for key, value in actual.items():
        reference = expected[key]
        if isinstance(value, dict): compare(value, reference)
        elif isinstance(value, float):
            if not np.isclose(value, reference, rtol=0., atol=1e-12):
                raise ValueError(f"Archived metric differs: {key}: {value} vs {reference}")
        elif value != reference:
            raise ValueError(f"Archived count differs: {key}: {value} vs {reference}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "examples/qualification")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    data = args.cache / "frozen_predictions.jsonl.gz"
    if hashlib.sha256(data.read_bytes()).hexdigest() != manifest["cache_sha256"]:
        raise ValueError("Cached predictions or offline labels changed")
    with gzip.open(data, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    output = {}
    for item in manifest["reports"]:
        file = ROOT / item["path"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Expected archived report changed")
        expected = json.loads(file.read_text())
        groups = set(expected["qualification_groups"])
        actual = summarize([row for row in rows if row["scene_id"] in groups])
        compare(actual, expected)
        output[file.stem] = actual
    result = {"status": "all_archived_metrics_reproduced", "models_run": False, "training_run": False,
              "new_qualification_decision": False, "original_raw_threshold": .35,
              "adapted_raw_threshold": .425, "results": output}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
