#!/usr/bin/env python3
"""Export exact already-scored development inputs/outputs for a qualitative demo."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"runs/improvement-12h/training/detector-dense-dataset-001"
EVAL = ROOT/"runs/improvement-12h/training/detector-full-dense1200-s7"
OUTPUT = ROOT/"presentation/demo"
THRESHOLD = .35
MATCH_M = 2.
PERIOD_NS = 200_000_000
LENGTH = 20


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    result = {}
    for line in path.open():
        if not line.strip(): continue
        row = json.loads(line);key = (row["scene_id"], row["frame_id"])
        if key in result: raise ValueError("Repeated source frame")
        result[key] = (row, hashlib.sha256(line.encode()).hexdigest())
    return result


def annotate(prediction, truth):
    used, annotations = set(), {}
    for index in sorted(range(len(prediction["boxes"])), key=lambda i: -prediction["boxes"][i]["score"]):
        box = prediction["boxes"][index]
        if box["score"] < THRESHOLD: continue
        choices = [(float(np.linalg.norm(np.asarray(box["center"])[:2]-np.asarray(gt["center"])[:2])), i)
                   for i, gt in enumerate(truth) if i not in used]
        distance, target = min(choices, default=(float("inf"), -1))
        matched = distance <= MATCH_M
        if matched: used.add(target)
        annotations[index] = {"source_prediction_index": index, "is_false_positive": not matched,
                              "matched_gt_index": target if matched else None,
                              "xy_center_error_m": distance if matched else None}
    return {"true_positives": len(used), "false_positives": sum(row["is_false_positive"] for row in annotations.values()),
            "false_negatives": len(truth)-len(used), "annotations": annotations}


def evenly(values, count):
    values = np.asarray(values, dtype=np.int64)
    if len(values) <= count: return values
    return values[np.floor(np.arange(count)*len(values)/count).astype(np.int64)]


def sample_points(points, budget=2000):
    available = np.flatnonzero(np.linalg.norm(points[:, :2], axis=1) <= 50.)
    above = available[points[available, 2] > .25]
    first = evenly(above, min(1000, budget))
    rest = available[~np.isin(available, first)]
    chosen = np.sort(np.concatenate((first, evenly(rest, budget-len(first)))))
    if len(np.unique(chosen)) != len(chosen): raise RuntimeError("Visual points repeated")
    return chosen


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename in ("replay.json", "replay_manifest.json", "source_rows.jsonl"):
        if (OUTPUT/filename).exists(): raise ValueError("Existing demo evidence is immutable; use a fresh folder")
    rule = json.loads((OUTPUT/"selection_rule.json").read_text())
    metadata = json.loads((DATA/"dataset.json").read_text());status = json.loads((EVAL/"status.json").read_text())
    if sha(DATA/"dataset.json") != status["source_dataset_sha256"] or sha(DATA/"val.jsonl") != metadata["manifest_sha256"]["val"]:
        raise ValueError("Scored development inputs changed")
    rows = read_rows(DATA/"val.jsonl");baseline = read_rows(EVAL/"pretrained_predictions.jsonl");adapted = read_rows(EVAL/"adapted_predictions.jsonl")
    if set(rows) != set(baseline) or set(rows) != set(adapted): raise ValueError("Scored frame sets differ")
    if {row[0]["scene_id"] for row in rows.values()} != {"field4-r1"}: raise ValueError("Use only the declared development run")
    grouped, annotations = {}, {}
    for key, (row, _) in rows.items():
        grouped.setdefault(row["car_id"], []).append(key)
        annotations[key] = {"baseline": annotate(baseline[key][0], row["gt_boxes"]), "adapted": annotate(adapted[key][0], row["gt_boxes"])}
    for keys in grouped.values(): keys.sort(key=lambda key: rows[key][0]["timestamp_ns"])
    source_runs = {Path(row[0]["source_run"]) for row in rows.values()}
    if len(source_runs) != 1: raise ValueError("Expected one actual development run")
    run = next(iter(source_runs));pose_path = run/"gt_poses.jsonl"
    poses = {row["timestamp_ns"]: {car["car_id"]: car for car in row["cars"]} for row in
             (json.loads(line) for line in pose_path.open() if line.strip())}
    def consecutive(keys):
        return len(keys) == LENGTH and all(rows[b][0]["timestamp_ns"]-rows[a][0]["timestamp_ns"] == PERIOD_NS for a,b in zip(keys, keys[1:]))
    def displacement(keys):
        first,last = rows[keys[0]][0],rows[keys[-1]][0]
        a = poses[first["timestamp_ns"]][first["car_id"]]["position_m"]
        b = poses[last["timestamp_ns"]][last["car_id"]]["position_m"]
        return float(np.linalg.norm(np.array(a)[:2]-np.array(b)[:2]))
    candidates = []
    for car, keys in grouped.items():
        for index in range(len(keys)-LENGTH+1):
            window = keys[index:index+LENGTH]
            if consecutive(window) and all(2 <= len(rows[key][0]["gt_boxes"]) <= 4 for key in window) and displacement(window) >= 5.:
                candidates.append(window)
    candidates.sort(key=lambda keys: (rows[keys[0]][0]["timestamp_ns"], rows[keys[0]][0]["car_id"]))
    if not candidates: raise ValueError("No window satisfies the declared moving overview rule")
    overview = candidates[0]
    failures = [(rows[key][0]["timestamp_ns"], rows[key][0]["car_id"], index, key)
                for key, values in annotations.items() for index, detail in values["adapted"]["annotations"].items() if detail["is_false_positive"]]
    if not failures: raise ValueError("No actual adapted FP exists at the declared threshold")
    _, car, source_prediction_index, anchor = min(failures)
    keys = grouped[car];anchor_index = keys.index(anchor)
    start = max(0, min(anchor_index-10, len(keys)-LENGTH));failure_window = keys[start:start+LENGTH]
    if not consecutive(failure_window): raise ValueError("First adapted failure has no consecutive20-frame context; do not silently substitute")
    clips, frames, originals, frame_provenance = [], [], [], []
    selected = [("overview", "First moving 2–4-target window", overview),
                ("first_adapted_fp", "First actual adapted false positive", failure_window)]
    for clip_id, title, keys in selected:
        clip_start = len(frames);start_ns = rows[keys[0]][0]["timestamp_ns"]
        for ordinal, key in enumerate(keys):
            row, dataset_line_sha = rows[key];point_path = DATA/row["points_file"]
            if sha(point_path) != row["points_sha256"]: raise ValueError("Selected measured cloud changed")
            raw = np.load(point_path, allow_pickle=False)
            if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all(): raise ValueError("Expected finite measured N×4 cloud")
            indices = sample_points(raw);visual = raw[indices]
            shown = {name: [box for box in source[key][0]["boxes"] if box["score"] >= THRESHOLD] for name,source in (("baseline",baseline),("adapted",adapted))}
            failure_ids = [f"{key[1]}:adapted:{i}" for i,detail in annotations[key]["adapted"]["annotations"].items() if detail["is_false_positive"]]
            frame = {"id": f"{clip_id}:{ordinal:02d}", "clip_id": clip_id, "frame_id": key[1], "scene_id": key[0], "car_id": row["car_id"],
                     "timestamp_ns": row["timestamp_ns"], "time_s": row["timestamp_ns"]/1e9, "relative_time_s": (row["timestamp_ns"]-start_ns)/1e9,
                     "points": visual.tolist(), "point_indices": indices.tolist(), "source_point_count": len(raw), "visual_point_count": len(visual),
                     "baseline": shown["baseline"], "adapted": shown["adapted"],
                     "baseline_all_predictions": baseline[key][0]["boxes"], "adapted_all_predictions": adapted[key][0]["boxes"],
                     "gt_boxes": row["gt_boxes"], "evaluation": annotations[key], "adapted_false_positive_ids": failure_ids,
                     "is_first_adapted_fp_anchor": key == anchor}
            for model in ("baseline","adapted"):
                for i,detail in frame["evaluation"][model]["annotations"].items():
                    detail["source_prediction_id"] = f"{key[1]}:{model}:{i}"
            if not np.array_equal(np.array(frame["points"], dtype=raw.dtype), raw[indices]): raise RuntimeError("Demo points are not an exact real subset")
            frames.append(frame)
            originals.append({"display_frame_id": frame["id"], "dataset_row": row, "baseline_prediction_row": baseline[key][0], "adapted_prediction_row": adapted[key][0]})
            frame_provenance.append({"display_frame_id": frame["id"], "scene_id": key[0], "frame_id": key[1], "timestamp_ns": row["timestamp_ns"],
                                     "car_id": row["car_id"], "point_file": str(point_path.relative_to(ROOT)), "points_sha256": row["points_sha256"],
                                     "dataset_source_line_sha256": dataset_line_sha, "baseline_source_line_sha256": baseline[key][1], "adapted_source_line_sha256": adapted[key][1]})
        clips.append({"id": clip_id, "title": title, "start_index": clip_start, "count": LENGTH, "frame_indices": list(range(clip_start, len(frames))),
                      "source_car_id": rows[keys[0]][0]["car_id"], "source_time_s": [rows[keys[0]][0]["timestamp_ns"]/1e9,rows[keys[-1]][0]["timestamp_ns"]/1e9],
                      "ego_endpoint_displacement_m": displacement(keys), "visible_gt_counts": [len(rows[key][0]["gt_boxes"]) for key in keys]})
    payload = {"schema_version": "real-lidar-replay-v1", "metadata": {"title": "Measured LiDAR and already-scored detector replay",
               "source_role": "development; qualitative illustrations, not a new benchmark", "coordinate_frame": "ego_x_forward_y_left_z_up",
               "point_features": ["x","y","z","intensity"], "prediction_evaluation_radius_m": 50., "visual_point_xy_radius_m": 50.,
               "raw_sensor_range_unchanged": True, "default_gt_visible": False, "default_score_threshold": THRESHOLD,
               "baseline_raw_score_threshold": THRESHOLD, "adapted_raw_score_threshold": THRESHOLD, "source_model_output_floor": .1,
               "matching_distance_m": MATCH_M, "capture_period_s": .2, "GT_visibility": {"minimum_inside_box_points": 3, "tolerance_m": .03, "range_m": 50},
               "gt_source": "offline exact-timestamp Supervisor poses; not model inputs", "point_sampling": rule["visual_sampling"],
               "source_predictions_unchanged": True, "model_inference_rerun": False, "training_calibration_or_model_selection": False,
               "baseline_checkpoint_sha256": status["source_checkpoint_sha256"], "adapted_checkpoint_sha256": status["candidate_checkpoint_sha256"],
               "adapted035_pipeline_deployment_status": "REJECTED; this is same-threshold development visualization",
               "separate_qualified_runtime_threshold": .425,
               "display_filter_note": "Optional UI filtering changes visualization only; it does not rerun inference or recompute benchmark claims."},
               "clips": clips, "frames": frames,
               "failure_anchor": {"frame_id": anchor[1], "scene_id": anchor[0], "car_id": car,
                    "timestamp_ns": rows[anchor][0]["timestamp_ns"], "source_prediction_index": source_prediction_index,
                    "source_prediction_id": f"{anchor[1]}:adapted:{source_prediction_index}", "box": adapted[anchor][0]["boxes"][source_prediction_index],
                    "annotation": annotations[anchor]["adapted"]["annotations"][source_prediction_index], "definition": "first actual adapted unmatched box at raw0.35 with2mXYassociation"}}
    (OUTPUT/"replay.json").write_text(json.dumps(payload, separators=(",",":"), allow_nan=False)+"\n")
    (OUTPUT/"source_rows.jsonl").write_text("".join(json.dumps(row, separators=(",",":"), allow_nan=False)+"\n" for row in originals))
    sources = [DATA/"dataset.json",DATA/"val.jsonl",EVAL/"pretrained_predictions.jsonl",EVAL/"adapted_predictions.jsonl",EVAL/"status.json",pose_path,run/"config.json",OUTPUT/"selection_rule.json",Path(__file__)]
    manifest = {"scope": "qualitative illustrative selection only; no reruns/benchmark/model selection", "selection_rule": rule,
                "source_file_sha256": {str(path.relative_to(ROOT)):sha(path) for path in sources}, "frames": frame_provenance,
                "display_frame_count": len(frames), "unique_source_frame_count": len({frame["frame_id"] for frame in frames}),
                "overlap_source_frames": sorted(set(overview)&set(failure_window)), "first_adapted_fp_anchor": payload["failure_anchor"],
                "source_raw_thresholds": {"baseline": THRESHOLD,"adapted":THRESHOLD}, "gt_default_visible": False,
                "exact_point_subsets_verified": True, "all_original_prediction_scores_preserved": True,
                "artifacts": {name: {"sha256":sha(OUTPUT/name),"bytes":(OUTPUT/name).stat().st_size} for name in ("replay.json","source_rows.jsonl")}}
    (OUTPUT/"replay_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"replay": str((OUTPUT/"replay.json").relative_to(ROOT)), "bytes":(OUTPUT/"replay.json").stat().st_size,
                      "frames":len(frames), "clips":clips, "failure_anchor":payload["failure_anchor"],
                      "unique_source_frames":manifest["unique_source_frame_count"]}, indent=2))


if __name__ == "__main__": main()
