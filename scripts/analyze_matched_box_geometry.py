"""Post hoc heading/box geometry on closed predictions; never fit or qualify.

The exact published score-greedy XY matcher is cross-checked for every setting.
Both models are summarized on their common target intersection. These custom
simulation diagnostics are not official nuScenes/A2RL AP or independent trials.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import shapely
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.qualify_sim_detector import matches


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def key(row):
    return row["scene_id"], row["frame_id"]


def heading_error(predicted, truth):
    error = abs(math.atan2(math.sin(predicted-truth), math.cos(predicted-truth)))
    return error, min(error, math.pi-error)


def footprint(box):
    length, width = np.asarray(box["size"], dtype=float)[:2]
    points = np.array([[-length/2,-width/2], [length/2,-width/2],
                       [length/2,width/2], [-length/2,width/2]])
    cosine, sine = math.cos(box["yaw"]), math.sin(box["yaw"])
    return Polygon(points @ np.array([[cosine,sine],[-sine,cosine]]) + np.asarray(box["center"])[:2])


def box_iou(left, right):
    a, b = footprint(left), footprint(right)
    intersection = a.intersection(b).area
    bev = intersection/(a.area+b.area-intersection)
    low = max(left["center"][2]-left["size"][2]/2,
              right["center"][2]-right["size"][2]/2)
    high = min(left["center"][2]+left["size"][2]/2,
               right["center"][2]+right["size"][2]/2)
    overlap = intersection*max(0., high-low)
    volume_a, volume_b = float(np.prod(left["size"])), float(np.prod(right["size"]))
    return float(bev), float(overlap/(volume_a+volume_b-overlap))


def assign_boxes(prediction, truth, cutoff):
    """Recover matched box identities and assert exact published match parity."""
    assigned = {}
    for frame in prediction:
        frame_key = key(frame)
        boxes = truth[frame_key]["gt_boxes"]
        used = set()
        for box in sorted(frame["boxes"], key=lambda value: -value["score"]):
            if box["score"] < cutoff:
                continue
            choices = [(float(np.linalg.norm(np.asarray(box["center"])[:2]-np.asarray(gt["center"])[:2])), index)
                       for index,gt in enumerate(boxes) if index not in used]
            distance, index = min(choices, default=(math.inf, -1))
            if distance <= 2.:
                used.add(index)
                assigned[(*frame_key,index)] = {"prediction":box,"truth":boxes[index],"center_xy_m":distance}
    published, false_positives = matches(prediction, truth, cutoff, 2.)
    if set(assigned) != set(published) or any(not math.isclose(assigned[k]["center_xy_m"],v,abs_tol=1e-12) for k,v in published.items()):
        raise ValueError("Box-identity reconstruction differs from the original qualifier")
    return assigned, false_positives


def summary(values):
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Need finite measured matched-box values")
    return {"count":len(array),"mean":float(np.mean(array)),"median":float(np.median(array)),
            "p95":float(np.quantile(array,.95))}


def geometry(assigned, common):
    records = []
    for identity in sorted(common):
        pair = assigned[identity]
        pred, gt = pair["prediction"], pair["truth"]
        directed, undirected = heading_error(pred["yaw"], gt["yaw"])
        bev, volume = box_iou(pred, gt)
        records.append({"center_xy_m":pair["center_xy_m"],"center_z_abs_m":abs(pred["center"][2]-gt["center"][2]),
                        "directed_heading_rad":directed,"undirected_box_heading_rad":undirected,
                        "bev_iou":bev,"upright_3d_iou":volume,
                        **{name:abs(pred["size"][index]-gt["size"][index]) for index,name in enumerate(
                            ("length_abs_error_m","width_abs_error_m","height_abs_error_m"))}})
    result = {name:summary([r[name] for r in records]) for name in records[0]}
    result["directed_heading_over_90deg"] = {"count":sum(r["directed_heading_rad"]>math.pi/2 for r in records),
                                            "fraction":float(np.mean([r["directed_heading_rad"]>math.pi/2 for r in records]))}
    return result


def analyze(dataset, evaluation):
    sources = {}
    def bind(path):
        sources[str(Path(path).resolve())] = digest(path)
        return Path(path)
    bind(Path(__file__));bind(ROOT/"scripts/qualify_sim_detector.py")
    metadata = json.loads(bind(dataset/"dataset.json").read_text())
    status = json.loads(bind(evaluation/"status.json").read_text())
    if status.get("training_executed") or not status.get("evaluation_only"):
        raise ValueError("Only closed evaluation-only evidence is permitted")
    if status["source_dataset_sha256"] != sources[str((dataset/"dataset.json").resolve())]:
        raise ValueError("Dataset differs from the actual evaluation")
    gt_path = bind(dataset/"val.jsonl")
    if metadata["manifest_sha256"]["val"] != digest(gt_path):
        raise ValueError("Frozen visible-target manifest changed")
    rows = read_rows(gt_path)
    truth = {key(row):row for row in rows}
    if len(truth) != len(rows) or metadata["min_measured_points_per_box"] != 3 or metadata["evaluation_radius_m"] != 50:
        raise ValueError("Unexpected target identities or scoring population")
    decision = json.loads(bind(evaluation/"qualification_decision.json").read_text())
    config = json.loads(bind(evaluation/"effective_config.json").read_text())
    model_floor = float(config["MODEL"]["DENSE_HEAD"]["POST_PROCESSING"]["SCORE_THRESH"])
    floors = []
    for name, minimum_cutoff in (("pretrained_metrics.json",.35),("adapted_metrics.json",.425)):
        path=bind(evaluation/name)
        if decision["artifact_sha256"][name] != digest(path):
            raise ValueError("Closed writer-floor evidence changed")
        metrics=json.loads(path.read_text())
        floor=max(model_floor,float(metrics["effective_prediction_score_floor"]))
        if minimum_cutoff < floor:
            raise ValueError("Requested cutoff is below cached scores; omitted predictions cannot be reconstructed")
        floors.append(floor)
    predictions = []
    for name in ("pretrained_predictions.jsonl","adapted_predictions.jsonl"):
        path = bind(evaluation/name)
        if decision["artifact_sha256"][name] != digest(path):
            raise ValueError("Closed prediction cache changed")
        values = read_rows(path)
        if len(values) != len(truth) or {key(row) for row in values} != set(truth):
            raise ValueError("Duplicate or missing prediction frames")
        predictions.append(values)
    reports = {}
    for label, left_cutoff, right_cutoff in (("deployed_pipeline",.35,.425),("common_raw0425",.425,.425)):
        left, fp_left = assign_boxes(predictions[0], truth, left_cutoff)
        right, fp_right = assign_boxes(predictions[1], truth, right_cutoff)
        common = set(left)&set(right)
        groups = {}
        for group in sorted({identity[0] for identity in truth}|{"AGGREGATE"}):
            ids = {identity for identity in common if group=="AGGREGATE" or identity[0]==group}
            groups[group] = {"common_matched_targets":len(ids),"original":geometry(left,ids),"adapted":geometry(right,ids)}
        reports[label] = {"raw_cutoffs":[left_cutoff,right_cutoff],"all_target_count":sum(len(r["gt_boxes"]) for r in truth.values()),
                          "matched_original":len(left),"matched_adapted":len(right),"FP_original":fp_left,"FP_adapted":fp_right,
                          "groups":groups}
    if any(digest(path)!=value for path,value in sources.items()):
        raise ValueError("Evidence changed during analysis")
    return {"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"role":"POSTHOC_GEOMETRY_DIAGNOSTIC_NO_FIT_OR_GATE_CHANGE",
            "frames":len(truth),"reports":reports,"source_sha256":sources,"shapely_version":shapely.__version__,
            "validated_stored_score_floors":floors,
            "raw_clouds_opened":False,"inference_executed":False,"training_use_allowed":False,
            "scope":["Same original visible-target population and exact score-greedy 2mXY associations.",
                     "Common-target summaries condition on both models matching; report recall/counts separately.",
                     "Directed yaw uses 2pi-period orientation; undirected box yaw identifies angles differing by pi.",
                     "3D IoU uses yaw-upright boxes, not full asset meshes/roll/pitch or official AP.",
                     "Correlated simulated frames; no independent-trial confidence interval or real-data claim."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True);p.add_argument("--evaluation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error("Choose a new diagnostic output; existing evidence is immutable")
    result=analyze(args.dataset,args.evaluation)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    values=result["reports"]["common_raw0425"]["groups"]["AGGREGATE"]
    print(json.dumps({"output":str(args.output),"frames":result["frames"],"common":values["common_matched_targets"],
        "heading_mean_rad":[values[name]["directed_heading_rad"]["mean"] for name in ("original","adapted")],
        "iou3d_mean":[values[name]["upright_3d_iou"]["mean"] for name in ("original","adapted")]},indent=2))


if __name__=="__main__":main()
