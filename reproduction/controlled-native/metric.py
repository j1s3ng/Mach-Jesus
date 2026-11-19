"""Verbatim archived numeric functions; see provenance.json."""
from __future__ import annotations
import math
import numpy as np
from shapely.geometry import Polygon

def matches(prediction, ground_truth, score, distance):
    matched, false_positives = {}, 0
    for frame in prediction:
        frame_key = (frame["scene_id"], frame["frame_id"])
        boxes = ground_truth[frame_key]["gt_boxes"]
        used = set()
        for box in sorted(frame["boxes"], key=lambda value: -value["score"]):
            if box["score"] < score: continue
            available = [(float(np.linalg.norm(np.asarray(box["center"])[:2]-np.asarray(gt["center"])[:2])), i)
                         for i, gt in enumerate(boxes) if i not in used]
            error, index = min(available, default=(float("inf"), -1))
            if error <= distance:
                used.add(index);matched[(*frame_key, index)] = error
            else:
                false_positives += 1
    return matched, false_positives


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


def frame_key(row):
    return row["scene_id"], row["frame_id"]


def matched_errors(predictions, truth, threshold=.35):
    """Score-ranked nearest unmatched GT, XY distance<=2m, keyed by GT index."""
    annotated = {frame_key(row): row["gt_boxes"] for row in truth}
    if len(annotated) != len(truth) or len({frame_key(row) for row in predictions}) != len(predictions):
        raise ValueError("Duplicate frame identity")
    if {frame_key(row) for row in predictions} != set(annotated):
        raise ValueError("Prediction/development frame identity mismatch")
    errors = {}
    fp = empty_fp = 0
    for prediction in predictions:
        key = frame_key(prediction)
        target = annotated[key]
        used = set()
        for box in sorted(prediction["boxes"], key=lambda item: -item["score"]):
            if not np.isfinite(box["score"]) or not np.all(np.isfinite(box["center"])):
                raise ValueError("Nonfinite prediction")
            if box["score"] < threshold:
                continue
            choices = [(float(np.linalg.norm(np.asarray(box["center"])[:2] - np.asarray(gt["center"])[:2])), i)
                       for i, gt in enumerate(target) if i not in used]
            distance, selected = min(choices, default=(float("inf"), -1))
            if distance <= 2.:
                used.add(selected)
                errors[(*key, selected)] = distance
            else:
                fp += 1
                empty_fp += int(not target)
    target_count = sum(len(boxes) for boxes in annotated.values())
    empty_count = sum(not boxes for boxes in annotated.values())
    far_targets = {(*frame_key(row), i) for row in truth for i, box in enumerate(row["gt_boxes"])
                   if 30. <= np.linalg.norm(np.asarray(box["center"])[:2]) <= 50.}
    tp = len(errors)
    return errors, {"score_threshold": threshold, "frames": len(truth), "targets": target_count,
        "true_positives": tp, "false_positives": fp, "false_negatives": target_count - tp,
        "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / target_count if target_count else 0.,
        "false_positives_per_frame": fp / len(truth), "empty_frames": empty_count,
        "empty_frame_false_positives": empty_fp,
        "false_positives_per_empty_frame": empty_fp / empty_count if empty_count else None,
        "matched_xy_center_mae_m": float(np.mean(list(errors.values()))) if errors else None,
        "far_30_50m": {"targets": len(far_targets), "matched": len(far_targets & errors.keys()),
                        "recall": len(far_targets & errors.keys()) / len(far_targets) if far_targets else None}}


def paired_errors(first, second):
    keys = sorted(first.keys() & second.keys())
    a = np.array([first[key] for key in keys])
    b = np.array([second[key] for key in keys])
    return {"common_gt_matches": len(keys), "first_xy_center_mae_m": float(a.mean()) if keys else None,
            "second_xy_center_mae_m": float(b.mean()) if keys else None,
            "paired_second_minus_first_mean_m": float((b - a).mean()) if keys else None,
            "scope": "Identical GT IDs matched by both at the stated threshold; descriptive only."}
