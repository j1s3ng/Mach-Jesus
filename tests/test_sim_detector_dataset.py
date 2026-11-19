import json
import hashlib
from types import SimpleNamespace

import numpy as np

from scripts.build_sim_detector_dataset import build
from scripts.qualify_sim_detector import matches, qualify


def make_run(path, points, entries):
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / "points").mkdir()
    for name, cloud in points.items():
        np.save(path / "points" / (name + ".npy"), cloud, allow_pickle=False)
    (path / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in entries))


def test_qualification_duplicate_filter_is_input_only_deterministic_and_never_trains_it(tmp_path):
    same = np.array([[10., 0, .5, 1]], np.float32)
    unique = np.array([[11., 0, .5, 1]], np.float32)
    base = {"car_id": "ego", "coordinate_frame": "ego_x_forward_y_left_z_up", "gt_boxes": []}
    train_entry = {**base, "frame_id": "train", "timestamp_ns": 1, "points_file": "points/a.npy"}
    duplicate = {**base, "frame_id": "duplicate", "timestamp_ns": 2, "points_file": "points/a.npy"}
    fresh = {**base, "frame_id": "unique", "timestamp_ns": 3, "points_file": "points/b.npy"}
    train, val = tmp_path / "train_run", tmp_path / "validation_run"
    make_run(train, {"a": same}, [train_entry])
    make_run(val, {"a": same, "b": unique}, [duplicate, fresh])
    outputs = []
    for index in range(2):
        if index:
            # Changing offline labels must not change which input is excluded.
            duplicate["gt_boxes"] = [{"center": [10, 0, .5], "size": [4, 2, 1], "yaw": 0}]
            (val / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [duplicate, fresh]))
        output = tmp_path / f"export{index}"
        args = SimpleNamespace(train_runs=[train], val_runs=[val], output=output,
                               diagnostic_only=False, drop_duplicate_validation=True, stride=1,
                               max_frames_per_run=0, positive_only=False, min_points=1, max_range=50.)
        report = build(args)
        training = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        validation = [json.loads(line) for line in (output / "val.jsonl").read_text().splitlines()]
        assert report["input_frame_counts_before_dedup"]["val"] == 2
        assert len(report["excluded_duplicate_validation_frames"]) == 1
        assert report["excluded_duplicate_validation_frames"][0]["frame_id"] == "duplicate"
        assert [row["source_frame_id"] for row in training] == ["train"]
        assert [row["source_frame_id"] for row in validation] == ["unique"]
        assert not {row["points_sha256"] for row in training} & {row["points_sha256"] for row in validation}
        outputs.append((output / "val.jsonl").read_text())
    assert outputs[0] == outputs[1]


def test_paired_center_gate_uses_the_same_matched_ground_truth_targets():
    box = {"center": [10, 0, .5], "size": [4, 2, 1], "yaw": 0}
    other = {**box, "center": [30, 0, .5]}
    truth = {("run", "frame"): {"gt_boxes": [box, other]}}
    baseline = [{"scene_id": "run", "frame_id": "frame", "boxes": [{**box, "center": [10.1, 0, .5], "score": .9}]}]
    candidate = [{"scene_id": "run", "frame_id": "frame", "boxes": [
        {**box, "center": [10.15, 0, .5], "score": .9},
        {**other, "center": [31, 0, .5], "score": .8}]}]
    old, old_fp = matches(baseline, truth, .35, 2.)
    new, new_fp = matches(candidate, truth, .35, 2.)
    common = set(old) & set(new)
    assert common == {("run", "frame", 0)}
    assert old_fp == new_fp == 0
    assert np.mean([new[key] for key in common]) < .2
    assert np.mean(list(new.values())) > .5  # Added harder matches do not alter the paired target set.


def test_separate_group_gates_cannot_be_laundered_by_a_large_safe_group(tmp_path):
    dataset, evaluation = tmp_path / "data", tmp_path / "evaluation"
    dataset.mkdir();evaluation.mkdir()
    box = {"center": [10, 0, .5], "size": [4, 2, 1], "yaw": 0}
    truth, before, after = [], [], []
    for group, count in [("large_safe", 100), ("small_bad", 1)]:
        for index in range(count):
            key = {"scene_id": group, "frame_id": str(index)}
            truth.append({**key, "gt_boxes": [box]})
            before.append({**key, "boxes": [{**box, "score": .9}]})
            boxes = [{**box, "score": .9}]
            if group == "small_bad": boxes.append({**box, "center": [40, 0, .5], "score": .8})
            after.append({**key, "boxes": boxes})
    for path, values in [(dataset/"val.jsonl", truth), (evaluation/"pretrained_predictions.jsonl", before),
                         (evaluation/"adapted_predictions.jsonl", after)]:
        path.write_text("".join(json.dumps(row)+"\n" for row in values))
    metadata = {"split_unit": "whole_groups", "input_frame_counts_by_group": {"val": {"large_safe": 100, "small_bad": 1}}}
    (dataset/"dataset.json").write_text(json.dumps(metadata))
    status = {"training_executed": False, "evaluation_only": True, "candidate_checkpoint_sha256": "0"*64,
              "source_dataset_sha256": hashlib.sha256((dataset/"dataset.json").read_bytes()).hexdigest()}
    (evaluation/"status.json").write_text(json.dumps(status))
    common = {"candidate_sha256": "0"*64, "operating_score_threshold": .35, "center_matching_threshold_m": 2.,
              "false_positives_per_frame": {"maximum_increase": .01, "absolute_maximum": .02},
              "center_error_common_matched_targets": {"adapted_mean_max_m": .5, "maximum_mean_degradation_m": .1},
              "far_range_m": [30, 50], "far_minimum_targets_for_additional_gate": 30}
    reports = {}
    for label, groups in [("large", ["large_safe"]), ("small", ["small_bad"]), ("aggregate", ["large_safe", "small_bad"])]:
        definition = tmp_path / (label+".json")
        definition.write_text(json.dumps({**common, "qualification_runs": groups}))
        reports[label] = qualify(dataset, evaluation, definition, evaluation/(label+"_gate.json"))
    assert reports["large"]["passed"]
    assert not reports["small"]["passed"]
    assert reports["aggregate"]["passed"]
    assert not all(reports[group]["passed"] for group in ("large", "small"))
