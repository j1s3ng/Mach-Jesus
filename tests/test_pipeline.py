"""Functional tests of recording boundaries and custom evaluation semantics."""

import json
from pathlib import Path

import numpy as np
import pytest

from race_lidar.baseline import detect
from race_lidar.demo import generate_demo
from race_lidar.io import COORDINATE_FRAME, read_frame, read_manifest, write_frame
from race_lidar.metrics import evaluate_frames
from race_lidar.replay import render_bev, replay


def box(x=10, y=0, score=.9):
    return dict(center=[x, y, 0], size=[4, 2, 1.5], yaw=0, score=score)


def test_frame_round_trip_and_coordinates(tmp_path):
    points = np.array([[10, 2, 0, .7], [20, -3, 1, .2]], dtype=np.float32)
    image = np.zeros((10, 20, 3), np.uint8)
    image[0, 0] = [255, 2, 3]
    write_frame(tmp_path, "0001", points, 100, "scene", [box()], image)
    entries = read_manifest(tmp_path)
    assert entries[0]["coordinate_frame"] == COORDINATE_FRAME
    frame = read_frame(entries[0], tmp_path)
    np.testing.assert_array_equal(points, frame["points"])
    np.testing.assert_array_equal(image, frame["image"])
    assert frame["gt_boxes"][0]["center"] == [10, 0, 0]
    with pytest.raises(ValueError, match="Duplicate"):
        write_frame(tmp_path, "0001", points, 101, "scene")


def test_paths_and_coordinate_validation(tmp_path):
    write_frame(tmp_path, "a", np.empty((0, 4)), 0, "scene")
    entry = read_manifest(tmp_path)[0]
    with pytest.raises(ValueError, match="coordinate"):
        read_frame(dict(entry, coordinate_frame="camera_z_forward"), tmp_path)
    with pytest.raises(ValueError, match="inside"):
        read_frame(dict(entry, points_file="../escape.npy"), tmp_path)


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.zeros((4,)), np.full((2, 4), np.nan), [["x"] * 4]])
def test_malformed_clouds_rejected(bad):
    with pytest.raises(ValueError):
        detect(bad)


def test_empty_cloud():
    assert detect(np.empty((0, 4), dtype=np.float32)) == []


def test_demo_deterministic_and_no_ground_truth_leakage(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    generate_demo(first, frames=2, seed=7)
    generate_demo(second, frames=2, seed=7)
    entry = read_manifest(first)[0]
    original = read_frame(entry, first)
    other = read_frame(read_manifest(second)[0], second)
    np.testing.assert_array_equal(original["points"], other["points"])
    assert len(original["points"]) > 1000
    assert len(original["gt_boxes"]) >= 2
    assert np.median(original["points"][:, 2]) == pytest.approx(0.0, abs=.02)
    assert original["gt_boxes"][0]["center"][2] == pytest.approx(.75)
    assert all(b["center"][2] - b["size"][2] / 2 >= -1e-6 for b in original["gt_boxes"])
    metadata = json.loads((first / "sensor_metadata.json").read_text())
    assert metadata["coordinate_origin"] == "ego_vehicle_reference_at_ground"
    assert metadata["lidar_to_ego_translation_m"] == [0., 0., 1.4]
    predictions = detect(original["points"])
    assert predictions
    original["gt_boxes"] = [box(1000)]
    assert detect(original["points"]) == predictions
    assert all(p["score_type"] == "uncalibrated_geometric_heuristic" for p in predictions)


def test_metrics_one_to_one_false_positives_and_scenes():
    labels = [dict(scene_id="a", frame_id="1", gt_boxes=[box()]),
              dict(scene_id="b", frame_id="1", gt_boxes=[box(20)])]
    predictions = [dict(scene_id="a", frame_id="1", boxes=[box(), box(score=.8)]),
                   dict(scene_id="b", frame_id="1", boxes=[box()])]
    result = evaluate_frames(predictions, labels)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 1
    assert result["recall"] == .5
    assert result["metric_type"].endswith("non_official")


def test_metrics_empty_annotations_and_score_ranking():
    labels = [dict(scene_id="a", frame_id="1", gt_boxes=[box()]),
              dict(scene_id="a", frame_id="2", gt_boxes=[])]
    predictions = [dict(scene_id="a", frame_id="1", boxes=[box()]),
                   dict(scene_id="a", frame_id="2", boxes=[box(score=.99)])]
    result = evaluate_frames(predictions, labels)
    assert result["false_positives"] == 1
    assert result["ap_101"] == pytest.approx(.5)
    assert evaluate_frames([], []) ["ap_101"] is None
    unannotated = evaluate_frames([dict(scene_id="a", frame_id="3", boxes=[box()])], labels)
    assert unannotated["excluded_unannotated_prediction_frames"] == 1


def test_heading_error_wraps_and_uses_only_center_matches():
    target = dict(box(), yaw=np.pi - .05)
    prediction = dict(box(), yaw=-np.pi + .05)
    predictions = [dict(scene_id="a", frame_id="1", boxes=[prediction, dict(box(100), yaw=1.)])]
    labels = [dict(scene_id="a", frame_id="1", gt_boxes=[target])]
    result = evaluate_frames(predictions, labels)
    assert result["matched_heading_error_mean_rad"] == pytest.approx(.1)
    assert result["matched_heading_error_p95_rad"] == pytest.approx(.1)
    flipped = evaluate_frames([dict(scene_id="a", frame_id="1", boxes=[dict(target, yaw=target["yaw"] + np.pi)])], labels)
    assert flipped["matched_heading_error_mean_rad"] == pytest.approx(np.pi)
    empty = evaluate_frames([], labels)
    assert empty["matched_heading_error_mean_rad"] is None
    assert empty["matched_heading_error_p95_rad"] is None


def test_replay_artifacts(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    generate_demo(source, frames=3)
    summary = replay(source, output, make_video=False)
    assert summary["frames"] == 3
    assert summary["inference_ms_mean"] > 0
    assert (output / "contact_sheet.jpg").stat().st_size > 0
    assert len(list((output / "bev").glob("*.png"))) == 3
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert "gt_boxes" not in rows[0]
    assert summary["evaluation"]["ground_truth_objects"] > 0


def test_bev_ego_left_is_image_left():
    image = render_bev(np.array([[10, 10, 1, 1]], np.float32), [], size=(960, 720))
    # x=10 m forward, y=10 m left must appear left of the image center.
    pixels = np.asarray(image)
    assert np.any(np.all(pixels[550:600, 350:420] == [155, 190, 175], axis=-1))
