import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iteration", ROOT/"scripts/assemble_detector_iteration.py")
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def export(root, split, samples):
    root.mkdir();(root/"points").mkdir();rows = []
    for name, scene, value in samples:
        point = root/"points"/(name+".npy")
        np.save(point, np.full((3, 4), value, dtype=np.float32), allow_pickle=False)
        rows.append({"frame_id": name, "scene_id": scene, "points_file": "points/"+name+".npy",
                     "points_sha256": hashlib.sha256(point.read_bytes()).hexdigest(), "gt_boxes": []})
    manifest = root/(split+".jsonl");manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    metadata = {"min_measured_points_per_box": 3, "evaluation_radius_m": 50., "point_in_box_tolerance_m": .03,
                "ground_origin": "ego ground", "timestamp_join": "exact", "source_file_sha256": {},
                "manifest_sha256": {split: hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    (root/"dataset.json").write_text(json.dumps(metadata))


def test_explicit_role_change_preserves_sources_links_points_and_removes_only_input_duplicates(tmp_path):
    old, promoted, dev = tmp_path/"old", tmp_path/"former-qualification", tmp_path/"development"
    export(old, "train", [("a", "old", 1.)]);export(promoted, "val", [("b", "released", 2.)])
    export(dev, "val", [("c", "dev", 2.), ("d", "dev", 3.)])
    original = (promoted/"val.jsonl").read_bytes()
    protocol = tmp_path/"protocol.json";protocol.write_text(json.dumps({"training_groups": ["old", "released"], "development_groups": ["dev"], "role_change_history": "explicit"}))
    output = tmp_path/"new"
    result = module.assemble([(old, "train"), (promoted, "val")], (dev, "val"), protocol, output)
    assert result["splits"]["train"]["frames"] == 2 and result["splits"]["val"]["frames"] == 1
    assert result["excluded_duplicate_validation_frames"][0]["frame_id"] == "c"
    assert (output/"points/b.npy").stat().st_ino == (promoted/"points/b.npy").stat().st_ino
    assert (promoted/"val.jsonl").read_bytes() == original
    assert result["assembly_protocol"]["role_change_history"] == "explicit"
    with pytest.raises(ValueError, match="fresh output"):
        module.assemble([(old, "train"), (promoted, "val")], (dev, "val"), protocol, output)
