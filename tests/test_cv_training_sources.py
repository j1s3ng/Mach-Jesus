import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.build_sim_detector_dataset import build
from scripts.cv_training_sources import validate_export_training_sources
from scripts.training_sources import TrainingSourceError
from scripts.train_sim_detector import validate_evaluation_authority


def args(train, val, output):
    return SimpleNamespace(train_runs=[train], val_runs=[val], output=output,
                           diagnostic_only=False, drop_duplicate_validation=True,
                           stride=1, max_frames_per_run=0, positive_only=False,
                           min_points=1, max_range=50., training_releases=None)


def test_builder_exclusion_guard_runs_before_capture_reads(tmp_path):
    train = tmp_path/"cv_qualification";train.mkdir()
    (train/"config.json").write_text(json.dumps({"excluded_from_training": True}))
    output = tmp_path/"output"
    # No capture manifest exists: policy must reject before trying to read it.
    with pytest.raises(TrainingSourceError, match="excluded"):
        build(args(train, tmp_path/"validation", output))
    assert not output.exists()


def test_export_guard_binds_exact_archived_configs_before_point_arrays_open(tmp_path):
    train, val = tmp_path/"training", tmp_path/"development"
    for index, path in enumerate((train, val)):
        path.mkdir();(path/"config.json").write_text("{}")
        np.save(path/"cloud.npy", np.full((3, 4), index+1., dtype=np.float32), allow_pickle=False)
        entry = {"frame_id": "frame", "timestamp_ns": index+1, "points_file": "cloud.npy", "gt_boxes": []}
        (path/"manifest.jsonl").write_text(json.dumps(entry)+"\n")
    output = tmp_path/"export";build(args(train, val, output))
    assert validate_export_training_sources(output)["frames"] == 1
    archive = json.loads((output/"training_sources.json").read_text())
    # Even removing every cloud cannot hide a config-integrity failure.
    for cloud in (output/"points").glob("*.npy"): cloud.unlink()
    config = output/archive["sources"][0]["config_archive"]
    config.write_text(json.dumps({"excluded_from_training": True}))
    with pytest.raises(TrainingSourceError, match="hash mismatch"):
        validate_export_training_sources(output)


def test_reserved_evaluation_requires_explicit_authority_and_never_fitting(tmp_path):
    authority = tmp_path/"authority.json";authority.write_text("{}")
    metadata = {"validation_purpose": "qualification_only_no_fitting_or_development_selection",
                "qualification_authority_sha256": hashlib.sha256(authority.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="evaluation-only"):
        validate_evaluation_authority(metadata, None, authority)
    with pytest.raises(ValueError, match="authority"):
        validate_evaluation_authority(metadata, Path("candidate.pth"), None)
    validate_evaluation_authority(metadata, Path("candidate.pth"), authority)
