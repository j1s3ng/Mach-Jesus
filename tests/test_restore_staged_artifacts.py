import hashlib
import json

import pytest

from scripts.restore_staged_artifacts import restore


def fixture(tmp_path):
    root, staging = tmp_path / "repository", tmp_path / "artifacts"
    (root / "artifacts").mkdir(parents=True);staging.mkdir()
    raw = b"small verified checkpoint fixture"
    sha = hashlib.sha256(raw).hexdigest()
    reference = {"role": "test", "source": "runs/test/model.bin", "sha256": sha}
    (root / "artifacts/downloads.json").write_text(json.dumps({"artifacts": [reference]}))
    (staging / "model.bin").write_bytes(raw)
    row = {"role": "test", "file": "model.bin", "original_project_relative_path": reference["source"], "sha256": sha}
    (staging / "manifest.json").write_text(json.dumps({"files": [row]}))
    return root, staging, raw


def test_valid_restore_is_idempotent(tmp_path):
    root, staging, raw = fixture(tmp_path)
    assert restore(staging, root) == ["runs/test/model.bin"]
    assert restore(staging, root) == ["runs/test/model.bin"]
    assert (root / "runs/test/model.bin").read_bytes() == raw


def test_corrupt_source_copies_nothing(tmp_path):
    root, staging, _ = fixture(tmp_path)
    (staging / "model.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        restore(staging, root)
    assert not (root / "runs").exists()


def test_existing_different_result_is_preserved(tmp_path):
    root, staging, _ = fixture(tmp_path)
    (root / "runs/test").mkdir(parents=True)
    target = root / "runs/test/model.bin";target.write_bytes(b"original result")
    with pytest.raises(ValueError, match="overwritten"):
        restore(staging, root)
    assert target.read_bytes() == b"original result"


def test_target_symlink_escape_is_refused(tmp_path):
    root, staging, _ = fixture(tmp_path)
    outside = tmp_path / "outside";outside.mkdir()
    (root / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        restore(staging, root)
    assert not list(outside.iterdir())
