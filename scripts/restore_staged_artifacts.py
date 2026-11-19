#!/usr/bin/env python3
"""Restore verified companion artifacts locally; no downloading or overwrites."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def restore(staging, root):
    staging, root = Path(staging).resolve(), Path(root).resolve()
    expected = {row["role"]: row for row in json.loads((root / "artifacts/downloads.json").read_text())["artifacts"]}
    supplied = json.loads((staging / "manifest.json").read_text())["files"]
    if {row["role"] for row in supplied} != set(expected) or len(supplied) != len(expected):
        raise ValueError("Companion artifact roles differ from the source manifest")
    operations = []
    for row in supplied:
        reference = expected[row["role"]]
        if row["sha256"] != reference["sha256"] or row["original_project_relative_path"] != reference["source"]:
            raise ValueError("Companion artifact identity/path differs")
        filename = Path(row["file"])
        relative = Path(reference["source"])
        if filename.name != row["file"] or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact path escapes its permitted root")
        source, target = (staging / filename).resolve(), (root / relative).resolve()
        if not source.is_relative_to(staging) or not target.is_relative_to(root):
            raise ValueError("Artifact symlink/path escapes its permitted root")
        if digest(source) != reference["sha256"]:
            raise ValueError("Companion artifact SHA256 differs")
        if target.exists() and digest(target) != reference["sha256"]:
            raise ValueError("Different existing artifact will not be overwritten")
        operations.append((source, target, reference["sha256"]))
    # No writes occur until every role, path, source hash and existing target
    # has passed. This protects old results even if a later source is corrupt.
    restored = []
    for source, target, sha in operations:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if digest(target) != sha:
            raise ValueError("Restored artifact verification failed")
        restored.append(target.relative_to(root).as_posix())
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps({"verified_and_available": restore(args.staging, args.root), "downloads": False}, indent=2))


if __name__ == "__main__": main()
