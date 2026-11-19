"""Portable, hash-bound source eligibility audits for measured CV exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.training_sources import (TrainingSourceError, load_training_releases,
                                     validate_frozen_training_source,
                                     validate_training_metadata, validate_training_source)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def releases(path):
    return load_training_releases(path) if path else ()


def archive_training_sources(dataset, release_path=None):
    """Validate full live lineage, then archive exact configs without edits."""
    dataset = Path(dataset)
    target = dataset/"training_sources.json"
    if target.exists(): raise ValueError("Existing source eligibility audit is immutable")
    metadata = json.loads((dataset/"dataset.json").read_text())
    validate_training_metadata(metadata, context="CV export metadata")
    rows = [json.loads(line) for line in (dataset/"train.jsonl").read_text().splitlines() if line.strip()]
    if not rows or any(not row.get("source_run") for row in rows):
        raise TrainingSourceError("Every training frame needs an original source_run")
    allowed = releases(release_path)
    records = []
    for source in sorted({row["source_run"] for row in rows}):
        run = Path(source)
        audit = validate_training_source(run, require_config=True, releases=allowed)
        config = run/"config.json"
        if not config.is_file(): raise TrainingSourceError("Measured CV source requires its exact run config")
        digest = sha(config)
        expected = metadata.get("source_file_sha256", {}).get(str(config))
        if expected != digest: raise TrainingSourceError("Run config differs from captured export provenance")
        archived = []
        for lineage in audit["lineage_configs"]:
            path = Path(lineage["path"]);raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != lineage["sha256"]: raise TrainingSourceError("Live lineage changed while archiving")
            relative = "source_configs/"+lineage["sha256"]+".json"
            destination = dataset/relative;destination.parent.mkdir(exist_ok=True)
            if destination.exists() and destination.read_bytes() != raw: raise TrainingSourceError("Config archive hash collision")
            if not destination.exists(): destination.write_bytes(raw)
            archived.append({**lineage, "archive": relative})
        records.append({"source_path": str(run.resolve()), "config_sha256": digest,
                        "config_archive": "source_configs/"+digest+".json", "live_lineage_audit": audit,
                        "archived_lineage_configs": archived})
    result = {"dataset_metadata_sha256": sha(dataset/"dataset.json"), "train_manifest_sha256": sha(dataset/"train.jsonl"),
              "release_manifest_sha256": sha(release_path) if release_path else None,
              "sources": records, "policy": "complete live lineage audited before portable archive; no observations added"}
    target.write_text(json.dumps(result, indent=2)+"\n")
    validate_export_training_sources(dataset, release_path)
    return result


def validate_export_training_sources(dataset, release_path=None):
    """Fail before point arrays open if copied configs, roles or lineage differ."""
    dataset = Path(dataset)
    metadata = json.loads((dataset/"dataset.json").read_text())
    validate_training_metadata(metadata, context="CV export metadata")
    path = dataset/"training_sources.json"
    if not path.is_file(): raise TrainingSourceError("CV export needs a portable training_sources.json audit")
    archive = json.loads(path.read_text())
    if archive["dataset_metadata_sha256"] != sha(dataset/"dataset.json") or archive["train_manifest_sha256"] != sha(dataset/"train.jsonl"):
        raise TrainingSourceError("Dataset changed after source eligibility audit")
    if archive["release_manifest_sha256"] != (sha(release_path) if release_path else None):
        raise TrainingSourceError("Use the exact authorized immutable source release manifest")
    allowed, audits = releases(release_path), []
    for record in archive["sources"]:
        captured = metadata.get("source_file_sha256", {}).get(str(Path(record["source_path"])/"config.json"))
        if captured != record["config_sha256"]:
            raise TrainingSourceError("Archived source config differs from captured provenance")
        config = (dataset/record["config_archive"]).resolve()
        if not config.is_relative_to(dataset.resolve()): raise TrainingSourceError("Archived config escapes dataset")
        audit = validate_frozen_training_source(record["source_path"], config,
                    expected_config_sha256=record["config_sha256"], releases=allowed)
        lineage = record["live_lineage_audit"]
        if lineage["source"] != record["source_path"] or not record["archived_lineage_configs"]:
            raise TrainingSourceError("Portable archive lacks complete original lineage audit")
        expected_lineage = {(entry["path"], entry["sha256"]) for entry in lineage["lineage_configs"]}
        actual_lineage = {(entry["path"], entry["sha256"]) for entry in record["archived_lineage_configs"]}
        if expected_lineage != actual_lineage or (str(Path(record["source_path"])/"config.json"), record["config_sha256"]) not in expected_lineage:
            raise TrainingSourceError("Portable source lineage archive is incomplete")
        for entry in record["archived_lineage_configs"]:
            frozen = (dataset/entry["archive"]).resolve()
            if not frozen.is_relative_to(dataset.resolve()) or sha(frozen) != entry["sha256"]:
                raise TrainingSourceError("Archived source lineage config changed")
            is_released_root = audit["release"] and entry["path"] == str(Path(audit["release"]["path"])/"config.json")
            if not is_released_root: validate_training_metadata(json.loads(frozen.read_text()), context=entry["path"])
        audits.append(audit)
    rows = [json.loads(line) for line in (dataset/"train.jsonl").read_text().splitlines() if line.strip()]
    if {row.get("source_run") for row in rows} != {record["source_path"] for record in archive["sources"]}:
        raise TrainingSourceError("Training rows differ from audited original source set")
    for row in rows: validate_training_metadata(row, context="CV training frame")
    return {"source_audit_sha256": sha(path), "release_manifest_sha256": archive["release_manifest_sha256"],
            "dataset_metadata_sha256": archive["dataset_metadata_sha256"], "train_manifest_sha256": archive["train_manifest_sha256"],
            "sources": audits, "frames": len(rows)}
