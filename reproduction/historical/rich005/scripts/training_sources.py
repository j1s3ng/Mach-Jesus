"""Fail-closed source eligibility checks shared by offline NN dataset builders.

Check eligibility before opening a trace, point manifest, or training bundle.
This reads only source-lineage ``config.json`` metadata, never observations or
labels. A historical release needs an exact run path, immutable config hash and
written authorization reason; there is no generic allow-sealed switch.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

POLICY_VERSION = "source_eligibility_v1"
SEALED_NAMES = ("formalcases", "bootstrap", "holdout", "confirmation", "cvqualification")


class TrainingSourceError(ValueError):
    """A source is reserved, ambiguous, or has changed since its release."""


def _normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _sealed(value):
    value = _normalized(value)
    return any(name in value for name in SEALED_NAMES)


def _metadata_violations(value, location="config"):
    violations = []
    if isinstance(value, dict):
        for key, item in value.items():
            where = f"{location}.{key}"
            if _normalized(key) == "excludedfromtraining":
                if item is not False:
                    violations.append(f"{where} must be the JSON boolean false (got {item!r})")
            elif _normalized(key) == "datasetsplit":
                if not isinstance(item, str) or not item.strip():
                    violations.append(f"{where} must be a nonempty split string")
                elif _sealed(item):
                    violations.append(f"{where}={item!r} is sealed")
            if isinstance(item, (dict, list)):
                violations.extend(_metadata_violations(item, where))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(_metadata_violations(item, f"{location}[{index}]"))
    return violations


def validate_training_metadata(metadata, *, context="metadata"):
    """Reject exclusion/split flags even if nested in a copied manifest."""
    if not isinstance(metadata, dict):
        raise TrainingSourceError(f"{context} must be a JSON object")
    violations = _metadata_violations(metadata, context)
    if violations:
        raise TrainingSourceError("; ".join(violations))
    return metadata


def load_training_releases(path):
    """Read an explicit immutable release manifest; callers must audit its hash."""
    path = Path(path)
    document = json.loads(path.read_text())
    if not isinstance(document, (dict, list)):
        raise TrainingSourceError("Training release manifest must be an object or list")
    rows = document if isinstance(document, list) else document.get("sources", document.get("releases", document.get("exceptions")))
    if not isinstance(rows, list):
        raise TrainingSourceError("Training release manifest must contain a sources list")
    return tuple(rows)


def _release_record(row):
    if not isinstance(row, dict):
        raise TrainingSourceError("A source release must be a JSON object")
    root = row.get("path", row.get("run_path", row.get("source_path")))
    digest = row.get("config_sha256")
    reason = row.get("reason")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise TrainingSourceError("A source release needs an absolute exact run path")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise TrainingSourceError("A source release needs an exact lowercase config_sha256")
    if not isinstance(reason, str) or not reason.strip():
        raise TrainingSourceError("A source release needs its authorization reason")
    return {"path": str(Path(root).resolve()), "config_sha256": digest, "reason": reason}


def validate_training_source(source, *, require_config=False, releases=()):
    """Return an auditable eligibility record, or raise before source data opens.

    Both lexical and symlink-resolved lineage are checked. Every ancestor config
    is inspected, so moving a file under an innocuous child directory or adding
    a child ``excluded_from_training=false`` cannot override a reserved parent.
    ``releases`` may contain only explicit ``{path, config_sha256, reason}``
    records. A release covers its exact immutable run config and descendants;
    it never overrides another excluded config in that lineage.
    """
    lexical = Path(os.path.abspath(os.fspath(source)))
    resolved = lexical.resolve()
    path_violations = [str(path) for path in (lexical, resolved) if any(_sealed(part) for part in path.parts)]
    configs = {}
    for path in (lexical, resolved):
        start = path if path.is_dir() else path.parent
        for parent in (start, *start.parents):
            candidate = parent / "config.json"
            if candidate.is_file():
                key = str(candidate.resolve())
                if key in configs:
                    continue
                raw = candidate.read_bytes()
                try:
                    metadata = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as error:
                    raise TrainingSourceError(f"Cannot validate malformed source config {candidate}") from error
                if not isinstance(metadata, dict):
                    raise TrainingSourceError(f"Source config must be an object: {candidate}")
                configs[key] = {"path": key, "sha256": hashlib.sha256(raw).hexdigest(),
                                "violations": _metadata_violations(metadata, key)}
    release = None
    for row in releases:
        candidate = _release_record(row)
        root = Path(candidate["path"])
        if not resolved.is_relative_to(root) or not lexical.is_relative_to(root):
            continue
        config = configs.get(str(root / "config.json"))
        if config is None or config["sha256"] != candidate["config_sha256"]:
            raise TrainingSourceError(f"Released source config is missing or changed: {root}")
        release = candidate
        break
    blocked_configs = [record for record in configs.values() if record["violations"]
                       and (release is None or record["path"] != str(Path(release["path"]) / "config.json"))]
    if blocked_configs or (path_violations and release is None):
        details = [f"sealed path: {path}" for path in path_violations] if release is None else []
        details.extend(item for record in blocked_configs for item in record["violations"])
        raise TrainingSourceError("Training source excluded; " + "; ".join(details))
    if require_config and not configs:
        raise TrainingSourceError(f"Measured training source needs lineage config.json: {source}")
    return {"policy_version": POLICY_VERSION, "source": str(resolved),
            "lineage_configs": [{"path": row["path"], "sha256": row["sha256"]} for row in configs.values()],
            "release": release}


def validate_frozen_training_source(source_path, frozen_config, *, expected_config_sha256, releases=()):
    """Validate a portable archived run config without opening the original run.

    ``source_path`` is the canonical original absolute *run directory* recorded
    by the local builder. ``frozen_config`` is an exact config-file path or its
    raw bytes, not reserialized JSON. The local builder must first validate the
    complete live lineage with :func:`validate_training_source` and archive that
    audit beside this config. CUDA hosts can then enforce its metadata/hash and
    canonical source name without access to the Mac's original capture folder.
    """
    if not isinstance(source_path, (str, Path)) or not Path(source_path).is_absolute():
        raise TrainingSourceError("Frozen source_path must be an absolute canonical run directory")
    source = Path(os.path.normpath(os.fspath(source_path)))
    raw = frozen_config if isinstance(frozen_config, bytes) else Path(frozen_config).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not isinstance(expected_config_sha256, str) or digest != expected_config_sha256:
        raise TrainingSourceError(f"Frozen source config hash mismatch: {source}")
    try:
        metadata = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise TrainingSourceError(f"Malformed frozen source config: {source}") from error
    if not isinstance(metadata, dict):
        raise TrainingSourceError(f"Frozen source config must be an object: {source}")
    violations = _metadata_violations(metadata, str(source / "config.json"))
    release = None
    for row in releases:
        candidate = _release_record(row)
        if candidate["path"] == str(source):
            if candidate["config_sha256"] != digest:
                raise TrainingSourceError(f"Released frozen source config changed: {source}")
            release = candidate
            break
    if release is None and (violations or any(_sealed(part) for part in source.parts)):
        raise TrainingSourceError("Frozen training source excluded; " + "; ".join(violations or [f"sealed path: {source}"]))
    return {"policy_version": POLICY_VERSION, "source": str(source), "config_sha256": digest,
            "frozen_config": True, "release": release}
