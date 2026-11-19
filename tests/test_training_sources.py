import hashlib
import json

import pytest
import numpy as np

from scripts.training_sources import (TrainingSourceError, load_training_releases,
                                      validate_frozen_training_source, validate_training_metadata,
                                      validate_training_source)


def config(run, **metadata):
    run.mkdir(parents=True, exist_ok=True)
    path = run / "config.json"
    path.write_text(json.dumps(metadata))
    return path


@pytest.mark.parametrize("split", ["holdout", "CONFIRMATION", "bootstrap", "cv_qualification",
                                  "CV-Qualification", "hold_out", "formal_cases", "Formal Cases / seed2",
                                  "championship", "geometric-confirmation"])
def test_reserved_split_cannot_be_renamed_to_an_innocuous_path(tmp_path, split):
    run = tmp_path / "engineering/safe-looking-name"
    config(run, dataset_split=split, excluded_from_training=False)
    with pytest.raises(TrainingSourceError, match="excluded"):
        validate_training_source(run)


@pytest.mark.parametrize("flag", [True, "true", "false", 0, 1, None, [], {}])
def test_exclusion_metadata_is_fail_closed_unless_exact_json_false(tmp_path, flag):
    run = tmp_path / "engineering/scene"
    config(run, metadata={"provenance": [{"excluded_from_training": flag}]})
    with pytest.raises(TrainingSourceError, match="boolean false"):
        validate_training_source(run)


def test_child_config_cannot_override_excluded_ancestor(tmp_path):
    parent = tmp_path / "engineering/scene"
    config(parent, excluded_from_training=True)
    child = parent / "cars/R01"
    config(child, excluded_from_training=False, dataset_split="engineering")
    with pytest.raises(TrainingSourceError, match="excluded_from_training"):
        validate_training_source(child / "policy_actions.jsonl")


def test_both_symlink_target_and_lexical_alias_are_checked(tmp_path):
    reserved = tmp_path / "HoLd-Out/scene"
    safe = tmp_path / "engineering/scene"
    config(reserved, dataset_split="engineering")
    config(safe, dataset_split="engineering")
    alias = tmp_path / "engineering/alias"
    alias.symlink_to(reserved, target_is_directory=True)
    alias2 = tmp_path / "confirmation-alias"
    alias2.symlink_to(safe, target_is_directory=True)
    for path in (alias, alias2):
        with pytest.raises(TrainingSourceError, match="sealed path"):
            validate_training_source(path)


def test_malformed_or_missing_required_config_rejects_before_data_open(tmp_path):
    run = tmp_path / "engineering/scene"
    run.mkdir(parents=True)
    with pytest.raises(TrainingSourceError, match="needs lineage"):
        validate_training_source(run, require_config=True)
    (run / "config.json").write_text("not json")
    with pytest.raises(TrainingSourceError, match="malformed"):
        validate_training_source(run)


def test_exact_release_is_audited_and_changed_config_or_sibling_cannot_use_it(tmp_path):
    run = tmp_path / "bootstrap/field4-r0"
    path = config(run, dataset_split="bootstrap", excluded_from_training=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    release = {"source_path": str(run), "config_sha256": digest, "reason": "Explicit historical development release"}
    manifest = tmp_path / "releases.json"
    manifest.write_text(json.dumps({"exceptions": [release]}))
    releases = load_training_releases(manifest)
    audit = validate_training_source(run / "cars/R01/policy_actions.jsonl", releases=releases)
    assert audit["release"]["reason"] == release["reason"]
    assert audit["lineage_configs"][0]["sha256"] == digest
    sibling = tmp_path / "bootstrap/new-formal-scene"
    config(sibling, dataset_split="bootstrap", excluded_from_training=True)
    with pytest.raises(TrainingSourceError):
        validate_training_source(sibling, releases=releases)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(TrainingSourceError, match="changed"):
        validate_training_source(run, releases=releases)


def test_release_does_not_override_a_new_excluded_child(tmp_path):
    run = tmp_path / "bootstrap/field4-r0"
    path = config(run, excluded_from_training=True)
    release = {"path": str(run), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reason": "Root authorization"}
    child = run / "cars/R01"
    config(child, excluded_from_training=True)
    with pytest.raises(TrainingSourceError, match="R01"):
        validate_training_source(child, releases=[release])


def test_portable_snapshot_checks_canonical_original_path_and_exact_bytes(tmp_path):
    source = "/unmounted/mac/engineering/scene"
    raw = b'{"dataset_split":"engineering","excluded_from_training":false}\n'
    digest = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "frozen_config.json"
    path.write_bytes(raw)
    for frozen in (raw, path):
        audit = validate_frozen_training_source(source, frozen, expected_config_sha256=digest)
        assert audit["source"] == source
        assert audit["config_sha256"] == digest
    with pytest.raises(TrainingSourceError, match="hash mismatch"):
        validate_frozen_training_source(source, raw + b"\n", expected_config_sha256=digest)
    with pytest.raises(TrainingSourceError, match="excluded"):
        validate_frozen_training_source("/unmounted/mac/cv_qualification/scene", raw, expected_config_sha256=digest)


def test_portable_release_requires_exact_original_run_and_config(tmp_path):
    source = "/unmounted/mac/bootstrap/field4-r0"
    raw = b'{"excluded_from_training":true,"dataset_split":"bootstrap"}'
    digest = hashlib.sha256(raw).hexdigest()
    release = {"source_path": source, "config_sha256": digest, "reason": "Explicit engineering-only grandfathering"}
    audit = validate_frozen_training_source(source, raw, expected_config_sha256=digest, releases=[release])
    assert audit["release"]["config_sha256"] == digest
    with pytest.raises(TrainingSourceError, match="excluded"):
        validate_frozen_training_source(source + "-fresh", raw, expected_config_sha256=digest, releases=[release])


def test_copied_bundle_metadata_must_not_hide_nested_exclusion_flags():
    with pytest.raises(TrainingSourceError, match="sealed"):
        validate_training_metadata({"source_configs": [{"dataset_split": "cv qualification"}]})
    assert validate_training_metadata({"dataset_split": "engineering", "excluded_from_training": False})


@pytest.mark.parametrize("module_name,function_name", [("scripts.train_driver", "build_dataset"),
                                                       ("scripts.train_race_driver", "prepare")])
def test_legacy_trace_builders_reject_before_opening_observations(tmp_path, module_name, function_name):
    import importlib
    module = importlib.import_module(module_name)
    run = tmp_path / "innocent-looking-scene"
    config(run, dataset_split="confirmation", excluded_from_training=True)
    trace = run / "control_trace.jsonl"
    trace.write_text("not valid JSON; must never be parsed")
    arguments = [trace, tmp_path / "reference.npz"]
    if function_name == "prepare":
        arguments.append(tmp_path / "output")
    with pytest.raises(TrainingSourceError, match="excluded"):
        getattr(module, function_name)(*arguments)


def test_legacy_bundle_exclusion_is_checked_before_feature_arrays_load(tmp_path):
    from scripts.train_race_driver import train
    bundle = tmp_path / "harmless-name.npz"
    np.savez(bundle, metadata_json=json.dumps({"dataset_split": "holdout"}),
             train_x=np.array([{"invalid": "object arrays must not load"}], dtype=object))
    with pytest.raises(TrainingSourceError, match="sealed"):
        train(bundle, tmp_path / "output", [7], 1, 128, "cpu")


def test_legacy_bundle_original_source_path_survives_renaming(tmp_path):
    from scripts.train_race_driver import train
    bundle = tmp_path / "renamed-bundle.npz"
    np.savez(bundle, metadata_json=json.dumps({"source_trace": "/unmounted/holdout/trace.jsonl"}))
    with pytest.raises(TrainingSourceError, match="sealed path"):
        train(bundle, tmp_path / "output", [7], 1, 128, "cpu")
