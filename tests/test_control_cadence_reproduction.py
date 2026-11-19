"""Frozen saved-motion parity and meaningful input-boundary rejection tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


FOLDER = Path(__file__).resolve().parents[1] / "reproduction/control-cadence"
SPEC = importlib.util.spec_from_file_location("control_cadence_reproduction_tested", FOLDER / "reproduce.py")
cadence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cadence)


@pytest.fixture(scope="module")
def bundle():
    return cadence.load_bundle(FOLDER)


def first_arrays(bundle):
    cases, arrays, *_ = bundle
    return {key: value.copy() for key, value in arrays[cases[0]["id"]].items()}, cases[0]


def test_all_three_frozen_motion_reports_and_costs_match():
    result = cadence.reproduce(FOLDER)
    assert result["status"] == "VERIFIED_SAVED_MOTION_CPU_REPRODUCTION"
    assert result["case_ids"] == ["solo_20ms", "solo_10ms", "solo_5ms"]
    assert result["recorded_50hz_samples"] == 27_801
    assert result["maximum_frozen_float_absolute_error"] <= 1e-12
    assert result["cases"]["solo_20ms"]["timing"]["first_reference_crossing_scalar_s"] == pytest.approx(188.78722571100607, abs=1e-10)
    assert result["cases"]["solo_10ms"]["frozen_metrics"]["body"]["XY_difference_body_lateral_jerk_rms_mps3"] == pytest.approx(10.874451800465152, abs=1e-10)
    assert result["cases"]["solo_5ms"]["wall_cost_ratio_vs_20ms"] == pytest.approx(2.992509452074287, abs=1e-12)
    assert all(case["outcome"] == "complete_first_circuit" and case["anchors_within_half_metre"] == 18 for case in result["cases"].values())
    assert not result["new_simulation_inference_or_training"]
    assert not result["convergence_claimed"] and not result["promotion_claimed"]


def test_short_terminal_pose_brackets_never_enter_metric_samples(bundle):
    cases, arrays, methods, *_ = bundle
    for case in cases:
        data = arrays[case["id"]]
        assert not data["terminal_sample"].any()
        timing = cadence.validate_arrays(data, case)
        assert 0 < timing["bracket_duration_s"] <= .0200001
        assert timing["interpolated_metric_samples"] == 0
        assert data["bracket_time_s"][1] > data["time_s"][-1]
        metrics, _ = cadence.compute_metrics(data, methods)
        assert metrics["steering"]["sample_count"] == len(data["time_s"])


def test_compressed_saved_measurements_remain_under_five_megabytes():
    assert sum(path.stat().st_size for path in FOLDER.glob("*.npz")) < 5_000_000


def test_npz_tampering_is_rejected_before_scoring(tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(FOLDER, copied)
    with (copied / "solo_20ms.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum changed"):
        cadence.load_bundle(copied)


def test_manifest_cannot_drop_a_recorded_condition(tmp_path):
    manifest = json.loads((FOLDER / "manifest.json").read_text())
    manifest["case_ids"].pop()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="three-case schema"):
        cadence.load_bundle(tmp_path)


def test_saved_case_metadata_cannot_redirect_array_paths(bundle):
    cases, *_ = bundle
    changed = json.loads(json.dumps(cases))
    changed[0]["array_file"] = "../outside.npz"
    with pytest.raises(ValueError, match="array filename"):
        cadence.validate_case_metadata(changed)


def test_saved_case_metadata_rejects_missing_fields(bundle):
    cases, *_ = bundle
    changed = json.loads(json.dumps(cases))
    changed[0].pop("wall_seconds")
    with pytest.raises(ValueError, match="saved-case metadata"):
        cadence.validate_case_metadata(changed)


def test_saved_conditions_must_share_the_same_physical_reference(bundle):
    cases, *_ = bundle
    changed = json.loads(json.dumps(cases))
    changed[1]["physical_reference_length_m"] += 1.
    with pytest.raises(ValueError, match="one physical reference"):
        cadence.validate_case_metadata(changed)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_native_array_schema_is_exact(bundle, mutation):
    data, case = first_arrays(bundle)
    if mutation == "missing":
        data.pop("progress_m")
    else:
        data["invented_sample"] = np.zeros(case["sample_count"])
    with pytest.raises(ValueError, match="Missing or extra"):
        cadence.validate_arrays(data, case)


def test_nonfinite_measurement_is_rejected(bundle):
    data, case = first_arrays(bundle)
    data["yaw_rad"][20] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        cadence.validate_arrays(data, case)


def test_nonuniform_timestamps_are_not_resampled(bundle):
    data, case = first_arrays(bundle)
    data["time_s"][20] += .001
    with pytest.raises(ValueError, match="uniform 50 Hz"):
        cadence.validate_arrays(data, case)


def test_precision_and_shape_are_not_silently_coerced(bundle):
    data, case = first_arrays(bundle)
    data["time_s"] = data["time_s"].astype(np.float32)
    with pytest.raises(ValueError, match="native float array"):
        cadence.validate_arrays(data, case)
    data, case = first_arrays(bundle)
    data["position_xy_m"] = np.column_stack((data["position_xy_m"], np.zeros(case["sample_count"])))
    with pytest.raises(ValueError, match="actual XY"):
        cadence.validate_arrays(data, case)


def test_terminal_command_injection_is_rejected(bundle):
    data, case = first_arrays(bundle)
    data["terminal_sample"][-1] = True
    with pytest.raises(ValueError, match="Terminal command/body"):
        cadence.validate_arrays(data, case)


def test_unbracketed_crossing_is_not_clamped_or_extrapolated(bundle):
    data, case = first_arrays(bundle)
    data["bracket_progress_m"][1] = data["bracket_progress_m"][0]
    with pytest.raises(ValueError, match="do not bracket"):
        cadence.validate_arrays(data, case)


def test_pose_bracket_cannot_skip_a_recording_interval(bundle):
    data, case = first_arrays(bundle)
    data["bracket_time_s"][1] += .02
    with pytest.raises(ValueError, match="no longer than one recording interval"):
        cadence.validate_arrays(data, case)


def test_import_and_scoring_do_not_import_torch():
    program = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('isolated_cadence', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.reproduce()
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", program, str(FOLDER / "reproduce.py")], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_cli_requires_fresh_output_and_preserves_previous_results(tmp_path):
    output = tmp_path / "result"
    command = [sys.executable, str(FOLDER / "reproduce.py"), "--output-dir", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    before = (output / "result.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and "fresh output directory" in second.stderr
    assert (output / "result.json").read_bytes() == before
    assert {p.name for p in output.iterdir()} == {"result.json", "summary.csv"}


def test_public_provenance_has_no_private_path_or_service_identity():
    forbidden = ("/Users/", "/home/", "127.0.0.1", "GPU-")
    for name in ("manifest.json", "provenance.json", "case_metadata.json", "expected_reports.json"):
        content = (FOLDER / name).read_text()
        assert not any(token in content for token in forbidden)
