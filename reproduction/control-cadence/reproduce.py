#!/usr/bin/env python3
"""CPU reproduction of three sealed, saved 50 Hz motion measurements."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parent
CASE_IDS = ("solo_20ms", "solo_10ms", "solo_5ms")
STEPS = (20, 10, 5)
FLOAT_FIELDS = ("time_s", "world_time_s", "yaw_rad", "steering_left_rad", "measured_yaw_rate_rps",
                "progress_m", "speed_mps", "selected_path_cross_error_m")
ARRAY_KEYS = {*FLOAT_FIELDS, "position_xy_m", "steering_lateral_budget_limited", "steering_method",
              "geometry_source", "terminal_sample", "bracket_time_s", "bracket_world_time_s",
              "bracket_progress_m", "bracket_position_m"}
CASE_FIELDS = {"id", "physics_control_step_ms", "control_hz", "recording_hz", "array_file", "sample_count",
               "start_world_s", "physical_reference_length_m", "terminal_rows_in_metric_arrays", "after_bracket_pose_is_terminal",
               "wall_seconds", "first_reference_crossing_scalar_s", "median_filter_samples", "native_safety_clean",
               "observed_map_anchors", "anchors_within_half_metre", "maximum_anchor_error_m", "outcome"}
REL_TOL = ABS_TOL = 1e-12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def check_equal(actual, expected, where="metrics"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Parity schema differs: " + where)
        for key in expected:
            check_equal(actual[key], expected[key], where + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "Parity list differs: " + where)
        for i, (a, b) in enumerate(zip(actual, expected)):
            check_equal(a, b, f"{where}[{i}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), "Frozen numeric parity differs: " + where)
    else:
        require(type(actual) is type(expected) and actual == expected, "Frozen value differs: " + where)


def max_float_error(actual, expected):
    if isinstance(expected, dict):
        return max((max_float_error(actual[key], value) for key, value in expected.items()), default=0.)
    if isinstance(expected, list):
        return max((max_float_error(a, b) for a, b in zip(actual, expected)), default=0.)
    return abs(actual - expected) if isinstance(expected, float) else 0.


def validate_arrays(data, case):
    require(set(data) == ARRAY_KEYS, "Missing or extra native-array fields")
    require(all(isinstance(value, np.ndarray) for value in data.values()), "Native fields must be NumPy arrays")
    n = case["sample_count"]
    require(type(n) is int and 64 <= n <= 100_000, "Invalid native sample count")
    for key in FLOAT_FIELDS:
        a = data[key]
        require(a.dtype == np.dtype("float64") and a.shape == (n,) and np.isfinite(a).all(), "Invalid/nonfinite native float array: " + key)
    require(data["position_xy_m"].dtype == np.dtype("float64") and data["position_xy_m"].shape == (n, 2) and np.isfinite(data["position_xy_m"]).all(), "Invalid/nonfinite actual XY array")
    for key in ("terminal_sample", "steering_lateral_budget_limited"):
        require(data[key].dtype == np.dtype("bool") and data[key].shape == (n,), "Invalid native boolean array: " + key)
    require(not data["terminal_sample"].any(), "Terminal command/body rows are forbidden in metric arrays")
    for key in ("steering_method", "geometry_source"):
        require(data[key].dtype.kind == "U" and data[key].shape == (n,), "Invalid native metadata array: " + key)
    dt = np.diff(data["time_s"])
    require(np.all(np.abs(dt - .02) < 1e-7), "Actual metric rows must be uniform 50 Hz; resampling is forbidden")
    require(abs(data["time_s"][0]) < 1e-9, "Metric arrays must begin at the recorded race start")
    require(np.allclose(data["world_time_s"], data["time_s"] + case["start_world_s"], rtol=0., atol=1e-9), "Native race/world clocks disagree")
    for key in ("bracket_time_s", "bracket_world_time_s", "bracket_progress_m"):
        require(data[key].dtype == np.dtype("float64") and data[key].shape == (2,) and np.isfinite(data[key]).all(), "Invalid actual pose bracket: " + key)
    require(data["bracket_position_m"].dtype == np.dtype("float64") and data["bracket_position_m"].shape == (2, 3) and np.isfinite(data["bracket_position_m"]).all(), "Invalid actual bracket positions")
    bracket_dt = float(np.diff(data["bracket_time_s"])[0])
    require(0. < bracket_dt <= .02 + 1e-7, "Crossing requires an actual pose bracket no longer than one recording interval")
    step = case["physics_control_step_ms"] / 1000.
    require(abs(bracket_dt / step - round(bracket_dt / step)) < 1e-6, "Pose bracket is incompatible with the declared control step")
    require(np.allclose(data["bracket_world_time_s"], data["bracket_time_s"] + case["start_world_s"], rtol=0., atol=1e-9), "Bracket race/world clocks disagree")
    require(abs(data["bracket_time_s"][0] - data["time_s"][-1]) < 1e-9 and
            abs(data["bracket_progress_m"][0] - data["progress_m"][-1]) < 1e-9 and
            np.allclose(data["bracket_position_m"][0, :2], data["position_xy_m"][-1], rtol=0., atol=1e-9), "Timing bracket is not bound to the final actual metric pose")
    target = case["physical_reference_length_m"]
    low, high = data["bracket_progress_m"]
    require(low < target <= high and high > low, "Actual poses do not bracket the fixed physical reference crossing")
    fraction = float((target - low) / (high - low))
    crossing = float(data["bracket_time_s"][0] + fraction * bracket_dt)
    require(math.isclose(crossing, case["first_reference_crossing_scalar_s"], rel_tol=0., abs_tol=1e-10), "Recorded scalar crossing differs from the actual pose bracket")
    return {"first_reference_crossing_scalar_s": crossing,
            "actual_pose_bracket_race_s": data["bracket_time_s"].tolist(), "bracket_duration_s": bracket_dt,
            "last_metric_row_to_crossing_gap_s": crossing - float(data["time_s"][-1]),
            "distance_short_of_reference_m": float(target - data["progress_m"][-1]),
            "exact_physical_substep_clock_claimed": False, "interpolated_metric_samples": 0}


def validate_case_metadata(cases):
    require(isinstance(cases, list) and len(cases) == 3, "All three declared outcomes must remain included")
    require(all(isinstance(case, dict) and set(case) == CASE_FIELDS for case in cases), "Missing or extra saved-case metadata")
    require([case["id"] for case in cases] == list(CASE_IDS) and [case["physics_control_step_ms"] for case in cases] == list(STEPS), "All three declared outcomes must remain included")
    for case, step in zip(cases, STEPS):
        require(case["array_file"] == case["id"] + ".npz", "Saved-case array filename differs")
        require(case["recording_hz"] == 50 and case["control_hz"] == 1000. / step and case["terminal_rows_in_metric_arrays"] == 0 and case["median_filter_samples"] == 5, "Declared recording/control cadence differs")
        for key in ("wall_seconds", "first_reference_crossing_scalar_s", "physical_reference_length_m", "maximum_anchor_error_m", "start_world_s"):
            require(isinstance(case[key], (int, float)) and not isinstance(case[key], bool) and math.isfinite(case[key]), "Nonfinite saved-case metadata: " + key)
        require(case["wall_seconds"] > 0. and case["physical_reference_length_m"] > 0., "Invalid saved duration or physical length")
        require(case["start_world_s"] > 0. and case["first_reference_crossing_scalar_s"] > 0., "Invalid saved start/crossing clock")
        require(type(case["native_safety_clean"]) is bool and type(case["after_bracket_pose_is_terminal"]) is bool, "Saved outcome flags must be booleans")
    lengths = [case["physical_reference_length_m"] for case in cases]
    require(max(lengths) - min(lengths) < 1e-8, "Cadence cases must share one physical reference length")


def load_bundle(folder=BUNDLE):
    folder = Path(folder)
    manifest = read(folder / "manifest.json")
    require(manifest.get("schema_version") == 1 and manifest.get("case_ids") == list(CASE_IDS), "Frozen three-case schema changed")
    require(manifest.get("manifest_sha256") == hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"})).hexdigest(), "Manifest semantic checksum changed")
    required = {"README.md", "requirements.txt", "reproduce.py", "export_bundle.py", "metric.py", "provenance.json", "case_metadata.json", "expected_reports.json", *(case + ".npz" for case in CASE_IDS)}
    require(set(manifest["files"]) == required, "Complete immutable bundle inventory required")
    for name, pin in manifest["files"].items():
        require(Path(name).name == name and (folder / name).is_file(), "Missing or unsafe bundled path")
        require((folder / name).stat().st_size == pin["bytes"] and sha(folder / name) == pin["sha256"], "Bundled file checksum changed: " + name)
    require(sys.version_info[:2] == (3, 11), "Use Python3.11 for the pinned AST/numeric reproduction")
    require(np.__version__ == "1.26.4" and scipy.__version__ == "1.16.2", "Use the pinned NumPy/SciPy versions in requirements.txt")
    cases = read(folder / "case_metadata.json")
    validate_case_metadata(cases)
    provenance = read(folder / "provenance.json")
    sources = provenance["cases"]
    require([case["source_id"] for case in sources] == list(CASE_IDS), "Source provenance case identities differ")
    require(len({case["reference_sha256"] for case in sources}) == 1 and len({case["policy_model_sha256"] for case in sources}) == 1, "Physical reference or original model identity differs")
    require(all(case["runtime_module_sha256"] == sources[0]["runtime_module_sha256"] for case in sources) and len(sources[0]["runtime_module_sha256"]) == 17, "Original runtime source identities differ")
    functions = {n.name: n for n in ast.parse((folder / "metric.py").read_text()).body if isinstance(n, ast.FunctionDef)}
    for entry in provenance["function_ast_provenance"]:
        if "exported_function_ast_sha256" in entry:
            actual = hashlib.sha256(ast.dump(functions[entry["function"]], include_attributes=False).encode()).hexdigest()
            require(actual == entry["exported_function_ast_sha256"], "Frozen metric function AST changed")
        else:
            actual = [hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest() for n in functions["derive_arrays"].body[:-1]]
            require(actual == entry["copied_statement_ast_sha256"], "Frozen derivative/filter statements changed")
    spec = importlib.util.spec_from_file_location("control_cadence_frozen_metric", folder / "metric.py")
    methods = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(methods)
    data = {}
    for case in cases:
        with np.load(folder / case["array_file"], allow_pickle=False) as archive:
            data[case["id"]] = {key: archive[key].copy() for key in archive.files}
        validate_arrays(data[case["id"]], case)
    return cases, data, methods, read(folder / "expected_reports.json"), manifest


def compute_metrics(data, methods):
    a = {"time": data["time_s"], "steer": data["steering_left_rad"], "yaw": np.unwrap(data["yaw_rad"]),
         "position": data["position_xy_m"], "speed": data["speed_mps"], "cross": data["selected_path_cross_error_m"],
         "geometry": data["geometry_source"], "method": data["steering_method"], "limited": data["steering_lateral_budget_limited"]}
    a = methods.derive_arrays(a)
    result = methods.motion_metrics(a, np.ones(len(a["time"]), dtype=bool))
    raw_gyro_hf = methods.rms(methods.highpass(data["measured_yaw_rate_rps"], 1. / a["median_dt"]))
    return result, raw_gyro_hf


def reproduce(folder=BUNDLE):
    cases, arrays, methods, expected, manifest = load_bundle(folder)
    results = {}
    maximum_error = 0.
    for case in cases:
        key = case["id"]
        metrics, raw_gyro_hf = compute_metrics(arrays[key], methods)
        check_equal(metrics, expected[key], key)
        maximum_error = max(maximum_error, max_float_error(metrics, expected[key]))
        results[key] = {"physics_control_step_ms": case["physics_control_step_ms"], "control_hz": case["control_hz"],
                        "recording_hz": 50, "sample_count": case["sample_count"], "outcome": case["outcome"],
                        "native_safety_clean": case["native_safety_clean"], "anchors_within_half_metre": case["anchors_within_half_metre"],
                        "wall_seconds": case["wall_seconds"], "wall_cost_ratio_vs_20ms": case["wall_seconds"] / cases[0]["wall_seconds"],
                        "median_filter_sample_horizon_ms": case["median_filter_samples"] * case["physics_control_step_ms"],
                        "timing": validate_arrays(arrays[key], case), "frozen_metrics": metrics,
                        "additional_raw_gyro_above1hz_rms_radps": raw_gyro_hf}
    return {"status": "VERIFIED_SAVED_MOTION_CPU_REPRODUCTION", "case_ids": list(CASE_IDS), "cases": results,
            "frozen_numeric_relative_tolerance": REL_TOL, "frozen_numeric_absolute_tolerance": ABS_TOL,
            "maximum_frozen_float_absolute_error": maximum_error, "recorded_50hz_samples": sum(case["sample_count"] for case in cases),
            "manifest_sha256": manifest["manifest_sha256"], "python_version": sys.version.split()[0], "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "new_simulation_inference_or_training": False, "gpu_or_network_calls": False,
            "physical_tire_force_claimed": False, "pure_physics_isolation_claimed": False,
            "all_frequency_band_improvement_claimed": False, "convergence_claimed": False, "promotion_claimed": False,
            "interpretation": "Three n=1 clear saved trajectories. Integration/control cadence and sample-count filter horizon change together. Lower TV/jerk can coexist with greater 1–6 Hz band power and wall cost. Published yaw HF uses pose-derived yaw rate; raw gyro HF is separately named."}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Choose a fresh output directory; existing results are never overwritten")
    result = reproduce()
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (output / "summary.csv").open("x", newline="") as stream:
        fields = ["case_id", "control_step_ms", "lap_scalar_s", "wall_s", "wall_ratio_vs_20ms", "command_TV_rad_per_s", "pose_yaw_HF_rms_radps", "yaw_1to6hz_power", "steering_1to6hz_power", "XY_lateral_jerk_rms_mps3", "FD_yaw_acceleration_rms_radps2"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, case in result["cases"].items():
            m = case["frozen_metrics"]
            writer.writerow(dict(zip(fields, [key, case["physics_control_step_ms"], case["timing"]["first_reference_crossing_scalar_s"], case["wall_seconds"], case["wall_cost_ratio_vs_20ms"], m["steering"]["total_variation_rad_per_s"], m["body"]["yaw_rate_above1hz_rms_radps"], m["body"]["yaw_rate_band_1to6hz_power_rad2ps2"], m["body"]["steering_band_1to6hz_power_rad2"], m["body"]["XY_difference_body_lateral_jerk_rms_mps3"], m["body"]["forward_difference_yaw_acceleration_rms_radps2"]])))
    print(json.dumps({"status": result["status"], "cases": len(CASE_IDS), "maximum_frozen_float_absolute_error": result["maximum_frozen_float_absolute_error"], "output_files": ["result.json", "summary.csv"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, TypeError) as error:
        print("REPRODUCTION_REFUSED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
