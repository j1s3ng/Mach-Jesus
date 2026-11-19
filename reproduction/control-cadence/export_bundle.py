#!/usr/bin/env python3
"""Export a fresh, path-normalized bundle from explicitly supplied CLOSED evidence.

The local evidence specification is not included in the public bundle. Its cases
have id, physics_step_ms, case_dir and evaluation fields; evaluator/legacy fields
identify the frozen original Python measurement sources.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

EVALUATOR_SHA = "46f012aaa2b96eef8f736aa8ee5e8048f8cf8b298af17caaf739e65c86074f49"
LEGACY_SHA = "a0c3dd76ea7022fb44c1296c6d6a2e323f8bbabefc1eebe9681db5f436cd130c"
EVALUATOR_FUNCTIONS = ("rms", "highpass", "band_power", "motion_metrics")
LEGACY_FUNCTIONS = ("percentile", "direction_changes", "metric")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def extract_methods(evaluator, legacy, output):
    if sha(evaluator) != EVALUATOR_SHA or sha(legacy) != LEGACY_SHA:
        raise ValueError("Frozen measurement source identity differs")
    trees = {"measurement-evaluator": ast.parse(Path(evaluator).read_text()),
             "steering-metric": ast.parse(Path(legacy).read_text())}
    functions = {key: {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)} for key, tree in trees.items()}
    copied, provenance = [], []
    for source, names in (("steering-metric", LEGACY_FUNCTIONS), ("measurement-evaluator", EVALUATOR_FUNCTIONS)):
        for name in names:
            node = functions[source][name]
            copied.append(copy.deepcopy(node))
            provenance.append({"source_id": source, "function": name, "source_line": node.lineno,
                               "source_function_ast_sha256": ast_sha(node), "exported_function_ast_sha256": ast_sha(node)})
    original = functions["measurement-evaluator"]["array_records"]
    begin = next(i for i, node in enumerate(original.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "dt")
    statements = copy.deepcopy(original.body[begin:-1])
    derived = ast.FunctionDef(name="derive_arrays", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="a")],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=statements + [ast.Return(value=ast.Name(id="a", ctx=ast.Load()))], decorator_list=[])
    copied.append(ast.fix_missing_locations(derived))
    provenance.append({"source_id": "measurement-evaluator", "function": "array_records", "source_line": original.lineno,
                       "source_function_ast_sha256": ast_sha(original),
                       "extraction": "Unchanged tail from dt=np.diff(a['time']) through body_lat_highpass; return adapter only.",
                       "copied_statement_ast_sha256": [ast_sha(n) for n in original.body[begin:-1]]})
    header = ('"""Frozen metric AST extracts; source IDs and hashes are in provenance.json."""\n'
              'import numpy as np\nfrom scipy.signal import butter, sosfiltfilt, welch\nfrom types import SimpleNamespace\n\n')
    code = header + "\n\n".join(ast.unparse(n) for n in copied)
    code += '\n\nsteering = SimpleNamespace(metric=metric)\n'
    with Path(output).open("x") as stream:
        stream.write(code)
    evaluate = functions["measurement-evaluator"]["evaluate"]
    timing = [node for node in ast.walk(evaluate) if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("fraction", "first_cross")
              and isinstance(node.value, ast.BinOp)]
    timing.sort(key=lambda node: node.lineno)
    if len(timing) != 2:
        raise ValueError("Expected the frozen scalar crossing interpolation statements")
    return provenance, {"source_id": "measurement-evaluator", "function": "evaluate",
                        "source_function_ast_sha256": ast_sha(evaluate),
                        "scalar_crossing_statement_ast_sha256": [ast_sha(node) for node in timing],
                        "scope": "Only the scalar clock is interpolated between actual poses; metric arrays never receive an endpoint sample."}


def normalized_config(value, root):
    if isinstance(value, dict):
        return {k: ("<CASE_LABEL>" if k == "label" and v == root.name else normalized_config(v, root))
                for k, v in value.items() if k != "optimization_protocol"}
    if isinstance(value, list):
        return [normalized_config(v, root) for v in value]
    return value.replace(str(root), "<CASE>") if isinstance(value, str) else value


def export_case(spec, output):
    root, evaluation_file = Path(spec["case_dir"]).resolve(), Path(spec["evaluation"]).resolve()
    terminal = read(root / "terminal.json")
    if terminal["status"] != "CLOSED" or terminal["exit_code"] != 0 or terminal["error"] or terminal["source_verification_error"] or terminal["cleanup"]["survivors"]:
        raise ValueError("Only source-clean CLOSED/exit0 cases may be exported")
    evaluation = read(evaluation_file)
    case = evaluation["candidate"]
    evaluator_digest = evaluation.get("measurement_definition_sha256", evaluation.get("script_sha256"))
    if sha(spec["evaluator_source"]) != evaluator_digest:
        raise ValueError("Actual cadence evaluator source pin differs")
    if Path(case["attempt"]).resolve() != root or not case["closed_input_verified"]:
        raise ValueError("Evaluation/CLOSED case binding differs")
    for path, digest in case["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Frozen evaluation input changed")
    prep, config = read(root / "preparation.json"), read(root / "race/config.json")
    if sha(root / "race/config.json") != prep["config_sha256"]:
        raise ValueError("Case config pin differs")
    for name, digest in prep["files_sha256"].items():
        if sha(Path(prep["source"]) / name) != digest:
            raise ValueError("Prepared source asset changed")
    car = root / "race/cars/R20"
    summary = read(car / "controller_summary.json")
    step = spec["physics_step_ms"]
    if summary["physics_step_ms"] != step or summary["telemetry_period_ms"] != 20:
        raise ValueError("Declared control/recording cadence differs")
    if config["cars"]["R20"]["path_tracking"]["yaw_rate_damping"]["enabled"]:
        raise ValueError("These three declared cases require yaw damping OFF")
    raw = [json.loads(line) for line in (car / "control_trace.jsonl").open() if line.strip()]
    first_terminal = next(r["simulation_time_s"] for r in raw if r.get("terminal_sample"))
    start = float(config.get("start_time_s", 1.))
    end = start + case["actual_telemetry"]["authoritative_episode_duration_s"]
    rows = [r for r in raw if start - 1e-9 <= r["simulation_time_s"] <= end + 1e-7 and r["simulation_time_s"] < first_terminal - 1e-9]
    if len(rows) != case["actual_telemetry"]["samples"]:
        raise ValueError("Frozen metric population includes a terminal/non-native row")
    target = config["reference_length_m"]
    poses = []
    for row in raw:
        if row["simulation_time_s"] >= start - 1e-9:
            poses.append(row)
        if row.get("terminal_sample"):
            break
    before, after = next((a, b) for a, b in zip(poses[:-1], poses[1:]) if a["progress_m"] < target <= b["progress_m"])
    get = lambda key: np.asarray([r[key] for r in rows], dtype=np.float64)
    meta = lambda key, default: [r.get("path_tracking", {}).get(key, default) for r in rows]
    arrays = {"time_s": get("time_s"), "world_time_s": get("simulation_time_s"), "position_xy_m": np.asarray([r["position_m"][:2] for r in rows], dtype=np.float64),
              "yaw_rad": get("yaw_rad"), "steering_left_rad": get("steering_left_rad"), "measured_yaw_rate_rps": get("measured_yaw_rate_rps"),
              "progress_m": get("progress_m"), "speed_mps": get("speed_mps"),
              "selected_path_cross_error_m": np.asarray(meta("selected_path_cross_error_m", 0.), dtype=np.float64),
              "steering_lateral_budget_limited": np.asarray(meta("steering_lateral_budget_limited", False), dtype=np.bool_),
              "steering_method": np.asarray(meta("steering_method", "none"), dtype="<U48"),
              "geometry_source": np.asarray(meta("geometry_source", "none"), dtype="<U48"),
              "terminal_sample": np.zeros(len(rows), dtype=np.bool_),
              "bracket_time_s": np.asarray([before["time_s"], after["time_s"]], dtype=np.float64),
              "bracket_world_time_s": np.asarray([before["simulation_time_s"], after["simulation_time_s"]], dtype=np.float64),
              "bracket_progress_m": np.asarray([before["progress_m"], after["progress_m"]], dtype=np.float64),
              "bracket_position_m": np.asarray([before["position_m"], after["position_m"]], dtype=np.float64)}
    path = output / (spec["id"] + ".npz")
    if path.exists():
        raise FileExistsError("Fresh exported array path required")
    np.savez_compressed(path, **arrays)
    fraction = (target - before["progress_m"]) / (after["progress_m"] - before["progress_m"])
    timing = before["time_s"] + fraction * (after["time_s"] - before["time_s"])
    if abs(timing - case["first_original_reference_circuit_crossing_race_s"]) > 1e-10:
        raise ValueError("Actual pose bracket does not reproduce frozen scalar crossing")
    provenance = {"source_id": spec["id"], "raw_trace_sha256": sha(car / "control_trace.jsonl"),
                  "native_archive_sha256": sha(evaluation_file.with_name("measured_arrays.npz")), "evaluation_sha256": sha(evaluation_file),
                  "evaluator_sha256": evaluation.get("measurement_definition_sha256", evaluation.get("script_sha256")),
                  "terminal_sha256": sha(root / "terminal.json"), "preparation_sha256": sha(root / "preparation.json"),
                  "configuration_sha256": sha(root / "race/config.json"), "physical_world_sha256": sha(prep["world"]),
                  "reference_sha256": summary["reference_sha256"], "policy_model_sha256": summary["model_sha256"],
                  "runtime_module_sha256": {f"runtime-module-{i:02d}": value for i, (_, value) in enumerate(sorted(summary["runtime_source_sha256"].items()), 1)}}
    public = {"id": spec["id"], "physics_control_step_ms": step, "control_hz": 1000. / step, "recording_hz": 50,
              "array_file": path.name, "sample_count": len(rows), "start_world_s": start, "physical_reference_length_m": target,
              "terminal_rows_in_metric_arrays": 0, "after_bracket_pose_is_terminal": bool(after.get("terminal_sample")),
              "wall_seconds": terminal["wall_seconds"], "first_reference_crossing_scalar_s": timing,
              "median_filter_samples": config["cars"]["R20"]["path_tracking"]["longitudinal_accel_median_samples"],
              "native_safety_clean": case["native_and_source_checks"]["native_safety_clean"],
              "observed_map_anchors": case["map_anchor_summary"]["observed_map_anchors"],
              "anchors_within_half_metre": case["map_anchor_summary"]["map_anchors_within_half_metre_of_planned_xy"],
              "maximum_anchor_error_m": case["map_anchor_summary"]["actual_to_planned_anchor_xy_error_max_m"],
              "outcome": "complete_first_circuit" if case["first_original_reference_circuit_crossing_race_s"] is not None else "partial"}
    return public, provenance, case["whole_run"], normalized_config(config, root), summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evidence-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.evidence_spec)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("manifest.json", "provenance.json", "expected_reports.json", "metric.py")):
        raise FileExistsError("Bundle artifacts already exist")
    methods, timing_method = extract_methods(spec["evaluator"], spec["legacy"], out / "metric.py")
    cases, source_cases, expected, configs, summaries = [], [], {}, [], []
    for item in spec["cases"]:
        public, provenance, metrics, config, summary = export_case(item, out)
        cases.append(public); source_cases.append(provenance); expected[public["id"]] = metrics
        configs.append(config); summaries.append(summary)
    if [case["physics_control_step_ms"] for case in cases] != [20, 10, 5] or not all(c == configs[0] for c in configs):
        raise ValueError("Exactly the declared same-config20/10/5ms conditions are required")
    if not all(s["model_sha256"] == summaries[0]["model_sha256"] and s["runtime_source_sha256"] == summaries[0]["runtime_source_sha256"] for s in summaries):
        raise ValueError("Runtime source/model equality failed")
    write_json(out / "provenance.json", {"schema_version": 1, "source_path_policy": "Logical source IDs and SHA256 only; no local paths or raw logs shipped.",
        "method_sources": [{"source_id": "measurement-evaluator", "sha256": EVALUATOR_SHA}, {"source_id": "steering-metric", "sha256": LEGACY_SHA}],
        "function_ast_provenance": methods, "scalar_timing_ast_provenance": timing_method,
        "cases": source_cases, "same_runtime_source_and_model_bytes": True,
        "new_inference_training_or_simulation": False})
    write_json(out / "expected_reports.json", expected)
    write_json(out / "case_metadata.json", cases)
    print(json.dumps({"exported_cases": [c["id"] for c in cases], "array_bytes": sum((out / c["array_file"]).stat().st_size for c in cases)}, indent=2))


if __name__ == "__main__":
    main()
