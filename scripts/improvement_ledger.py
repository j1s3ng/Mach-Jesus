#!/usr/bin/env python3
"""Durable experiment ledger, frozen snapshots, scenarios, and promotion checks.

This module never starts a simulator, trains a model, or changes a controller.
The orchestrator owns those processes and records their independent audit here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

OFFICIAL_DETECTOR_SHA256 = "9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b"
CONTROLLER_RULE_KEYS = frozenset({"control_version", "visible_distance_m"})


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def environment_rules(rules):
    """Separate controller choices from common physical experiment conditions."""
    if not isinstance(rules, dict):
        raise ValueError("Environment rules must be an object")
    if isinstance(rules.get("rules"), dict):
        rules = rules["rules"]
    return {key: value for key, value in rules.items() if key not in CONTROLLER_RULE_KEYS}


def environment_rules_hash(rules):
    return digest(environment_rules(rules))


def control_parameters(value):
    if not isinstance(value, dict):
        raise ValueError("control_parameters must be an object")
    prohibited = {"max_speed_mps", "lateral_accel_budget_mps2", "friction_coefficient", "physics_step_ms",
                  "sensor_period_ms", "lidar_range_m", "engine_max_power_w", "brake_decel_mps2"}
    if prohibited & value.keys():
        raise ValueError("Physical environment limits cannot be relabeled as controller parameters")
    normalized = json.loads(canonical(value))  # Reject NaN/non-JSON values and detach caller mutations.
    if "visible_distance_m" in normalized:
        distance = normalized["visible_distance_m"]
        if not isinstance(distance, (int, float)) or isinstance(distance, bool) or not 0 < distance <= 100.:
            raise ValueError("Controller visibility budget must be positive and no greater than the 100 m physical LiDAR range")
        normalized["visible_distance_m"] = float(distance)
    return normalized


def frozen_policy_parameters(manifest, policy_key):
    frozen = manifest.get("metadata", {}).get("policy_parameters")
    if frozen is None:
        return None
    if isinstance(frozen, dict) and isinstance(frozen.get(policy_key), dict):
        frozen = frozen[policy_key]
    return control_parameters(frozen)


def scenario_identity(values):
    """Exactly the world/scenario dictionary used by run_improvement_heat CLI.

    Policy/model/guard/detector choices are intentionally not scenario identity.
    """
    get = values.get if isinstance(values, dict) else lambda key, default: getattr(values, key, default)
    payload = {"scenario": str(get("scenario", "race")), "cars": int(get("cars", 4)),
               "rotation": int(get("rotation", 0)), "seed": int(get("seed", 7)),
               "spacing": float(get("spacing", 12.)), "seconds": float(get("seconds", 480.)),
               "obstacle_progress": float(get("obstacle_progress", 100.)), "grid_jitter": float(get("grid_jitter", 0.)),
               "ego_progress": float(get("ego_progress", 0.)), "ego_lane": float(get("ego_lane", 0.)),
               "blocked_side": str(get("blocked_side", "right"))}
    if payload["scenario"] not in {"race", "stopped", "parking"} or payload["blocked_side"] not in {"left", "right"}:
        raise ValueError("Invalid scenario type or blocked side")
    if not 1 <= payload["cars"] <= 64 or payload["seconds"] <= 0 or payload["spacing"] <= 0 or payload["grid_jitter"] < 0:
        raise ValueError("Invalid scenario size, duration, spacing or jitter")
    canonical(payload)
    return payload


def _read_json(path):
    return json.loads(Path(path).read_text())


def _manifest_identity(manifest):
    return {"schema_version": 1, "metadata": manifest["metadata"],
            "files": [{key: row[key] for key in ("path", "bytes", "sha256")} for row in manifest["files"]]}


def freeze_snapshot(destination, files, metadata=None):
    """Copy explicitly mapped source/weight files; never mutate the originals.

    ``files`` maps relative names to source paths. Content identity excludes
    creation time and original absolute paths, so equal snapshots reproduce.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists: {destination}")
    if not files:
        raise ValueError("A snapshot must include source or weight files")
    sources = {}
    for name, source in sorted(files.items()):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name == "manifest.json":
            raise ValueError(f"Unsafe snapshot relative name: {name}")
        source = Path(source).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Not a regular source file: {source}")
        sources[str(relative)] = source
    before = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in sources.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=destination.parent))
    try:
        rows = []
        for name, source in sources.items():
            data = source.read_bytes()
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            target.chmod(0o444)
            rows.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "original_path": str(source)})
        after = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in sources.items()}
        if after != before:
            raise RuntimeError("A source changed while taking the snapshot; freeze again after editing stops")
        manifest = {"schema_version": 1, "created_at": utc_now(), "metadata": metadata or {}, "files": rows}
        manifest["snapshot_sha256"] = digest(_manifest_identity(manifest))
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        manifest_path.chmod(0o444)
        temporary.rename(destination)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_snapshot(directory):
    directory = Path(directory)
    manifest = _read_json(directory / "manifest.json")
    if digest(_manifest_identity(manifest)) != manifest["snapshot_sha256"]:
        raise ValueError("Snapshot manifest identity is invalid")
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Snapshot manifest contains an unsafe path")
        data = (directory / relative).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Frozen file changed: {row['path']}")
    return manifest


def candidate_descriptor(registration):
    """Select one immutable policy/weight treatment from a runnable bundle."""
    descriptor = registration.get("descriptor")
    if descriptor is None and "policy_key" in registration:
        descriptor = {key: registration.get(key) for key in ("policy_key", "model_sha256")}
        if "detector_sha256" in registration:
            descriptor["detector_sha256"] = registration["detector_sha256"]
        if "control_parameters" in registration:
            descriptor["control_parameters"] = registration["control_parameters"]
    if not isinstance(descriptor, dict) or not descriptor.get("policy_key") or "model_sha256" not in descriptor:
        raise ValueError("Candidate descriptor requires policy_key and model_sha256")
    policy, model = descriptor["policy_key"], descriptor["model_sha256"]
    if not isinstance(policy, str):
        raise ValueError("Candidate policy_key must be a string")
    if policy == "reference":
        if model is not None:
            raise ValueError("The reference descriptor must have model_sha256=None")
    elif not isinstance(model, str) or len(model) != 64 or any(c not in "0123456789abcdef" for c in model):
        raise ValueError("A learned policy requires its full lowercase model SHA-256")
    detector = descriptor.get("detector_sha256", OFFICIAL_DETECTOR_SHA256)
    if not isinstance(detector, str) or len(detector) != 64 or any(c not in "0123456789abcdef" for c in detector):
        raise ValueError("A formal candidate requires its full detector SHA-256")
    normalized = {"policy_key": policy, "model_sha256": model, "detector_sha256": detector}
    if "control_parameters" in descriptor:
        normalized["control_parameters"] = control_parameters(descriptor["control_parameters"])
    return normalized


def entrant_descriptor(entrant):
    descriptor = {"policy_key": entrant.get("policy_key", entrant.get("policy")),
                  "model_sha256": entrant.get("model_sha256", entrant.get("checkpoint_sha256")),
                  "detector_sha256": entrant.get("runtime_detector_checkpoint_sha256", entrant.get("detector_sha256"))}
    parameters = entrant.get("runtime_policy_parameters", entrant.get("control_parameters"))
    if parameters is not None:
        try:
            descriptor["control_parameters"] = control_parameters(parameters)
        except ValueError:
            descriptor["control_parameters"] = None
    return descriptor


def _bundle_contains_descriptor(manifest, descriptor):
    hashes = {row["sha256"] for row in manifest["files"]}
    model_ok = descriptor["model_sha256"] is None or descriptor["model_sha256"] in hashes
    detector_ok = descriptor["detector_sha256"] == OFFICIAL_DETECTOR_SHA256 or descriptor["detector_sha256"] in hashes
    parameters = frozen_policy_parameters(manifest, descriptor["policy_key"])
    parameter_ok = ((parameters is None and "control_parameters" not in descriptor)
                    or (parameters is not None and descriptor.get("control_parameters") == parameters))
    return model_ok and detector_ok and parameter_ok


def cyclic_grid(cars, rotation, reverse=False):
    """Identity at each grid slot; four shifts cover every policy/slot pairing."""
    if cars < 1:
        raise ValueError("cars must be positive")
    identities = list(range(cars))
    if reverse:
        identities.reverse()
    rotation %= cars
    return identities[rotation:] + identities[:rotation]


def scenario_suite(split="development"):
    if split not in {"development", "holdout", "confirmation"}:
        raise ValueError("Unknown scenario split")
    seed = {"development": 1103, "holdout": 9203, "confirmation": 18731}[split]
    cases = []
    for cars in (4, 8, 20):
        for rotation in range(4):
            cases.append({"scenario_id": f"{split}-race-{cars}-r{rotation}", "split": split,
                          "kind": "race", "tier": f"cars-{cars}", "cars": cars,
                          "evaluated_entrants": cars, "grid_rotation": rotation,
                          "reverse_grid": split != "development", "seed": seed + cars,
                          "max_sim_seconds": 480., "record_video": False})
    for index, (name, objective, lane) in enumerate((
        ("stopped-bypass", "pass_without_overlap", 0.),
        ("stopped-braking", "pass_after_braking_without_overlap", 2.5))):
        cases.append({"scenario_id": f"{split}-{name}", "split": split, "kind": "hazard",
                      "tier": "stopped-hazard", "cars": 2, "evaluated_entrants": 1,
                      "is_hazard": True, "objective": objective, "seed": seed + index,
                      "hazard": {"kind": "stopped_car", "lane_m": lane,
                                 "progress_fraction": (.20 if split == "development" else .60) + .07 * index},
                      "max_sim_seconds": 120., "record_video": False})
    for case in cases:
        case["scenario_sha256"] = digest(case)
    return cases


def can_schedule(now, deadline, estimated_wall_seconds, reserve_seconds=300.):
    """Leave time to stop processes, audit, checkpoint, and write the final report."""
    parse = lambda value: dt.datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return (parse(deadline) - parse(now)).total_seconds() >= estimated_wall_seconds + reserve_seconds


def _decode_records(data):
    if data and not data.endswith(b"\n"):
        raise ValueError("Incomplete ledger tail; call recover_tail() before continuing")
    records, previous = [], "0" * 64
    for line in data.splitlines():
        record = json.loads(line)
        claimed = record.pop("sha256")
        if record["sequence"] != len(records) + 1 or record["previous_sha256"] != previous or digest(record) != claimed:
            raise ValueError("Ledger hash chain is corrupted")
        record["sha256"] = claimed
        records.append(record); previous = claimed
    return records


def _state(records):
    state = {"run": None, "incumbent_id": None, "candidates": {}, "attempts": {}, "decisions": [],
             "bootstrap_decisions": [], "engineering_finalizations": {}}
    for record in records:
        kind, value = record["event_type"], record["payload"]
        if kind == "run_initialized":
            if state["run"] is not None:
                raise ValueError("A ledger can only initialize once")
            state["run"] = value
            state["incumbent_id"] = value.get("incumbent_id")
        elif kind == "candidate_frozen":
            cid = value["candidate_id"]
            if cid in state["candidates"]:
                raise ValueError(f"Candidate ID is already frozen: {cid}")
            if "descriptor" in value or "policy_key" in value:
                candidate_descriptor(value)
            state["candidates"][cid] = dict(value, ledger_registered_at_utc=record["utc"])
        elif kind == "attempt_scheduled":
            aid = value["attempt_id"]
            if aid in state["attempts"]:
                raise ValueError(f"Attempt ID is already scheduled: {aid}")
            state["attempts"][aid] = dict(value, state="scheduled",
                                           stage=value.get("stage", value.get("category", "promotion")),
                                           ledger_scheduled_at_utc=record["utc"])
        elif kind in {"attempt_started", "attempt_finished"}:
            attempt = state["attempts"].get(value["attempt_id"])
            if attempt is None:
                raise ValueError("Register an attempt before starting or finishing it")
            if attempt["state"] == "finished":
                raise ValueError("A terminal attempt cannot be overwritten or retried under the same ID")
            immutable = ("candidate_id", "scenario_id", "split", "stage", "category", "comparison_key",
                         "scenario_sha256", "rules_sha256", "snapshot_sha256", "expected_entrants",
                         "tier", "is_hazard", "output_dir")
            if any(key in value and value[key] != attempt.get(key) for key in immutable):
                raise ValueError("Scheduled experiment identity cannot change after launch")
            attempt.update(value)
            attempt["state"] = "finished" if kind == "attempt_finished" else "started"
            if kind == "attempt_finished":
                attempt["ledger_finished_at_utc"] = record["utc"]
        elif kind == "bootstrap_decision":
            state["bootstrap_decisions"].append(value)
        elif kind == "engineering_finalized":
            aid = value["attempt_id"]
            state["engineering_finalizations"][aid] = value
            if aid in state["attempts"]:
                state["attempts"][aid]["performance_status"] = value.get("performance_status", value["status"])
                state["attempts"][aid]["eligible_for_promotion"] = False
        elif kind == "incumbent_bootstrapped":
            if state["incumbent_id"] is not None:
                raise ValueError("An incumbent already exists; use matched promotion")
            cid = value["candidate_id"]
            registration = state["candidates"].get(cid, {})
            descriptor = candidate_descriptor(registration)
            approved = any(d.get("approved") and d.get("candidate_id") == cid
                           and value.get("decision_sha256")
                           and d.get("decision_sha256") == value["decision_sha256"]
                           and d.get("descriptor") == descriptor
                           and d.get("snapshot_sha256") == registration.get("snapshot_sha256")
                           for d in state["bootstrap_decisions"])
            if not approved:
                raise ValueError("Bootstrap requires a recorded verified complete-suite decision")
            state["incumbent_id"] = cid
        elif kind == "promotion_decision":
            state["decisions"].append(value)
        elif kind == "candidate_promoted":
            if state["incumbent_id"] != value["incumbent_id"]:
                raise ValueError("Incumbent changed before promotion")
            descriptor_bindings = {cid: candidate_descriptor(state["candidates"].get(cid, {}))
                                   for cid in (value["incumbent_id"], value["candidate_id"])}
            snapshot_bindings = {cid: state["candidates"][cid].get("snapshot_sha256") for cid in descriptor_bindings}
            approved = any(d.get("approved") and d.get("candidate_id") == value["candidate_id"]
                           and d.get("incumbent_id") == value["incumbent_id"]
                           and value.get("decision_sha256")
                           and d.get("decision_sha256") == value.get("decision_sha256")
                           and d.get("candidate_descriptors") == descriptor_bindings
                           and d.get("snapshot_sha256s") == snapshot_bindings for d in state["decisions"])
            if not approved:
                raise ValueError("Promotion requires a recorded approved decision")
            state["incumbent_id"] = value["candidate_id"]
    state["attempts"] = list(state["attempts"].values())
    state["record_count"] = len(records)
    return state


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    def records(self):
        if not self.path.exists():
            return []
        with self.path.open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            return _decode_records(stream.read())

    def state(self):
        return _state(self.records())

    def register_candidate(self, candidate_id, snapshot_path, descriptor, **metadata):
        manifest = verify_snapshot(snapshot_path)
        descriptor = candidate_descriptor({"descriptor": descriptor})
        if not _bundle_contains_descriptor(manifest, descriptor):
            raise ValueError("Registered model, detector or control parameters are not frozen in the bundle")
        return self.append("candidate_frozen", {**metadata, "candidate_id": candidate_id,
            "snapshot_path": str(Path(snapshot_path).resolve()), "snapshot_sha256": manifest["snapshot_sha256"],
            "descriptor": descriptor})

    def bootstrap_incumbent(self, candidate_id, required_scenarios, split="bootstrap"):
        report = validate_bootstrap(self.state(), candidate_id, required_scenarios, split)
        report["decision_sha256"] = digest(report)
        self.append("bootstrap_decision", report)
        if report["approved"]:
            self.append("incumbent_bootstrapped", {"candidate_id": candidate_id,
                                                    "decision_sha256": report["decision_sha256"]})
        return report

    def append(self, event_type, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            records = _decode_records(stream.read())
            record = {"schema_version": 1, "sequence": len(records) + 1, "utc": utc_now(),
                      "event_type": event_type, "payload": payload,
                      "previous_sha256": records[-1]["sha256"] if records else "0" * 64}
            record["sha256"] = digest(record)
            _state(records + [record])  # Reject invalid lifecycle transitions before writing.
            stream.seek(0, os.SEEK_END)
            stream.write(canonical(record) + b"\n"); stream.flush(); os.fsync(stream.fileno())
            return record

    def recover_tail(self):
        """Preserve an interrupted final append, then restore its valid prefix."""
        if not self.path.exists():
            return None
        with self.path.open("r+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            data = stream.read()
            if not data or data.endswith(b"\n"):
                _decode_records(data)
                return None
            boundary = data.rfind(b"\n") + 1
            _decode_records(data[:boundary])
            tail = data[boundary:]
            backup = self.path.with_name(self.path.name + ".interrupted-" + hashlib.sha256(tail).hexdigest()[:16])
            if backup.exists():
                if backup.read_bytes() != tail:
                    raise ValueError("Interrupted-tail backup does not match the preserved bytes")
            else:
                with backup.open("xb") as saved:
                    saved.write(tail); saved.flush(); os.fsync(saved.fileno())
            stream.truncate(boundary); stream.flush(); os.fsync(stream.fileno())
        self.append("ledger_tail_recovered", {"backup": str(backup), "bytes": len(tail),
                                               "sha256": hashlib.sha256(tail).hexdigest()})
        return str(backup)

    def promote(self, candidate_id, required_scenarios, split="holdout"):
        state = self.state()
        report = compare_candidates(state["attempts"], state["incumbent_id"], candidate_id,
                                    required_scenarios, split, snapshot_registry=state["candidates"])
        report["decision_sha256"] = digest(report)
        self.append("promotion_decision", report)
        if report["approved"]:
            self.append("candidate_promoted", {key: report[key] for key in (
                "incumbent_id", "candidate_id", "decision_sha256")})
        return report


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_candidates(attempts, incumbent_id, candidate_id, required_scenarios,
                       split="holdout", snapshot_registry=None, minimum_pace_gain=.02):
    """Conservative promotion gate for a fixed, paired confirmation suite.

    Missing/failed/unverified scheduled attempts are never silently discarded.
    Hazard objectives are separate from full-lap completion and must all pass.
    """
    required = {case if isinstance(case, str) else case["scenario_id"]: case
                for case in required_scenarios}
    issues, paired = [], {}
    selected = [a for a in attempts if a.get("candidate_id") in {incumbent_id, candidate_id}
                and a.get("split") == split and a.get("stage", a.get("category", "promotion"))
                in {"promotion", "bootstrap"}]
    excluded = [a.get("attempt_id") for a in attempts if a.get("candidate_id") in {incumbent_id, candidate_id}
                and a not in selected]
    scores, identities, snapshots, descriptors = {}, {}, {}, {}
    for cid in (incumbent_id, candidate_id):
        group = [a for a in selected if a.get("candidate_id") == cid]
        keys, seen_scenarios = {}, set()
        totals = defaultdict(lambda: Counter(expected=0, clean=0, valid=0))
        identity = {}
        snapshot_ids = {a.get("snapshot_sha256") for a in group}
        if len(snapshot_ids) != 1 or None in snapshot_ids:
            issues.append(f"{cid}:missing_or_changed_snapshot")
        snapshots[cid] = next(iter(snapshot_ids)) if len(snapshot_ids) == 1 else None
        if snapshot_registry is not None:
            registration = snapshot_registry.get(cid, {})
            manifest = None
            try:
                manifest = verify_snapshot(registration["snapshot_path"])
                if manifest["snapshot_sha256"] != snapshots[cid]:
                    issues.append(f"{cid}:registered_snapshot_mismatch")
                if registration.get("snapshot_sha256") != manifest["snapshot_sha256"]:
                    issues.append(f"{cid}:registration_snapshot_identity_mismatch")
            except (KeyError, OSError, ValueError) as error:
                issues.append(f"{cid}:snapshot_verification_failed:{type(error).__name__}")
            try:
                descriptors[cid] = candidate_descriptor(registration)
                if manifest is None or not _bundle_contains_descriptor(manifest, descriptors[cid]):
                    issues.append(f"{cid}:registered_descriptor_not_frozen_in_bundle")
            except ValueError:
                issues.append(f"{cid}:missing_or_invalid_registered_descriptor")
        hazard_count = hazard_passes = 0
        for attempt in group:
            aid = attempt.get("attempt_id", "unknown")
            scenario = attempt.get("scenario_id")
            if scenario not in required:
                issues.append(f"{aid}:unexpected_confirmation_scenario")
            seen_scenarios.add(scenario)
            key = attempt.get("comparison_key")
            if not key or key in keys:
                issues.append(f"{aid}:missing_or_duplicate_comparison_key")
            keys[key] = attempt
            outcome = attempt.get("outcome") or {}
            if attempt.get("state", "finished") != "finished" or attempt.get("status") not in {"completed", "finished"}:
                issues.append(f"{aid}:attempt_not_completed")
            if outcome.get("audit_verified") is not True or outcome.get("snapshot_verified") is not True:
                issues.append(f"{aid}:unverified_outcome")
            if cid in descriptors and outcome.get("homogeneous_field") is not True:
                issues.append(f"{aid}:formal_field_not_verified_homogeneous")
            if cid in descriptors:
                for field in ("snapshot_sha256", "rules_sha256"):
                    if outcome.get(field) != attempt.get(field):
                        issues.append(f"{aid}:outcome_{field}_not_scheduled_identity")
            expected = attempt.get("expected_entrants")
            if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
                issues.append(f"{aid}:missing_expected_entrants"); expected = 0
            entrants = outcome.get("entrants") or []
            if len(entrants) != expected or len({e.get("car_id") for e in entrants}) != len(entrants):
                issues.append(f"{aid}:missing_or_duplicate_entrant_outcomes")
            definition = required.get(scenario)
            if isinstance(definition, dict) and definition.get("scenario_sha256") != attempt.get("scenario_sha256"):
                issues.append(f"{aid}:scenario_definition_changed")
            hazard = attempt.get("is_hazard", False)
            if isinstance(definition, dict):
                hazard = definition.get("is_hazard", definition.get("kind") == "hazard")
            if hazard:
                hazard_count += 1
                hazard_passes += int(outcome.get("hazard_pass") is True)
                if cid == candidate_id and outcome.get("hazard_pass") is not True:
                    issues.append(f"{aid}:stopped_hazard_not_passed")
            tier = attempt.get("tier", f"cars-{expected}")
            for entrant in entrants:
                if cid in descriptors:
                    expected_descriptor = descriptors[cid]
                    actual_descriptor = entrant_descriptor(entrant)
                    if actual_descriptor["policy_key"] != expected_descriptor["policy_key"]:
                        issues.append(f"{aid}:entrant_policy_not_registered_candidate")
                    if actual_descriptor["model_sha256"] != expected_descriptor["model_sha256"]:
                        issues.append(f"{aid}:entrant_checkpoint_not_registered_candidate")
                    if actual_descriptor["detector_sha256"] != expected_descriptor["detector_sha256"]:
                        issues.append(f"{aid}:entrant_detector_not_registered_candidate")
                    if ("control_parameters" in expected_descriptor
                            and actual_descriptor.get("control_parameters") != expected_descriptor["control_parameters"]):
                        issues.append(f"{aid}:entrant_control_parameters_not_registered_candidate")
                valid = entrant.get("valid_finish") is True
                clean = entrant.get("clean_finish") is True
                finish = entrant.get("finish_time_s")
                if clean and not valid:
                    issues.append(f"{aid}:clean_without_valid_finish")
                if valid and (not _finite(finish) or finish <= 0):
                    issues.append(f"{aid}:invalid_finish_clock"); valid = clean = False
                if entrant.get("clean_finish") is None:
                    issues.append(f"{aid}:unknown_incident_audit")
                if not hazard:
                    identity[(key, entrant.get("car_id"))] = (valid, clean, finish if clean else None, tier)
                    totals[tier]["clean"] += int(clean); totals[tier]["valid"] += int(valid)
            if not hazard:
                totals[tier]["expected"] += expected
        if seen_scenarios != set(required):
            issues.append(f"{cid}:incomplete_required_suite")
        paired[cid] = keys
        identities[cid] = identity
        counts = sum(totals.values(), Counter())
        scores[cid] = {"expected": counts["expected"], "clean": counts["clean"], "valid": counts["valid"],
                       "hazard_attempts": hazard_count, "hazard_passes": hazard_passes,
                       "clean_rate": counts["clean"] / counts["expected"] if counts["expected"] else 0.,
                       "valid_rate": counts["valid"] / counts["expected"] if counts["expected"] else 0.,
                       "by_tier": dict(totals)}
    if not incumbent_id or incumbent_id == candidate_id:
        issues.append("distinct_incumbent_and_candidate_required")
    old_keys, new_keys = paired.get(incumbent_id, {}), paired.get(candidate_id, {})
    if set(old_keys) != set(new_keys):
        issues.append("unmatched_scheduled_attempts")
    for key in old_keys.keys() & new_keys.keys():
        old, new = old_keys[key], new_keys[key]
        for field in ("scenario_id", "scenario_sha256", "rules_sha256", "expected_entrants", "tier"):
            if old.get(field) is None or old.get(field) != new.get(field):
                issues.append(f"{key}:mismatched_{field}")
        old_ids = {e.get("car_id") for e in (old.get("outcome") or {}).get("entrants", [])}
        new_ids = {e.get("car_id") for e in (new.get("outcome") or {}).get("entrants", [])}
        if old_ids != new_ids:
            issues.append(f"{key}:mismatched_entrant_identities")
    old, new = scores.get(incumbent_id, {}), scores.get(candidate_id, {})
    old_tiers, new_tiers = old.get("by_tier", {}), new.get("by_tier", {})
    improvements = []
    for tier in old_tiers.keys() | new_tiers.keys():
        a, b = old_tiers.get(tier, {}), new_tiers.get(tier, {})
        if a.get("expected") != b.get("expected"):
            issues.append(f"{tier}:unequal_scheduled_entrants")
        first = (a.get("clean", 0), a.get("valid", 0))
        second = (b.get("clean", 0), b.get("valid", 0))
        if second < first:
            issues.append(f"{tier}:clean_or_valid_regression")
        elif second > first:
            improvements.append(f"{tier}:clean_or_valid_improvement")
    if new.get("hazard_passes", 0) > old.get("hazard_passes", 0):
        improvements.append("stopped_hazard_improvement")
    old_identity, new_identity = identities.get(incumbent_id, {}), identities.get(candidate_id, {})
    ratios = []
    identical_completion_sets = old_identity.keys() == new_identity.keys()
    for key in old_identity.keys() & new_identity.keys():
        a, b = old_identity[key], new_identity[key]
        identical_completion_sets &= a[:2] == b[:2]
        if a[1] and b[1]:
            ratios.append(b[2] / a[2])
    median_ratio = statistics.median(ratios) if ratios else None
    if (identical_completion_sets and len(ratios) >= 4 and median_ratio <= 1-minimum_pace_gain
            and max(ratios) <= 1.):
        improvements.append("paired_clean_pace_improvement")
    if not improvements:
        issues.append("no_verified_strict_improvement")
    if not required:
        issues.append("empty_confirmation_suite")
    return {"schema_version": 1, "incumbent_id": incumbent_id, "candidate_id": candidate_id,
            "split": split, "approved": not issues, "issues": sorted(set(issues)),
            "verified_improvements": improvements, "scores": scores,
            "snapshot_sha256s": snapshots, "clean_pace_pairs": len(ratios),
            "candidate_descriptors": descriptors,
            "median_clean_pace_ratio": median_ratio, "identical_completion_identities": identical_completion_sets,
            "scored_attempt_ids": [a.get("attempt_id") for a in selected],
            "non_scoring_attempt_ids_retained": excluded,
            "scope": "Verified improvement on the frozen matched suite; not statistical superiority on unseen tracks."}


def validate_bootstrap(state, candidate_id, required_scenarios, split="bootstrap"):
    """Verify a fresh baseline suite without pretending it beat an incumbent."""
    issues = []
    required = {case if isinstance(case, str) else case["scenario_id"]: case for case in required_scenarios}
    registration = state["candidates"].get(candidate_id, {})
    descriptor = manifest = None
    if state.get("incumbent_id") is not None:
        issues.append("incumbent_already_exists")
    try:
        descriptor = candidate_descriptor(registration)
        manifest = verify_snapshot(registration["snapshot_path"])
        if registration.get("snapshot_sha256") != manifest["snapshot_sha256"]:
            issues.append("registration_snapshot_identity_mismatch")
        if not _bundle_contains_descriptor(manifest, descriptor):
            issues.append("registered_descriptor_not_frozen_in_bundle")
    except (KeyError, ValueError, OSError) as error:
        issues.append(f"registered_candidate_not_verified:{type(error).__name__}")
    selected = [a for a in state["attempts"] if a.get("candidate_id") == candidate_id
                and a.get("split") == split and a.get("stage") in {"bootstrap", "promotion"}]
    seen, keys = set(), set()
    total = Counter(expected=0, clean=0, valid=0, hazards=0, hazard_passes=0)
    for attempt in selected:
        aid = attempt["attempt_id"]; scenario = attempt.get("scenario_id")
        seen.add(scenario)
        if scenario not in required:
            issues.append(f"{aid}:unexpected_bootstrap_scenario")
        key = attempt.get("comparison_key")
        if not key or key in keys:
            issues.append(f"{aid}:missing_or_duplicate_comparison_key")
        keys.add(key)
        scheduled = attempt.get("ledger_scheduled_at_utc", "")
        registered = registration.get("ledger_registered_at_utc", "")
        try:
            parse = lambda text: dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
            fresh = bool(scheduled and registered and parse(scheduled) >= parse(registered))
            run = state.get("run") or {}
            campaign_start = run.get("started_at", run.get("started_at_utc"))
            if campaign_start:
                fresh = fresh and parse(scheduled) >= parse(campaign_start)
        except (TypeError, ValueError):
            fresh = False
        if not fresh or attempt.get("registered_after_launch"):
            issues.append(f"{aid}:not_fresh_preregistered_trial")
        outcome = attempt.get("outcome") or {}
        if attempt.get("state") != "finished" or attempt.get("status") not in {"completed", "finished"}:
            issues.append(f"{aid}:trial_not_completed")
        if not all(outcome.get(key) is True for key in ("audit_verified", "snapshot_verified", "homogeneous_field")):
            issues.append(f"{aid}:trial_not_verified_homogeneous")
        if manifest is not None and attempt.get("snapshot_sha256") != manifest["snapshot_sha256"]:
            issues.append(f"{aid}:wrong_frozen_snapshot")
        for field in ("snapshot_sha256", "rules_sha256"):
            if not attempt.get(field) or outcome.get(field) != attempt[field]:
                issues.append(f"{aid}:outcome_{field}_mismatch")
        definition = required.get(scenario)
        if isinstance(definition, dict) and definition.get("scenario_sha256") != attempt.get("scenario_sha256"):
            issues.append(f"{aid}:scenario_definition_changed")
        entrants = outcome.get("entrants") or []
        expected = attempt.get("expected_entrants")
        if not isinstance(expected, int) or expected < 1 or len(entrants) != expected:
            issues.append(f"{aid}:missing_scheduled_entrants")
        if len({e.get("car_id") for e in entrants}) != len(entrants):
            issues.append(f"{aid}:duplicate_entrant_identities")
        for entrant in entrants:
            actual = entrant_descriptor(entrant)
            if descriptor is not None and any(actual.get(key) != value for key, value in descriptor.items()):
                issues.append(f"{aid}:entrant_not_registered_candidate")
            if entrant.get("clean_finish") is None:
                issues.append(f"{aid}:unknown_incident_audit")
            if entrant.get("valid_finish") and (not _finite(entrant.get("finish_time_s")) or entrant["finish_time_s"] <= 0):
                issues.append(f"{aid}:invalid_finish_clock")
        hazard = attempt.get("is_hazard", False)
        if hazard:
            total["hazards"] += 1
            total["hazard_passes"] += int(outcome.get("hazard_pass") is True)
            if outcome.get("hazard_pass") is not True:
                issues.append(f"{aid}:stopped_hazard_not_passed")
        else:
            total["expected"] += expected if isinstance(expected, int) and expected > 0 else 0
            total["clean"] += sum(e.get("clean_finish") is True for e in entrants)
            total["valid"] += sum(e.get("valid_finish") is True for e in entrants)
    if not required or seen != set(required):
        issues.append("incomplete_predeclared_bootstrap_suite")
    return {"schema_version": 1, "approved": not issues, "candidate_id": candidate_id, "split": split,
            "issues": sorted(set(issues)), "scored_attempt_ids": [a["attempt_id"] for a in selected],
            "required_scenario_ids": sorted(required), "descriptor": descriptor,
            "snapshot_sha256": (manifest or {}).get("snapshot_sha256"), "scores": dict(total),
            "scope": "Verified fresh baseline initialization, not a claim of superiority over another candidate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init"); init.add_argument("ledger", type=Path)
    init.add_argument("--start", default=None); init.add_argument("--hours", type=float, default=12.)
    init.add_argument("--incumbent", default=None)
    event = commands.add_parser("event"); event.add_argument("ledger", type=Path)
    event.add_argument("event_type"); event.add_argument("payload", type=Path)
    for name in ("state", "recover"):
        sub = commands.add_parser(name); sub.add_argument("ledger", type=Path)
    snapshot = commands.add_parser("snapshot"); snapshot.add_argument("destination", type=Path)
    snapshot.add_argument("files", type=Path, help="JSON mapping relative snapshot names to source paths")
    snapshot.add_argument("--metadata", type=Path)
    verify = commands.add_parser("verify"); verify.add_argument("directory", type=Path)
    register = commands.add_parser("register"); register.add_argument("ledger", type=Path)
    register.add_argument("candidate_id"); register.add_argument("snapshot", type=Path)
    register.add_argument("policy_key"); register.add_argument("--model-sha256")
    register.add_argument("--detector-sha256", default=OFFICIAL_DETECTOR_SHA256)
    register.add_argument("--control-parameters", type=Path)
    bootstrap = commands.add_parser("bootstrap"); bootstrap.add_argument("ledger", type=Path)
    bootstrap.add_argument("candidate_id"); bootstrap.add_argument("required_scenarios", type=Path)
    bootstrap.add_argument("--split", default="bootstrap")
    suite = commands.add_parser("suite"); suite.add_argument("--split", default="development")
    compare = commands.add_parser("compare"); compare.add_argument("ledger", type=Path)
    compare.add_argument("candidate"); compare.add_argument("required_scenarios", type=Path)
    compare.add_argument("--split", default="holdout"); compare.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        start = dt.datetime.fromisoformat((args.start or utc_now()).replace("Z", "+00:00"))
        if start.tzinfo is None or args.hours <= 0:
            parser.error("Use a timezone-aware start and positive duration")
        deadline = start + dt.timedelta(hours=args.hours)
        value = Ledger(args.ledger).append("run_initialized", {"started_at": start.isoformat(),
            "deadline": deadline.isoformat(), "duration_hours": args.hours, "incumbent_id": args.incumbent})
    elif args.command == "event":
        value = Ledger(args.ledger).append(args.event_type, _read_json(args.payload))
    elif args.command == "state":value = Ledger(args.ledger).state()
    elif args.command == "recover":value = {"recovered_tail": Ledger(args.ledger).recover_tail()}
    elif args.command == "snapshot":
        value = freeze_snapshot(args.destination, _read_json(args.files),
                                _read_json(args.metadata) if args.metadata else None)
    elif args.command == "verify":value = verify_snapshot(args.directory)
    elif args.command == "register":
        descriptor = {"policy_key": args.policy_key, "model_sha256": args.model_sha256,
                      "detector_sha256": args.detector_sha256}
        if args.control_parameters:
            descriptor["control_parameters"] = _read_json(args.control_parameters)
        value = Ledger(args.ledger).register_candidate(args.candidate_id, args.snapshot, descriptor)
    elif args.command == "bootstrap":
        value = Ledger(args.ledger).bootstrap_incumbent(args.candidate_id,
                    _read_json(args.required_scenarios), args.split)
    elif args.command == "suite":value = scenario_suite(args.split)
    elif args.command == "compare":
        ledger = Ledger(args.ledger); required = _read_json(args.required_scenarios)
        if args.promote:value = ledger.promote(args.candidate, required, args.split)
        else:
            state = ledger.state()
            value = compare_candidates(state["attempts"], state["incumbent_id"], args.candidate, required,
                                       args.split, snapshot_registry=state["candidates"])
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
