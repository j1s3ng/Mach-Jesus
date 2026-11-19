#!/usr/bin/env python3
"""Build final race jobs or an isolated CPU structural preview; never enqueue."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]
from scripts.improvement_ledger import (Ledger, OFFICIAL_DETECTOR_SHA256, digest, environment_rules_hash,
                                        freeze_snapshot, verify_snapshot)
from scripts.generate_improvement_cases import verify_case
from scripts.prepare_finalist_selection import numerical_model_identity
from scripts.analyze_final_championship import normalize_roster
from scripts.run_improvement_heat import prepare
from scripts.watch_improvement_evaluation import physics_binding, world_issues

RUNTIME = ("webots/controllers/tournament_racer/tournament_racer.py", "src/race_lidar/race_policy.py",
           "src/race_lidar/race_control.py", "src/race_lidar/hotlap_driver.py", "src/race_lidar/learned_driver.py",
           "src/race_lidar/lidar_hazards.py", "src/race_lidar/rich_driver.py", "src/race_lidar/remote.py")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def capture_authority(campaign):
    """Bind the prospective same-race LiDAR recording decision only."""
    path = Path(campaign) / 'plans/first_championship_lidar_capture_20261114T1530Z.json'
    checksum = '7bb34df81390f5129fe6a435e88558416c15add05945f12e696704e93893e53e'
    if sha(path) != checksum:
        raise ValueError('First-heat LiDAR capture authority changed')
    plan = read(path)
    if (plan.get('status') != 'ROOT_APPROVED_PROSPECTIVE_FIRST_HEAT_LIDAR_CAPTURE'
            or plan.get('first_heat_raw_pointcloud_stride_frames') != 5
            or plan.get('other27_raw_pointcloud_stride_frames') != 0
            or plan.get('actual_LiDAR_detector_period_ms_unchanged') != 200):
        raise ValueError('Recording must preserve the declared sensor cadence')
    for name, expected in plan['unchanged_case_file_sha256'].items():
        if sha(Path(campaign) / 'final-evaluation-plan' / name) != expected:
            raise ValueError('Recording authority does not bind unchanged final cases')
    records = Ledger(Path(campaign) / 'ledger.jsonl').records()
    events = [r for r in records if r['event_type'] == 'first_championship_lidar_capture_predeclared'
              and r['payload'].get('file_sha256') == checksum]
    if len(events) != 1 or any(r['sequence'] <= events[0]['sequence'] for r in records
            if r['event_type'] == 'attempt_scheduled'
            and r['payload'].get('split') in ('championship', 'geometric-confirmation')):
        raise ValueError('LiDAR capture must be declared before every final case')
    return {'path': str(path.resolve()), 'file_sha256': checksum,
            'ledger_declaration_sha256': events[0]['sha256'], 'first_heat_point_stride': 5,
            'other27_point_stride': 0, 'detector_period_ms': 200}


def require_case_predeclarations(campaign, cases, geometry):
    """Bind exact canonical cases to the FIRST immutable plan ledger events."""
    if (len(cases) != 4 or {c["scenario"]["rotation"] for c in cases} != {0, 1, 2, 3}
            or any(c["scenario"]["cars"] != 20 or c["is_hazard"] or c["split"] != "championship" for c in cases)):
        raise ValueError("Exactly four reserved20-car rotations0through3 required")
    if (len(geometry) != 6 or len({c["scenario_id"] for c in geometry}) != 6
            or any(not c["is_hazard"] or c["split"] != "geometric-confirmation" for c in geometry)):
        raise ValueError("Exactly six unique reserved geometry cases required")
    for case in cases + geometry: verify_case(case)
    records = Ledger(Path(campaign) / "ledger.jsonl").records()
    for name, definitions in (("championship20", cases), ("geometric-confirmation", geometry)):
        declaration = next((r for r in records if r["event_type"] == "final_evaluation_predeclared"
                            and r["payload"].get("plan") == name), None)
        if declaration is None or declaration["payload"].get("definition_sha256") != digest(definitions):
            raise ValueError("Final cases differ from original ledger predeclaration:" + name)


def require_common_roster(snapshot, manifest, descriptors):
    if (len(descriptors) != 4 or descriptors[0]["policy_key"] != "reference"
            or sum(row["policy_key"] == "reference" for row in descriptors) != 1
            or any(row["model_sha256"] is None for row in descriptors[1:])):
        raise ValueError("Exactly reference-first plus three learned policies required")
    policies = manifest.get("metadata", {}).get("policies", [])
    if [row.get("policy") for row in policies] != [row["policy_key"] for row in descriptors]:
        raise ValueError("Frozen loader policy order differs from actual roster order")
    numerical = []
    for policy, descriptor in zip(policies[1:], descriptors[1:]):
        relative = Path(policy["model_relpath"])
        if relative.is_absolute() or ".." in relative.parts: raise ValueError("Unsafe selected model path")
        model = Path(snapshot) / relative
        if sha(model) != descriptor["model_sha256"]: raise ValueError("Selected policy checkpoint differs")
        numerical.append(numerical_model_identity(model)["numerical_model_sha256"])
    if len(set(numerical)) != 3: raise ValueError("Three numerically distinct learned drivers required")


def freeze_common_bundle(base, roster, models, destination, *, structural_preview=False):
    """Copy frozen base runtime plus exactly the selected models, without registration."""
    base = Path(base).resolve()
    manifest = verify_snapshot(base)
    if len(roster) != 4 or roster[0]["policy_key"] != "reference":
        raise ValueError("Reference plus three ordered learned descriptors required")
    if any(row["detector_sha256"] != OFFICIAL_DETECTOR_SHA256 or row["control_parameters"] != {"visible_distance_m": 70.} for row in roster):
        raise ValueError("Final driver comparison requires official detector and70m for every policy")
    files = {row["path"]: base / row["path"] for row in manifest["files"]
             if not row["path"].startswith(("models/", "detectors/"))}
    policies, numeric = [], []
    for descriptor in roster:
        policy = descriptor["policy_key"]
        entry = {"policy": policy, "label": policy}
        if policy != "reference":
            path = Path(models[policy]).resolve()
            if sha(path) != descriptor["model_sha256"]:
                raise ValueError("Selected checkpoint binding failed:" + policy)
            numeric.append(numerical_model_identity(path)["numerical_model_sha256"])
            name = "models/" + policy + ".npz"
            files[name] = path
            entry["model_relpath"] = name
        policies.append(entry)
    if len(set(numeric)) != 3:
        raise ValueError("Three numerically distinct learned models required")
    frozen = freeze_snapshot(destination, files, metadata={"candidate_id": "final_common_bundle",
        "policies": policies, "policy_parameters": {"visible_distance_m": 70.}, "protocol": "race-improvement-v2",
        "structural_preview_only": structural_preview, "qualification_claim": False})
    frozen_paths = {row["path"]: row["sha256"] for row in frozen["files"]}
    base_paths = {row["path"]: row["sha256"] for row in manifest["files"]}
    if any(frozen_paths.get(name) != base_paths.get(name) for name in RUNTIME):
        raise ValueError("Common runnable control sources differ from frozen reference")
    normalize_roster(roster, frozen)
    return frozen


def build_pack(campaign, snapshot, roster, *, output_root="runs/improvement-12h/final-championship-001", actual_roster=None):
    campaign, snapshot = Path(campaign).resolve(), Path(snapshot).resolve()
    manifest = verify_snapshot(snapshot)
    descriptors = normalize_roster(roster, manifest)
    require_common_roster(snapshot, manifest, descriptors)
    if any(row["detector_sha256"] != OFFICIAL_DETECTOR_SHA256 or row["control_parameters"] != {"visible_distance_m": 70.} for row in descriptors):
        raise ValueError("Official9409/70 required")
    cases = read(campaign / "final-evaluation-plan/championship20.json")
    geometry = read(campaign / "final-evaluation-plan/geometric-confirmation.json")
    require_case_predeclarations(campaign, cases, geometry)
    recording = capture_authority(campaign)
    reference_manifest = verify_snapshot(campaign / "snapshots/formal-v1-fast70")
    expected_sources = {row["path"]: row["sha256"] for row in reference_manifest["files"]}
    actual_sources = {row["path"]: row["sha256"] for row in manifest["files"]}
    if any(actual_sources.get(name) != expected_sources.get(name) for name in RUNTIME):
        raise ValueError("Common actual8module source differs from frozen Fast70")
    if physics_binding(manifest) != physics_binding(reference_manifest):
        raise ValueError("Reference/vehicle/generator physics differ from frozen Fast70")
    roster_hash = None
    if actual_roster:
        declared = read(actual_roster)
        if declared["snapshot_sha256"] != manifest["snapshot_sha256"] or declared["policies"] != descriptors:
            raise ValueError("Actual declared roster does not bind this bundle/order")
        roster_hash = digest(declared)
    jobs, race_paths, geo_paths = [], [], []
    common = ["--source-root", str(snapshot), "--reference", str(snapshot / "reference.npz"),
        "--visibility", "70", "--vision-url", "http://127.0.0.1:8770", "--secondary-url", "http://127.0.0.1:8771",
        "--detector-sha", OFFICIAL_DETECTOR_SHA256, "--stage", "engineering", "--pointcloud-stride", "0"]
    for case in cases:
        rotation = case["scenario"]["rotation"]
        relative = "heats/shift" + str(rotation)
        args = list(common)
        if rotation == 0:
            args[args.index('--pointcloud-stride') + 1] = '5'
        args += ["--candidate-id", "final-championship-001", "--split", "championship", "--secondary-cars", "8"] + case["cli_args"]
        if rotation != 0: args += ["--no-video"]
        jobs.append({"job_id": "final-championship20-shift" + str(rotation), "output": output_root + "/" + relative,
            "estimated_wall_seconds": 900, "max_wall_seconds": 1800, "args": args,
            "final_execution_binding": {"roster_sha256": roster_hash, "snapshot_sha256": manifest["snapshot_sha256"],
                "case_sha256": case["scenario_sha256"], "split": "championship"}})
        race_paths.append({"path": relative})
    for descriptor in descriptors:
        policy = descriptor["policy_key"]
        for case in geometry:
            relative = "geometry/" + policy + "/" + case["case_id"]
            args = common + ["--candidate-id", "final-geometry-" + policy, "--single-policy", policy,
                "--split", "geometric-confirmation", "--secondary-cars", "0", "--no-video"] + case["cli_args"]
            jobs.append({"job_id": "final-geometry-" + policy + "-" + case["case_id"], "output": output_root + "/" + relative,
                "estimated_wall_seconds": 120, "max_wall_seconds": 300, "args": args,
                "final_execution_binding": {"roster_sha256": roster_hash, "snapshot_sha256": manifest["snapshot_sha256"],
                    "case_sha256": case["scenario_sha256"], "split": "geometric-confirmation"}})
            geo_paths.append({"path": relative})
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STRUCTURAL_PREVIEW_NOT_EXECUTABLE" if not roster_hash else "DECLARED_ROSTER_BOUND_REQUIRES_ROOT_ENQUEUE_REVIEW",
        "snapshot_path": str(snapshot), "snapshot_sha256": manifest["snapshot_sha256"], "policies": descriptors,
        "actual_roster_path": str(Path(actual_roster).resolve()) if actual_roster else None, "actual_roster_sha256": roster_hash,
        "output_root": output_root, "jobs": jobs, "tournament": {"heats": race_paths, "roster_sha256": roster_hash},
        "geometry_manifest": geo_paths, "planned_case_counts": {"championship": 4, "geometric-confirmation": 24},
        "capture_authority": recording,
        "excluded_from_training": True, "fresh_confirmation_is_separate": True,
        "reserve_seconds": 300, "estimated_complete_pack_seconds": sum(row["estimated_wall_seconds"] for row in jobs),
        "case_file_sha256": {name: sha(campaign / "final-evaluation-plan" / name) for name in ("championship20.json", "geometric-confirmation.json")},
        "video": "First heat records video and1Hz rawclouds for aligned replay; detector/driver remain5Hz. Other27 cases omit rawclouds. Physical sensors/step/driver inputs remain identical; recording is not an NN feature.",
        "queues_or_campaign_roster_changed": False}


def require_execution_bindings(pack, campaign, now=None):
    """Fail closed before root dispatches/enqueues a real final pack."""
    if not pack.get("actual_roster_sha256") or pack.get("status") == "STRUCTURAL_PREVIEW_NOT_EXECUTABLE":
        raise ValueError("A structural placeholder pack cannot execute")
    roster = read(pack["actual_roster_path"])
    if digest(roster) != pack["actual_roster_sha256"]:
        raise ValueError("Actual frozen roster changed")
    state = Ledger(Path(campaign) / "ledger.jsonl")
    if not any(r["event_type"] == "championship_roster_predeclared" and r["payload"].get("roster_sha256") == pack["actual_roster_sha256"] for r in state.records()):
        raise ValueError("No actual campaign roster declaration")
    manifest = verify_snapshot(pack["snapshot_path"])
    if manifest["snapshot_sha256"] != pack["snapshot_sha256"] or manifest["metadata"].get("structural_preview_only"):
        raise ValueError("Source bundle changed or remains a placeholder")
    expected = build_pack(campaign, pack["snapshot_path"], roster["policies"],
                          output_root=pack["output_root"], actual_roster=pack["actual_roster_path"])
    for key in ("jobs", "tournament", "geometry_manifest", "case_file_sha256", "capture_authority", "estimated_complete_pack_seconds", "reserve_seconds"):
        if pack.get(key) != expected[key]:
            raise ValueError("Declared final pack changed:" + key)
    deadline = datetime.fromisoformat(read(Path(campaign) / "campaign.json")["deadline_utc"])
    now = now or datetime.now(timezone.utc)
    if (deadline - now).total_seconds() < pack["estimated_complete_pack_seconds"] + pack["reserve_seconds"]:
        raise ValueError("Complete final pack does not fit declared deadline reserve")
    if len(pack["jobs"]) != 28:
        raise ValueError("Complete four-plus24 planned pack required")
    return True


def preview(campaign, destination):
    """Run actual prepare/world-generation in an isolated temporary bundle only."""
    campaign, destination = Path(campaign).resolve(), Path(destination)
    if destination.exists(): raise ValueError("Choose a new preview report path")
    plan = read(campaign / "plans/finalist_roster_selection_draft_20261114T1415Z.json")
    roster = [{"policy_key": "reference", "model_sha256": None, "detector_sha256": OFFICIAL_DETECTOR_SHA256,
               "control_parameters": {"visible_distance_m": 70.}}]
    models = {}
    for row in plan["primary_candidates"]:
        roster.append({"policy_key": row["policy_key"], "model_sha256": row["model_sha256"],
                       "detector_sha256": OFFICIAL_DETECTOR_SHA256, "control_parameters": {"visible_distance_m": 70.}})
        models[row["policy_key"]] = row["model_path"]
    temp = Path(tempfile.mkdtemp(prefix="final-race-structural-preview-"))
    frozen = freeze_common_bundle(campaign / "snapshots/formal-v1-fast70", roster, models, temp / "bundle", structural_preview=True)
    pack = build_pack(campaign, temp / "bundle", roster)
    cases = read(campaign / "final-evaluation-plan/championship20.json")
    geometry = read(campaign / "final-evaluation-plan/geometric-confirmation.json")
    slots, checked = defaultdict(list), []
    for index, job in enumerate(pack["jobs"]):
        case = cases[index] if index < 4 else geometry[(index - 4) % 6]
        scenario = case["scenario"]
        single = None if index < 4 else roster[(index - 4) // 6]["policy_key"]
        args = argparse.Namespace(**scenario, output=temp / "outputs" / job["job_id"], source_root=temp / "bundle",
            reference=temp / "bundle/reference.npz", policies=None, single_policy=single, visibility=70.,
            no_video=index != 0, pointcloud_stride=5 if index == 0 else 0, capacity=False, secondary_cars=8 if index < 4 else 0,
            secondary_url="http://127.0.0.1:8771", vision_url="http://127.0.0.1:8770",
            detector_sha=OFFICIAL_DETECTOR_SHA256, split=case["split"])
        config_path, world = prepare(args)
        config = read(config_path)
        problems = world_issues(config, case)
        urls = Counter(entry.get("vision_url", config["vision_url"]) for entry in config["entrants"] if not entry.get("is_static"))
        if index < 4:
            if urls != {"http://127.0.0.1:8770": 12, "http://127.0.0.1:8771": 8}: problems.append("12plus8_endpoint_placement_failed")
            for entry in config["entrants"]:
                expected = roster[(entry["grid_index"] + scenario["rotation"]) % 4]
                if entry["policy"] != expected["policy_key"]: problems.append("cyclic_assignment_failed")
                slots[entry["policy"]].append(entry["grid_index"])
                if expected["model_sha256"] is not None and sha(entry["model_path"]) != expected["model_sha256"]: problems.append("model_checksum_failed")
        if not config["excluded_from_training"] or config["snapshot_sha256"] != frozen["snapshot_sha256"]:
            problems.append("frozen_or_training_exclusion_failed")
        text = world.read_text()
        if "optimalThreadCount 8" not in text or f"randomSeed {scenario['seed']}" not in text:
            problems.append("world_physics_or_seed_failed")
        if config["physics_threads"] != 8 or config["rules"]["physics_step_ms"] != 20 or config["sensor_period_ms"] != 200:
            problems.append("common_cadence_failed")
        if config['pointcloud_stride_frames'] != (5 if index == 0 else 0):
            problems.append('raw_cloud_capture_stride_failed')
        checked.append({"job_id": job["job_id"], "case_sha256": case["scenario_sha256"], "config_sha256": sha(config_path),
            "world_sha256": sha(world), "rules_sha256": environment_rules_hash(config["rules"]),
            "cars": len(config["entrants"]), "vision_endpoint_counts": dict(urls), "record_video": config["record_video"],
            "pointcloud_stride_frames": config['pointcloud_stride_frames'],
            "excluded_from_training": config["excluded_from_training"], "issues": problems})
    for policy, values in slots.items():
        if sorted(values) != list(range(20)): raise ValueError("Full exact slot distribution failed:" + policy)
    try: require_execution_bindings(pack, campaign)
    except ValueError: blocked = True
    else: blocked = False
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "UNQUALIFIED_STRUCTURAL_PREVIEW_ONLY",
        "verified": blocked and all(not row["issues"] for row in checked), "placeholder_execution_blocked": blocked,
        "temporary_directory": str(temp), "actual_roster_declared": False, "finalist_selected": False,
        "Webots_or_inference_launched": False, "ledger_or_queue_mutated": False,
        "runtime8_sha256": {name: sha(temp / "bundle" / name) for name in RUNTIME},
        "frozen_snapshot_sha256": frozen["snapshot_sha256"], "pack": pack, "case_checks": checked,
        "exact_full_grid_slots_by_policy": dict(slots), "source_sha256": sha(__file__)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, value in (("tournament.json", pack["tournament"]), ("geometry_manifest.json", pack["geometry_manifest"]), ("jobs.json", pack["jobs"])):
        (temp / name).write_text(json.dumps(value, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "runs/improvement-12h")
    parser.add_argument("--preview-output", type=Path, required=True)
    args = parser.parse_args()
    report = preview(args.campaign, args.preview_output)
    print(json.dumps({key: report[key] for key in ("verified", "placeholder_execution_blocked", "temporary_directory", "actual_roster_declared")}))


if __name__ == "__main__": main()
