import copy
import datetime as dt
import json
from types import SimpleNamespace

import pytest
import scripts.prepare_final_race_jobs as final


def setup_pack(tmp_path, monkeypatch):
    roster = {"snapshot_sha256": "bundle", "policies": ["placeholder-contract"]}
    path = tmp_path/"roster.json"; path.write_text(json.dumps(roster))
    identity = final.digest(roster)
    pack = {"status": "DECLARED_ROSTER_BOUND_REQUIRES_ROOT_ENQUEUE_REVIEW", "actual_roster_path": str(path),
        "actual_roster_sha256": identity, "snapshot_path": "bundle-path", "snapshot_sha256": "bundle",
        "jobs": [{"id": i} for i in range(28)], "tournament": {"heats": list(range(4))},
        "geometry_manifest": list(range(24)), "case_file_sha256": {"fixed": "cases"},
        "capture_authority": {"fixture": "bound-capture"},
        "estimated_complete_pack_seconds": 6480, "reserve_seconds": 300, "output_root": "final-only"}
    monkeypatch.setattr(final, "Ledger", lambda _: SimpleNamespace(records=lambda: [{"event_type": "championship_roster_predeclared", "payload": {"roster_sha256": identity}}]))
    monkeypatch.setattr(final, "verify_snapshot", lambda _: {"snapshot_sha256": "bundle", "metadata": {"structural_preview_only": False}})
    expected = copy.deepcopy(pack)
    monkeypatch.setattr(final, "build_pack", lambda *args, **kwargs: expected)
    (tmp_path/"campaign.json").write_text(json.dumps({"deadline_utc": "2026-11-14T21:50:47+00:00"}))
    return pack


def test_structural_placeholders_never_execute(tmp_path):
    with pytest.raises(ValueError, match="placeholder"):
        final.require_execution_bindings({"status": "STRUCTURAL_PREVIEW_NOT_EXECUTABLE", "actual_roster_sha256": None}, tmp_path)


def test_declared_pack_detects_changed_job_or_missing_geometry(tmp_path, monkeypatch):
    pack = setup_pack(tmp_path, monkeypatch)
    now = dt.datetime.fromisoformat("2026-11-14T19:30:00+00:00")
    assert final.require_execution_bindings(pack, tmp_path, now)
    bad = copy.deepcopy(pack); bad["jobs"][0]["args"] = ["--single-policy", "favorable-driver"]
    with pytest.raises(ValueError, match="pack changed:jobs"):
        final.require_execution_bindings(bad, tmp_path, now)
    bad = copy.deepcopy(pack); bad["geometry_manifest"].pop()
    with pytest.raises(ValueError, match="pack changed:geometry_manifest"):
        final.require_execution_bindings(bad, tmp_path, now)


def test_deadline_reserves_whole_pack_not_just_one_fast_heat(tmp_path, monkeypatch):
    pack = setup_pack(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="deadline reserve"):
        final.require_execution_bindings(pack, tmp_path, dt.datetime.fromisoformat("2026-11-14T21:00:00+00:00"))


def test_replaced_case_cannot_redefine_original_ledger_predeclaration(tmp_path):
    from scripts.prepare_final_evaluation_protocols import championship_cases, geometric_confirmation_cases
    from scripts.improvement_ledger import Ledger, digest
    cases, geometry = championship_cases(), geometric_confirmation_cases()
    ledger = Ledger(tmp_path/"ledger.jsonl")
    for name, data in (("championship20", cases), ("geometric-confirmation", geometry)):
        ledger.append("final_evaluation_predeclared", {"plan": name, "definition_sha256": digest(data)})
    final.require_case_predeclarations(tmp_path, cases, geometry)
    changed = copy.deepcopy(cases)
    # A new internally consistent replacement case must still fail the original ledger binding.
    from scripts.prepare_final_evaluation_protocols import case_definition
    scene = copy.deepcopy(changed[0]["scenario"]); scene["seed"] += 1
    changed[0] = case_definition(changed[0]["case_id"], "championship", scene, changed[0]["kind"])
    ledger.append("final_evaluation_predeclared", {"plan": "championship20", "definition_sha256": digest(changed)})
    with pytest.raises(ValueError, match="original ledger predeclaration"):
        final.require_case_predeclarations(tmp_path, changed, geometry)
    duplicate = copy.deepcopy(geometry); duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="six unique"):
        final.require_case_predeclarations(tmp_path, cases, duplicate)


def test_direct_build_contract_rejects_reference_in_another_slot():
    wrong = [{"policy_key": "neural1", "model_sha256": "one"},
             {"policy_key": "reference", "model_sha256": None},
             {"policy_key": "neural2", "model_sha256": "two"},
             {"policy_key": "neural3", "model_sha256": "three"}]
    with pytest.raises(ValueError, match="reference-first"):
        final.require_common_roster("unused", {}, wrong)
