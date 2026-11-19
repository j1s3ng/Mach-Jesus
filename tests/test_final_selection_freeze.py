"""Final commit mechanics; synthetic qualification is isolated at its API boundary."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from race_lidar.rich_driver import save_rich_model
from scripts import freeze_final_engineering_selection as freeze
from scripts.improvement_ledger import Ledger, OFFICIAL_DETECTOR_SHA256, digest, freeze_snapshot
from scripts.prepare_finalist_selection import numerical_model_identity


def setup(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    (campaign / "formal-cases").mkdir(parents=True)
    (campaign / "queue.json").write_text("[]\n")
    (campaign / "formal-cases/confirmation.json").write_text(json.dumps(freeze.formal_cases("confirmation")))
    ledger = Ledger(campaign / "ledger.jsonl")
    ledger.append("run_initialized", {"started_at_utc": datetime.now(timezone.utc).isoformat()})
    raw = tmp_path / "raw"
    raw.mkdir()
    files = {}
    for name in (*freeze.RUNTIME, "reference.npz"):
        source = raw / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("synthetic test source for " + name)
        files[name] = source
    metadata = {"policies": [{"policy": "reference"}], "policy_parameters": {"visible_distance_m": 70.}}
    base = campaign / "snapshots/formal-v1-fast70"
    base_manifest = freeze_snapshot(base, files, metadata)
    reference = {"policy_key": "reference", "model_sha256": None,
        "detector_sha256": OFFICIAL_DETECTOR_SHA256, "control_parameters": {"visible_distance_m": 70.}}
    ledger.register_candidate(freeze.BASELINE, base, reference)
    specs, candidates = [], []
    for index, cid in enumerate(freeze.PRIMARY):
        policy = f"neural_test_{index}"
        model = raw / (policy + ".npz")
        save_rich_model(model, [np.full((6, 48), .001 * (index + 1), dtype=np.float32)],
            [np.zeros(6, dtype=np.float32)], np.zeros(48), np.ones(48), {"seed": index})
        identity = numerical_model_identity(model)
        desc = {"policy_key": policy, "model_sha256": identity["checkpoint_sha256"],
            "detector_sha256": OFFICIAL_DETECTOR_SHA256, "control_parameters": {"visible_distance_m": 70.}}
        snap = campaign / "snapshots" / cid
        manifest = freeze_snapshot(snap, {**files, "models/model.npz": model}, {
            "policies": [{"policy": policy, "model_relpath": "models/model.npz"}],
            "policy_parameters": {"visible_distance_m": 70.}})
        ledger.register_candidate(cid, snap, desc)
        specs.append({"registered_engineering_id": cid, "model_sha256": identity["checkpoint_sha256"],
                      "snapshot_sha256": manifest["snapshot_sha256"]})
        candidates.append({"candidate_id": cid, "policy_key": policy, "model_sha256": identity["checkpoint_sha256"],
            "numerical_model_sha256": identity["numerical_model_sha256"], "mean_finish_time_s": 280. - index,
            "selection20_run": str(campaign / "engineering" / cid), "eligible": True, "known": True})
    plan = {"primary_candidates": specs, "CEM_candidate": {"state_path": "/training/declared-cem/state.json"},
        "common_actual_runtime_source8_sha256": {r["path"]: r["sha256"] for r in base_manifest["files"] if r["path"] in freeze.RUNTIME},
        "selection20_reference_sha256": next(r["sha256"] for r in base_manifest["files"] if r["path"] == "reference.npz")}
    ranked = freeze.eligible_ranking(candidates)
    report = {"status": "VERIFIED_ENGINEERING_SELECTION", "formal_outcomes_used": False,
        "selection_protocol_sha256": freeze.APPROVAL_SHA, "prospective_draft_sha256": freeze.DRAFT_SHA,
        "primary_candidates": candidates, "CEM": {"known": True, "eligible": False},
        "all_primary_and_CEM_outcomes_known": True, "provisional_eligible_rank": ranked,
        "selected_finalist_id": ranked[0]["candidate_id"],
        "selected_roster_learned_ids": [r["candidate_id"] for r in ranked]}
    report["selection_semantic_sha256"] = digest(report)
    report_path = campaign / "engineering-report.json"
    report_path.write_text(json.dumps(report))
    # Positive tests focus on concrete immutable bundle/declaration ordering.
    # No synthetic fixture pretends to pass the real eligibility auditor.
    monkeypatch.setattr(freeze, "verify_engineering_selection", lambda _p, _c: copy.deepcopy(report))
    monkeypatch.setattr(freeze, "protocol", lambda _c: copy.deepcopy(plan))
    return campaign, ledger, report_path, report, campaign / "final-selection-test", plan


def test_preparation_never_freezes_registers_declares_or_changes_queue(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    before, queue = ledger.path.read_bytes(), (campaign / "queue.json").read_bytes()
    blueprint = freeze.selection_blueprint(campaign, report_path, destination)
    assert blueprint["finalist_id"] == report["selected_finalist_id"]
    assert not destination.exists() and ledger.path.read_bytes() == before
    assert (campaign / "queue.json").read_bytes() == queue
    with pytest.raises(ValueError, match="Explicit --freeze"):
        freeze.freeze_selection(campaign, report_path, destination)


def test_actual_test_commit_reuses_existing_aliases_and_binds_rank_before_declaration(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    registered_before = copy.deepcopy(ledger.state()["candidates"])
    queue_before = (campaign / "queue.json").read_bytes()
    result = freeze.freeze_selection(campaign, report_path, destination, execute=True)
    assert result["finalist_id"] == report["selected_finalist_id"]
    assert ledger.state()["candidates"] == registered_before
    assert (campaign / "queue.json").read_bytes() == queue_before
    receipt = freeze.read(result["selection_receipt"])
    roster = freeze.read(result["roster_path"])
    assert receipt["selected_roster_learned_ids"] == report["selected_roster_learned_ids"]
    assert roster["policies"][0]["policy_key"] == "reference"
    assert [r["policy_key"] for r in roster["policies"][1:]] == [r["policy_key"] for r in report["provisional_eligible_rank"]]
    bound = freeze.read(destination / "roster_selection_receipt.json")
    assert bound["roster"] == roster
    assert bound["engineering_selection_report"]["sha256"] == freeze.sha(report_path)
    events = {r["event_type"]: r for r in ledger.records()}
    assert events["final_roster_engineering_selection_bound"]["sequence"] < events["championship_roster_predeclared"]["sequence"]
    assert events["championship_roster_predeclared"]["sequence"] < events["finalist_frozen_for_confirmation"]["sequence"]
    assert receipt["formal_outcomes_used_for_selection"] is False
    assert not (Path(result["selection_receipt"]).stat().st_mode & 0o222)
    assert ledger.state()["incumbent_id"] is None  # Bootstrap remains separate.
    from scripts.final_roster_binding import verify_frozen_roster_chain
    chain = verify_frozen_roster_chain(campaign, result["roster_path"], result["selection_receipt"])
    assert chain["verified"] and chain["finalist_id"] == report["selected_finalist_id"]
    assert chain["roster"]["policies"] == roster["policies"]


@pytest.mark.parametrize("split", ["confirmation", "championship", "geometric-confirmation"])
def test_reserved_pending_queue_job_blocks_freeze_before_any_output(tmp_path, monkeypatch, split):
    campaign, ledger, report_path, _, destination, _ = setup(tmp_path, monkeypatch)
    (campaign / "queue.json").write_text(json.dumps([{"job_id": "too-early", "args": ["--split", split]}]))
    with pytest.raises(ValueError, match="already queued"):
        freeze.selection_blueprint(campaign, report_path, destination)
    assert not destination.exists()


def test_any_prior_reserved_schedule_or_freeze_is_rejected(tmp_path, monkeypatch):
    campaign, ledger, report_path, _, destination, _ = setup(tmp_path, monkeypatch)
    records = [{"event_type": "attempt_scheduled", "payload": {"split": "confirmation"}}]
    with pytest.raises(ValueError, match="scheduled before"):
        freeze.require_first_freeze(campaign, destination, records)
    for event in freeze.FREEZE_EVENTS:
        with pytest.raises(ValueError, match="already exists"):
            freeze.require_first_freeze(campaign, destination, [{"event_type": event, "payload": {}}])


def test_rank_tampering_and_substituted_finalist_are_rejected(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    report["selected_finalist_id"] = report["provisional_eligible_rank"][-1]["candidate_id"]
    with pytest.raises(ValueError, match="exact prospective"):
        freeze.selection_blueprint(campaign, report_path, destination)


def test_underqualified_or_nonprospective_selection_is_rejected(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    report["status"] = "INCOMPLETE"
    with pytest.raises(ValueError, match="Complete approved"):
        freeze.selection_blueprint(campaign, report_path, destination)
    report["status"], report["selection_protocol_sha256"] = "VERIFIED_ENGINEERING_SELECTION", "different"
    with pytest.raises(ValueError, match="Complete approved"):
        freeze.selection_blueprint(campaign, report_path, destination)


def test_real_verifier_boundary_rejects_a_three_field_placeholder(tmp_path, monkeypatch):
    path = tmp_path / "placeholder.json"
    path.write_text(json.dumps({"status": "VERIFIED_ENGINEERING_SELECTION", "selected_finalist_id": "fake", "formal_outcomes_used": False}))
    with pytest.raises(ValueError, match="semantic hash"):
        freeze.verify_engineering_selection(path, tmp_path)


def test_changed_common_reference_or_runtime8_cannot_enter_bundle(tmp_path, monkeypatch):
    campaign, ledger, report_path, _, destination, plan = setup(tmp_path, monkeypatch)
    plan["selection20_reference_sha256"] = "wrong"
    with pytest.raises(ValueError, match="runtime8/reference"):
        freeze.selection_blueprint(campaign, report_path, destination)


def test_genuine_extra_eligible_candidate_cannot_be_invented(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    report["primary_candidates"].append({**report["primary_candidates"][0], "candidate_id": "unapproved-extra"})
    with pytest.raises(ValueError, match="outside the approved"):
        freeze.selection_blueprint(campaign, report_path, destination)


def test_known_ineligible_cem_winner_does_not_block_three_eligible_primaries(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    report["CEM"] = {"known": True, "eligible": False, "candidate_id": "declared-cem-g1-c1", "search_id": "declared-cem"}
    blueprint = freeze.selection_blueprint(campaign, report_path, destination)
    assert blueprint["learned_ids_in_rank_order"] == report["selected_roster_learned_ids"]


def test_numerical_clone_cannot_supply_third_rank_slot(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    report["primary_candidates"][-1]["numerical_model_sha256"] = report["primary_candidates"][0]["numerical_model_sha256"]
    with pytest.raises(ValueError, match="exact prospective"):
        freeze.selection_blueprint(campaign, report_path, destination)


def add_conditional_backup(campaign, ledger, report):
    from scripts.conditional_rich006_backup import BACKUP_ID, POLICY
    model = campaign / "synthetic-backup.npz"
    shapes = [(192, 48), (192, 192), (96, 192), (6, 96)]
    save_rich_model(model, [np.full(s, .004, np.float32) for s in shapes],
        [np.zeros(s[0], np.float32) for s in shapes], np.zeros(48), np.ones(48), {"seed": 7})
    ident = numerical_model_identity(model)
    base = campaign / "snapshots/formal-v1-fast70"
    files = {r["path"]: base / r["path"] for r in freeze.verify_snapshot(base)["files"]}
    snap = campaign / "snapshots/conditional-backup"
    freeze_snapshot(snap, {**files, "models/backup.npz": model}, {"policies": [
        {"policy": POLICY, "model_relpath": "models/backup.npz"}], "policy_parameters": {"visible_distance_m": 70.}})
    ledger.register_candidate(BACKUP_ID, snap, {"policy_key": POLICY, "model_sha256": ident["checkpoint_sha256"],
        "detector_sha256": OFFICIAL_DETECTOR_SHA256, "control_parameters": {"visible_distance_m": 70.}})
    row = {"candidate_id": BACKUP_ID, "policy_key": POLICY, "model_sha256": ident["checkpoint_sha256"],
        "numerical_model_sha256": ident["numerical_model_sha256"], "mean_finish_time_s": 270.,
        "selection20_run": "/engineering/synthetic-backup", "eligible": True, "known": True}
    report["conditional_backup_candidate"] = row
    report["fallback006seed7"] = {"required": True, "activation_verified": True, "status": "PHYSICALLY_QUALIFIED"}
    return row


def test_only_verified_conditional_backup_fills_actual_primary_shortfall(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    report["primary_candidates"][-1]["eligible"] = False
    row = add_conditional_backup(campaign, ledger, report)
    rank = freeze.eligible_ranking([*report["primary_candidates"], row])
    report.update(provisional_eligible_rank=rank, selected_finalist_id=rank[0]["candidate_id"],
                  selected_roster_learned_ids=[r["candidate_id"] for r in rank])
    blueprint = freeze.selection_blueprint(campaign, report_path, destination)
    assert blueprint["finalist_id"] == row["candidate_id"]
    assert len(blueprint["learned_ids_in_rank_order"]) == 3
    report["fallback006seed7"]["activation_verified"] = False
    with pytest.raises(ValueError, match="authorized fitting and all seven"):
        freeze.selection_blueprint(campaign, report_path, destination)


def test_backup_cannot_displace_three_eligible_primaries_even_with_fabricated_flags(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    add_conditional_backup(campaign, ledger, report)
    with pytest.raises(ValueError, match="authorized fitting and all seven"):
        freeze.selection_blueprint(campaign, report_path, destination)


def test_failure_is_retained_and_blocks_silent_second_freeze(tmp_path, monkeypatch):
    campaign, ledger, report_path, _, destination, _ = setup(tmp_path, monkeypatch)
    def fail(*_a, **_k): raise OSError("test interrupted bundle copy")
    monkeypatch.setattr(freeze, "freeze_common_bundle", fail)
    with pytest.raises(OSError):
        freeze.freeze_selection(campaign, report_path, destination, execute=True)
    assert destination.exists()
    assert any(r["event_type"] == "final_selection_freeze_failed_preserved" for r in ledger.records())
    with pytest.raises(ValueError, match="already exists"):
        freeze.selection_blueprint(campaign, report_path, campaign / "another-destination")


def test_whitespace_only_report_rewrite_during_copy_does_not_commit_stale_byte_binding(tmp_path, monkeypatch):
    campaign, ledger, report_path, report, destination, _ = setup(tmp_path, monkeypatch)
    calls = []
    def verify(*_a, **_k):
        calls.append(True)
        if len(calls) == 2:
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        return copy.deepcopy(report)
    monkeypatch.setattr(freeze, "verify_engineering_selection", verify)
    with pytest.raises(ValueError, match="changed during commit"):
        freeze.freeze_selection(campaign, report_path, destination, execute=True)
    assert not (destination / "selection_receipt.json").exists()
    assert not any(r["event_type"] == "finalist_frozen_for_confirmation" for r in ledger.records())
