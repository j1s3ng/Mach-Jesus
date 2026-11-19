"""Current evidence, rather than terminal markers, closes CEM selection."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_engineering_finalist_selection as audit
from race_lidar.rich_driver import save_rich_model


@pytest.fixture
def search(tmp_path, monkeypatch):
    obj = audit.Auditor.__new__(audit.Auditor)
    obj.campaign, obj.cache_dir, obj.inventory = tmp_path, tmp_path / "cache", {}
    foundation = tmp_path / "foundation.npz"
    foundation.write_bytes(b"foundation")
    identity = {"checkpoint_sha256": audit.sha(foundation), "numerical_model_sha256": "foundation"}
    obj.plan = {"CEM_candidate": {"state_path": str(tmp_path / "state.json")},
                "ranking": {"identity_clone_proof": {"foundation": identity}}}
    names = ("scripts/race_reward_cem.py", "scripts/cem_evidence.py", "scripts/improvement_queue_edit.py",
             "scripts/training_sources.py", "scripts/evaluate_improvement_attempt.py", "scripts/improvement_ledger.py")
    state = {"search_id": "search", "status": "complete", "reason": "no_verified_development_gain",
             "output": str(tmp_path), "foundation_model": str(foundation), "foundation_sha256": audit.sha(foundation),
             "source_sha256": {n: "source" for n in names}, "seed": 7, "generation_limit": 2,
             "candidates_per_generation": 6, "formal_sources_for_learning": [], "generations": []}
    artifacts, rewards, evidence = {}, {}, {}
    mean, std = audit.IDENTITY.copy(), audit.INITIAL_STD.copy()
    for gen in range(2):
        rows = []
        for index, params in enumerate(audit.sample_generation(mean, std, gen, 7)):
            cid = f"search-g{gen}-c{index}"
            candidate = {"candidate_id": cid, "policy": cid, "parameters": params.tolist(), "generation": gen,
                "index": index, "model_path": str(tmp_path / cid / "model.npz"), "model_sha256": "weights-" + cid,
                "snapshot_path": str(tmp_path / cid / "snapshot"), "snapshot_sha256": "source-" + cid}
            candidate["expected_cases"] = {j["job_id"]: {"job_id": j["job_id"]} for j in audit.job_batch(candidate)}
            artifacts[str(Path(candidate["model_path"]).parent / "candidate.json")] = copy.deepcopy(candidate)
            cases = [{"run": str(tmp_path / "engineering" / j["job_id"]), "kind": "hazard" if k < 4 else "field",
                      "usable": True, "passed": True} for k, j in enumerate(audit.job_batch(candidate))]
            reward = {"feasible": True, "reward": 100. + gen + index / 10., "reason": "all_required_cases_clean",
                      "mean_finish_time_s": 250. - gen - index, "cases": cases}
            candidate["reward"] = copy.deepcopy(reward)
            artifacts[str(Path(candidate["model_path"]).parent / "physical_reward.json")] = copy.deepcopy(reward)
            rewards[cid] = copy.deepcopy(reward)
            rows.append(candidate)
        mean, std, elites = audit.update_distribution(rows, mean, std)
        state["generations"].append({"index": gen, "completed": True, "candidates": rows, "elite_ids": elites})
    state.update(mean=mean.tolist(), std=std.tolist())
    candidates = [c for g in state["generations"] for c in g["candidates"]]
    winner, baseline = candidates[-1], candidates[0]
    state["winner"] = copy.deepcopy(winner)
    validation_jobs = [j for c in (baseline, winner) for j in audit.job_batch(c, seed=211, rotation=1, validation=True)]
    state["validation"] = {"baseline": copy.deepcopy(baseline), "winner": copy.deepcopy(winner), "seed": 211,
        "rotation": 1, "used_for_cem_updates": False, "jobs": [j["job_id"] for j in validation_jobs],
        "expected_cases": {j["job_id"]: {"job_id": j["job_id"]} for j in validation_jobs},
        "development_improvement": False, "results": {}}
    obj.jobs = [j for c in candidates for j in audit.job_batch(c)] + validation_jobs
    obj.job_by_id = {j["job_id"]: j for j in obj.jobs}
    obj.completions = {j["job_id"]: {"exit_code": 0} for j in obj.jobs}
    obj.state = {"candidates": {}, "attempts": []}
    for role, elapsed in (("baseline", 245.), ("winner", 247.)):
        row = {"passed": True, "usable": True, "audit_verified": True, "source_contract_verified": True,
               "mean_finish_time_s": elapsed, "outcome_sha256": role}
        evidence[state["validation"][role]["candidate_id"] + "-validation-field4"] = row
        state["validation"]["results"][role] = copy.deepcopy(row)
    for c in candidates:
        for j in audit.job_batch(c):
            evidence[j["job_id"]] = {"usable": True, "audit_verified": True, "source_contract_verified": True, "passed": True}
    original_remember = audit.Auditor.remember
    def remember(path, expected=None):
        path = Path(path)
        if path.name == "state.json": return copy.deepcopy(state)
        if str(path) in artifacts: return copy.deepcopy(artifacts[str(path)])
        if path == foundation: return audit.sha(foundation)
        if str(path).endswith(".py"): return "source"
        return original_remember(obj, path, expected)
    monkeypatch.setattr(obj, "remember", remember)
    monkeypatch.setattr(audit, "numerical_model_identity", lambda p: copy.deepcopy(identity))
    monkeypatch.setattr(obj, "registered", lambda spec: ({}, {"numerical_model_sha256": "foundation" if spec["registered_engineering_id"].endswith("g0-c0") else spec["registered_engineering_id"]}))
    monkeypatch.setattr(audit, "cem_fold_matches", lambda *_a: True)
    monkeypatch.setattr(audit, "bind_jobs", lambda _c, jobs: {j["job_id"]: {"job_id": j["job_id"]} for j in jobs})
    monkeypatch.setattr(audit, "candidate_reward", lambda c, *_a: copy.deepcopy(rewards.get(c["candidate_id"])))
    monkeypatch.setattr(obj, "cem_closed_job", lambda _c, job, _e: copy.deepcopy(evidence[job["job_id"]]) if evidence[job["job_id"]].get("usable") else (_ for _ in ()).throw(ValueError("unusable current case")))
    (tmp_path / "state.json").write_text("{}")
    return obj, state, artifacts, rewards, evidence


def test_complete_negative_reaudits_all_twelve_and_both_actual_validation_runs(search):
    obj, _state, _artifacts, _rewards, _evidence = search
    report = obj.cem()
    assert report["known"] and not report["eligible"]
    assert len(report["independent_twelve_proposal_fitness"]) == 12
    assert all(r["all_scored_cases_reaudited"] for r in report["independent_twelve_proposal_fitness"])
    assert set(report["validation_evidence"]) == {"baseline", "winner"}
    assert report["paired_validation_gain_s"] == -2.


@pytest.mark.parametrize("mutation", ["missing_reward", "tampered_reward", "missing_generation", "winner_substitution", "changed_weights", "wrong_seed", "orphan_attempt"])
def test_saved_negative_marker_cannot_hide_missing_or_changed_current_evidence(search, monkeypatch, mutation):
    obj, state, _artifacts, rewards, _evidence = search
    cid = state["generations"][0]["candidates"][2]["candidate_id"]
    if mutation == "missing_reward": rewards.pop(cid)
    elif mutation == "tampered_reward": rewards[cid]["reward"] += 1.
    elif mutation == "missing_generation": state["generations"].pop()
    elif mutation == "winner_substitution": state["winner"] = copy.deepcopy(state["generations"][0]["candidates"][0])
    elif mutation == "changed_weights": monkeypatch.setattr(audit, "cem_fold_matches", lambda *_a: False)
    elif mutation == "wrong_seed": state["seed"] += 1
    else: obj.state["attempts"].append({"candidate_id": cid, "attempt_id": "undeclared", "stage": "engineering", "split": "development", "state": "finished"})
    assert obj.cem()["known"] is False


@pytest.mark.parametrize("kind", ["missing", "infrastructure", "physical_failure", "wrong_reason"])
def test_validation_requires_current_usable_evidence_but_retains_real_physical_failure(search, kind):
    obj, state, _artifacts, _rewards, evidence = search
    aid = state["winner"]["candidate_id"] + "-validation-field4"
    if kind == "missing":
        evidence.pop(aid)
    elif kind == "infrastructure":
        evidence[aid]["usable"] = False
    elif kind == "physical_failure":
        evidence[aid].update(passed=False, mean_finish_time_s=None)
        state["validation"]["results"]["winner"] = copy.deepcopy(evidence[aid])
    else:
        state["reason"] = "development_gain_verified"
    result = obj.cem()
    assert result["known"] is (kind == "physical_failure")
    if kind == "physical_failure":
        assert not result["eligible"] and result["validation_evidence"]["winner"]["passed"] is False


def test_mutable_stopped_marker_never_replaces_missing_historical_backlog_inputs(search):
    obj, state, _artifacts, _rewards, _evidence = search
    state.update(status="stopped", reason="deadline_reserve_prevents_validation")
    state.pop("validation")
    result = obj.cem()
    assert result["known"] is False
    assert "CEM_historical_budget_queue_completion_inputs_not_preserved" in result["issues"]


def test_exact_fold_verifies_actual_arrays_not_only_calibration_metadata(tmp_path):
    weights = [np.arange(6 * 48, dtype=np.float32).reshape(6, 48) / 100.]
    biases = [np.arange(6, dtype=np.float32) / 10.]
    foundation = save_rich_model(tmp_path / "foundation.npz", weights, biases, np.zeros(48), np.ones(48), {})
    parameters = [1.02, .4, .1, -.05, -.05]
    final_w, final_b = weights[0].copy(), biases[0].copy()
    final_w[:3] *= np.float32(parameters[0])
    final_b[:3] *= np.float32(parameters[0]); final_b[:3] += np.float32(parameters[1] / 60.)
    final_b[3:] += np.asarray(parameters[2:], np.float32)
    metadata = {"calibration_parameters": parameters, "calibration_generation": 1,
                "calibration_foundation_sha256": audit.sha(foundation)}
    candidate = {"model_path": str(tmp_path / "candidate.npz"), "parameters": parameters, "generation": 1}
    save_rich_model(candidate["model_path"], [final_w], [final_b], np.zeros(48), np.ones(48), metadata)
    assert audit.cem_fold_matches(foundation, candidate)
    final_w[0, 0] += .1
    save_rich_model(candidate["model_path"], [final_w], [final_b], np.zeros(48), np.ones(48), metadata)
    assert not audit.cem_fold_matches(foundation, candidate)


@pytest.mark.parametrize("functional_clone", [True, False])
def test_identity_terminal_requires_verified_twelve_and_functional_clone_without_inventing_validation(search, monkeypatch, functional_clone):
    obj, state, _artifacts, _rewards, _evidence = search
    candidates = [c for g in state["generations"] for c in g["candidates"]]
    candidates[0]["reward"]["reward"] = 200.
    state.update(reason="identity_foundation_remains_best", winner=copy.deepcopy(candidates[0]))
    state.pop("validation")
    obj.jobs = [j for j in obj.jobs if "-validation-" not in j["job_id"]]
    obj.job_by_id = {j["job_id"]: j for j in obj.jobs}
    monkeypatch.setattr(obj, "cem_fitness", lambda _s: (candidates, [{"audited": True}] * 12, []))
    monkeypatch.setattr(obj, "registered", lambda _s: ({}, {"numerical_model_sha256": "foundation" if functional_clone else "different"}))
    result = obj.cem()
    assert result["known"] is functional_clone
    assert not result["eligible"] and "validation_evidence" not in result


def test_unlisted_whole_same_search_candidate_cannot_escape_inventory(search):
    obj, _state, _artifacts, _rewards, _evidence = search
    obj.state["candidates"]["search-g2-c0"] = {"unlisted": "registration"}
    result = obj.cem()
    assert result["known"] is False
    assert "search-g2-c0" in result["search_engineering_inventory"]
    assert "CEM_whole_same_search_proposal_omitted_from_state" in result["issues"]


def test_same_search_active_attempt_cannot_be_hidden_by_terminal_marker(search):
    obj, state, _artifacts, _rewards, _evidence = search
    obj.state["attempts"].append({"candidate_id": state["winner"]["candidate_id"], "attempt_id": "unlisted-active",
                                 "stage": "engineering", "split": "development", "state": "scheduled"})
    assert obj.cem()["known"] is False
