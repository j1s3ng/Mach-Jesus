import csv
import json

import pytest

from race_lidar.race_results import (
    analyze_directory, evaluate_heat, render_report, summarize_tournament, write_analysis,
)


RULES = {"reference_length_m": 1000.0, "laps": 1, "sensor_period_ms": 200,
         "physics_step_ms": 20, "max_speed_mps": 60.0,
         "lateral_accel_budget_mps2": 8.0, "friction_coefficient": 1.5,
         "vehicle_model": "identical physical Car"}


def config(reverse=False):
    entrants = [{"car_id": "a", "policy": "v1", "grid_index": 1 if reverse else 0},
                {"car_id": "b", "policy": "v3", "grid_index": 0 if reverse else 1}]
    return {"rules": dict(RULES), "entrants": entrants}


def result(time=50.0, **kwargs):
    return {"status": "completed", "completed": True, "finish_time_s": time,
            "progress_m": 1000.2, "valid_lap": True, "collision_samples": 0,
            "position_resets_after_spawn": 0, "track_violation_steps": 0, **kwargs}


def heat(reverse=False, **kwargs):
    return {"heat_id": "reverse" if reverse else "forward", "config": config(reverse),
            "results": {"a": result(51.0), "b": result(50.0)}, **kwargs}


def test_finish_flag_cannot_replace_full_distance_or_positive_time():
    rows = evaluate_heat(config(), {"a": result(progress_m=999), "b": result(time=float("nan"))})
    assert not any(row["valid_finish"] for row in rows)
    assert not any(row["race_winner"] for row in rows)
    by_id = {row["car_id"]: row for row in rows}
    assert "incomplete_ordered_distance" in by_id["a"]["issues"]
    assert "missing_positive_finish_time" in by_id["b"]["issues"]


def test_finishers_precede_dnfs_and_missing_cars_even_at_negative_grid_progress():
    cfg = config()
    cfg["entrants"].append({"car_id": "c", "policy": "v2", "grid_index": 2})
    rows = evaluate_heat(cfg, {"a": result(time=100),
                               "b": result(status="timeout", completed=False,
                                           finish_time_s=None, progress_m=-10)})
    assert [row["car_id"] for row in rows] == ["a", "b", "c"]
    assert rows[0]["race_winner"]
    assert "missing_result" in rows[2]["issues"]


def test_supervisor_audit_overrides_controller_and_unknown_is_not_clean():
    rows = evaluate_heat(config(), {"a": result(collision_samples=None),
                                   "b": result(collision_samples=0)}, {"a": 0, "b": 3})
    assert rows[0]["clean_finish"] is True
    assert rows[1]["clean_finish"] is False
    unknown = evaluate_heat(config(), {"a": result(collision_samples=None), "b": result()})
    assert unknown[0]["clean_finish"] is None
    assert unknown[0]["valid_finish"]
    assert not unknown[0]["incident_audit_complete"]


@pytest.mark.parametrize("invalid", [
    {"position_resets_after_spawn": 1}, {"valid_lap": False},
    {"max_speed_mps": 90}, {"track_violation_steps": -1},
])
def test_invalid_laps_resets_changed_caps_and_negative_audits_reject_finish(invalid):
    rows = evaluate_heat(config(), {"a": result(**invalid), "b": result(55)})
    assert rows[0]["car_id"] == "b"
    assert rows[0]["race_winner"]
    assert not rows[1]["valid_finish"]


def test_tied_finishes_share_position_and_deterministic_display():
    rows = evaluate_heat(config(), {"b": result(50), "a": result(50)})
    assert [row["car_id"] for row in rows] == ["a", "b"]
    assert [row["position"] for row in rows] == [1, 1]
    assert all(row["race_winner"] for row in rows)


def test_balanced_two_heat_comparison_identifies_tested_winner_and_measured_capacity():
    context = "includes background postprocessing of another heat"
    summary = summarize_tournament([heat(wall_seconds=100, simulation_seconds=50, wall_clock_context=context), heat(True)])
    assert summary["best_policy_supported"]
    assert summary["best_policies_among_tested"] == ["v3"]
    assert summary["heats"][0]["simulation_to_wall_ratio"] == .5
    assert summary["heats"][0]["wall_clock_context"] == context
    assert context in render_report(summary)
    assert summary["policies"][0]["attempts"] == 2
    assert summary["policies"][0]["mean_grid_index"] == .5
    assert summary["policies"][0]["mean_position"] == 1


def test_equal_grid_means_do_not_qualify_as_balanced_full_grid_distributions():
    cfg = config()
    cfg["entrants"] = [{"car_id": cid, "policy": cid, "grid_index": i}
                       for i, cid in enumerate(("a", "b", "c", "d"))]
    reverse = {**cfg, "entrants": [{**e, "grid_index": 3-e["grid_index"]} for e in cfg["entrants"]]}
    first = {"heat_id": "first", "config": cfg,
             "results": {cid: result(40 if cid == "b" else 50) for cid in ("a", "b", "c", "d")}}
    second = {"heat_id": "second", "config": reverse,
              "results": {cid: result(40 if cid == "c" else 50) for cid in ("a", "b", "c", "d")}}
    summary = summarize_tournament([first, second])
    assert not summary["best_policy_supported"]
    assert summary["best_policies_among_tested"] == []
    assert "unequal_full_starting_grid_distributions" in summary["qualification_issues"]
    assert all(row["mean_grid_index"] == 1.5 for row in summary["policies"])
    assert not summary["identical_full_grid_distributions"]
    assert summary["policy_grid_distributions"]["a"] == [0, 3]
    assert summary["policy_grid_distributions"]["b"] == [1, 2]
    assert [row["grid_index"] for row in summary["winner_starting_grids"]] == [1, 1]
    report = render_report(summary)
    assert "All valid heat winners started at grid index 1" in report
    assert "does not establish statistical superiority" in report


def test_four_cyclic_shifts_balance_every_policy_across_all_twenty_slots():
    heats=[]
    for rotation in range(4):
        entrants=[{'car_id':f'R{slot+1:02d}','policy':f'policy{(slot+rotation)%4}',
                   'grid_index':slot} for slot in range(20)]
        heats.append({'heat_id':str(rotation),'config':{'rules':dict(RULES),'entrants':entrants},
                      'results':{entry['car_id']:result(50.+int(entry['policy'][-1])+slot/100.)
                                 for slot,entry in enumerate(entrants)}})
    summary=summarize_tournament(heats)
    assert summary['best_policy_supported']
    assert summary['best_policies_among_tested']==['policy0']
    assert all(slots==list(range(20)) for slots in summary['policy_grid_distributions'].values())
    # Full grid balance cannot waive the independent clean-finishing gate.
    for item in heats:
        for row in item['results'].values():row['collision_samples']=1
    dirty=summarize_tournament(heats)
    assert dirty['identical_full_grid_distributions'] and not dirty['best_policy_supported']
    assert 'no_clean_finishing_policy' in dirty['qualification_issues']


def test_clean_completion_rate_precedes_one_fast_dirty_finish_and_one_dnf():
    first, second = heat(), heat(True)
    first["results"]["b"] = result(40, collision_samples=3)
    second["results"]["b"] = result(status="timeout", completed=False, finish_time_s=None, progress_m=900)
    summary = summarize_tournament([first, second])
    assert summary["best_policies_among_tested"] == ["v1"]
    assert summary["policies"][1]["completions"] == 1
    assert summary["policies"][1]["mean_finish_time_s"] == 40


def test_all_dnfs_cannot_prove_best_policy_but_retain_distance_ranking():
    first, second = heat(), heat(True)
    for item in (first, second):
        item["results"] = {"a": result(status="timeout", completed=False, finish_time_s=None, progress_m=900),
                           "b": result(status="timeout", completed=False, finish_time_s=None, progress_m=800)}
    summary = summarize_tournament([first, second])
    assert not summary["best_policy_supported"]
    assert summary["best_policies_among_tested"] == []
    assert summary["observed_leading_policies"] == ["v1"]
    assert "no_clean_finishing_policy" in summary["qualification_issues"]
    assert summary["fastest_valid_finishers"] == []
    assert all(not item["winner_car_ids"] for item in summary["heats"])


def test_valid_dirty_finishes_keep_fastest_winner_without_supported_best_claim():
    first, second = heat(), heat(True)
    for item in (first, second):
        for outcome in item["results"].values():
            outcome["collision_samples"] = 1
    summary = summarize_tournament([first, second])
    assert not summary["best_policy_supported"]
    assert "no_clean_finishing_policy" in summary["qualification_issues"]
    assert summary["observed_leading_policies"] == ["v3"]
    assert [row["car_id"] for row in summary["fastest_valid_finishers"]] == ["b", "b"]
    assert all(item["winner_car_ids"] == ["b"] for item in summary["heats"])
    report = render_report(summary)
    assert "Fastest valid finish:" in report
    assert "No overall best is declared." in report
    assert "balanced best-policy claim" not in report
    assert "| Race wins |" in report
    assert summary["policies"][0]["race_wins"] == 2


def test_single_heat_and_biased_grid_are_limited_observations():
    summary = summarize_tournament([heat()])
    assert not summary["best_policy_supported"]
    assert summary["observed_leading_policies"] == ["v3"]
    assert "fewer_than_two_heats" in summary["qualification_issues"]
    assert "unequal_mean_starting_grid_position" in summary["qualification_issues"]
    assert not summarize_tournament([heat(), heat()])["best_policy_supported"]


def test_changed_rules_unknown_audits_and_missing_results_prevent_strong_claim():
    changed = heat(True)
    changed["config"]["rules"]["max_speed_mps"] = 80
    changed["results"]["a"]["collision_samples"] = None
    changed["results"].pop("b")
    summary = summarize_tournament([heat(), changed])
    assert not summary["best_policy_supported"]
    assert {"common_rules_differ_between_heats", "incomplete_incident_audits", "missing_car_results"} <= set(
        summary["qualification_issues"])
    assert all(row["attempts"] == 2 for row in summary["policies"])


def test_duplicate_ids_raise_and_empty_tournament_is_serializable():
    cfg = config()
    cfg["entrants"].append(cfg["entrants"][0])
    with pytest.raises(ValueError, match="unique"):
        evaluate_heat(cfg, {})
    json.dumps(summarize_tournament([]), allow_nan=False)


def test_policy_cannot_change_checkpoint_between_heats_under_same_label():
    first, second = heat(), heat(True)
    first["results"]["b"]["model_sha256"] = "first-checkpoint"
    second["results"]["b"]["model_sha256"] = "different-checkpoint"
    summary = summarize_tournament([first, second])
    assert not summary["best_policy_supported"]
    assert "policy_checkpoint_changed:v3" in summary["qualification_issues"]
    assert len(summary["policies"][0]["checkpoint_sha256s"]) == 2


def test_artifact_analysis_uses_controller_outcomes_and_supervisor_incident_audit(tmp_path):
    (tmp_path / "tournament.json").write_text(json.dumps({
        "rules": RULES, "heats": [{"heat_id": "h1", "path": "h1"},
                                   {"heat_id": "h2", "path": "h2"}]}))
    for name, reverse in [("h1", False), ("h2", True)]:
        directory = tmp_path / name
        directory.mkdir()
        cfg = config(reverse)
        cfg.pop("rules")
        (directory / "config.json").write_text(json.dumps(cfg))
        (directory / "heat_results.json").write_text(json.dumps({
            "cars": {"a": result(10), "b": result(10)},
            "collisions_by_car": {"a": 0, "b": 0}, "wall_seconds": 100, "simulation_seconds": 50}))
        for car_id, seconds in [("a", 51), ("b", 50)]:
            car_directory = directory / "cars" / car_id
            car_directory.mkdir(parents=True)
            (car_directory / "result.json").write_text(json.dumps(result(seconds, collision_samples=None)))
    summary = analyze_directory(tmp_path)
    assert summary["best_policies_among_tested"] == ["v3"]
    assert summary["heats"][0]["standings"][0]["finish_time_s"] == 50
    write_analysis(tmp_path, summary)
    assert "Best among the tested policies: **v3**" in render_report(summary)
    assert json.loads((tmp_path / "results.json").read_text())["best_policy_supported"]
    with (tmp_path / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["heat_id"] == "h1"


@pytest.mark.parametrize("status,reason,expected_status", [
    ("running", "awaiting_finish", "timeout"),
    ("incomplete", "simulation_closed_before_finish", "timeout"),
    ("incomplete", "other_early_shutdown", "incomplete"),
    ("failed", "simulation_closed_before_finish", "failed"),
])
def test_analysis_normalizes_only_cutoff_statuses_without_changing_race_files(
        tmp_path, status, reason, expected_status):
    (tmp_path / "config.json").write_text(json.dumps(config()))
    running = result(status=status, reason=reason, completed=False, finish_time_s=None, progress_m=700)
    source = tmp_path / "cars" / "a" / "result.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(running))
    original = source.read_bytes()
    aggregate = {"status": "time_limit", "cars": {"a": running, "b": result()},
                 "collisions_by_car": {"a": 0, "b": 0}}
    (tmp_path / "heat_results.json").write_text(json.dumps(aggregate))
    summary = analyze_directory(tmp_path)
    by_id = {row["car_id"]: row for row in summary["heats"][0]["standings"]}
    assert by_id["a"]["status"] == expected_status
    if expected_status == "timeout":
        assert by_id["a"]["reported_status"] == status
        assert "nonterminal_at_time_limit" in by_id["a"]["issues"]
    else:
        assert "reported_status" not in by_id["a"]
    assert not by_id["a"]["valid_finish"]
    assert by_id["b"]["status"] == "finished"
    assert source.read_bytes() == original
    assert running["status"] == status
    assert json.loads((tmp_path / "heat_results.json").read_text()) == aggregate
