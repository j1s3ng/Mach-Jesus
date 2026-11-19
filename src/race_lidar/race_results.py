"""Validate physical race finishes and compare policies across balanced heats.

Results are observations, not controller inputs. In particular, collision counts
may come from a supervisor's independent ground-truth audit.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


RULE_ALIASES = {
    "reference_length_m": ("reference_length_m", "track_length_m"),
    "laps": ("laps", "target_laps"),
    "sensor_period_ms": ("sensor_period_ms",),
    "physics_step_ms": ("physics_step_ms",),
    "max_speed_mps": ("max_speed_mps", "max_speed"),
    "lateral_accel_budget_mps2": ("lateral_accel_budget_mps2", "lateral_accel"),
    "friction_coefficient": ("friction_coefficient", "friction", "mu"),
    "vehicle_model": ("vehicle_model",),
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _first(data: Mapping, names: Sequence[str], default=None):
    return next((data[key] for key in names if key in data), default)


def _rules(config: Mapping) -> dict:
    """Retain all explicit common rules and normalize familiar top-level keys."""
    rules = dict(config.get("rules") or {})
    for canonical, aliases in RULE_ALIASES.items():
        value = _first(rules, aliases, _first(config, aliases))
        if value is not None:
            rules[canonical] = value
        for alias in aliases:
            if alias != canonical:
                rules.pop(alias, None)
    return rules


def _records(records: Any) -> dict[str, dict]:
    if isinstance(records, Mapping):
        if "cars" in records or "results" in records:
            return _records(records.get("cars", records.get("results")))
        return {str(key): dict(value) for key, value in records.items()
                if isinstance(value, Mapping)}
    return {str(row.get("car_id", row.get("id"))): dict(row)
            for row in (records or []) if isinstance(row, Mapping)
            and row.get("car_id", row.get("id")) is not None}


def _collision_count(collisions: Any, car_id: str, record: Mapping):
    """An explicit supervisor count takes precedence over a controller count."""
    if isinstance(collisions, Mapping) and car_id in collisions:
        value = collisions[car_id]
        if isinstance(value, Mapping):
            value = _first(value, ("collision_samples", "samples", "count"))
        return _number(value)
    if isinstance(collisions, list):
        # Event lists must be explicitly complete to imply zero; an empty list
        # passed by the supervisor is such an audit.
        return float(sum(1 for event in collisions if isinstance(event, Mapping)
                         and car_id in [str(x) for x in event.get("car_ids", [])]))
    return _number(_first(record, ("collision_samples", "conservative_envelope_overlap_samples")))


def evaluate_heat(config: Mapping, car_results: Any, collisions: Any = None) -> list[dict]:
    """Return ordered validated standings, including missing cars and DNFs.

    ``config['entrants']`` is authoritative: unlisted result files cannot add a
    racer. A completed flag alone cannot prove a lap; ordered progress and a
    positive common-clock finish time are required. Collision overlap samples
    are conservative observations, not automatically contact events.
    """
    records = _records(car_results)
    entrants = config.get("entrants") or [
        {"car_id": key, "policy": row.get("policy", "unknown"), "grid_index": index}
        for index, (key, row) in enumerate(records.items())]
    car_ids = [str(item.get("car_id", item.get("id"))) for item in entrants]
    if len(set(car_ids)) != len(car_ids):
        raise ValueError("Each entrant must have a unique car_id")
    rules = _rules(config)
    length = _number(rules.get("reference_length_m"))
    laps = _number(rules.get("laps", 1))
    target = length * laps if length and laps and length > 0 and laps > 0 else None
    rows = []
    for index, entrant in enumerate(entrants):
        car_id = car_ids[index]
        record = records.get(car_id, {})
        issues = []
        status = str(record.get("status", "missing" if not record else "unknown"))
        claimed = bool(_first(record, ("completed", "lap_complete"), False)) or status in {
            "completed", "finished"}
        finish = _number(_first(record, ("finish_time_s", "race_time_s", "lap_time_s")))
        progress = _number(_first(record, ("progress_m", "ordered_progress_m")))
        resets = _number(record.get("position_resets_after_spawn"))
        violations = _number(_first(record, ("track_violation_steps", "track_violation_samples")))
        overlaps = _collision_count(collisions, car_id, record)
        if not record:
            issues.append("missing_result")
        if target is None:
            issues.append("missing_valid_race_distance")
        if progress is None:
            issues.append("missing_ordered_progress")
        if claimed and (finish is None or finish <= 0):
            issues.append("missing_positive_finish_time")
        if claimed and target is not None and progress is not None and progress < target - 1e-3:
            issues.append("incomplete_ordered_distance")
        if record.get("valid_lap") is False or record.get("valid_finish") is False:
            issues.append("explicitly_invalid_lap")
        if resets is not None and resets != 0:
            issues.append("position_reset")
        if any(value is not None and value < 0 for value in (resets, violations, overlaps)):
            issues.append("invalid_negative_audit_count")
        observed_rules = _rules(record)
        for key in set(rules) & set(observed_rules):
            if key != "vehicle_model" and observed_rules[key] != rules[key]:
                issues.append(f"reported_rule_mismatch:{key}")
        valid = claimed and not issues
        audited = all(value is not None and value >= 0 for value in (resets, violations, overlaps))
        clean = bool(valid and resets == 0 and violations == 0 and overlaps == 0) if audited else None
        if not audited:
            issues.append("incomplete_incident_audit")
        rows.append({
            "car_id": car_id,
            "policy": str(entrant.get("policy", record.get("policy", "unknown"))),
            "label": str(entrant.get("label", entrant.get("policy", car_id))),
            "checkpoint_sha256": _first(record, ("checkpoint_sha256", "model_sha256",
                                                  "driver_model_sha256", "neural_driver_checkpoint_sha256"),
                                         entrant.get("checkpoint_sha256")),
            "grid_index": _number(entrant.get("grid_index", index)),
            "status": "finished" if valid else ("invalid_finish" if claimed else status),
            "reason": record.get("reason"),
            "valid_finish": bool(valid), "clean_finish": clean,
            "finish_time_s": finish if valid else None,
            "reported_finish_time_s": finish,
            "progress_m": progress, "target_progress_m": target,
            "collision_samples": overlaps, "track_violation_steps": violations,
            "position_resets_after_spawn": resets, "incident_audit_complete": audited,
            "issues": issues,
        })
    rows.sort(key=lambda row: (not row["valid_finish"],
                              row["finish_time_s"] if row["valid_finish"] else
                              -(row["progress_m"] if row["progress_m"] is not None else -math.inf),
                              row["car_id"]))
    previous_key = None
    for index, row in enumerate(rows):
        score = (row["valid_finish"], row["finish_time_s"] if row["valid_finish"] else row["progress_m"])
        if score != previous_key:
            rank = index + 1
        row["position"] = rank
        row["race_winner"] = bool(row["valid_finish"] and rank == 1)
        previous_key = score
    return rows


def summarize_tournament(heats: Sequence[Mapping]) -> dict:
    """Rank tested policies, distinguishing an observed leader from a fair claim.

    Ranking keys are clean-completion rate, completion rate, mean position, and
    mean valid finish time. All attempts count, including missing cars and DNFs.
    Missing incident audits or unequal full grid distributions prevent a
    supported best-policy claim.
    """
    evaluated = []
    by_policy = defaultdict(list)
    qualification_issues = []
    common_rules = None
    policy_counts = []
    for index, heat in enumerate(heats):
        config = heat.get("config") or {}
        rows = heat.get("standings")
        if rows is None:
            rows = evaluate_heat(config, heat.get("results", heat.get("car_results", [])),
                                 heat.get("collisions"))
        # Normalize only the analysis view. Preserve the controller's original
        # status and leave source records/files untouched for forensic review.
        rows = [dict(row, issues=list(row["issues"])) for row in rows]
        if heat.get("status") == "time_limit":
            for row in rows:
                cutoff_closure = (row["status"] == "incomplete"
                                  and row.get("reason") == "simulation_closed_before_finish")
                if not row["valid_finish"] and (row["status"] in {"running", "initializing", "unknown"}
                                                or cutoff_closure):
                    row["reported_status"] = row["status"]
                    row["status"] = "timeout"
                    row["issues"].append("nonterminal_at_time_limit")
        heat_id = str(heat.get("heat_id", config.get("heat_id", f"heat-{index + 1}")))
        rules = _rules(config)
        if common_rules is None:
            common_rules = rules
        elif rules != common_rules:
            qualification_issues.append("common_rules_differ_between_heats")
        counts = Counter(row["policy"] for row in rows)
        policy_counts.append(counts)
        for row in rows:
            by_policy[row["policy"]].append(dict(row, heat_id=heat_id))
        wall = _number(heat.get("wall_seconds"))
        simulation = _number(heat.get("simulation_seconds"))
        evaluated.append({"heat_id": heat_id, "status": heat.get("status"), "standings": rows,
                          "winner_car_ids": [row["car_id"] for row in rows if row["race_winner"]],
                          "wall_seconds": wall, "simulation_seconds": simulation,
                          "wall_clock_context": heat.get("wall_clock_context"),
                          "simulation_to_wall_ratio": simulation / wall if wall and simulation else None,
                          "rules": rules})
    if len(evaluated) < 2:
        qualification_issues.append("fewer_than_two_heats")
    if len(by_policy) < 2:
        qualification_issues.append("fewer_than_two_policies")
    if not common_rules:
        qualification_issues.append("common_rules_not_recorded")
    elif any(key not in common_rules for key in RULE_ALIASES):
        qualification_issues.append("incomplete_common_rules")
    if policy_counts and (any(count != policy_counts[0] for count in policy_counts[1:])
                          or len(set(policy_counts[0].values())) > 1):
        qualification_issues.append("unequal_policy_entries_between_heats")
    summaries = []
    for policy, rows in by_policy.items():
        completed = [row for row in rows if row["valid_finish"]]
        grids = [row["grid_index"] for row in rows if row["grid_index"] is not None]
        checkpoints = sorted({str(row["checkpoint_sha256"]) for row in rows if row.get("checkpoint_sha256")})
        if len(checkpoints) > 1:
            qualification_issues.append(f"policy_checkpoint_changed:{policy}")
        summaries.append({
            "policy": policy, "attempts": len(rows), "completions": len(completed),
            "checkpoint_sha256s": checkpoints,
            "clean_completions": sum(row["clean_finish"] is True for row in rows),
            "unknown_incident_audits": sum(not row["incident_audit_complete"] for row in rows),
            "completion_rate": len(completed) / len(rows),
            "clean_completion_rate": sum(row["clean_finish"] is True for row in rows) / len(rows),
            "mean_position": mean(row["position"] for row in rows),
            "mean_finish_time_s": mean(row["finish_time_s"] for row in completed) if completed else None,
            "race_wins": sum(row["race_winner"] for row in rows),
            "mean_grid_index": mean(grids) if len(grids) == len(rows) else None,
            "collision_samples": sum(row["collision_samples"] or 0 for row in rows),
            "track_violation_steps": sum(row["track_violation_steps"] or 0 for row in rows),
            "entries": [{"heat_id": row["heat_id"], "car_id": row["car_id"],
                         "position": row["position"], "grid_index": row["grid_index"]} for row in rows],
        })
    if any(row["unknown_incident_audits"] for row in summaries):
        qualification_issues.append("incomplete_incident_audits")
    if any("missing_result" in row["issues"] for rows in by_policy.values() for row in rows):
        qualification_issues.append("missing_car_results")
    grid_means = [row["mean_grid_index"] for row in summaries]
    if not grid_means or any(value is None for value in grid_means) or max(grid_means) - min(grid_means) > 1e-6:
        qualification_issues.append("unequal_mean_starting_grid_position")
    def score(row):
        return (-row["clean_completion_rate"], -row["completion_rate"], row["mean_position"],
                row["mean_finish_time_s"] if row["mean_finish_time_s"] is not None else math.inf)
    summaries.sort(key=lambda row: (score(row), row["policy"]))
    previous_score = None
    for index, row in enumerate(summaries):
        key = score(row)
        if key != previous_score:
            rank = index + 1
        row["rank"] = rank
        previous_score = key
    leaders = [row["policy"] for row in summaries if row["rank"] == 1]
    if not any(row["rank"] == 1 and row["clean_completions"] > 0 for row in summaries):
        qualification_issues.append("no_clean_finishing_policy")
    valid_finishes = [dict(heat_id=heat["heat_id"], car_id=row["car_id"], policy=row["policy"],
                          finish_time_s=row["finish_time_s"], clean_finish=row["clean_finish"])
                      for heat in evaluated for row in heat["standings"] if row["valid_finish"]]
    fastest_time = min((row["finish_time_s"] for row in valid_finishes), default=None)
    fastest = [row for row in valid_finishes if row["finish_time_s"] == fastest_time]
    grid_distributions = {policy: sorted(row["grid_index"] for row in rows if row["grid_index"] is not None)
                          for policy, rows in sorted(by_policy.items())}
    identical_grids = len({tuple(values) for values in grid_distributions.values()}) <= 1
    if not identical_grids:
        qualification_issues.append("unequal_full_starting_grid_distributions")
    winner_grids = [dict(heat_id=heat["heat_id"], car_id=row["car_id"], policy=row["policy"],
                         grid_index=row["grid_index"])
                    for heat in evaluated for row in heat["standings"] if row["race_winner"]]
    issues = sorted(set(qualification_issues))
    eligible = not issues and bool(leaders)
    return {"protocol_version": 1, "heat_count": len(evaluated), "policy_count": len(summaries),
            "best_policy_supported": eligible,
            "best_policies_among_tested": leaders if eligible else [],
            "fastest_valid_finishers": fastest,
            "policy_grid_distributions": grid_distributions,
            "identical_full_grid_distributions": identical_grids,
            "winner_starting_grids": winner_grids,
            "observed_leading_policies": leaders, "qualification_issues": issues,
            "ranking_order": ["clean_completion_rate_desc", "completion_rate_desc", "mean_position_asc",
                              "mean_valid_finish_time_s_asc"],
            "policies": summaries, "heats": evaluated,
            "scope": "Tested policies on this simulated track, vehicle, traffic field, and grid schedule; not general racing ability."}


def _read(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def analyze_directory(directory: str | Path) -> dict:
    """Read a tournament manifest or a single heat directory without writing."""
    directory = Path(directory)
    tournament = _read(directory / "tournament.json", {})
    heat_specs = tournament.get("heats")
    if not heat_specs:
        heat_specs = [{"path": str(path.parent.relative_to(directory))}
                      for path in sorted(directory.glob("heat*/config.json"))] or [{"path": "."}]
    heats = []
    for index, spec in enumerate(heat_specs):
        spec = {"path": spec} if isinstance(spec, str) else dict(spec)
        heat_directory = directory / spec.get("path", spec.get("directory", "."))
        config = dict(tournament.get("config") or {})
        if tournament.get("rules"):
            config["rules"] = tournament["rules"]
        config.update(_read(heat_directory / "config.json", {}) or spec.get("config") or {})
        aggregate = _read(heat_directory / "heat_results.json", {})
        records = _records(aggregate)
        for entrant in config.get("entrants", []):
            car_id = str(entrant.get("car_id", entrant.get("id")))
            result = _read(heat_directory / "cars" / car_id / "result.json")
            if result is None:
                result = _read(heat_directory / car_id / "result.json")
            if result is not None:
                records[car_id] = result
        collisions = _first(aggregate, ("collisions_by_car", "collision_samples_by_car", "collisions"))
        heats.append({"heat_id": spec.get("heat_id", spec.get("name", f"heat-{index + 1}")),
                      "status": aggregate.get("status"),
                      "config": config, "results": records, "collisions": collisions,
                      "wall_seconds": aggregate.get("wall_seconds", spec.get("wall_seconds")),
                      "wall_clock_context": aggregate.get("wall_clock_context", spec.get("wall_clock_context")),
                      "simulation_seconds": aggregate.get("simulation_seconds", spec.get("simulation_seconds"))})
    return summarize_tournament(heats)


def render_report(summary: Mapping) -> str:
    leaders = ", ".join(summary["observed_leading_policies"]) or "none"
    claim = (f"Best among the tested policies: **{leaders}**." if summary["best_policy_supported"]
             else f"Observed leader: **{leaders}**. No overall best is declared.")
    lines = ["# AI race results", "", claim, "", summary["scope"], "",
             "| Policy | Clean / attempts | Finished | Race wins | Mean position | Mean finish (s) | Mean grid |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    for row in summary["policies"]:
        lines.append(f"| {row['policy']} | {row['clean_completions']} / {row['attempts']} | "
                     f"{row['completions']} | {row['race_wins']} | {fmt(row['mean_position'])} | "
                     f"{fmt(row['mean_finish_time_s'])} | {fmt(row['mean_grid_index'])} |")
    lines += ["", "Clean completion means a validated full-distance finish with zero recorded track "
              "violation steps, zero conservative car-envelope overlap samples, and zero pose resets. "
              "Unknown incident audits do not count as clean.", ""]
    if summary.get("fastest_valid_finishers"):
        fastest = summary["fastest_valid_finishers"]
        names = ", ".join(f"{row['car_id']} ({row['policy']}, {row['heat_id']})" for row in fastest)
        lines += [f"Fastest valid finish: {names}, {fastest[0]['finish_time_s']:.3f} s. "
                  "This time is reported separately from clean-completion policy ranking.", ""]
    if summary["qualification_issues"]:
        lines += ["Comparison limitations:", ""]
        lines += [f"- {issue.replace('_', ' ')}" for issue in summary["qualification_issues"]]
        lines.append("")
    if not summary.get("identical_full_grid_distributions", True):
        lines += ["The policies have different full distributions of starting slots. Equal mean grid "
                  "positions do not remove the influence of particular slots and traffic. This comparison "
                  "is limited to the observed grid schedule and does not establish statistical superiority.", ""]
        for policy, positions in summary["policy_grid_distributions"].items():
            lines.append(f"- {policy}: grid indices " + ", ".join(f"{position:g}" for position in positions))
        lines.append("")
    if summary.get("winner_starting_grids"):
        winners = summary["winner_starting_grids"]
        description = "; ".join(f"{row['heat_id']} {row['car_id']} ({row['policy']}): {row['grid_index']:g}"
                                for row in winners if row["grid_index"] is not None)
        lines += [f"Valid race winners' starting grid indices (zero based): {description}.", ""]
        slots = {row["grid_index"] for row in winners}
        if len(winners) > 1 and len(slots) == 1 and None not in slots:
            lines += [f"All valid heat winners started at grid index {next(iter(slots)):g}. "
                      "That shared starting slot limits attribution of the wins to policy alone.", ""]
    for heat in summary["heats"]:
        winners = ", ".join(heat["winner_car_ids"]) or "none (no valid finisher)"
        lines += [f"## {heat['heat_id']}", "", f"Race winner: {winners}.", "",
                  "| Position | Car | Policy | Grid | Status | Finish (s) | Clean |", 
                  "| ---: | --- | --- | ---: | --- | ---: | --- |"]
        for row in heat["standings"]:
            clean = "unknown" if row["clean_finish"] is None else ("yes" if row["clean_finish"] else "no")
            lines.append(f"| {row['position']} | {row['car_id']} | {row['policy']} | "
                         f"{fmt(row['grid_index'])} | {row['status']} | {fmt(row['finish_time_s'])} | {clean} |")
        if heat["simulation_to_wall_ratio"] is not None:
            lines += ["", f"Measured simulation / wall-clock ratio: {heat['simulation_to_wall_ratio']:.3f}×."]
        if heat.get("wall_clock_context"):
            lines += ["", f"Wall-clock measurement context: {heat['wall_clock_context']}. "
                      "Use the separate unloaded capacity benchmarks for hardware-capacity comparisons."]
        lines.append("")
    return "\n".join(lines)


def write_analysis(directory: str | Path, summary: Mapping) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (directory / "report.md").write_text(render_report(summary))
    fields = ["heat_id", "position", "car_id", "policy", "grid_index", "status", "valid_finish",
              "clean_finish", "finish_time_s", "progress_m", "collision_samples", "track_violation_steps"]
    with (directory / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for heat in summary["heats"]:
            for row in heat["standings"]:
                writer.writerow(dict(row, heat_id=heat["heat_id"]))
