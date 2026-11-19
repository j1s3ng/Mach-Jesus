#!/usr/bin/env python3
"""Fetch a public historical F1 layout/pace reference without an account.

Default: Norris's 2024 Abu Dhabi qualifying pole lap. OpenF1 positions do not
resolve lateral track placement, so this is not an exact driver racing line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

API = "https://api.openf1.org/v1/"
ROOT = Path(__file__).resolve().parents[1]


def fetch_json(kind, query, output):
    url = API + kind + "?" + urlencode(query)
    with urlopen(url, timeout=30) as response:
        rows = json.load(response)
    (output / f"openf1_{kind}.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows, url


def fetch_reference(output, session_key=9658, driver_number=4):
    output = Path(output)
    if (output / "reference.npz").exists():
        raise ValueError("Reference already exists; use a fresh --output to preserve provenance")
    output.mkdir(parents=True, exist_ok=True)
    jobs = {
        "sessions": {"session_key": session_key},
        "laps": {"session_key": session_key, "driver_number": driver_number},
        "drivers": {"session_key": session_key, "driver_number": driver_number},
        "session_result": {"session_key": session_key, "driver_number": driver_number},
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(fetch_json, key, query, output) for key, query in jobs.items()}
        initial = {key: future.result() for key, future in futures.items()}
    sessions, laps, drivers = (initial[key][0] for key in ["sessions", "laps", "drivers"])
    if not sessions or not drivers:
        raise ValueError("No matching historical session/driver returned")
    usable = [lap for lap in laps if lap.get("lap_duration") and not lap.get("is_pit_out_lap")]
    if not usable:
        raise ValueError("No complete non-out lap returned")
    lap = min(usable, key=lambda row: row["lap_duration"])
    start = datetime.fromisoformat(lap["date_start"])
    duration = float(lap["lap_duration"])
    query = {"session_key": session_key, "driver_number": driver_number,
             "date>": (start - timedelta(seconds=2)).isoformat(),
             "date<": (start + timedelta(seconds=duration + 2)).isoformat()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {key: pool.submit(fetch_json, key, query, output) for key in ["location", "car_data"]}
        detail = {key: future.result() for key, future in futures.items()}
    positions = sorted(detail["location"][0], key=lambda row: row["date"])
    car = sorted(detail["car_data"][0], key=lambda row: row["date"])
    if len(positions) < 10 or len(car) < 10:
        raise ValueError("Insufficient historical position or car samples")
    pt = np.array([(datetime.fromisoformat(row["date"]) - start).total_seconds() for row in positions])
    ct = np.array([(datetime.fromisoformat(row["date"]) - start).total_seconds() for row in car])
    if not (pt[0] < 0 < duration < pt[-1] and ct[0] <= 0 < duration <= ct[-1]):
        raise ValueError("Downloaded samples do not bracket both lap boundaries")
    time_s = np.r_[0., pt[(pt > 0) & (pt < duration)], duration]
    x_m = np.interp(time_s, pt, [row["x"] / 10. for row in positions])
    y_m = np.interp(time_s, pt, [row["y"] / 10. for row in positions])
    speed_mps = np.interp(time_s, ct, [row["speed"] / 3.6 for row in car])
    arrays = {"time_s": time_s, "x_m": x_m, "y_m": y_m, "speed_mps": speed_mps}
    arrays["throttle_pct"] = np.interp(time_s, ct, [row["throttle"] for row in car])
    previous = np.clip(np.searchsorted(ct, time_s, side="right") - 1, 0, len(car) - 1)
    arrays["gear"] = np.array([row["n_gear"] for row in car], dtype=np.int32)[previous]
    arrays["brake"] = np.array([row["brake"] > 0 for row in car], dtype=bool)[previous]
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Nonfinite telemetry values returned")
    if not np.all(np.diff(time_s) > 0) or np.any(speed_mps < 0):
        raise ValueError("Invalid time/speed channels")
    np.savez(output / "reference.npz", **arrays)
    driver, session = drivers[0], sessions[0]
    results = initial["session_result"][0]
    metadata = {
        "source": "OpenF1 public historical API", "season": session["year"],
        "event": session.get("location"), "circuit": session["circuit_short_name"],
        "session": session["session_name"], "session_key": session_key,
        "driver": driver["name_acronym"], "driver_name": driver["full_name"],
        "driver_number": driver_number, "team": driver["team_name"],
        "session_position": results[0].get("position") if results else None,
        "lap_number": lap["lap_number"], "lap_time_s": duration,
        "source_lap_start_utc": start.isoformat(), "telemetry_samples": len(time_s),
        "original_position_samples": len(positions), "original_car_samples": len(car),
        "time_first_s": float(time_s[0]), "time_last_s": float(time_s[-1]),
        "sample_dt_median_s": float(np.median(np.diff(time_s))),
        "endpoint_gap_m": float(np.hypot(x_m[-1] - x_m[0], y_m[-1] - y_m[0])),
        "polyline_length_m": float(np.hypot(np.diff(x_m), np.diff(y_m)).sum()),
        "speed_min_kph": float(speed_mps.min() * 3.6), "speed_max_kph": float(speed_mps.max() * 3.6),
        "position_units_source": "F1-feed decimetres per FastF1's data model; divide by 10",
        "position_units_output": "metres", "speed_units_source": "km/h", "speed_units_output": "m/s",
        "coordinate_frame": "arbitrary F1 timing track-local XY",
        "position_quality": "Approximate OpenF1 location at about 3.7 Hz; lacks reliable lateral placement. Layout/pace reference, not an exact professional racing line.",
        "processing": "Original position sample times retained; linearly interpolated boundary positions, speed and throttle. Previous sample for gear/brake. Approximate source lap-start timestamp. No forced geometric closure.",
        "sources": ["https://openf1.org/docs/#location", "https://openf1.org/docs/#laps",
                    "https://raw.githubusercontent.com/theOehrly/Fast-F1/master/fastf1/core.py",
                    *[item[1] for item in initial.values()], *[item[1] for item in detail.values()]],
        "reference_sha256": hashlib.sha256((output / "reference.npz").read_bytes()).hexdigest(),
    }
    (output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/f1-reference")
    parser.add_argument("--session-key", type=int, default=9658)
    parser.add_argument("--driver-number", type=int, default=4)
    args = parser.parse_args()
    try:
        metadata = fetch_reference(args.output, args.session_key, args.driver_number)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
