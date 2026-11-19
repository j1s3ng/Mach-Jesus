#!/usr/bin/env python3
"""Finite campaign lease for exact owned detector processes; no broad kills."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(os.environ.get("RACE_GPU_PROJECT", str(Path(__file__).resolve().parents[2])))
DEADLINE = datetime(2026, 11, 14, 21, 50, 47, tzinfo=timezone.utc)
CUTOFF = DEADLINE-timedelta(seconds=300)
ALLOWED = {str(PROJECT/"code/scripts"/name) for name in
           ("serve_race_vision.py", "serve_supported_vision.py", "serve_leased_candidate.py")}


def now():
    return datetime.now(timezone.utc)


def identity(pid):
    root = Path("/proc")/str(pid)
    stat = (root/"stat").read_text().rsplit(")", 1)[1].split()
    uid = next(line.split()[1:] for line in (root/"status").read_text().splitlines() if line.startswith("Uid:"))
    raw = (root/"cmdline").read_bytes()
    return {"pid": pid, "uid": [int(value) for value in uid], "start_ticks": int(stat[19]), "state": stat[0],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "cmdline": [value.decode() for value in raw.rstrip(b"\0").split(b"\0")],
            "cmdline_sha256": hashlib.sha256(raw).hexdigest()}


def matches(record):
    try: actual = identity(record["pid"])
    except (OSError, ValueError, StopIteration): return False
    return actual["state"] != "Z" and all(actual[key] == record[key] for key in
           ("pid", "uid", "start_ticks", "boot_id", "cmdline", "cmdline_sha256"))


def service_signature(record):
    command = record.get("cmdline", [])
    try:
        return (record.get("port") in (8770, 8771, 8772) and record.get("script") in ALLOWED and
                record.get("uid") == [os.getuid()]*4 and command[0] == str(PROJECT/"environment/bin/python") and
                record["script"] in command and command[command.index("--port")+1] == str(record["port"]))
    except (IndexError, ValueError): return False


def read(path):
    value = json.loads(path.read_text())
    if value["deadline_utc"] != DEADLINE.isoformat() or value["launch_cutoff_utc"] != CUTOFF.isoformat() or value["owner_uid"] != os.getuid():
        raise ValueError("Registry lease or owner differs from this finite campaign")
    return value


def save(path, value):
    temporary = path.with_name(path.name+"."+str(os.getpid())+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n");os.replace(temporary, path)


def log(path, event, **fields):
    with path.open("a") as handle:
        handle.write(json.dumps({"utc": now().isoformat(), "event": event, **fields})+"\n");handle.flush()


def lease_check(path):
    read(path)
    if now() >= CUTOFF: raise ValueError("Campaign launch cutoff reached; no new detector launch is permitted")


def register(path, pid, port, script):
    lease_check(path)
    if port not in (8770, 8771, 8772) or str(script) not in ALLOWED:
        raise ValueError("Only the three explicit detector ports/task scripts may be leased")
    record = identity(pid)
    record.update(port=port, script=str(script), registered_utc=now().isoformat())
    if not service_signature(record): raise ValueError("Process UID/interpreter/task script/port differs from the owned service")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX);registry = read(path)
        previous = registry["services"].get(str(port))
        if previous and matches(previous) and previous["pid"] != pid:
            raise ValueError("An existing live owned service already holds that port lease")
        registry["services"][str(port)] = record;save(path, registry)
    return record


def interrupt_exact(record):
    if not service_signature(record): return "refused_unowned_service_signature"
    if not matches(record): return "signature_mismatch_or_exited"
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            descriptor = os.pidfd_open(record["pid"])
            if not matches(record): return "signature_changed_before_signal"
            signal.pidfd_send_signal(descriptor, signal.SIGINT)
        else:
            if not matches(record): return "signature_changed_before_signal"
            os.kill(record["pid"], signal.SIGINT)
        return "SIGINT_sent"
    except ProcessLookupError:
        return "exited_before_signal"
    finally:
        if descriptor is not None: os.close(descriptor)


def watch(path, logfile):
    registry = read(path)
    log(logfile, "watchdog_started", pid=os.getpid(), deadline_utc=registry["deadline_utc"], launch_cutoff_utc=registry["launch_cutoff_utc"])
    while now() < DEADLINE:
        time.sleep(min(30., max(.01, (DEADLINE-now()).total_seconds())))
    registry = read(path)
    for port, record in registry["services"].items():
        # UID, PID, bootID, start ticks and full command/port are checked again
        # immediately before signaling. Reused PIDs and unrelated jobs survive.
        log(logfile, "deadline_cleanup", port=port, pid=record["pid"], result=interrupt_exact(record))
    cleanup_end = time.monotonic()+65.
    while time.monotonic() < cleanup_end and any(matches(record) for record in registry["services"].values()):
        time.sleep(1.)
    survivors = {port: record["pid"] for port, record in registry["services"].items() if matches(record)}
    log(logfile, "watchdog_finished", still_owned_processes=survivors, escalation="none; SIGINT only")


def start(path, logfile):
    lease_check(path)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX);registry = read(path)
        if registry.get("watchdog") and matches(registry["watchdog"]): return registry["watchdog"]
        with logfile.open("a") as output:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watch", "--registry", str(path), "--log", str(logfile)],
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True)
        time.sleep(.1)
        if process.poll() is not None: raise RuntimeError("Detached watchdog did not start")
        record = identity(process.pid);registry["watchdog"] = record;save(path, registry)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "register", "check", "start", "watch"))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--script", type=Path)
    args = parser.parse_args()
    if args.action == "init":
        if args.registry.exists(): parser.error("Existing service registry is immutable except explicit registration")
        args.registry.parent.mkdir(parents=True, exist_ok=True)
        save(args.registry, {"campaign": "improvement-12h-20261114T095047Z", "owner_uid": os.getuid(),
                            "deadline_utc": DEADLINE.isoformat(), "launch_cutoff_utc": CUTOFF.isoformat(), "services": {}})
        result = read(args.registry)
    elif args.action == "register":
        if not all((args.pid, args.port, args.script)): parser.error("Registration requires exact pid,port,script")
        result = register(args.registry, args.pid, args.port, args.script)
    elif args.action == "check": lease_check(args.registry);result = {"launch_allowed": True, "cutoff_utc": CUTOFF.isoformat()}
    elif args.action == "start":
        if not args.log: parser.error("A persistent log path is required")
        result = start(args.registry, args.log)
    else:
        if not args.log: parser.error("A persistent log path is required")
        watch(args.registry, args.log);return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
