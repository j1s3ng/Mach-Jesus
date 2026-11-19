"""Atomic, serialized edits for the physical queue's producers.

The existing run_improvement_queue.py remains the sole consumer. This writer
uses queue_edit.lock, never the consumer's queue.lock or queue_status.json.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import tempfile
import time
from pathlib import Path


def validate_jobs(jobs):
    if not isinstance(jobs, list):
        raise ValueError("Physical queue must be a job list")
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Every queue job must be an object")
        identity = job.get("job_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Queue job_id must be a unique nonempty string")
        output = job.get("output")
        if not isinstance(output, str) or not output or Path(output).is_absolute() or ".." in Path(output).parts:
            raise ValueError("Queue output must stay in the project as a relative path")
        args = job.get("args")
        if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
            raise ValueError("Queue args must be a string list")
        if any(value.split("=", 1)[0] in ("--output", "--attempt-id") for value in args):
            raise ValueError("The sole consumer owns output and attempt-id arguments")
        for key in ("estimated_wall_seconds", "max_wall_seconds"):
            value = job.get(key, 900 if key.startswith("estimated") else 3600)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0.:
                raise ValueError(f"{key} must be positive and finite")
        for requirement in job.get("requires", []):
            if not isinstance(requirement, dict) or requirement.get("job_id") not in seen:
                raise ValueError("Queue dependencies must refer to an earlier job")
            fields = requirement.get("true_fields", ["audit_verified"])
            if not isinstance(fields, list) or not fields or not all(isinstance(key, str) and key for key in fields):
                raise ValueError("Dependency true_fields must be a nonempty string list")
        seen.add(identity)
    return jobs


def atomic_json(path, value):
    """Replace JSON on the same filesystem; readers see one complete document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def edit_queue(queue_path, *, timeout_s=10.):
    """Yield the current list under queue_edit.lock; validate and atomically save.

    Exceptions inside the context leave the prior queue unchanged. Read/modify/
    replace must all happen inside this context to prevent lost producer edits.
    """
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = queue_path.with_name("queue_edit.lock")
    with lock_path.open("a+") as lock:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Queue writer lock busy: {lock_path}")
                time.sleep(.025)
        try:
            jobs = json.loads(queue_path.read_text()) if queue_path.exists() else []
            validate_jobs(jobs)
            yield jobs
            validate_jobs(jobs)
            atomic_json(queue_path, jobs)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def append_jobs(queue_path, new_jobs, *, timeout_s=10.):
    """Idempotently append identical jobs; a conflicting identity is rejected."""
    added = []
    with edit_queue(queue_path, timeout_s=timeout_s) as jobs:
        by_id = {job["job_id"]: job for job in jobs}
        for job in new_jobs:
            old = by_id.get(job.get("job_id"))
            if old is not None:
                if old != job:
                    raise ValueError(f"Conflicting immutable queue job: {job['job_id']}")
                continue
            jobs.append(job); by_id[job.get("job_id")] = job; added.append(job.get("job_id"))
    return added
