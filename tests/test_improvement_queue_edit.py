import concurrent.futures
import json

import pytest

from scripts.improvement_queue_edit import append_jobs, edit_queue, validate_jobs


def job(identity, **extra):
    return {"job_id": identity, "output": "runs/engineering/" + identity, "args": ["--no-video"], **extra}


def test_parallel_producers_preserve_all_jobs_without_consumer_state(tmp_path):
    path = tmp_path / "queue.json"
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda index: append_jobs(path, [job(str(index))]), range(12)))
    assert len(results) == 12
    assert {row["job_id"] for row in json.loads(path.read_text())} == {str(index) for index in range(12)}
    assert (tmp_path / "queue_edit.lock").exists()
    assert not (tmp_path / "queue.lock").exists()
    assert not (tmp_path / "queue_status.json").exists()


def test_idempotent_append_and_conflicts_leave_original_queue(tmp_path):
    path = tmp_path / "queue.json"
    assert append_jobs(path, [job("a")]) == ["a"]
    assert append_jobs(path, [job("a")]) == []
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Conflicting"):
        append_jobs(path, [job("a", args=["different"])])
    assert path.read_bytes() == before


def test_failed_or_invalid_transaction_never_replaces_queue(tmp_path):
    path = tmp_path / "queue.json"
    append_jobs(path, [job("a")]); before = path.read_bytes()
    with pytest.raises(RuntimeError):
        with edit_queue(path) as jobs:
            jobs.append(job("b")); raise RuntimeError("abort")
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="earlier"):
        append_jobs(path, [job("b", requires=[{"job_id": "missing"}])])
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", [{"output": "../outside"}, {"args": ["--output", "owned"]},
                                    {"estimated_wall_seconds": float("inf")}, {"job_id": ""}])
def test_invalid_job_cannot_escape_consumer_contract(change):
    with pytest.raises(ValueError):
        validate_jobs([job("a", **change)])
