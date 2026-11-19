from datetime import timedelta
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lease", ROOT/"scripts/watch_owned_gpu_services.py")
lease = importlib.util.module_from_spec(spec);spec.loader.exec_module(lease)


def record():
    script = str(lease.PROJECT/"code/scripts/serve_race_vision.py")
    return {"pid": 123, "uid": [9]*4, "start_ticks": 88, "boot_id": "boot", "state": "S", "cmdline_sha256": "cmd",
            "cmdline": [str(lease.PROJECT/"environment/bin/python"), script, "--port", "8770"], "port": 8770, "script": script}


def test_exact_process_signal_refuses_pid_reuse_wrong_ports_and_unrelated_scripts(monkeypatch):
    expected = record();signals = []
    monkeypatch.setattr(lease, "os", SimpleNamespace(getuid=lambda: 9, kill=lambda *args: signals.append(args), close=lambda fd: None))
    monkeypatch.setattr(lease, "identity", lambda pid: {**expected, "start_ticks": 999})
    assert lease.interrupt_exact(expected) == "signature_mismatch_or_exited" and not signals
    monkeypatch.setattr(lease, "identity", lambda pid: expected)
    assert lease.interrupt_exact({**expected, "script": "/tmp/unrelated.py"}) == "refused_unowned_service_signature"
    assert lease.interrupt_exact({**expected, "port": 8771}) == "refused_unowned_service_signature"
    assert lease.interrupt_exact(expected) == "SIGINT_sent" and signals == [(123, lease.signal.SIGINT)]


def test_launch_lease_closes_five_minutes_before_deadline(tmp_path, monkeypatch):
    registry = tmp_path/"registry.json"
    registry.write_text(json.dumps({"deadline_utc": lease.DEADLINE.isoformat(), "launch_cutoff_utc": lease.CUTOFF.isoformat(), "owner_uid": lease.os.getuid()}))
    monkeypatch.setattr(lease, "now", lambda: lease.CUTOFF-timedelta(seconds=1));lease.lease_check(registry)
    monkeypatch.setattr(lease, "now", lambda: lease.CUTOFF)
    with pytest.raises(ValueError, match="cutoff"): lease.lease_check(registry)
