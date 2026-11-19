"""Clean-checkout launcher contracts without starting Webots or a GPU."""
import fcntl
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_launcher(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("launcher_under_test_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hotlap_fixture(tmp_path, monkeypatch, extra=()):
    launcher = load_launcher("run_hotlap", monkeypatch)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    reference, webots = tmp_path / "reference.npz", tmp_path / "webots-stub"
    reference.write_bytes(b"reference fixture")
    webots.write_bytes(b"Webots fixture; never executed")
    output = tmp_path / "attempt"
    monkeypatch.setattr(sys, "argv", ["run_hotlap.py", "--reference", str(reference),
        "--webots", str(webots), "--output", str(output), *extra])
    generated, launched = [], []

    def generate(command, check):
        generated.append(command)
        target = Path(command[command.index("--output") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("generated world fixture")

    def launch(command, env):
        launched.append((command, env))
        return 7

    monkeypatch.setattr(launcher.subprocess, "run", generate)
    monkeypatch.setattr(launcher.subprocess, "call", launch)
    return launcher, output, generated, launched


@pytest.mark.parametrize("explicit_world", [False, True])
def test_hotlap_generates_default_zero_backmarker_world_but_preserves_explicit_world(tmp_path, monkeypatch, explicit_world):
    extra = []
    supplied = tmp_path / "supplied.wbt"
    if explicit_world:
        supplied.write_text("user supplied world")
        extra = ["--world", str(supplied)]
    launcher, _, generated, launched = hotlap_fixture(tmp_path, monkeypatch, extra)
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert stopped.value.code == 7
    assert len(launched) == 1
    if explicit_world:
        assert not generated
        assert supplied.read_text() == "user supplied world"
        assert launched[0][0][-1] == str(supplied)
    else:
        assert len(generated) == 1
        command = generated[0]
        assert command[command.index("--backmarkers") + 1] == "0"
        assert Path(launched[0][0][-1]).read_text() == "generated world fixture"
        assert launched[0][1]["RACE_BACKMARKERS"] == "0"


def test_hotlap_rejects_existing_evidence_before_regenerating_a_world(tmp_path, monkeypatch):
    launcher, output, generated, launched = hotlap_fixture(tmp_path, monkeypatch)
    output.mkdir()
    sentinel = output / "existing-result.json"
    sentinel.write_text("preserve this evidence")
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert stopped.value.code == 2
    assert sentinel.read_text() == "preserve this evidence"
    assert not generated and not launched


@pytest.mark.parametrize("option,value", [("--max-seconds", "nan"), ("--max-seconds", "inf"), ("--max-speed", "inf")])
def test_hotlap_rejects_nonfinite_limits_before_writing_output(tmp_path, monkeypatch, option, value):
    launcher, output, generated, launched = hotlap_fixture(tmp_path, monkeypatch, [option, value])
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert stopped.value.code == 2
    assert not output.exists() and not generated and not launched


def test_improvement_launcher_creates_first_run_lock_and_releases_it_on_exit(tmp_path, monkeypatch):
    launcher = load_launcher("run_improvement_heat", monkeypatch)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    lock_path = tmp_path / "runs/improvement-12h/simulator.lock"
    output = tmp_path / "attempt"
    config, world = tmp_path / "config.json", tmp_path / "world.wbt"
    config.write_text(json.dumps({"rules": {}}))
    world.write_text("world fixture")
    events = []

    def prepare(args):
        assert lock_path.is_file()
        with lock_path.open("a+") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return config, world

    class Ledger:
        def __init__(self, path):
            assert path.parent == lock_path.parent

        def append(self, event, value):
            events.append((event, value))

    monkeypatch.setattr(launcher, "prepare", prepare)
    monkeypatch.setattr(launcher, "Ledger", Ledger)
    monkeypatch.setattr(launcher.subprocess, "call", lambda command, env: 19)
    monkeypatch.setattr(sys, "argv", ["run_improvement_heat.py", "--output", str(output)])
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert stopped.value.code == 19
    assert [name for name, _ in events] == ["attempt_scheduled", "attempt_started", "attempt_finished"]
    assert events[-1][1]["status"] == "failed"
    with lock_path.open("a+") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("seconds", ["nan", "inf"])
def test_improvement_launcher_rejects_nonfinite_duration_before_creating_a_run(tmp_path, monkeypatch, seconds):
    launcher = load_launcher("run_improvement_heat", monkeypatch)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_improvement_heat.py", "--output", str(tmp_path / "attempt"), "--seconds", seconds])
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert stopped.value.code == 2
    assert not (tmp_path / "runs").exists()
