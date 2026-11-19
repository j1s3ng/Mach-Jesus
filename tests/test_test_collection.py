"""Regression tests must be collected without maintaining a hidden allowlist."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("include_archive", [False, True])
def test_new_tests_run_by_default_and_only_declared_archive_requires_opt_in(tmp_path, include_archive):
    project = tmp_path / "fixture"
    tests = project / "tests"
    tests.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("conftest.py"), tests / "conftest.py")
    (project / "PORTABLE_TESTS.json").write_text(json.dumps({
        "default_files": ["test_known.py"], "archived_files": ["test_campaign.py"]}))
    (tests / "test_known.py").write_text("def test_known(): assert True\n")
    (tests / "test_new.py").write_text("def test_new(): assert True\n")
    (tests / "test_campaign.py").write_text("def test_campaign(): assert False, 'archive fixture was selected'\n")
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if include_archive:
        command.append("--include-archived-campaign-tests")
    command.append("tests")
    result = subprocess.run(command, cwd=project, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=30)
    if include_archive:
        assert result.returncode == 1, result.stdout + result.stderr
        assert "1 failed, 2 passed" in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "2 passed" in result.stdout
