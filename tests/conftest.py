"""Only explicitly archived campaign tests require opt-in.

New tests run by default, so adding a regression cannot silently hide it behind
an out-of-date allowlist. The manifest also records the known portable files.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_FILES = frozenset(json.loads((ROOT / "PORTABLE_TESTS.json").read_text())["archived_files"])


def pytest_addoption(parser):
    parser.addoption("--include-archived-campaign-tests", action="store_true", default=False)


def pytest_ignore_collect(collection_path, config):
    if config.getoption("--include-archived-campaign-tests"):
        return False
    path = Path(str(collection_path))
    if path.parent == ROOT / "tests":
        return path.name in ARCHIVED_FILES
    return False
