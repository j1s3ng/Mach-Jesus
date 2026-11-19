#!/usr/bin/env python3
"""Verify externally supplied artifacts without downloading or accepting terms."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest = json.loads((args.root / "artifacts/downloads.json").read_text())
    failures = []
    for row in manifest["artifacts"]:
        path = args.root / row["source"]
        status = "MISSING: acquisition/hosting pending"
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            status = "VERIFIED" if digest == row["sha256"] else "HASH MISMATCH"
        print(row["role"] + ": " + status)
        if status != "VERIFIED": failures.append(row["role"])
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__": main()
