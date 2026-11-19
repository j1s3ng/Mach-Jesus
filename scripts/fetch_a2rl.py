#!/usr/bin/env python3
"""Acquire the official gated toolchain; downloading all sensor data is opt-in."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = "a2rl-vmax/a2rl-vmax"
REVISION = "4280b7f76e0e20e8aa0fc943a088a6f7051f6d8d"


def fetch(output, revision=REVISION, data=False, metadata_only=False):
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": REPO, "requested_revision": revision,
                "requested_scope": "metadata_only" if metadata_only else "all_data" if data else "toolchain_only",
                "time_utc": datetime.now(timezone.utc).isoformat(), "status": "started"}
    try:
        info = HfApi().dataset_info(REPO, revision=revision)
        manifest.update(revision=info.sha, gated=info.gated)
        if metadata_only:
            manifest.update(status="metadata_inspected", files=[x.rfilename for x in info.siblings])
        else:
            patterns = None if data else ["README.md", "extract.py", "code/**"]
            snapshot_download(repo_id=REPO, repo_type="dataset", revision=info.sha,
                              local_dir=output, allow_patterns=patterns)
            manifest.update(status="downloaded", sensor_data_downloaded=bool(data))
    except GatedRepoError:
        manifest.update(status="blocked_access", action="Request/accept dataset access on the official Hugging Face page, then run hf auth login locally. No credentials belong in source files or chat.")
    except HfHubHTTPError as exc:
        manifest.update(status="download_failed", http_status=getattr(exc.response, "status_code", None),
                        action="Check connectivity, revision, and the locally authenticated account's access")
    except OSError as exc:
        manifest.update(status="download_failed", error_type=type(exc).__name__, action="Check network connectivity and available local storage")
    (output / "acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("external/a2rl-vmax"))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--data", action="store_true", help="Download the complete ~67 GB release, including sensor data")
    parser.add_argument("--metadata-only", action="store_true", help="Inspect public file metadata without downloading the toolchain")
    args = parser.parse_args()
    if args.data and args.metadata_only:
        parser.error("Choose --data or --metadata-only")
    result = fetch(args.output, args.revision, args.data, args.metadata_only)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"downloaded", "metadata_inspected"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
