"""Report real capabilities without claiming an unrun CUDA test passed."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path


def inspect(openpcdet_root=None, data_root=None, checkpoint=None):
    report = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
              "cuda_pilot_executed": False, "requirements": {}, "blockers": []}
    try:
        import torch
        report.update(torch=torch.__version__, cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                              "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                              "capability": list(torch.cuda.get_device_capability(i))}
                             for i in range(torch.cuda.device_count())]
        else:
            report["blockers"].append("VoxelNeXt requires Linux/NVIDIA CUDA; Mac Metal cannot execute its CUDA sparse-convolution/NMS stack")
    except ImportError:
        report["blockers"].append("PyTorch is not installed")
    for name in ("spconv", "cumm", "pcdet", "nuscenes", "yaml", "huggingface_hub"):
        try:
            report["requirements"][name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            report["requirements"][name] = False
    for name in ("spconv", "cumm", "pcdet", "nuscenes", "yaml"):
        if not report["requirements"][name]:
            report["blockers"].append(f"Required package is missing: {name}")
    if openpcdet_root:
        path = Path(openpcdet_root)
        checks = {"vmax_dataset_config": path / "tools/cfgs/dataset_configs/vmax_dataset.yaml",
                  "voxelnext_model": path / "pcdet/models/detectors/voxelnext.py"}
        report["files"] = {key: value.is_file() for key, value in checks.items()}
        if not all(report["files"].values()):
            report["blockers"].append("A2RL bundled vmax_OpenPCDet fork is missing or incomplete")
    if data_root and not (Path(data_root) / "v1.0-mini/sample.json").is_file():
        report["blockers"].append("Exported A2RL nuScenes metadata is missing")
    if checkpoint and not Path(checkpoint).is_file():
        report["blockers"].append("Official pretrained checkpoint is missing")
    report["status"] = "blocked" if report["blockers"] else "preflight_only_not_a_pilot"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpcdet-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.openpcdet_root, args.data_root, args.checkpoint)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return bool(report["blockers"])


if __name__ == "__main__":
    raise SystemExit(main())
