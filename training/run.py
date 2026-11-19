"""Execute a genuine pretrained CUDA pilot, matched fit, or selected evaluation."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .workflow import load_yaml, set_training_mode, sha256, transfer_state


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execute(args):
    import numpy as np
    import torch
    from easydict import EasyDict

    plan = json.loads(args.plan.read_text())
    specs = {x["name"]: x for x in plan["experiments"]}
    if args.experiment not in specs:
        raise ValueError(f"Unknown experiment; choose {list(specs)}")
    spec = specs[args.experiment]
    for path, expected in plan["source_identity"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Frozen study input changed: {path}")
    if sha256(Path(spec["config"])) != spec["config_sha256"] or sha256(Path(plan["split_manifest"])) != plan["split_sha256"]:
        raise ValueError("Frozen configuration or scene split changed")
    if args.phase != "evaluate" and spec["mode"] == "baseline":
        raise ValueError("The pretrained baseline is evaluation only")
    if args.split == "heldout":
        if args.phase != "evaluate" or not args.selection:
            raise ValueError("Held-out data is evaluation only: provide a frozen --selection JSON containing experiment and checkpoint_sha256")
        selection = json.loads(args.selection.read_text())
        selected_checkpoint = args.checkpoint or Path(plan["pretrained_checkpoint"])
        if selection.get("experiment") != args.experiment or selection.get("checkpoint_sha256") != sha256(selected_checkpoint):
            raise ValueError("Held-out evaluation must match the frozen development selection")
        if selection.get("plan_sha256") != sha256(args.plan):
            raise ValueError("Held-out selection must use this exact frozen study plan")
    if not torch.cuda.is_available():
        raise RuntimeError("No NVIDIA CUDA device; use the LAN GPU host for VoxelNeXt")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite an existing run; use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "phase": args.phase, "experiment": spec,
              "started_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha256(args.plan),
              "training_executed": False, "pilot_executed": False, "evaluation_executed": False}
    write_json(args.output / "status.json", status)
    try:
        root = Path(plan["openpcdet_root"])
        sys.path.insert(0, str(root))
        from pcdet.datasets import build_dataloader
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
        from pcdet.utils.spconv_utils import find_all_spconv_keys

        torch.cuda.set_device(args.device)
        seed = args.seed if args.seed is not None else spec["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        config = EasyDict(load_yaml(Path(spec["config"]), root / "tools"))
        training = args.phase != "evaluate"
        if not training:
            config.DATA_CONFIG.INFO_PATH.test = [plan["info_paths"][args.split]]
        logger = common_utils.create_logger(log_file=str(args.output / "execution.log"), rank=0)
        dataset, loader, _ = build_dataloader(
            dataset_cfg=config.DATA_CONFIG, class_names=config.CLASS_NAMES,
            batch_size=int(config.OPTIMIZATION.BATCH_SIZE_PER_GPU) if training else 1,
            dist=False, workers=args.workers, seed=seed, logger=logger, training=training,
        )
        if not len(dataset) or not len(loader):
            raise ValueError("Empty dataset: inspect export/version/info paths")
        sample_points = dataset.get_lidar_with_sweeps(0, max_sweeps=1)
        raw_path = Path(config.DATA_CONFIG.DATA_PATH) / config.DATA_CONFIG.VERSION / dataset.infos[0]["lidar_path"]
        raw_points = np.fromfile(raw_path, dtype=np.float32).reshape(-1, 5)
        if (sample_points.shape != raw_points.shape or not np.isfinite(sample_points).all()
                or not np.array_equal(sample_points[:, :4], raw_points[:, :4])
                or not np.array_equal(sample_points[:, 4], np.zeros(len(sample_points), dtype=np.float32))):
            raise ValueError("Fork input audit failed: expected unchanged x/y/z/intensity and zero time-lag, discarding raw ring")
        write_json(args.output / "input_layout_audit.json", {"verified": True, "sample_token": str(dataset.infos[0]["token"]),
                   "raw_columns": ["x", "y", "z", "intensity", "ring"],
                   "model_columns": ["x", "y", "z", "intensity", "zero_time_lag"], "points": len(sample_points)})
        model = build_network(model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=dataset)
        checkpoint_path = args.checkpoint or Path(plan["pretrained_checkpoint"])
        if training and args.checkpoint:
            raise ValueError("Every comparison fit starts from the same original checkpoint; omit --checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        disk_state = checkpoint.get("model_state", checkpoint.get("state_dict"))
        if disk_state is None:
            raise ValueError("Checkpoint needs model_state or state_dict")
        transfer = transfer_state(model, disk_state, find_all_spconv_keys(model))
        write_json(args.output / "checkpoint_transfer.json", transfer)
        del checkpoint, disk_state
        model.cuda()
        status.update(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                      seed=seed, checkpoint_path=str(checkpoint_path.resolve()), checkpoint_sha256=sha256(checkpoint_path), dataset_frames=len(dataset),
                      checkpoint_size_bytes=checkpoint_path.stat().st_size)
        write_json(args.output / "status.json", status)
        if training:
            run_fit(args, model, loader, spec, config, status, load_data_to_gpu)
        else:
            run_evaluation(args, model, loader, dataset, config, status, load_data_to_gpu)
        status.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "status.json", status)
        return status
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(args.output / "status.json", status)
        raise


def run_fit(args, model, loader, spec, config, status, load_data_to_gpu):
    import torch

    names = set_training_mode(model, spec["mode"])
    if not names:
        raise ValueError("No trainable parameters")
    write_json(args.output / "trainable_parameters.json", {"mode": spec["mode"], "names": names})
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=spec["lr"],
                                  weight_decay=float(config.OPTIMIZATION.WEIGHT_DECAY))
    step, started = 0, time.perf_counter()
    epochs = math.ceil(args.pilot_steps / len(loader)) if args.phase == "pilot" else spec["epochs"]
    with (args.output / "losses.jsonl").open("w") as losses:
        for epoch in range(epochs):
            for batch in loader:
                set_training_mode(model, spec["mode"])
                optimizer.zero_grad(set_to_none=True)
                load_data_to_gpu(batch)
                loss_dict, _, _ = model(batch)
                loss = loss_dict["loss"].mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                parameters = [p for p in model.parameters() if p.requires_grad]
                norm = torch.nn.utils.clip_grad_norm_(parameters, float(config.OPTIMIZATION.GRAD_NORM_CLIP))
                if not math.isfinite(float(norm)):
                    raise ValueError("Nonfinite gradient norm")
                optimizer.step()
                if hasattr(model, "update_global_step"):
                    model.update_global_step()
                step += 1
                status.update(training_executed=True, training_steps=step)
                if step == 1:
                    write_json(args.output / "status.json", status)
                losses.write(json.dumps({"epoch": epoch + 1, "step": step, "loss": float(loss.detach()),
                                         "gradient_norm": float(norm), "learning_rate": spec["lr"]}) + "\n")
                losses.flush()
                if args.phase == "pilot" and step >= args.pilot_steps:
                    break
            checkpoint_path = args.output / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "optimizer_state": optimizer.state_dict(), "epoch": epoch + 1, "it": step,
                        "study_mode": spec["mode"], "seed": status["seed"]}, checkpoint_path)
            # A real save/reload plus forward pass is required for pilot completion.
            status.update(epochs_completed=epoch + 1, final_checkpoint=str(checkpoint_path))
            write_json(args.output / "status.json", status)
            if args.phase == "pilot" and step >= args.pilot_steps:
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                transfer_state(model, saved["model_state"])
                del saved
                model.eval()
                next_batch = next(iter(loader))
                next_batch.pop("gt_boxes", None)
                load_data_to_gpu(next_batch)
                with torch.inference_mode():
                    predictions, _ = model(next_batch)
                status.update(pilot_executed=True, pilot_forward_after_reload=True,
                              pilot_prediction_count=sum(len(x["pred_boxes"]) for x in predictions))
                break
    torch.cuda.synchronize()
    status.update(training_executed=True, training_steps=step, epochs_completed=epoch + 1,
                  training_seconds=time.perf_counter() - started,
                  final_checkpoint=str(checkpoint_path), final_checkpoint_sha256=sha256(checkpoint_path))


def run_evaluation(args, model, loader, dataset, config, status, load_data_to_gpu):
    import numpy as np
    import torch
    from race_lidar.metrics import evaluate_frames

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    warmup_batch = next(iter(loader))
    warmup_batch.pop("gt_boxes", None)
    load_data_to_gpu(warmup_batch)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(warmup_batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    token_to_info = {str(info["token"]): info for info in dataset.infos}
    sample_path = Path(config.DATA_CONFIG.DATA_PATH) / config.DATA_CONFIG.VERSION / "sample.json"
    token_to_scene = {str(x["token"]): str(x["scene_token"]) for x in json.loads(sample_path.read_text())}
    predictions, truth, latencies = [], [], []
    car_label = config.CLASS_NAMES.index("car") + 1
    with torch.inference_mode(), (args.output / "predictions.jsonl").open("w") as out:
        for batch in loader:
            tokens = [str(metadata["token"]) for metadata in batch["metadata"]]
            batch.pop("gt_boxes", None)  # labels are never inputs to inference
            started = time.perf_counter()
            load_data_to_gpu(batch)
            result, _ = model(batch)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000)
            for token, row in zip(tokens, result):
                boxes = row["pred_boxes"].detach().cpu().numpy()
                labels = row["pred_labels"].detach().cpu().numpy()
                scores = row["pred_scores"].detach().cpu().numpy()
                prediction = {"scene_id": token_to_scene[token], "frame_id": token, "boxes": []}
                for box, label, score in zip(boxes, labels, scores):
                    if int(label) == car_label and np.linalg.norm(box[:2]) <= 50.0:
                        prediction["boxes"].append({"center": box[:3].tolist(), "size": box[3:6].tolist(),
                                                    "yaw": float(box[6]), "score": float(score)})
                info = token_to_info[token]
                gt = {"scene_id": token_to_scene[token], "frame_id": token, "gt_boxes": []}
                point_counts = np.asarray(info.get("num_lidar_pts", np.ones(len(info["gt_boxes"]))))
                for box, name, count in zip(info["gt_boxes"], info["gt_names"], point_counts):
                    if str(name) == "car" and count >= 1 and np.linalg.norm(box[:2]) <= 50.0:
                        gt["gt_boxes"].append({"center": box[:3].tolist(), "size": box[3:6].tolist(), "yaw": float(box[6])})
                predictions.append(prediction)
                truth.append(gt)
                out.write(json.dumps(prediction, allow_nan=False) + "\n")
    metrics = evaluate_frames(predictions, truth)
    metrics.update(evaluation_radius_m=50.0, split=args.split, batch_size=1, warmup_passes=args.warmup,
                   latency_scope="host-to-device copy plus model forward/NMS, after CPU voxel preparation",
                   latency_median_ms=float(np.median(latencies)), latency_p95_ms=float(np.percentile(latencies, 95)),
                   throughput_frames_per_second=1000.0 / float(np.mean(latencies)),
                   peak_torch_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_torch_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                   official_a2rl_metrics_executed=False)
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "ground_truth_for_audit.json", truth)
    status.update(evaluation_executed=True, evaluated_frames=len(predictions), metrics_file=str(args.output / "metrics.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--phase", choices=["pilot", "train", "evaluate"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Evaluation checkpoint; comparison fits always use the original")
    parser.add_argument("--split", choices=["development", "heldout"], default="development")
    parser.add_argument("--selection", type=Path, help="Frozen selection JSON required for held-out evaluation")
    parser.add_argument("--seed", type=int, help="Use for the second-seed repeat of the selected configuration")
    parser.add_argument("--pilot-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA index; use CUDA_VISIBLE_DEVICES to choose a GPU")
    args = parser.parse_args()
    if args.pilot_steps < 1 or args.warmup < 0 or args.workers < 0:
        parser.error("pilot-steps must be positive; warmup/workers must be nonnegative")
    try:
        result = execute(args)
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
