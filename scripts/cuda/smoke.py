"""Real pretrained GPU pipeline test with a simulator cloud and one toy label.

This is not A2RL training or an accuracy evaluation. It checks full weight reuse,
forward inference, head gradients, optimizer update, save/reload, and inference.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcdet-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.pcdet_root.resolve()))
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils.spconv_utils import find_all_spconv_keys
    from training.workflow import load_yaml, sha256, transfer_state, set_training_mode

    torch.manual_seed(7)
    np.random.seed(7)
    config = EasyDict(load_yaml(args.config, args.pcdet_root / "tools"))
    config.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST = []
    config.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = []
    for processor in config.DATA_CONFIG.DATA_PROCESSOR:
        if processor.NAME == "shuffle_points":
            processor.SHUFFLE_ENABLED.test = False
    config.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights[-2:] = [0., 0.]
    cloud = np.load(args.points, allow_pickle=False).astype(np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 4 or not np.isfinite(cloud).all():
        parser.error("Expected finite N x 4 simulator point cloud")
    features = np.column_stack((cloud, np.zeros(len(cloud), np.float32)))
    train_data = DatasetTemplate(config.DATA_CONFIG, config.CLASS_NAMES, training=True)
    eval_data = DatasetTemplate(config.DATA_CONFIG, config.CLASS_NAMES, training=False)
    model = build_network(config.MODEL, len(config.CLASS_NAMES), train_data)
    disk = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    transfer = transfer_state(model, disk["model_state"], find_all_spconv_keys(model))
    (args.output / "checkpoint_transfer.json").write_text(json.dumps(transfer, indent=2) + "\n")
    del disk
    model.cuda().eval()

    def inference():
        item = eval_data.prepare_data({"points": features.copy(), "frame_id": "simulator_smoke"})
        batch = eval_data.collate_batch([item])
        assert "gt_boxes" not in batch
        load_data_to_gpu(batch)
        with torch.inference_mode():
            rows, _ = model(batch)
        torch.cuda.synchronize()
        row = rows[0]
        return {"box_count_all_classes": len(row["pred_boxes"]),
                "car_count": int((row["pred_labels"] == 1).sum()),
                "finite_boxes": bool(torch.isfinite(row["pred_boxes"]).all())}

    started = time.perf_counter()
    before_inference = inference()
    trainable = set_training_mode(model, "head")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    # This box is a pipeline fixture, never used as an inference input.
    item = train_data.prepare_data({"points": features.copy(), "frame_id": "simulator_smoke",
        "gt_names": np.array(["car"]),
        "gt_boxes": np.array([[18., 2.5, .57, 5., 2.2, 1.14, .28, 0., 0.]], np.float32)})
    batch = train_data.collate_batch([item])
    load_data_to_gpu(batch)
    parameters = [p for p in model.parameters() if p.requires_grad]
    first_before = parameters[0].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss_dict, _, _ = model(batch)
    loss = loss_dict["loss"].mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite pilot loss")
    loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("Missing or nonfinite gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.)
    optimizer.step()
    changed = bool((parameters[0].detach() != first_before).any())
    if not changed:
        raise RuntimeError("Optimizer did not update the first head parameter")
    output_checkpoint = args.output / "smoke_head_step.pth"
    torch.save({"model_state": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                "smoke_only": True, "source_checkpoint_sha256": sha256(args.checkpoint)}, output_checkpoint)
    saved = torch.load(output_checkpoint, map_location="cpu", weights_only=True)
    transfer_state(model, saved["model_state"])
    model.eval()
    after_inference = inference()
    report = {"status":"completed", "dataset":"simulator cloud with one toy pilot annotation",
        "a2rl_training_executed":False, "accuracy_evaluation_executed":False,
        "gpu":torch.cuda.get_device_name(), "torch":torch.__version__, "cuda":torch.version.cuda,
        "points":len(cloud), "input_sha256":sha256(args.points),
        "source_checkpoint_sha256":sha256(args.checkpoint),
        "source_config_sha256":sha256(args.config),
        "loaded_tensors":transfer["loaded_tensors"], "model_tensors":transfer["model_tensors"],
        "trainable_parameter_tensors":len(trainable), "gradient_tensors":len(gradients),
        "loss":float(loss.detach()), "gradient_norm_before_clip":float(gradient_norm),
        "optimizer_changed_first_head_tensor":changed,
        "before_inference":before_inference,"after_reload_inference":after_inference,
        "saved_checkpoint_sha256":sha256(output_checkpoint),
        "seconds":time.perf_counter()-started}
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
