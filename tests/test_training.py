from __future__ import annotations

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from training.workflow import (
    NUSCENES_CLASSES, configure_source_model, experiment_specs, load_yaml,
    set_training_mode, split_scenes, transfer_state,
)


def source_config():
    return {
        "CLASS_NAMES": list(NUSCENES_CLASSES),
        "DATA_CONFIG": {
            "POINT_CLOUD_RANGE": [-54, -54, -5, 54, 54, 3],
            "POINT_FEATURE_ENCODING": {"used_feature_list": ["x", "y", "z", "intensity", "timestamp"],
                                       "src_feature_list": ["x", "y", "z", "intensity", "timestamp"]},
            "DATA_PROCESSOR": [{"NAME": "transform_points_to_voxels", "VOXEL_SIZE": [0.075, 0.075, 0.2]},
                               {"NAME": "shuffle_points", "SHUFFLE_ENABLED": {"train": True, "test": True}}],
            "DATA_AUGMENTOR": {"AUG_CONFIG_LIST": [{"NAME": "gt_sampling"}, {"NAME": "random_world_flip"}]},
        },
        "MODEL": {"NAME": "VoxelNeXt", "DENSE_HEAD": {
            "CLASS_NAMES_EACH_HEAD": [["car"], NUSCENES_CLASSES[1:]],
            "POST_PROCESSING": {"SCORE_THRESH": 0.1},
            "LOSS_CONFIG": {"LOSS_WEIGHTS": {"code_weights": [1.0] * 10}}}},
        "OPTIMIZATION": {"WEIGHT_DECAY": 0.01, "GRAD_NORM_CLIP": 10.0},
    }


class SceneSplitTests(unittest.TestCase):
    def setUp(self):
        self.train = [{"token": str(i)} for i in range(8)]
        self.val = [{"token": "v0"}, {"token": "v1"}]
        self.mapping = {str(i): f"scene{i // 2}" for i in range(8)} | {"v0": "val_scene", "v1": "val_scene"}

    def test_scene_integrity_and_reproducibility(self):
        fit, dev, manifest = split_scenes(self.train, self.val, self.mapping, seed=42)
        self.assertFalse(set(manifest["fit_scenes"]) & set(manifest["development_scenes"]))
        self.assertFalse(set(manifest["heldout_scenes"]) & set(manifest["fit_scenes"]))
        self.assertEqual(len(fit) + len(dev), len(self.train))
        other = split_scenes(list(reversed(self.train)), self.val, self.mapping, seed=42)[2]
        self.assertEqual(manifest, other)

    def test_released_overlap_fails(self):
        self.mapping["v0"] = "scene0"
        with self.assertRaisesRegex(ValueError, "scenes overlap"):
            split_scenes(self.train, self.val, self.mapping)

    def test_missing_and_duplicate_tokens_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            split_scenes(self.train + self.train[:1], self.val, self.mapping)
        with self.assertRaisesRegex(ValueError, "missing"):
            split_scenes(self.train + [{"token": "unknown"}], self.val, self.mapping)


class ConfigurationTests(unittest.TestCase):
    def test_source_head_geometry_and_input_preserved(self):
        source = source_config()
        before = copy.deepcopy(source)
        cfg = configure_source_model(source, {"DATASET": "NuScenesDataset", "POINT_CLOUD_RANGE": [-300] * 6},
                                     Path("/tmp/data"), {"fit": Path("/tmp/fit.pkl"), "development": Path("/tmp/dev.pkl")})
        self.assertEqual(source, before)
        self.assertEqual(cfg["CLASS_NAMES"], NUSCENES_CLASSES)
        self.assertEqual(cfg["DATA_CONFIG"]["POINT_CLOUD_RANGE"], source["DATA_CONFIG"]["POINT_CLOUD_RANGE"])
        self.assertEqual(cfg["DATA_CONFIG"]["DATA_PROCESSOR"][0]["VOXEL_SIZE"], [0.075, 0.075, 0.2])
        self.assertEqual(cfg["DATA_CONFIG"]["MAX_SWEEPS"], 1)
        self.assertFalse(cfg["DATA_CONFIG"]["BALANCED_RESAMPLING"])
        self.assertNotIn("gt_sampling", [x["NAME"] for x in cfg["DATA_CONFIG"]["DATA_AUGMENTOR"]["AUG_CONFIG_LIST"]])
        self.assertEqual(cfg["MODEL"]["DENSE_HEAD"]["LOSS_CONFIG"]["LOSS_WEIGHTS"]["code_weights"][-2:], [0, 0])

    def test_ten_class_checkpoint_contract_is_required(self):
        cfg = source_config()
        cfg["CLASS_NAMES"] = ["car"]
        with self.assertRaisesRegex(ValueError, "ten nuScenes"):
            configure_source_model(cfg, {}, Path("/tmp"), {})

    def test_experiments_share_budget_and_seed(self):
        specs = experiment_specs(epochs=12, seed=9)
        fits = [x for x in specs if x["mode"] != "baseline"]
        self.assertEqual({x["epochs"] for x in fits}, {12})
        self.assertEqual({x["seed"] for x in fits}, {9})
        self.assertEqual([x["lr"] for x in fits], [1e-4, 1e-4, 1e-5])

    def test_yaml_base_resolution_and_cycle_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text("A: {one: 1, two: 2}\n")
            (root / "model.yaml").write_text("DATA_CONFIG:\n  _BASE_CONFIG_: base.yaml\n  A: {two: 3}\n")
            self.assertEqual(load_yaml(root / "model.yaml", root), {"DATA_CONFIG": {"A": {"one": 1, "two": 3}}})
            (root / "cycle.yaml").write_text("_BASE_CONFIG_: cycle.yaml\n")
            with self.assertRaisesRegex(ValueError, "Recursive"):
                load_yaml(root / "cycle.yaml", root)

    def test_prepare_writes_real_subsets_and_unexecuted_plan(self):
        import numpy as np
        import yaml
        from training.prepare import prepare

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pcdet = root / "pcdet"
            model_dir = pcdet / "tools/cfgs/nuscenes_models"
            dataset_dir = pcdet / "tools/cfgs/dataset_configs"
            loader_dir = pcdet / "pcdet/datasets/nuscenes"
            metadata = root / "data/v1.0-mini"
            for path in (model_dir, dataset_dir, loader_dir, metadata):
                path.mkdir(parents=True)
            (model_dir / "cbgs_voxel0075_voxelnext.yaml").write_text(yaml.safe_dump(source_config()))
            (dataset_dir / "vmax_dataset.yaml").write_text("DATASET: NuScenesDataset\n")
            (loader_dir / "nuscenes_dataset.py").write_text("# fixture marker: get_lidar_with_sweeps; [:, :4]\n")
            np.asarray([[1, 2, 3, 4, 5]], dtype=np.float32).tofile(metadata / "cloud.bin")
            rows = [{"token": f"t{i}", "scene_token": f"s{i // 2}"} for i in range(6)]
            (metadata / "sample.json").write_text(json.dumps(rows))
            infos = [{"token": row["token"], "lidar_path": "cloud.bin", "gt_boxes": np.zeros((0, 7), dtype=np.float32),
                      "gt_names": np.asarray([], dtype=str)} for row in rows]
            train, val = root / "train.pkl", root / "val.pkl"
            train.write_bytes(pickle.dumps(infos[:4]))
            val.write_bytes(pickle.dumps(infos[4:]))
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"hash-only preparation fixture; not a model")
            output = root / "prepared"
            args = SimpleNamespace(output=output, openpcdet_root=pcdet, source_config=None,
                                   checkpoint=checkpoint, data_root=root / "data", train_info=train,
                                   val_info=val, development_fraction=0.5, seed=42, epochs=3, batch_size=1)
            plan = prepare(args)
            self.assertEqual(plan["status"], "prepared_not_executed")
            self.assertEqual(len(plan["experiments"]), 4)
            self.assertTrue((output / "plan.json").is_file())
            with (output / "infos_fit.pkl").open("rb") as handle:
                fit = pickle.load(handle)
            with (output / "infos_development.pkl").open("rb") as handle:
                dev = pickle.load(handle)
            self.assertFalse({x["token"] for x in fit} & {x["token"] for x in dev})
            self.assertEqual(fit[0]["gt_boxes"].shape[1], 9)
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                prepare(args)


class TinyDetector(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
        self.dense_head = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.BatchNorm1d(2))

    def forward(self, x):
        return self.dense_head(self.backbone(x))


@unittest.skipIf(torch is None, "Optional checkpoint/freeze tests require PyTorch")
class TransferAndFreezeTests(unittest.TestCase):
    def test_full_parameter_transfer_and_missing_weight_failure(self):
        model = TinyDetector()
        state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        report = transfer_state(model, state)
        self.assertEqual(report["loaded_tensors"], len(model.state_dict()))
        self.assertTrue(all(torch.equal(value, state[key]) for key, value in model.state_dict().items()))
        del state["backbone.0.weight"]
        with self.assertRaisesRegex(ValueError, "do not fully match"):
            transfer_state(model, state)

    def test_missing_batch_counter_is_explicit_and_allowed(self):
        model = TinyDetector()
        state = {key: value for key, value in model.state_dict().items() if not key.endswith("num_batches_tracked")}
        report = transfer_state(model, state)
        self.assertEqual(sum(x["status"] == "reset_counter" for x in report["entries"]), 2)

    def test_spconv_permutation_only_for_identified_sparse_weights(self):
        model = torch.nn.Linear(3, 2, bias=False)
        state = {"weight": torch.arange(6.0).reshape(3, 2)}
        with self.assertRaises(ValueError):
            transfer_state(model, state)
        report = transfer_state(model, state, {"weight"})
        self.assertEqual(report["entries"][0]["conversion"], "spconv_transpose_last_two")
        self.assertTrue(torch.equal(model.weight, state["weight"].T))

    def test_spconv_output_first_conversion(self):
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.zeros(5, 2, 3, 4, 6)))
        disk = torch.arange(2 * 3 * 4 * 6 * 5.0).reshape(2, 3, 4, 6, 5)
        report = transfer_state(model, {"weight": disk}, {"weight"})
        self.assertEqual(report["entries"][0]["conversion"], "spconv_output_first")
        self.assertTrue(torch.equal(model.weight, disk.permute(4, 0, 1, 2, 3)))

    def test_head_only_freezes_backbone_gradients_and_batchnorm_statistics(self):
        model = TinyDetector()
        initial = model.backbone[1].running_mean.clone()
        initial_head = model.dense_head[1].running_mean.clone()
        trainable = set_training_mode(model, "head")
        self.assertTrue(all(name.startswith("dense_head.") for name in trainable))
        model(torch.randn(8, 3) + 5).square().mean().backward()
        self.assertTrue(torch.equal(model.backbone[1].running_mean, initial))
        self.assertFalse(torch.equal(model.dense_head[1].running_mean, initial_head))
        self.assertIsNone(model.backbone[0].weight.grad)
        self.assertIsNotNone(model.dense_head[0].weight.grad)
        set_training_mode(model, "full")
        self.assertTrue(model.backbone[1].training)
        self.assertTrue(model.backbone[0].weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
