import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from race_lidar.pipeline_artifact import tensor_state_sha256
from race_lidar.point_support import config_sha256


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_tensor_identity_is_order_independent_but_binds_shape_dtype_and_content():
    a = {"a": torch.tensor(1), "b": torch.tensor([1., 2.])}
    assert tensor_state_sha256(a) == tensor_state_sha256(dict(reversed(list(a.items()))))
    for b in ({**a, "a": torch.tensor([1])}, {**a, "a": torch.tensor(1.)}, {**a, "b": torch.tensor([1., 3.])}):
        assert tensor_state_sha256(a) != tensor_state_sha256(b)


def test_packaging_preserves_tensors_and_runtime_health_binds_artifact_and_support(tmp_path, monkeypatch):
    parent, artifact = tmp_path/"parent.pth", tmp_path/"supported.pth"
    config_file = tmp_path/"model.yaml";config_file.write_text("config fixture")
    torch.save({"model_state": {"weight": torch.tensor([1., 2.])}, "dataset_manifest_sha256": {"train": "train-hash"}}, parent)
    original = sha(parent)
    support = {"name": "measured_above_ground_box_support", "version": 1, "minimum_points": 2,
               "above_ground_z_m": .25, "box_margin_m": 0., "minimum_xy_diameter_m": .2}
    report = tmp_path/"calibration_report.json";report.write_text("{}")
    selection = {"fresh_inputs_opened": False, "parent_checkpoint_sha256": original,
                 "postprocess": support, "postprocess_sha256": config_sha256(support),
                 "training_manifest_sha256": "train-hash", "protocol_sha256": "protocol-hash",
                 "calibration_report_sha256": sha(report)}
    selected = tmp_path/"selected_config.json";selected.write_text(json.dumps(selection))
    (tmp_path/"status.json").write_text(json.dumps({"status": "completed", "parent_checkpoint_sha256": original,
        "protocol_sha256": "protocol-hash", "source_postprocess_sha256": sha(ROOT/"src/race_lidar/point_support.py"),
        "model": {"config_sha256": sha(config_file)}}))
    packaged = script("package_supported_detector").package(parent, selected, artifact)
    assert sha(parent) == original and packaged["tensor_state_unchanged_verified"]
    assert sha(artifact) != original
    server = script("serve_supported_vision")
    detection = {"center": [0., 0., .5], "size": [4., 2., 1.], "yaw": 0., "score": .8}
    def fake_load(*args):
        def predictor(clouds): return [[detection] for _ in clouds]
        predictor.metadata = {"checkpoint_sha256": sha(artifact), "transfer": {"loaded_tensors": 542}}
        return predictor
    monkeypatch.setattr(server, "load_batched_predictor", fake_load)
    predictor = server.load_supported_predictor(tmp_path, config_file, artifact, sha(artifact))
    assert predictor.metadata["checkpoint_sha256"] == sha(artifact)
    assert predictor.metadata["weights_parent_checkpoint_sha256"] == original
    assert predictor.metadata["postprocess"] == support
    cloud = np.array([[-.2, 0., .5, 1.], [.2, 0., .5, 1.]], dtype=np.float32)
    assert predictor([cloud, np.empty((0, 4), dtype=np.float32)]) == [[detection], []]
    with pytest.raises(ValueError, match="artifact differs"):
        server.load_supported_predictor(tmp_path, config_file, artifact, original)
    modified = torch.load(artifact, map_location="cpu", weights_only=True)
    modified["model_state"]["weight"][0] = 9.
    torch.save(modified, artifact)
    with pytest.raises(ValueError, match="Inherited detector tensors differ"):
        server.load_supported_predictor(tmp_path, config_file, artifact, sha(artifact))
