"""Bounded, concurrent point-cloud RPC with one CUDA microbatch worker.

Each request remains an independent example. Only inference weights are shared;
tracking, driving decisions, labels, and camera frames never enter this service.
"""
from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .remote import MAX_BYTES


def validate_cloud(points):
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("Expected N×4 x,y,z,intensity points")
    if points.dtype.kind != "f" or not np.isfinite(points).all():
        raise ValueError("Points must be finite floating point numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        cloud = points.astype(np.float32, copy=False)
    if not np.isfinite(cloud).all():
        raise ValueError("Points must remain finite in float32")
    return cloud


def load_batched_predictor(pcdet_root, config_path, checkpoint_path, device="cuda:0"):
    """Load the identical, strictly transferred official VoxelNeXt checkpoint.

    This follows training.inference.load_predictor, but collates independent
    examples into one forward pass. Five input features include zero time lag.
    """
    import torch
    from easydict import EasyDict
    from training.workflow import NUSCENES_CLASSES, load_yaml, sha256, transfer_state

    root = Path(pcdet_root).resolve()
    if not (root / "pcdet/models/detectors/voxelnext.py").is_file() or not (root / "pcdet/datasets/dataset.py").is_file():
        raise ValueError("Expected official VoxelNeXt/OpenPCDet source")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("VoxelNeXt inference needs NVIDIA CUDA")
    sys.path.insert(0, str(root))
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils.spconv_utils import find_all_spconv_keys

    config = EasyDict(load_yaml(Path(config_path), root / "tools"))
    if config.CLASS_NAMES != NUSCENES_CLASSES or config.MODEL.NAME != "VoxelNeXt":
        raise ValueError("Expected the retained ten-class nuScenes VoxelNeXt configuration")
    torch_device = torch.device(device)
    with torch.cuda.device(torch_device):
        dataset = DatasetTemplate(dataset_cfg=config.DATA_CONFIG, class_names=config.CLASS_NAMES, training=False)
        model = build_network(model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=dataset)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint.get("state_dict"))
        if state is None:
            raise ValueError("Checkpoint needs model_state or state_dict")
        transfer = transfer_state(model, state, find_all_spconv_keys(model))
        model.to(torch_device).eval()
    car_label = config.CLASS_NAMES.index("car") + 1

    def predict_batch(point_clouds):
        result = [[] for _ in point_clouds]
        items, positions = [], []
        for index, points in enumerate(point_clouds):
            cloud = validate_cloud(points)
            if not len(cloud):
                continue
            features = np.column_stack((cloud, np.zeros(len(cloud), dtype=np.float32)))
            item = dataset.prepare_data(data_dict={"points": features, "frame_id": str(index)})
            if not len(item.get("voxel_coords", item.get("points", []))):
                continue
            items.append(item)
            positions.append(index)
        if not items:
            return result
        batch = dataset.collate_batch(items)
        with torch.cuda.device(torch_device), torch.inference_mode():
            load_data_to_gpu(batch)
            predictions, _ = model(batch)
            if len(predictions) != len(positions):
                raise RuntimeError("VoxelNeXt returned a different number of examples")
            for index, row in zip(positions, predictions):
                boxes = row["pred_boxes"].detach().cpu().numpy()
                scores = row["pred_scores"].detach().cpu().numpy()
                labels = row["pred_labels"].detach().cpu().numpy()
                result[index] = [
                    {"center": box[:3].tolist(), "size": box[3:6].tolist(),
                     "yaw": float(box[6]), "score": float(score)}
                    for box, score, label in zip(boxes, scores, labels)
                    if int(label) == car_label and np.linalg.norm(box[:2]) <= 50.0
                ]
        return result

    predict_batch.metadata = {
        "backend": "voxelnext", "device": str(torch_device),
        "gpu_name": torch.cuda.get_device_name(torch_device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_sha256": sha256(Path(checkpoint_path)),
        "config_sha256": sha256(Path(config_path)), "transfer": transfer,
        "scored_class": "car", "max_range_m": 50.0, "time_lag_seconds": 0.0,
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByUUID(os.environ["CUDA_VISIBLE_DEVICES"])
        def telemetry():
            memory = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            return {"memory_used_mib": memory.used / 2**20, "memory_total_mib": memory.total / 2**20,
                    "gpu_utilization_percent": utilization.gpu,
                    "torch_allocated_mib": torch.cuda.memory_allocated(torch_device) / 2**20,
                    "torch_reserved_mib": torch.cuda.memory_reserved(torch_device) / 2**20}
        predict_batch.telemetry = telemetry
    except (ImportError, KeyError, RuntimeError):
        pass
    return predict_batch


class Microbatcher:
    """Bound queue growth and serialize all GPU work on one dedicated thread."""

    def __init__(self, predict_batch, max_batch=16, batch_wait_ms=5.0, max_pending=64):
        if max_batch < 1 or batch_wait_ms < 0 or max_pending < max_batch:
            raise ValueError("Need positive max_batch, nonnegative wait, and max_pending >= max_batch")
        self.predict_batch = predict_batch
        self.max_batch = max_batch
        self.batch_wait_ms = batch_wait_ms
        self.pending = queue.Queue(max_pending)
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "failures": 0, "batches": 0, "fallback_batches": 0,
                      "batch_size_counts": Counter(), "worker_ms": 0.0}
        self.thread = threading.Thread(target=self._work, name="voxelnext-batcher", daemon=True)
        self.thread.start()

    def submit(self, points):
        points = validate_cloud(points)
        future = Future()
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("Detector is stopping")
            self.pending.put_nowait((points, future, time.perf_counter()))
        return future

    def health(self):
        with self.lock:
            stats = {**self.stats, "batch_size_counts": dict(self.stats["batch_size_counts"])}
        metadata = dict(getattr(self.predict_batch, "metadata", {}))
        if "transfer" in metadata:
            metadata["transfer"] = {key: value for key, value in metadata["transfer"].items() if key != "entries"}
        try:
            telemetry = getattr(self.predict_batch, "telemetry", lambda: {})()
        except Exception as error:
            telemetry = {"error": str(error)}
        return {"status": "stopping" if self.closed.is_set() else "ready", "pid": os.getpid(),
                "model": metadata, "gpu": telemetry,
                "batching": {"max_batch": self.max_batch, "batch_wait_ms": self.batch_wait_ms,
                             "max_pending": self.pending.maxsize, "queued": self.pending.qsize(), **stats}}

    def close(self):
        with self.lock:
            self.closed.set()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError("Detector worker did not stop within 60 seconds")

    def _work(self):
        while not self.closed.is_set() or not self.pending.empty():
            try:
                jobs = [self.pending.get(timeout=0.05)]
            except queue.Empty:
                continue
            deadline = time.perf_counter() + self.batch_wait_ms / 1000
            while len(jobs) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    jobs.append(self.pending.get(timeout=remaining))
                except queue.Empty:
                    break
            start = time.perf_counter()
            fallback = False
            try:
                results = self.predict_batch([job[0] for job in jobs])
                if len(results) != len(jobs):
                    raise RuntimeError("Detector returned a different number of examples")
            except Exception:
                # One problematic cloud must not discard other clients' frames.
                fallback = True
                results = []
                for points, _, _ in jobs:
                    try:
                        single = self.predict_batch([points])
                        if len(single) != 1:
                            raise RuntimeError("Detector returned a different number of examples")
                        results.append(single[0])
                    except Exception as error:
                        results.append(error)
            end = time.perf_counter()
            duration_ms = (end - start) * 1000
            failures = sum(isinstance(boxes, Exception) for boxes in results)
            with self.lock:
                self.stats["requests"] += len(jobs)
                self.stats["failures"] += failures
                self.stats["batches"] += 1
                self.stats["fallback_batches"] += int(fallback)
                self.stats["worker_ms"] += duration_ms
                self.stats["batch_size_counts"][len(jobs)] += 1
            for (_, future, submitted), boxes in zip(jobs, results):
                if isinstance(boxes, Exception):
                    future.set_exception(boxes)
                else:
                    future.set_result({"boxes": boxes, "inference_ms": duration_ms,
                                       "queue_ms": (start - submitted) * 1000,
                                       "server_ms": (end - submitted) * 1000,
                                       "batch_size": len(jobs), "fallback": fallback})
                self.pending.task_done()


def make_handler(batcher):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "Not found"})
                return
            self.respond(200, batcher.health())

        def do_POST(self):
            if self.path != "/predict":
                self.respond(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    raise ValueError("Expected a nonempty NPY body smaller than 16 MiB")
                self.connection.settimeout(30)
                points = validate_cloud(np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False))
                future = batcher.submit(points)
            except (ValueError, TypeError, EOFError, OSError) as error:
                self.respond(400, {"error": str(error)})
                return
            except queue.Full:
                self.respond(503, {"error": "Detector queue is full; reduce sensor frequency"})
                return
            except RuntimeError as error:
                self.respond(503, {"error": str(error)})
                return
            try:
                result = future.result(timeout=60)
                result["car_id"] = self.headers.get("X-Car-Id", "")[:128]
                result["gpu"] = getattr(batcher.predict_batch, "metadata", {}).get("gpu_name")
                self.respond(200, result)
            except Exception as error:
                self.respond(500, {"error": str(error)})

        def log_message(self, format, *args):
            pass
    return Handler


class RaceVisionServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64


def serve(predict_batch, port=8770, max_batch=16, batch_wait_ms=5.0, max_pending=64):
    batcher = Microbatcher(predict_batch, max_batch, batch_wait_ms, max_pending)
    server = RaceVisionServer(("127.0.0.1", port), make_handler(batcher))
    print(json.dumps({"event": "ready", "url": f"http://127.0.0.1:{server.server_port}", **batcher.health()}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        batcher.close()
