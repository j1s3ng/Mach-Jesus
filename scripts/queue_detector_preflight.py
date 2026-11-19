"""Attest and warm only the detector endpoints a queued physical trial uses.

No physical output directory, trial, model training, service restart or SSH
process is created here. A failed check invalidates its warm proof and leaves
the job pending. Every dispatch uses a fresh inference; no PID-only warm cache
is trusted. Predictions contain one pinned measured training cloud only.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SHA = "9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b"
CV425_SHA = "0a68a778cb676eea56d1da60131ac55ab190a78e2ba14537fec13c2b996d63f6"
CONFIG_SHA = "84f5ddcab780ef108af77412f9bad587c21299419530c8708121442449115a0d"
PRO_UUID = os.environ.get("RACE_PRO_GPU_UUID", "GPU-PRO6000")
RTX_UUID = os.environ.get("RACE_SECONDARY_GPU_UUID", "GPU-SECONDARY")
SAFE_CLOUD = Path("runs/improvement-12h/training/detector-dense-dataset-001/points/184c75ab91665e0ecc73d463.npy")
SAFE_CLOUD_SHA = "25be182a66b248d8b8f0e61c5e2ebde6081c64b2fb655aded68dabcf66dea5a5"
PIPELINE_EXPECTED = {
    "effective_car_score_threshold": .425,
    "source_model_config_sha256": CONFIG_SHA,
    "source_server_sha256": "dea51efdf1b31f4a177740b7857023671777d12c24e5e05750847ad38503cf54",
    "source_postprocess_sha256": "7bde005a94a0b592dd49be5dbe5f21839bd47bb36b6e3cc8ec772034db7ae6b5",
    "score_calibration_sha256": "6d05be728649789564641990a984cde9841e51fab1fea4ed36e9c1e05887b3aa",
    "weights_state_sha256": "4ae5edabfe8c1eb38f7e88cdf0b25c9441200823390cf25239cd475a40d2f22f",
}
VERSION = "queue_detector_preflight_v1"


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"); temporary.replace(path)


def flag(args, name, default):
    value = default
    for index, item in enumerate(args):
        if item == name:
            if index + 1 >= len(args):
                raise ValueError("Missing value for " + name)
            value = args[index + 1]
        elif item.startswith(name + "="):
            value = item.split("=", 1)[1]
    return value


def endpoints(job):
    """Mirror run_improvement_heat defaults and actual sensor assignments."""
    args = job["args"]
    count = int(flag(args, "--cars", 4)); secondary = int(flag(args, "--secondary-cars", 0))
    if not 1 <= count <= 64 or not 0 <= secondary <= count:
        raise ValueError("Invalid field/secondary sensor assignment")
    detector = str(flag(args, "--detector-sha", OFFICIAL_SHA))
    if detector not in (OFFICIAL_SHA, CV425_SHA):
        raise ValueError("No approved preflight profile for detector " + detector)
    primary_url = str(flag(args, "--vision-url", "http://127.0.0.1:8770")).rstrip("/")
    secondary_url = str(flag(args, "--secondary-url", "http://127.0.0.1:8771")).rstrip("/")
    live_grids = [0] if flag(args, "--scenario", "race") == "stopped" else list(range(count))
    result = []
    for grid in live_grids:
        url = secondary_url if grid >= count - secondary else primary_url
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("Detector URL must be an HTTP(S) base URL")
        if any(row["url"] == url for row in result):
            continue
        expected_gpu = {8770: PRO_UUID, 8771: RTX_UUID, 8772: PRO_UUID}.get(parsed.port)
        result.append({"url": url, "detector_sha256": detector, "config_sha256": CONFIG_SHA,
                       "effective_score_threshold": .425 if detector == CV425_SHA else None,
                       "expected_gpu_uuid": expected_gpu})
    return result


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def health_identity(response, spec):
    model = response.get("model", {})
    if response.get("status") != "ready" or not isinstance(response.get("pid"), int) or response["pid"] <= 0:
        raise ValueError("Detector is not a ready identified process")
    for key, expected in (("backend", "voxelnext"), ("device", "cuda:0"),
                          ("checkpoint_sha256", spec["detector_sha256"]), ("config_sha256", spec["config_sha256"]),
                          ("scored_class", "car"), ("max_range_m", 50.)):
        if model.get(key) != expected:
            raise ValueError("Detector health identity mismatch: " + key)
    gpu = model.get("cuda_visible_devices")
    if gpu not in (PRO_UUID, RTX_UUID) or spec["expected_gpu_uuid"] is not None and gpu != spec["expected_gpu_uuid"]:
        raise ValueError("Detector GPU identity differs")
    transfer = model.get("transfer", {})
    if transfer.get("loaded_tensors") != 542 or transfer.get("model_tensors") != 542 or transfer.get("missing") != [] or transfer.get("mismatched") != []:
        raise ValueError("Incomplete checkpoint tensor transfer")
    pipeline = model.get("pipeline")
    if spec["effective_score_threshold"] is None:
        if "effective_car_score_threshold" in model or pipeline is not None:
            raise ValueError("Original unwrapped detector pipeline changed")
    else:
        if model.get("effective_car_score_threshold") != .425 or not isinstance(pipeline, dict):
            raise ValueError("Calibrated detector score threshold differs")
        if pipeline.get("version") != 1 or any(pipeline.get(key) != value for key, value in PIPELINE_EXPECTED.items()):
            raise ValueError("Calibrated detector pipeline binding differs")
    requests = response.get("batching", {}).get("requests")
    if not isinstance(requests, int) or isinstance(requests, bool) or requests < 0:
        raise ValueError("Missing detector instance request counter")
    return {"pid": response["pid"], "checkpoint_sha256": model["checkpoint_sha256"],
            "config_sha256": model["config_sha256"], "gpu_uuid": gpu,
            "pipeline": pipeline, "effective_score_threshold": model.get("effective_car_score_threshold")}, requests


class HTTPTransport:
    def health(self, url, timeout):
        with urlopen(url + "/health", timeout=timeout) as response:
            return json.load(response)

    def predict(self, url, payload, car_id, timeout):
        request = Request(url + "/predict", data=payload,
                          headers={"Content-Type": "application/x-npy", "X-Car-Id": car_id}, method="POST")
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)


def valid_prediction(response, car_id, minimum_score=None):
    if response.get("car_id") != car_id or not isinstance(response.get("boxes"), list):
        raise ValueError("Prediction response identity/schema differs")
    if not response["boxes"]:
        raise ValueError("Known measured smoke cloud produced no car detection")
    for box in response["boxes"]:
        for key in ("center", "size"):
            values = box.get(key)
            if not isinstance(values, list) or len(values) != 3 or not all(number(value) for value in values):
                raise ValueError("Invalid measured-cloud box geometry")
        if any(value <= 0. for value in box["size"]) or not number(box.get("yaw")) or not number(box.get("score")) or not 0. <= box["score"] <= 1.:
            raise ValueError("Invalid measured-cloud prediction")
        if minimum_score is not None and box["score"] < minimum_score:
            raise ValueError("Returned box violates the declared effective score threshold")
    for key in ("inference_ms", "server_ms"):
        if not number(response.get(key)) or response[key] < 0.:
            raise ValueError("Missing actual inference clock")


class DetectorPreflight:
    def __init__(self, root=ROOT, campaign=None, *, transport=None, cloud_path=None, cloud_sha256=SAFE_CLOUD_SHA):
        self.root = Path(root)
        self.directory = Path(campaign or self.root / "runs/improvement-12h") / "detector_preflight"
        self.transport = transport or HTTPTransport()
        self.cloud_path = Path(cloud_path or self.root / SAFE_CLOUD)
        self.cloud_sha256 = cloud_sha256

    def _payload(self):
        raw = self.cloud_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.cloud_sha256:
            raise ValueError("Pinned measured preflight cloud changed")
        cloud = np.load(io.BytesIO(raw), allow_pickle=False)
        if cloud.ndim != 2 or cloud.shape[1] != 4 or len(cloud) < 32 or cloud.dtype.kind != "f" or not np.isfinite(cloud).all():
            raise ValueError("Preflight requires the actual finite Nx4 measured cloud")
        payload = io.BytesIO(); np.save(payload, cloud.astype(np.float32), allow_pickle=False)
        return payload.getvalue(), len(cloud)

    def _timeout(self, deadline, maximum):
        remaining = deadline - time.time()
        if remaining <= 0.:
            raise TimeoutError("Detector preflight deadline reached")
        return min(maximum, max(.05, remaining))

    def check(self, job, *, deadline):
        report = {"version": VERSION, "job_id": job["job_id"], "checked_epoch_s": time.time(),
                  "ready": False, "services": [], "cloud_path": str(self.cloud_path), "cloud_sha256": self.cloud_sha256,
                  "GT_sent": False, "physical_trial_launched": False}
        try:
            required = endpoints(job)
            for spec in required:
                url = spec["url"]
                try:
                    before = self.transport.health(url, self._timeout(deadline, 3.))
                    identity, requests = health_identity(before, spec)
                    # Fresh single RPC each job: the frozen health contract has
                    # no bootID/start-ticks tag, so no cross-job PID-only cache.
                    payload, points = self._payload()
                    car_id = f"queue-preflight-{job['job_id']}-{int(time.time()*1e6)}"[:128]
                    prediction = self.transport.predict(url, payload, car_id, self._timeout(deadline, 15.))
                    valid_prediction(prediction, car_id, spec["effective_score_threshold"])
                    after = self.transport.health(url, self._timeout(deadline, 3.))
                    after_identity, after_requests = health_identity(after, spec)
                    if after_identity != identity or after_requests < requests + 1:
                        raise ValueError("Detector instance changed or prediction was not processed by that instance")
                    result = {"url": url, "ready": True, "instance": identity, "warm_cache_reused": False,
                              "proof_epoch_s": time.time(), "requests": after_requests, "points": points,
                              "boxes": len(prediction["boxes"]), "inference_ms": prediction["inference_ms"],
                              "server_ms": prediction["server_ms"], "request_car_id": car_id}
                    report["services"].append(result)
                except Exception as error:
                    report["services"].append({"url": url, "ready": False, "error": f"{type(error).__name__}: {error}"})
            report["ready"] = bool(report["services"]) and all(row["ready"] for row in report["services"])
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        atomic_json(self.directory / "latest.json", report)
        with (self.directory / "checks.jsonl").open("a") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
        return report
