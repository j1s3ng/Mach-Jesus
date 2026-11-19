import copy
import hashlib
import io
import time

import numpy as np
import pytest

from scripts.queue_detector_preflight import (CONFIG_SHA, CV425_SHA, OFFICIAL_SHA, PIPELINE_EXPECTED,
    PRO_UUID, RTX_UUID, DetectorPreflight, endpoints)
from scripts.run_improvement_queue import detector_guarded_trial


def job(*args):
    return {"job_id": "trial", "output": "physical/trial", "estimated_wall_seconds": 1, "args": list(args)}


class WireFixture:
    def __init__(self, *, candidate=False):
        self.counter = {}; self.calls = []; self.fail_health = False; self.fail_predict = False
        self.change = {}; self.candidate = candidate; self.pid_changes_after_predict = False

    def health(self, url, timeout):
        self.calls.append(("health", url))
        if self.fail_health:
            raise ConnectionRefusedError("simulated missing SSH forward")
        candidate = self.candidate
        model = {"backend": "voxelnext", "device": "cuda:0", "checkpoint_sha256": CV425_SHA if candidate else OFFICIAL_SHA,
                 "config_sha256": CONFIG_SHA, "cuda_visible_devices": RTX_UUID if url.endswith(":8771") else PRO_UUID,
                 "transfer": {"loaded_tensors": 542, "model_tensors": 542, "missing": [], "mismatched": []},
                 "scored_class": "car", "max_range_m": 50.}
        if candidate:
            model.update(effective_car_score_threshold=.425, pipeline={"version": 1, **PIPELINE_EXPECTED})
        model.update(copy.deepcopy(self.change))
        return {"status": "ready", "pid": 102 if self.pid_changes_after_predict and self.counter.get(url, 0) else 101,
                "model": model, "batching": {"requests": self.counter.get(url, 0)}}

    def predict(self, url, payload, car_id, timeout):
        self.calls.append(("predict", url))
        if self.fail_predict:
            raise TimeoutError("simulated CUDA RPC timeout")
        points = np.load(io.BytesIO(payload), allow_pickle=False)
        assert points.shape == (64, 4) and points.dtype == np.float32
        self.counter[url] = self.counter.get(url, 0) + 1
        return {"car_id": car_id, "boxes": [{"center": [20., 0., .8], "size": [4., 2., 1.], "yaw": 0., "score": .8}],
                "inference_ms": 10., "server_ms": 15.}


def checker(tmp_path, wire):
    cloud = tmp_path / "measured.npy"
    np.save(cloud, np.ones((64, 4), np.float32))
    return DetectorPreflight(tmp_path, tmp_path / "campaign", transport=wire, cloud_path=cloud,
                             cloud_sha256=hashlib.sha256(cloud.read_bytes()).hexdigest())


def test_needed_endpoints_follow_actual_sensor_assignment_including_static_obstacles():
    assert [row["url"] for row in endpoints(job())] == ["http://127.0.0.1:8770"]
    assert len(endpoints(job("--cars", "20", "--secondary-cars", "8"))) == 2
    assert [row["url"] for row in endpoints(job("--cars", "4", "--secondary-cars", "4"))] == ["http://127.0.0.1:8771"]
    assert len(endpoints(job("--scenario", "stopped", "--cars", "3", "--secondary-cars", "1"))) == 1
    assert endpoints(job("--vision-url=http://127.0.0.1:8772", "--detector-sha=" + CV425_SHA))[0]["effective_score_threshold"] == .425


def test_refused_connection_never_launches_or_creates_trial_output_then_recovery_can(tmp_path):
    wire = WireFixture(); wire.fail_health = True; probe = checker(tmp_path, wire); launched = []
    def launch(proof):
        launched.append(proof); (tmp_path / "physical/trial").mkdir(parents=True); return {"status": "launched"}
    result, proof = detector_guarded_trial(job(), probe, time.time() + 1000., launch)
    assert result is None and not proof["ready"] and not launched
    assert not (tmp_path / "physical/trial").exists()
    assert not any(call[0] == "predict" for call in wire.calls)
    wire.fail_health = False
    result, proof = detector_guarded_trial(job(), probe, time.time() + 1000., launch)
    assert result == {"status": "launched"} and proof["ready"] and len(launched) == 1
    assert (tmp_path / "physical/trial").exists()


@pytest.mark.parametrize("change", [{"checkpoint_sha256": "0" * 64}, {"config_sha256": "0" * 64},
                                    {"cuda_visible_devices": "foreign-gpu"}, {"effective_car_score_threshold": .425},
                                    {"transfer": {"loaded_tensors": 1, "model_tensors": 542, "missing": [], "mismatched": []}}])
def test_wrong_original_pipeline_identity_never_dispatches_trial(tmp_path, change):
    wire = WireFixture(); wire.change = change; launched = []
    result, proof = detector_guarded_trial(job(), checker(tmp_path, wire), time.time() + 1000., lambda value: launched.append(1))
    assert result is None and not proof["ready"] and not launched
    assert not any(call[0] == "predict" for call in wire.calls)


def test_calibrated_pipeline_is_accepted_only_with_exact_threshold_and_pipeline_binding(tmp_path):
    wire = WireFixture(candidate=True); probe = checker(tmp_path, wire)
    selected = job("--vision-url", "http://127.0.0.1:8772", "--detector-sha", CV425_SHA)
    assert probe.check(selected, deadline=time.time() + 60.)["ready"]
    wire.change = {"effective_car_score_threshold": .35}
    assert not probe.check(selected, deadline=time.time() + 60.)["ready"]


def test_inference_timeout_or_instance_swap_never_launches(tmp_path):
    for mode in ("timeout", "swap"):
        folder = tmp_path / mode; folder.mkdir(); wire = WireFixture()
        wire.fail_predict = mode == "timeout"; wire.pid_changes_after_predict = mode == "swap"
        launched = []
        result, proof = detector_guarded_trial(job(), checker(folder, wire), time.time() + 1000., lambda value: launched.append(1))
        assert result is None and not proof["ready"] and not launched


def test_no_pid_only_cache_every_job_gets_its_own_actual_cloud_prediction(tmp_path):
    wire = WireFixture(); probe = checker(tmp_path, wire)
    assert probe.check(job(), deadline=time.time() + 60.)["ready"]
    assert probe.check(job(), deadline=time.time() + 60.)["ready"]
    assert sum(call[0] == "predict" for call in wire.calls) == 2


def test_missing_secondary_is_not_hidden_by_healthy_primary(tmp_path):
    class SecondaryRefused(WireFixture):
        def health(self, url, timeout):
            if url.endswith(":8771"):
                raise ConnectionRefusedError("secondary missing")
            return super().health(url, timeout)
    proof = checker(tmp_path, SecondaryRefused()).check(job("--cars", "20", "--secondary-cars", "8"), deadline=time.time() + 60.)
    assert not proof["ready"] and len(proof["services"]) == 2


def test_pinned_cloud_change_and_deadline_prevent_launch(tmp_path):
    wire = WireFixture(); probe = checker(tmp_path, wire)
    probe.cloud_path.write_bytes(b"corrupted source")
    launched = []
    result, proof = detector_guarded_trial(job(), probe, time.time() + 1000., lambda value: launched.append(1))
    assert result is None and not proof["ready"] and not launched
    assert not any(call[0] == "predict" for call in wire.calls)
    probe = checker(tmp_path, wire)
    result, proof = detector_guarded_trial(job(), probe, time.time() + 1., lambda value: launched.append(1))
    assert result is None and not proof["ready"] and not launched


def test_healthy_probe_cannot_spend_the_campaign_deadline_reserve(tmp_path):
    wire = WireFixture(); launched = []
    result, proof = detector_guarded_trial(job(), checker(tmp_path, wire), time.time() + 300.5, lambda value: launched.append(1))
    assert result is None and not proof["ready"] and proof["error"] == "deadline_reserve_after_detector_preflight" and not launched
