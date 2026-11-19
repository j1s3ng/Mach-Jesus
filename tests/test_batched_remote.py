import io
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from race_lidar.batched_remote import Microbatcher, RaceVisionServer, make_handler, validate_cloud


def cloud(value):
    return np.array([[value, 0, 0, 1]], dtype=np.float32)


@pytest.mark.parametrize("points", [np.zeros((2, 3)), np.zeros((2, 4), dtype=int),
                                   np.array([[np.nan, 0, 0, 1]]), np.array([[1e300, 0, 0, 1]])])
def test_invalid_or_unrepresentable_sensor_clouds_are_rejected(points):
    with pytest.raises(ValueError):
        validate_cloud(points)


def test_batch_preserves_independent_results_and_empty_samples():
    seen = []
    def predict_batch(clouds):
        seen.append(clouds)
        return [[{"value": float(points[0, 0])}] if len(points) else [] for points in clouds]
    worker = Microbatcher(predict_batch, max_batch=4, batch_wait_ms=50)
    try:
        futures = [worker.submit(cloud(2)), worker.submit(np.empty((0, 4), np.float32)), worker.submit(cloud(9))]
        results = [future.result(timeout=2) for future in futures]
        assert results[0]["boxes"] == [{"value": 2.0}]
        assert results[1]["boxes"] == []
        assert results[2]["boxes"] == [{"value": 9.0}]
        assert [result["batch_size"] for result in results] == [3, 3, 3]
        assert len(seen) == 1
    finally:
        worker.close()


def test_one_bad_example_isolated_and_worker_recovers():
    def predict_batch(clouds):
        if any(points[0, 0] < 0 for points in clouds):
            raise ValueError("Bad example")
        return [[float(points[0, 0])] for points in clouds]
    worker = Microbatcher(predict_batch, max_batch=4, batch_wait_ms=30)
    try:
        good, bad = worker.submit(cloud(2)), worker.submit(cloud(-1))
        assert good.result(timeout=2)["boxes"] == [2.0]
        with pytest.raises(ValueError, match="Bad example"):
            bad.result(timeout=2)
        assert worker.submit(cloud(8)).result(timeout=2)["boxes"] == [8.0]
        assert worker.health()["batching"]["failures"] == 1
    finally:
        worker.close()


def test_http_concurrent_clients_keep_identity_and_reject_invalid_npy():
    worker = Microbatcher(lambda clouds: [[float(points[0, 0])] for points in clouds], batch_wait_ms=40)
    server = RaceVisionServer(("127.0.0.1", 0), make_handler(worker))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    def request(index):
        buf = io.BytesIO()
        np.save(buf, cloud(index), allow_pickle=False)
        req = Request(url + "/predict", data=buf.getvalue(), headers={"X-Car-Id": str(index)})
        with urlopen(req, timeout=2) as response:
            return json.load(response)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(request, range(4)))
        assert [row["boxes"] for row in results] == [[0.0], [1.0], [2.0], [3.0]]
        assert [row["car_id"] for row in results] == ["0", "1", "2", "3"]
        assert any(row["batch_size"] > 1 for row in results)
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(url + "/predict", data=b"not an NPY"), timeout=2)
        assert caught.value.code == 400
        assert request(7)["boxes"] == [7.0]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        worker.close()


def test_pending_queue_is_bounded_and_stopping_rejects_new_work():
    entered, release = threading.Event(), threading.Event()
    def predict_batch(clouds):
        entered.set()
        release.wait(timeout=2)
        return [[] for _ in clouds]
    worker = Microbatcher(predict_batch, max_batch=1, max_pending=1, batch_wait_ms=0)
    try:
        first = worker.submit(cloud(1))
        assert entered.wait(timeout=2)
        second = worker.submit(cloud(2))
        with pytest.raises(queue.Full):
            worker.submit(cloud(3))
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    finally:
        release.set()
        worker.close()
    with pytest.raises(RuntimeError, match="stopping"):
        worker.submit(cloud(4))
