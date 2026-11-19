import threading
from http.server import HTTPServer

import numpy as np

from race_lidar.projection import project_points
from race_lidar.remote import make_handler, remote_detect


def test_projection_axes_and_sensor_translation():
    points = np.array([[11.6, 0, 1.2], [11.6, 1, 1.2], [11.6, 0, 2.2], [-1,0,0]])
    uv, valid = project_points(points)
    assert valid.tolist() == [True,True,True,False]
    np.testing.assert_allclose(uv[0], [319.5,179.5])
    assert uv[1,0] < uv[0,0]  # ego left goes image left
    assert uv[2,1] < uv[0,1]  # up goes image up


def test_http_detector_only_receives_points():
    seen = []
    def predict(points):
        seen.append(points.copy())
        return [{"center":[10,0,1],"size":[4,2,1],"yaw":0,"score":.7}]
    server = HTTPServer(("127.0.0.1",0),make_handler(predict))
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        points = np.array([[10,0,1,.4]],dtype=np.float32)
        result = remote_detect(points,f"http://127.0.0.1:{server.server_port}")
        assert result[0]["score"] == .7
        np.testing.assert_array_equal(seen[0],points)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_replay_evaluates_common_supported_range(tmp_path):
    from race_lidar.io import write_frame
    from race_lidar.replay import replay
    recording = tmp_path / "recording"
    boxes = [{"center":[x,0,1],"size":[4,2,1],"yaw":0} for x in [10,70]]
    write_frame(recording,"0",np.empty((0,4),dtype=np.float32),0,"scene",boxes)
    result = replay(recording,tmp_path/"replay",make_video=False)
    assert result["evaluation"]["evaluation_radius_m"] == 50
    assert result["evaluation"]["ground_truth_objects"] == 1
