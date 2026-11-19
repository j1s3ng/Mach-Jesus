"""Point-only detector transport. Bind loopback and use SSH forwarding for a GPU host."""
from __future__ import annotations

import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen

import numpy as np

MAX_BYTES = 16 * 1024 * 1024


def remote_detect(points, url="http://127.0.0.1:8765", timeout=30):
    """Only measured points cross the wire; labels/camera recognition never do."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(points, dtype=np.float32), allow_pickle=False)
    req = Request(url.rstrip("/") + "/predict", data=buf.getvalue(),
                  headers={"Content-Type": "application/x-npy"}, method="POST")
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)["boxes"]


def make_handler(predict):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            metadata = getattr(predict,"metadata",{})
            payload = json.dumps({"status":"ready", "pid":os.getpid(),
                                  "model":{key:metadata[key] for key in
                                           ("backend","device","gpu_name","cuda_visible_devices",
                                            "checkpoint_sha256","config_sha256","max_range_m")
                                           if key in metadata}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length",str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    raise ValueError("Expected a nonempty NPY body smaller than 16 MiB")
                points = np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False)
                if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 4:
                    raise ValueError("Expected float points of shape [N,4] (x,y,z,intensity)")
                if points.dtype.kind != "f" or not np.isfinite(points).all():
                    raise ValueError("Points must be finite floating point numbers")
                start = time.perf_counter()
                boxes = predict(points.astype(np.float32, copy=False))
                payload = json.dumps({"boxes": boxes, "inference_ms": 1000 * (time.perf_counter() - start)},
                                     allow_nan=False).encode()
            except (ValueError, TypeError, EOFError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return
    return Handler


def serve_detector(backend, port, pcdet_root=None, config=None, checkpoint=None):
    if backend == "cluster":
        from .baseline import detect
        predictor = detect
    else:
        if not all([pcdet_root, config, checkpoint]):
            raise SystemExit("voxelnext requires --pcdet-root, --config and --checkpoint")
        from training.inference import load_predictor
        predictor = load_predictor(pcdet_root, config, checkpoint)
    server = HTTPServer(("127.0.0.1", port), make_handler(predictor))
    print(f"{backend} detector ready on http://127.0.0.1:{port}; use ssh -L for LAN access", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
