# Simulated DEV BEV AP reproduction

Recomputes one BEV IoU 0.5 / AP101 column for six detector checkpoints from cached scored numeric boxes and labels. This is a known simulated development population, not a fresh holdout, official benchmark score, live inference, or a driving result. The compressed export includes all 6,144 frames per model, 8,155 eligible ground-truth cars, 10,277 excluded ground-truth boxes used as ignore regions, and scored post-NMS detections down to the common 0.1 floor. Lower-scored or suppressed boxes cannot be recovered. Read [PROTOCOL.md](PROTOCOL.md) for matching, ignore behavior, and interpolation before interpreting the numbers. Original raw source cache hashes are recorded in `source-provenance.json`.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q test_metric.py
.venv/bin/python reproduce.py --output /tmp/bev-ap-results.json
```

Choose an output path that does not already exist. `manifest.json` pins all input files and expected outputs. The reproduction verifies full frame/model coverage and recalculates every recall sample against the frozen expected output. No GPU, detector weights, raw point clouds, simulator, or network is required after dependencies are installed. Numeric box exports omit raw data paths and personnel identifiers. Fitting used one matched seed; adjacent frames are correlated.
