# CPU reproductions

These recipes recompute results from saved predictions or simulator measurements.
They need no GPU, Webots, model checkpoint, training run or network request after
dependencies are installed.

| Recipe | Saved inputs | What it recomputes |
| --- | --- | --- |
| [Detector qualification](../examples/qualification) | 8,955 frames | Matching and counts for the original detector qualification. |
| [Native view](controlled-native/README.md) | 6,144 frames | Controlled detector-development comparisons. |
| [Rotation diagnostic](controlled-rotation/README.md) | 512 frames | The saved rotation diagnostic. |
| [Motion and cadence](control-cadence/README.md) | 27,801 native samples from three solo runs | Pace, steering variation, body motion and the original wall-time tradeoff. |

Use Python 3.11 and install the pinned reproduction dependencies from the repo root:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test,reproduction]'
.venv/bin/python scripts/evaluate_cached_qualification.py --output runs/qualification.json
.venv/bin/python -I reproduction/controlled-native/reproduce.py --output runs/native.json
.venv/bin/python -I reproduction/controlled-rotation/reproduce.py --output runs/rotation.json
.venv/bin/python -I reproduction/control-cadence/reproduce.py --output-dir runs/control-cadence
```

Choose fresh output paths; the recipes preserve existing results. Each recipe
documents its checksums, metric definitions and limits. The motion recipe changes
physics and control cadence together, and retains opposing frequency-band and
compute results. A saved solo measurement does not qualify a controller in traffic.

For live experiments, see [setup](../docs/setup.md), the
[artifact catalog](../artifacts/README.md) and the
[engineering experiment guide](../experiments/README.md).
