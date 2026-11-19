# Cached controlled rotation reproduction

This folder reproduces the numerical rotation diagnostic from cached outputs.
It contains 512 frozen ground-truth frame rows (128 known development frames at
−90°, 0°, +90°, and 180°) and 3,072 prediction rows: the original detector and
five actual fitted models, each on all 512 inputs. It includes no point clouds,
model weights, simulator, private paths, or server connection information.

With Python 3.11 or 3.12 and the tested dependencies installed:

```sh
python -m pip install -r requirements.txt
python -I reproduce.py --output reproduced.json
python -I test_reproduction.py
```

The reproduction and tests work after copying this folder elsewhere. They do
not require the original repository, checkpoints, data directories, or network.
Choose a fresh output name on each run; existing output is preserved.
`verified_output.json` and `verification.json` record the supplied reproduction.
The pip command installs dependencies; the reproduction itself is offline.

The scorer checks every bundled checksum and requires all six models at every
frame, without duplicates. The fixed cutoff is RAW **0.425**. Lower thresholds
cannot be recovered from this exported cache. Predictions absent below that
floor are not reconstructed as detections. An empty box list is an actually
cached empty output, not a missing inference.

`expected_reports.json` retains all 24 model-by-angle reports from the saved
`result_scope_corrected_v2.json`, including TP, FP, recall, empty-frame FP,
paired-angle geometry, and all three fixed all-angle geometry summaries.
`expected_cohorts.json` gives exact scene/source-frame/target-index identities:
102 targets shared by the four core fits at every angle, 41 when adding the
original model, and a separately named 41-target intersection also including
the augmented fit. These are conditional common-target cohorts; they do not
replace the recall population of 174 targets at each angle.

Matching is the existing score-descending greedy nearest-unmatched **2 m XY**
rule; equal-distance ties retain the lower original GT index. `metric.py`
contains the original matching and geometry functions verbatim, with original
source and function hashes in `provenance.json`. Geometry reports center and
dimension errors, directed and π-symmetric heading errors, heading flips over
90°, BEV IoU, and yaw-upright 3D IoU. It does not use mesh, roll, or pitch IoU.
Counts, IDs, and field structure must match exactly; floating values are checked
at 1e-12 relative/absolute tolerance. On the tested environment, every saved
numeric value is exactly equal, including all geometry fields.

The five fitted models retain their actual checkpoint and loaded numerical
tensor-state hashes. The original cached endpoint recorded checkpoint/config
hashes but no tensor-state digest; that digest remains explicitly null. No
weights were reopened for this export. Provenance names are relative names of
the retained original evidence, not dependencies needed to run this folder.

The labels are **frozen visible-target sidecars**, created from measured LiDAR
using at least three box returns and a 50 m evaluation-radius filter before the
quarter-turn transforms. This package cannot recompute observability without
the original clouds. These correlated, previously inspected development frames
are a diagnostic reproduction, not fresh qualification, official A2RL/nuScenes
AP, independent generalization, a new detector selection, or a causal shortcut
test. There is no fitting, GPU inference, threshold search, or racer selection.
