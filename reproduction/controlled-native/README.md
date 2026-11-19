# Cached native-view controlled comparison

This folder reproduces numerical results from 6,144 frozen known-development
frames with 8,155 visible-target labels and six cached detector outputs per
frame: the original model, four controlled fits, and the quarter-turn augmented
fit. There are **36,864** prediction rows. No raw clouds, weights, simulator,
server connection details, or private absolute paths are included.

Using Python 3.11 or 3.12 and the tested dependencies:

```sh
python -m pip install -r requirements.txt
python -I reproduce.py --output reproduced.json
python -I test_reproduction.py
```

The reproduction runs offline after moving this folder, with no original
repository, checkpoint, or data paths. Choose a fresh output name each time.
`verified_output.json` and `verification.json` retain the supplied check. Only
the dependency-install command needs package access; scoring is CPU-only and
offline. Both fixed RAW cutoffs **0.35 and 0.425** are always reported.

The export retains only predicted boxes scoring at least 0.35, with their exact
numeric values and source ordering. `provenance.json` records each original
cache's model postprocessing floor, writer-floor field, and source hashes; each
effective source floor was checked to be at most 0.35 before export. Scores
below the **exported floor of 0.35 cannot be recovered**. Empty cached output
is distinct from missing output, and all six models must cover every frame.

The original archived comparisons are checked separately from newly computed
descriptive summaries:

- At 0.35, all original/four-fit metrics, original-paired center summaries, and
  the **3,953** common targets matched by the original plus four fits reproduce
  the main controlled report.
- The mixed/Aug pair reproduces all archived metrics and every paired geometry
  field at 0.35 and 0.425, using **7,671** and **7,431** common targets respectively.
- Original/head/uniform-high/uniform-low summaries at 0.425 were absent from the
  old native report's operating-point list. These four are explicitly new
  descriptive recomputations of the existing cache, not historical parity.
- The original-plus-four-fit intersection and the added-Aug intersection have
  separately named IDs, counts, and center summaries at each cutoff. Adding Aug
  never silently shrinks the original comparison cohort.

`metric.py` contains the original numeric functions verbatim; source/function
hashes are recorded. The rule is score-descending greedy nearest-unmatched
**2 m XY**, with original GT indices resolving equal-distance ties. Geometry
uses paired GT identities and reports center/dimension errors, directed and
π-symmetric heading, heading flips over 90°, BEV IoU, and yaw-upright 3D IoU.
These are custom diagnostics, not official AP or mesh/roll/pitch IoU.

Counts, frame IDs, and cohort IDs must match exactly. Floating fields use
1e-12 relative/absolute tolerance. The original main report iterated a Python
set when averaging its fixed-cohort centers; this reproduction sorts those
IDs for deterministic sums. A possible last-bit mean difference is reported
explicitly. All mixed/Aug archived metric and geometry fields are checked;
the verification records exact-value parity and the largest numeric difference.

The five fits retain their recorded actual checkpoint and numerical tensor-state
hashes. The original endpoint recorded checkpoint/config hashes but no numerical
tensor digest; its tensor digest remains null. No weights are reopened.

Labels are frozen ego-upright physical-asset box sidecars from at least three
measured returns within the original 50 m scoring population. Without raw
clouds this bundle cannot recompute visibility or observability. These are
correlated, previously inspected development measurements: no new fitting,
threshold/model selection, fresh qualification, racing claim, or independent
generalization follows from this reproduction.
