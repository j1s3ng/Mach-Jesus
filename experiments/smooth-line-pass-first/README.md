# Smooth line and pass-first overlay

**Status: CLOSED_ENGINEERING_CHASE_REVIEWED.** With the seven exact Python
runtime overrides here, R20 started P20 and reached sustained P1 at **299.74 s**,
compared with **325.50 s** for the prior 004 demo: **25.76 s (7.91%) earlier**.
All 20 entrants were active at closure. These changes are prepared locally
for review; publication is pending.

This is one matched engineering chase evaluating the combined
smooth-line and passing changes. It does not isolate component effects,
establish a measured lap-time gain or multi-seed improvement, or complete the
nominal 20-lap race. No new training or formal race-winner claim is made.

The prior 004 demo already passed on both sides: **11 observed left passes and
8 right passes**. This candidate adds a bounded smooth nominal racing line,
clear-corridor pass-first selection and stronger lane commitment. The policy
considers both absolute passing sides before settling into guarded following.
Rear/alongside conflicts, stopping checks and physical corner limits still
apply. Lane commitment and a minimum switch gain reduce repeated side changes.

The approximate simulator map remains the frame for lap progress, finish line,
ranking, traffic and learned features. Nominal steering looks ahead along the
new map-only line; passing uses absolute offsets in the original frame. The
current and destination curvature profiles both limit wide lane transitions.
Vehicle power, tires, grip, braking and acceleration are unchanged. Original
Rich005 seed-7 and VoxelNeXt weights are reused; no model was retrained.

## Files and diagnostics

`source/` preserves the seven module paths for a source overlay; it is not a
standalone simulator package. `manifest.json` pins every module, the original
models/reference, the external frozen plan and the relevant settings.

[PNG comparison](analysis/racing_line_comparison.png) ·
[PDF comparison](analysis/racing_line_comparison.pdf) ·
[Map-only diagnostics](analysis/map_only_report.json)

On the same map and the same 8/8/5 m/s² physical profile, curvature energy falls
14.40% and the model flying-lap estimate changes from 170.531 to 166.828 s.
**These are map/profile calculations, not measured simulator results.** The
bounded least-squares solver reached local `ftol` convergence at 173 evaluations
within a budget of 300; no global optimum is claimed. The nominal line stays
within ±2.8 m, and absolute passing targets within ±4 m. The finite software
planning ceiling is 100 m/s; actual vehicle performance remains physical.

Validation completed during preparation is recorded separately in the manifest:

- 20 controlled policy checks on synthetic scenes.
- 5 functional checks using saved 004 states and the actual frozen map plan.
- 11 line checks: 10 synthetic/loader checks and 1 using the actual reference.
- Independent integration review of the exact runtime source and settings.

The closed physical simulation audit additionally verified all frozen source,
model, detector and reference identities, identical shared physics/world/grid
settings, 19 unchanged rival configurations and no runtime line optimization.
The prior formal study and its results remain separate from this engineering
chase.

## Observed closed-chase comparison

| Observed measure | Prior 004 | Smooth line + pass-first |
| --- | ---: | ---: |
| Time to sustained lead, s | 325.50 | 299.74 |
| Active entrants at closure | 20/20 | 20/20 |
| Observed left / right progress crossings | 11 / 8 | 13 / 6 |
| All-car audited overlap samples, 50 Hz | 0 | 0 |
| All-car track violations / pose resets | 0 / 0 | 0 / 0 |
| Hero maximum reference error, m | 4.010 | 4.483 |
| Hero peak measured speed, km/h | 206.9 | 188.0 |
| Rapid opposite fixed-pass requests, whole chase | 21 | 1 |
| Braking time, whole chase | 31.15% | 34.96% |
| Braking time, common 299.74 s horizon | 32.14% | 34.96% |
| Steering total variation per second, common horizon, rad/s | 0.04866 | 0.05296 |

All 20 cars had zero audited overlaps, track violations and pose resets. The
hero's larger 4.483 m maximum reference error remained below the 6 m corridor
limit. Faster time to the lead and fewer rapid fixed-pass reversals came with
more braking time, slightly higher common-horizon steering variation, larger
tracking excursions and a lower peak speed. The result does not establish less
actual braking or universally smoother steering.

Rapid reversals use the same definition in both runs: opposite explicit fixed
outer passing corridors within 2 s of the preceding outer-corridor decision;
same-side decisions refresh the timer. Center, HOLD and varying nominal-line
offsets are excluded. The prior corridors were ±3.2 m and the new ones ±4 m.
The common-horizon reversal counts are 17 and 1, respectively. Braking fractions
integrate saved 10 Hz controls with brake command above 0.05. Overlap and
reference-error audits use the director at 50 Hz; track/reset results are also
checked against all 20 flushed controllers. Crossing sides are offline
original-map Frenet measurements and were never driver inputs; a crossing
count alone is not proof of a contact-free manoeuvre.

The pinned closed-audit JSON SHA-256 is
`d698fba90f1da369e33ac8b58467cd9a5c3cddcc7556e78a62d7f1fb82ebcf67`.
Sanitized observations, audit identities and measurement definitions are
recorded in `manifest.json`; private raw logs are excluded.

## Apply the source overlay

Start with a separate working copy of the Mach-Jesus repository, which already
contains the complete base-study source export. Copy the contents of this
experiment's `source/` folder onto that copy's repository root, preserving the
relative paths. From the eventual experiment location
`experiments/smooth-line-pass-first/`, this overlays `source/src/...` onto
`src/...` and `source/webots/...` onto `webots/...` in the separate copy.

The historical upstream provenance commit is
`ca985876a6539cf4630091d72554a91501ae3eef`; it identifies the original export's
source history and is not necessarily a publicly available Git ref. Checking
out that commit is not required. The external reference, models, Webots
vehicle/world assets and full race configuration are still required. This
experiment does not include those assets, binary NPZ files, model weights, raw
logs or machine-specific configuration.

For exact candidate execution, provide the external frozen plan with SHA-256
`b46bd54fdb842b33e599477fed58aaed76e4f8b7e059d03b43ae5ce02652a785`.
The loader verifies that hash, the exact original geometry, physical limits and
planner-source bytes. It loads the saved profiles without running optimization.
Set the hero's `racing_line_plan` to that artifact and
`racing_line_plan_sha256` to the same hash. Use the hero/guard/line settings in
`manifest.json`; retain the 19 rival configurations.

## Regenerate a plan from the external reference

Regeneration is an alternative map-only build step. It creates a new external
artifact; numerical-library versions and NPZ container metadata may change its
bytes. It is not a promise to recreate the frozen artifact hash. Record and use
the printed hash for a regenerated experiment, then repeat validation before
comparing its outcome with the frozen candidate.

From the complete base project with these overrides applied, use an environment
with NumPy and SciPy and provide paths for the original reference and output:

```sh
export RACING_REFERENCE_PATH=/path/to/reference.npz
export RACING_PLAN_OUTPUT=artifacts/map_line_plan.npz
PYTHONPATH=src python - <<'PYCODE'
from pathlib import Path
import hashlib, json, os
from race_lidar.hotlap_driver import RacingReference
from race_lidar.smooth_racing_line import SmoothRacingLine

reference_path = Path(os.environ["RACING_REFERENCE_PATH"])
expected_reference = "cfdea80ff8fb7f1e221d7eb4f9b2b8d4a18f8191c7eddc2e60f56d988e02df3e"
if hashlib.sha256(reference_path.read_bytes()).hexdigest() != expected_reference:
    raise ValueError("Use the original reference pinned in manifest.json")
reference = RacingReference.from_npz(
    reference_path, ignore_source_speed=True, max_speed=100.0,
    lateral_accel=8.0, brake_decel=8.0, acceleration=5.0,
)
line = SmoothRacingLine(
    reference, offset_limit_m=2.8, passing_limit_m=4.0,
    optimize=True, max_nfev=300, smoothing_distances_m=(6.0, 12.0, 20.0),
    offset_weight=1e-6, slope_weight=0.002, bend_weight=0.2,
    nominal_lateral_rate_mps=1.4,
)
output = Path(os.environ["RACING_PLAN_OUTPUT"])
output.parent.mkdir(parents=True, exist_ok=True)
plan_hash = line.save_plan(output)
loaded = SmoothRacingLine.from_npz(reference, output, expected_sha256=plan_hash)
assert not loaded.optimization_executed_this_instance
output.with_suffix(".report.json").write_text(json.dumps(line.report(), indent=2) + "\n")
print(json.dumps({"plan_sha256": plan_hash, "runtime_optimizer_required": False}))
PYCODE
```

Do not use a re-optimized path as the progress/ranking reference. Keep the
original reference and the separately loaded line in their existing roles.
