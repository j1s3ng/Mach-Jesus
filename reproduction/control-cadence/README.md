# Saved motion reproduction on CPU

Recompute three recorded one-car, clear-track outcomes at 20, 10 and 5 ms
integration/control steps. Every metric uses the original **50 Hz recording**.
This command reads the bundled measurements; it runs no simulator, detector,
neural inference, training, GPU operation or network request.

With Python 3.11 and the pinned NumPy/SciPy versions installed:

```sh
python -m pip install -r reproduction/control-cadence/requirements.txt
python reproduction/control-cadence/reproduce.py \
  --output-dir /tmp/control-cadence-result-001
```

Choose a fresh output directory. Existing results are never overwritten. The
command verifies checksums, schema, finite values, actual 50 Hz spacing, original
metric ASTs and frozen numeric results, then writes `result.json` and
`summary.csv`. No checkpoints, raw logs, Webots installation or Torch are needed.
The compressed measurements total about **1.86 MB**.

## What is reproduced

| Integration/control step | 20 ms | 10 ms | 5 ms |
|---|---:|---:|---:|
| Recorded native samples | 9,440 | 9,165 | 9,196 |
| First-reference crossing scalar, s | 188.787225711 | 183.282924162 | 183.903517100 |
| Original simulation wall time, s | 49.861995500 | 81.781050083 | 149.212492833 |
| Wall cost relative to 20 ms | 1 | 1.640147958 | 2.992509452 |
| Steering TV / recorded time, rad/s | .0740619513 | .0616944115 | .0597214981 |
| Pose-derived yaw rate above 1 Hz RMS, rad/s | .0278030734 | .0216114670 | .0220194557 |
| Yaw-rate 1–6 Hz integrated power, rad²/s² | .000601851110 | .000665697009 | .000682928510 |
| Steering 1–6 Hz integrated power, rad² | .0000539018150 | .0000698030817 | .0000689975567 |
| XY-derived lateral jerk RMS, m/s³ | 59.3810661 | 10.8744518 | 8.4187105 |
| Forward-difference yaw acceleration RMS, rad/s² | 1.78014034 | .447605097 | .443265169 |
| Planned map targets within 0.5 m | 18 / 18 | 18 / 18 | 18 / 18 |

All three saved episodes completed and passed their native safety checks. They
are **one case per condition**, with yaw damping OFF and planning budget 8. The
original vehicle, tyre and model settings remain declared in the source
evidence; this package exports no new trained model and promotes no controller.
Wall ratios describe those original runs, not the duration of this CPU command.

The tradeoff matters: the smaller steps lower TV and jerk, but increase yaw
1–6 Hz power and wall cost. The 5 ms lap is also slower than the 10 ms lap.
These observations do not establish an improvement in every frequency band,
convergence, traffic safety or measured tyre force.

## Timing is separate from sampled commands

The first-reference clock is a **scalar linear estimate between two actual
poses**, not an exact physical substep clock. The crossing brackets are:

| Step | Actual pose bracket, race seconds | Gap after last metric row |
|---|---|---:|
| 20 ms | [188.78, 188.80] | 7.226 ms |
| 10 ms | [183.28, 183.29] | 2.924 ms |
| 5 ms | [183.900, 183.905] | 3.517 ms |

A terminal pose can establish this bracket even when it falls between recording
timestamps. Its command and body row never enters the metric arrays. No samples
are appended, interpolated or upsampled for TV, derivatives, filters or Welch
power. Spatial endpoint shortfalls and the complete brackets are in the output.

## Method and limits

`metric.py` contains AST extracts from the frozen measurement evaluator and its
legacy steering metric. `provenance.json` records source/function AST hashes,
all three raw-trace/evaluator hashes, and source IDs without private paths. The
derivative/filter statements are copied unchanged; the small adapter supplies
the saved arrays.

The published yaw HF metric uses the gradient of unwrapped **pose yaw**. Raw
measured gyro HF is reported separately as
`additional_raw_gyro_above1hz_rms_radps`; it does not replace that metric. XY
jerk comes from XY gradients projected onto the body normal. HF uses the frozen
third-order 1 Hz Butterworth high-pass. Band power retains the frozen Welch
defaults, inclusive 1–6 Hz mask and trapezoidal integral.

Original float64 timestamps and their actual median interval are retained.
Forcing an exact 50 Hz floating-point frequency would subtly change boundary-bin
inclusion in the historical spectrum. Full frozen steering/body reports are
checked with relative and absolute tolerances of `1e-12`.

This is a **coupled cadence sensitivity**: integration and control change from
50 to 100 to 200 Hz, while recording stays at 50 Hz. The unchanged five-sample
acceleration median has nominal sample horizons of 100, 50 and 25 ms. It is not
an isolated physics-step experiment. Motion derivatives and software budgets
are not tyre-force measurements. No fair-model-tournament, generalization or
statistical-significance claim follows from these three clear cases.

`export_bundle.py` documents the path-normalized extraction. Re-export requires
explicit locally held CLOSED evidence and a fresh set of output files; that
private evidence specification is intentionally not shipped.

The focused repository checks are:

```sh
python -m pytest -q tests/test_control_cadence_reproduction.py
```
