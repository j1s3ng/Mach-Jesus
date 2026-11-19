# Smooth line and passing on both sides

Historical media links below resolve to the [local archive catalog](archive.md). Source/evidence receipts retain their original artifact identities.

[112-second demo](archive.md#pass-first-demo-mp4) ·
[Full normal-speed video](archive.md#pass-first-normal-speed-mp4) ·
[Source and pinned manifest](../experiments/smooth-line-pass-first/README.md) ·
[Line comparison](../experiments/smooth-line-pass-first/analysis/racing_line_comparison.png)

The controller compares clear passing corridors on both sides before choosing
guarded following. It considers traffic alongside, closing rear traffic and
the space swept during a lane change. It uses a 2.5-second commitment timer and
retains the corridor until measured clearance when needed, while allowing a
guarded alternative if the corridor becomes blocked. Braking still handles
corners and blocked paths. A bounded, periodic nominal
line reduces map curvature while the original map remains the frame for
progress, rankings, traffic and learned inputs.

The new and previous engineering chases use the same 20-car roster, grid,
seed, learned weights, original VoxelNeXt detector, power, tires and physical
limits. R20 starts last; each chase stops after holding a lead of at least
2.5 m for one second. This is a chase to the lead, not a completed 20-lap race.

| Measured result | Previous chase | Smooth line + pass-first |
| --- | ---: | ---: |
| Time to sustained P1 | 325.50 s | 299.74 s |
| Active cars at closure | 20 | 20 |
| Observed left / right crossings | 11 / 8 | 13 / 6 |
| Conservative overlap samples, entire field | 0 | 0 |
| Track violations / resets, entire field | 0 / 0 | 0 / 0 |
| Hero maximum reference error, 6 m limit | 4.010 m | 4.483 m |
| Hero peak speed | 206.9 km/h | 188.0 km/h |
| Braking time, whole chase | 31.15% | 34.96% |
| Braking time, common 299.74 s horizon | 32.14% | 34.96% |
| Rapid opposite fixed-pass requests | 21 | 1 |
| Steering variation, common horizon | 0.04866 rad/s | 0.05296 rad/s |

Time to the lead improves by **25.76 seconds (7.91%)** in this matched run.
Both sides already worked in the previous controller; the change makes clear
passing an explicit preference and reduces rapid opposite-side requests.
Actual braking time, steering variation and maximum reference error increase.
This result therefore supports faster progress through this field, while it
does not establish less braking or globally smoother steering.

Overlap and track counters come from the director at 50 Hz. Controls are saved
at 10 Hz; braking time integrates samples with brake above 0.05 after common
start. Rapid reversals count opposite explicit outer passing corridors within
two seconds of the preceding outer-corridor decision, excluding varying nominal
offsets. Actual crossing sides are observed offline from hero-minus-rival
lateral position in the original map frame at sustained progress crossings;
these observations never enter the driver. All 20 controller results flushed
without failures. Their full-distance status remains incomplete because the
chase ends before the nominal lap target.

Preparation passed 20 controlled policy checks, five saved-state functional
checks and 11 line checks. The closed audit verifies 55 frozen source files,
all 20 runtime source/model/detector/reference identities, the frozen plan,
unchanged rival settings and matched physics. The videos fully decode and
retain all 1,503 source camera frames: the short version is 112 seconds with
a 3.006x race segment; normal speed is 300.6 seconds at the native 5 fps.
The green ribbon in the scene is the original guide; gold on the map is the
new nominal plan. Planning-side badges describe intent; the separate crossing
counts describe observed progress crossings.

The line planner alone estimates a 3.70 s flying-lap gain and 14.40% lower
curvature energy on the approximate map. Those calculations are separate from
the measured chase result. This single combined-feature experiment does not
isolate line versus passing effects or establish performance across seeds.
No model was retrained, and the frozen study and presentation remain separate.

SHA-256:

- Short video: `c638a04fff59495ad77a19cb53826e826239c7380896931786207e5e8e0fe4a1`
- Normal speed: `3f7d403f9a6011f7f850fe5b99e330aa41c1088f39dcd6e13a41a7d59910738f`
- Closed audit: `d698fba90f1da369e33ac8b58467cd9a5c3cddcc7556e78a62d7f1fb82ebcf67`
- Media provenance: `9989941f8fc124eb9200af0fa6c68567b8ebbd7955e3eaac0df4e90c89c65d98`

The portable source manifest pins the supporting evidence; model weights,
binary map/plan arrays, raw logs and machine-specific service settings are
external artifacts.
