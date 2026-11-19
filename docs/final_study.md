# Source-verified study results

The complete 73-case study is independently source verified as of November 14,
2026, 20:47 UTC. This source copy includes compact cached reproductions and the
report/demo; full model, dataset and raw-evidence downloads are not hosted.
Reference qualifies among four tested hybrid policies. The neural finalist is
not promoted. The actual 116-second movie has passed full decode and source/frame verification.
The 311-second normal-speed race is retained; the historical 75-second movie
remains a separate recording.

VoxelNeXt detects cars from measured simulated LiDAR. A supervised neural driver
proposes speed and lane preferences using tracks and geometric hazard summaries.
A guard and conventional controller execute them. Own pose and the approximate
reference map are supplied. Opponent truth provides offline labels and scoring;
RGB is recorded footage. No real A2RL data, official AP, end-to-end RGB policy,
calibrated F1 vehicle model, or real-world safety claim is made.

## Completed detector comparisons

Four seed-7 fits share original initialization, 10,233 TRAIN clouds, mining,
batch order, 1,200 batch-2 updates and evaluation. On 6,144 development frames
with 8,155 visible targets at RAW 0.35:

| Model | Recall | False positives |
| --- | ---: | ---: |
| Original | 51.98% | 96 |
| Head only, LR 1e-4 | 76.87% | 33 |
| Full model, uniform LR 1e-4 | 95.77% | 51 |
| Full model, uniform LR 1e-5 | 96.27% | 75 |
| Full model, head/backbone LR 1e-4 / 1e-5 | 95.76% | 81 |

A separate matched quarter-turn TRAIN intervention worsens native-view recall
and directed heading. At 0.35, recall changes from 95.76% to 94.67% and FP from
81 to 105. Known rotated footprints can improve while native heading worsens.
The rotation probe is design-informed development evidence with correlated
rigid copies, not a fresh qualification or generalization claim.

An earlier frozen pipeline passes two new qualification groups, totaling 8,955
frames and 20,536 visible targets. Original RAW 0.35 has recall 54.07% and 256 FP;
adapted RAW 0.425 has recall 91.39% and 109 FP. The adapted cutoff was chosen
from TRAIN only. At the common RAW cutoff 0.425, original FP is 43 and adapted FP
is 109. Different-cutoff gains cannot be attributed to weights alone. Earlier
rejected models and unsupported point-filter hypotheses remain failures.

## Completed driving and bounded search evidence

The offline-qualified pipeline exposes a known driver halt. A fitted revision
passes the known hazard regressions but is 3.150 seconds slower in a paired
20-car original-perception engineering comparison. Both homogeneous fields
finish 20/20 clean; their means are 280.069 and 283.219 seconds. This is a
combined driver/teacher revision comparison, not a single-feature ablation or
fresh-layout generalization proof.

Bounded CEM searches affine outputs of the supervised Rich005 seed-17 driver.
Each proposal receives four hazards and a four-car fitness race only if the
hazards pass. Two generations of six proposals all pass, giving 60 fitness
cases. A separate paired development validation adds two races. The best
training mean improves from 246.514 to 246.139 seconds, a 0.3754-second gain.
Paired development validation changes 245.828 to 248.014 seconds, a gain of
−2.1868 seconds. It misses the required +1-second improvement and does not
advance to larger-field eligibility. No formal outcome guides this search.

## Complete final evaluation

| Suite | Cases | Verified outcome |
| --- | ---: | --- |
| Reference baseline | 15 | 127/128 clean laps; one retained 20-car DNF; 3/3 hazards |
| Paired confirmation | 30 | Each arm 128/128 clean laps and 3/3 hazards; neural finalist not promoted |
| Balanced championship | 4 heats | 80/80 valid and clean entries; Reference qualifies among tested policies |
| Fresh geometry | 24 | 24/24 passed; six cases for every frozen policy |

The original baseline strata contain 16/16, 32/32 and 79/80 clean laps. Their
valid-lap means are 242.747, 249.438 and 263.346 seconds. The last mean excludes
the DNF. In fresh confirmation, reference and the frozen Rich005 seed-7
finalist both complete 128/128 clean laps. Their 4/8/20-car means are
242.352/249.238/264.365 and 246.894/255.896/277.416 seconds. The median ratio over
128 corresponding clean pairs is 1.037373: the neural finalist is 3.74% slower
and has no verified strict improvement. Reference remains incumbent.

Every policy occupies every starting grid position once across four mixed
20-car heats. All policies complete 20/20 clean entries and win one heat. All four heat winners start at grid index 1, so heat wins alone do not isolate a policy effect.
Ranking compares clean completion, completion, mean position and then mean
valid time. Since completion rates tie, mean position decides this ranking:

| Rank | Policy | Clean | Mean position | Mean valid lap, s |
| ---: | --- | ---: | ---: | ---: |
| 1 | Reference | 20/20 | 8.75 | 268.808 |
| 2 | Rich005 seed 17 | 20/20 | 9.70 | 273.441 |
| 3 | Rich006 seed 17 | 20/20 | 11.75 | 280.401 |
| 4 | Rich005 seed 7 | 20/20 | 11.80 | 280.027 |

Rich005 seed 7 wins the first heat. Rich005 seed 17 has the fastest individual
finish, 236.124 seconds in heat 2. Neither determines the aggregate winner.
Reference also passes its six fresh geometry cases and is qualified among
these four tested policies. Every other frozen policy passes its six cases
too. This is not a universal best-racer or real-world transfer claim.

All final races use the original detector and the same common physics,
reference, runtime and 70 m driver horizon. Historical calibrated-detector
qualification is a separate pipeline comparison. Completeness, formal
promotion and qualified championship release remain separate fields:
`true`, `false` and `true`, respectively. An initial final-audit roster-path
spelling mismatch is repaired without changing scoring or outcomes; the failed
audit and old source remain archived.

## Measured capacity scope

All five clean 60-second simulated openers supply 300 frames per car and zero
client inference failures. The total is 56,400 verified frame/action pairs.

| Cars | Frames | Simulated/wall-time ratio | RPC p95 / p99, ms |
| ---: | ---: | ---: | ---: |
| 20 | 6,000 | 0.637 | 210.9 / 217.7 |
| 24 | 7,200 | 0.543 | 242.1 / 249.8 |
| 32 | 9,600 | 0.413 | 310.7 / 319.7 |
| 48 | 14,400 | 0.265 | 430.9 / 448.9 |
| 64 | 19,200 | 0.196 | 551.8 / 584.3 |

Streams are split equally across PRO 6000 and RTX 3090. The sensor cadence is
5 Hz per simulated second. RPC includes transport, serialization, queueing and
model work. These are not wall-time 5 Hz, 64-car full laps or a global hardware
maximum. Independent CPU work overlaps the 20/24/32 cases; deep analysis is
deferred during 48/64. GPU polling covers 0/0/2/42/60 samples respectively,
with partial windows for the larger fields. Sampled maxima are not total peaks
or hardware headroom.

The portable caches and [artifact catalog](../artifacts/downloads.json) support
reproduction within their stated scope. Full clouds and original private
provenance are separate verified local companions. See
[the provenance summary](final_artifact_provenance.json) for source hashes and
separate completion, promotion, championship and media fields. No hosted download or publication is claimed.
