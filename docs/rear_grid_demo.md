# Rear-grid chase

Historical media links below resolve to the [local archive catalog](archive.md). Source/evidence receipts retain their original artifact identities.

The [112-second demo](Mach_Jesus_Demo.mp4) and matching
[presentation](presentation.pdf) show a new engineering chase. Rich005 seed 7,
the winner of the earlier recorded heat, starts at P20 in a 20-car field.
Every car receives the same long race distance; the supervisor stops when the
hero holds the lead by total signed progress for one second.

| Recorded result | Value |
| --- | ---: |
| Start / stop position | P20 / P1 |
| Time at confirmed lead | 325.5 simulated seconds |
| Circuit position | Lap two; 1.4135 lap equivalents of signed progress |
| Active rivals passed in sampled progress | 19 |
| Lead at stop | 34.02 m |
| Peak speed | 206.85 km/h |
| Conservative overlap samples / hero track violations | 0 / 0 |
| Hero position resets after spawn | 0 |

All 20 cars remain active at the stop. This establishes a successful chase to
the lead; the nominal 20-lap race was not completed. The short video retains all
1,632 source camera frames and plays the 326.4-second recorded timeline at
3.264× between its introduction and results cards.

The original VoxelNeXt detector and learned driver weights are unchanged. The
hero's experimental configuration removes the blanket visibility pace cap and
recorded reference-speed constraint, narrows lateral uncertainty inflation,
commits to lanes faster, and adjusts following/reaction margins. Map/curvature,
braking and acceleration constraints remain. Actual power, grip and rival
configurations are unchanged. The finite planning ceiling is 100 m/s; measured
speed is determined by the vehicle and control constraints.

These changes were tested together, so the chase does not isolate the benefit
of any one component. A frozen-decision replay separately confirms that an
adjacent-lane fragment caused unnecessary braking in the earlier configuration.
The [source overrides and identities](../experiments/rear-grid-chase/README.md)
preserve the experimental implementation without replacing the study snapshot.

An earlier [literal no-lift test](archive.md#flat-out-test-mp4) forced full throttle and
zero braking. It left the reference corridor at 48.72 seconds. That failure is
retained; the successful chase allows lifting and braking. Neither demo changes
the [frozen study's](final_study.md) Reference qualification result or constitutes
new model training, official benchmark AP, or evidence of real F1 performance.

Video SHA-256:
`73b3f83b7ff3e5b69fc81c3b5f56bb27274ecf2db7cb755fa65c81c88380b7f4`.

Keep `Mach_Jesus_Demo.mp4` beside `presentation.pdf` when copying the PDF for its
local video link. The editable course presentation also embeds that exact movie.
