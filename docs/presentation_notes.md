# Explaining the project

Working notes for Bradley Yong. The original 73-case study has an evidence
cutoff of November 14, 2026, 20:47 UTC. Reference qualifies among the tested
policies; the neural finalist is not promoted. The current course deck pairs
with the [112-second rear-grid chase](LiDAR_Racing_Demo.mp4). The earlier
[116-second study movie](demo.mp4) is a different recording.
Review the code and results yourself before presenting these explanations.

## A 30-second explanation

We built a simulated autonomous racing system that detects other cars from
measured LiDAR points and uses those observations to choose a lane and speed.
We adapted VoxelNeXt to our simulation, tested several training choices, and
checked whether better detection scores actually improved driving. The key
finding so far is that stronger offline detection does not guarantee better
driving: confidence thresholds, heading errors, stale tracks, and the driver
policy all matter. We preserve failed runs and compare fixed versions under
the same physical conditions.

## What does each part do?

1. Webots simulates the vehicle, track, traffic, and LiDAR measurements.
2. VoxelNeXt receives the measured point array and returns 3D car boxes.
3. Tracking and a separate geometric hazard channel summarize nearby traffic.
4. A small learned driver proposes lane and speed preferences.
5. A guard checks stopping gaps and occupied lanes; a geometric controller
   turns the selected target into steering, acceleration, and braking.
6. A separate supervisor records positions and physical incidents for scoring.

The vehicle knows its own GPS/IMU pose and the reference map. Opponent ground
truth is used for offline scoring. RGB is recorded for the viewer; it is not
the driving model's input. The point arrays use a placeholder intensity of one.

## Did the driver learn the track from scratch?

The track geometry comes from a reference derived from public position data.
The neural driver learns a traffic-response policy from teacher-generated
examples. A separate cross-entropy search completed two generations of six small policy calibrations using physical engineering rewards. Its 0.3754-second training improvement failed paired development validation, where it was 2.1868 seconds slower than identity. It therefore did not advance. It is not end-to-end learning
from camera pixels, and the circuit and tires are approximations rather than
an exact Formula 1 digital twin. Explain which parts are learned and which
parts are supplied by the map, teacher, or controller.

## Why VoxelNeXt, and what did we change?

VoxelNeXt supplies an existing sparse 3D detection architecture. We retain its
architecture and compare changes to the trainable scope, learning rates, and
training augmentation. The controlled study uses four fits with the same
initial checkpoint, training data, seed, 1,200 batches, and sampling sequence.
That isolates these training choices better than comparing unrelated runs.
One additional matched fit adds quarter-turn augmentation.

## What do the detection numbers mean?

Our metrics are custom simulation diagnostics. A visible target must have at
least three LiDAR returns in its box within the scoring region. Detections are
matched greedily by score using a two-metre XY centre-distance threshold.
Recall measures how many visible targets were matched. False positives per
frame, centre error, orientation error, and footprint overlap show different
failure modes. These results are not official nuScenes or A2RL average precision.

On the 6,144-frame development set at raw score 0.35, recall is approximately
52.0% for the original model, 76.9% after head-only fitting, and 95.8–96.3%
after the controlled full-network fits. Frames are correlated and the matched
training comparison has one seed; this does not establish generalization to
real racing data.

## Why do we show two confidence-threshold comparisons?

The qualified adapted pipeline uses a threshold chosen from training data.
Across 8,955 reserved qualification frames, the deployed pipeline comparison
shows recall increasing from 54.1% to 91.4% and false positives decreasing from
256 to 109. That comparison changes both the weights and the operating point.

At the same raw threshold of 0.425, the original model has only 43 false
positives, while the adapted model still has 109. Adaptation increases false
positives at this common threshold. Present both comparisons so a threshold
change is not mistaken for an effect caused only by learning new weights.

## Did augmentation help?

It helped some rotated-view footprint measurements, but hurt native-view
recall and directed heading. A box pointing 180 degrees the wrong way can
still overlap the target well. We therefore report directed heading,
heading modulo 180 degrees, and footprint overlap separately. The rotation
probe reuses already inspected development scenes; it is a diagnostic study,
not a new qualification set.

## What is the strongest negative result?

Switching to a detector pipeline that passed offline qualification exposed a
halt in the existing learned driver in a known left-blocked scenario. A frozen-state intervention
suggested sensitivity to an aged geometric fragment. That is a specific
diagnostic hypothesis, not proof that false detections caused the halt.

A revised driver then passed the known engineering gates, including that
scenario. It was nevertheless 3.15 seconds slower on average than its paired
predecessor in the matched 20-car run. A repair can improve robustness without
improving pace.

## How do we know a lap is real?

Cars must follow the ordered route and cross the common finish after the full
lap distance. The supervisor checks physical overlap and track violations,
and the audit joins saved sensor/control records to the simulated poses.
Cars park physically after finishing. There are no position resets or
teleports to complete a lap. A short capacity opener is reported separately
from a full lap.

## What can the hardware actually handle?

The Mac runs simulation and control; the LAN RTX PRO 6000 and RTX 3090 run
LiDAR inference. Twenty-car full laps have completed. Matched 60-second
openers sustained every requested stream for 20, 24, 32, 48 and 64 cars:

| Cars | Successful frames | Simulation/wall-time ratio | RPC p95 |
| ---: | ---: | ---: | ---: |
| 20 | 6,000 | 0.637 | 211 ms |
| 24 | 7,200 | 0.543 | 242 ms |
| 32 | 9,600 | 0.413 | 311 ms |
| 48 | 14,400 | 0.265 | 431 ms |
| 64 | 19,200 | 0.196 | 552 ms |

The sensor cadence is 5 Hz in simulated time. A ratio below one means that
simulation takes longer than real time. RPC timing includes transport,
serialization, queueing, and compute. No GPU samples were recovered for 20 or 24 cars. There are only two late samples for 32 cars, and partial polling windows for 48 and 64, so sampled maxima do not establish total peaks or headroom. Sixty-four cars completed a clean 60-second simulated opener, not a full lap or a global hardware maximum. Reference qualifies among the four frozen policies in the balanced final ranking, as explained below.

## Who won, and what did the neural policy improve?

All 73 required cases are verified. The original baseline retains 127/128 clean
laps, one DNF and three passed hazards. Fresh paired confirmation gives both
reference and Rich005 seed 7 128/128 clean laps and three passed hazards, but the
neural policy is 3.74% slower by paired median. It is not promoted.

Four balanced mixed-policy heats give every policy every grid position once.
All 80 entries finish clean, and all 24 fresh geometry cases pass. All four heat winners start at grid index 1, so heat wins alone do not isolate a policy effect. Reference
ranks first by mean position (8.75), ahead of Rich005 seed 17 (9.70), Rich006
seed 17 (11.75) and Rich005 seed 7 (11.80). It qualifies among these four tested
policies. Rich005 seed 7 wins the first heat; Rich005 seed 17 has the fastest
single finish. Explain why a single win, fastest lap and aggregate winner are
different statements.

The selected course presentation uses the 112-second rear-grid chase. Its
recorded P20-to-P1 result is an unequal-configuration engineering demonstration,
separate from the formal championship. The 116-second study movie is a
different recording. The source is public at
[github.com/j1s3ng/lidar-racing](https://github.com/j1s3ng/lidar-racing);
a YouTube or shareable Drive link is still needed for the course final video.

## What changed after instructor feedback?

The [BEV AP analysis](../reproduction/controlled-bev-ap/README.md) reports
score-ranked AP101 at IoU 0.5 on the known simulated development set: 81.557%
for the original detector and 97.671% for the best full fit. These use the
cached post-NMS score floor of 0.1; they are not official benchmark scores or
the center-matched recall numbers at cutoff 0.35.

CARLA 0.9.16 and the released camera-only LEAD environment were prepared for a
recognized closed-loop baseline, but startup attempts timed out before a camera
frame. There is no CARLA learned-policy rollout or benchmark score yet. The
updated [report](report.pdf) discusses both this limitation and a temporal
planning direction.

## What should the audience remember?

Explain the chain from detection to tracking to action. Show one controlled
comparison and one failure that changed our understanding. Keep engineering
tests, reserved qualification, final confirmation, and mixed-policy racing
distinct. Name the final winner only after the scheduled comparisons and
source audits are complete.

## Before presenting the delivered version

Use the current report and the 112-second movie paired with the course deck.
Keep the verified negative confirmation, retained baseline DNF, CEM development
failure and 64-car short-opener limitation. The qualified winner is Reference
among four tested policies, not a global claim. The editable course PowerPoint
embeds the 112-second movie; the PDF links to the same companion file. Test
playback and rehearse the talk before presenting.
