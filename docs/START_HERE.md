# Start here: understanding LiDAR Racing

This guide explains the project before you read the detailed report or code.
Start with the question **“Do better boxes make a better racer?”** The project
evaluates an existing 3D car detector and the simulated driving system that
uses it. The central result is that improved offline detection did not
automatically improve the complete driving loop.

## Follow one observation

1. **Sense:** Webots produces a simulated LiDAR point cloud for each car.
2. **Detect:** VoxelNeXt predicts car boxes and confidence scores from the points.
3. **Track:** measured boxes and a separate raw-point geometry channel summarize
   traffic and uncertainty over time.
4. **Propose:** a supervised neural driver uses structured ego, route and traffic
   features to request a lane and speed.
5. **Execute:** conventional traffic guards check the request; a controller
   commands steering, throttle and braking. The vehicle moves and the next
   LiDAR observation changes.

That feedback makes this a *closed loop*. Scoring a fixed set of saved clouds
is *offline evaluation*. A box can be good on a saved frame yet lead to a poor
decision once it changes the car's trajectory and future observations.

The detector and driver are learned components. Own-car pose and the approximate
reference map are supplied; traffic guards and low-level control are engineered.
Opponent simulator truth is used for offline labels, scoring and labeled replay,
not for driving control. RGB is viewer footage, not input to this racing policy.
[Architecture and source map](architecture.md)

## Three results to keep separate

**Detector development.** At a common score cutoff of 0.35, custom
center-distance recall increased from **51.98%** for the original detector to
**96.27%** for one full-network fit. On the same 6,144 known simulated
development frames, the later BEV IoU 0.5 AP101 score sweep gives **81.56%** and
**97.67%**, respectively. AP uses cached post-NMS outputs down to score 0.1;
the two metrics have different matching and threshold rules. These are
single-seed development observations, not official benchmark or real-track
results. [AP definitions](../reproduction/controlled-bev-ap/PROTOCOL.md)

**The driver halt.** In one matched stopped-car case, the original detector
pipeline passed and the adapted pipeline exposed a halt with the same driver and
guard. On one saved state, removing an aged geometric fragment changed the
driver's lane choice. That is a useful sensitivity hypothesis, not proof that
false detections caused the complete failure. [Report](report.pdf)

**Final driver comparison.** In paired confirmation, Reference and the frozen
neural finalist both completed 128/128 clean laps, but the neural finalist was
**3.74% slower** by the median paired time ratio and was not promoted. Reference
ranked first among four tested policies in balanced 20-car heats. Those final
races used the original detector, so they must not be attributed to the adapted
detector from the halt case. [Complete results](final_study.md)

## The videos demonstrate different things

| Recording | What it shows | Limit |
| --- | --- | --- |
| [Original 112-second chase](LiDAR_Racing_Demo.mp4) | P20 to sustained P1 at 325.5 simulated race seconds; it matches the [shortened course deck](presentation.pptx). | Edited and accelerated; hero and rivals use 100 and 60 m/s planning ceilings. It is not a fair policy comparison or a completed 20-lap race. |
| [Later 121.44-second highlight](LiDAR_Racing_Qualified_Highlight.mp4) | A different two-circuit engineering chase, with observed P1 at 85.4 simulated seconds. | Hero and rivals have different software limits/policies; the clip exceeds the course's two-minute submission limit. |
| [Later full flying lap](LiDAR_Racing_Full_Flying_Lap_1x.mp4) | The 172.44-second native-speed video enclosing a 172.390-second crossing-to-crossing circuit. | Same unequal-condition engineering demonstration, not a fair model ranking. |

The original short movie is a GitHub reference. The final course submission
requires a separately accessible YouTube or Google Drive video link.

## What can I run?

For a quick CPU check, follow [the commands in the README](../README.md#reproduce-the-saved-results).
The cached recipes recompute saved metrics; the synthetic demo uses geometric
clustering. Neither runs VoxelNeXt or a physical race. Full inference and
Webots races need external weights, reference data and compatible hardware;
[setup](setup.md) and the [artifact catalog](../artifacts/README.md) list those
requirements.

The original study ran 20-car full laps. Larger 20–64-car tests were
60-**simulated**-second openers, and the simulator ran slower than wall time.
The 5 Hz LiDAR rate refers to simulation time, not a guaranteed real-time
service rate. [Capacity details](final_study.md#measured-capacity-scope)

## What remains open?

The dataset and experiments use one approximate course, one simulated vehicle,
correlated frames and one matched detector-fitting seed. The A2RL dataset
remained gated; it was not used. CARLA 0.9.16 and a released camera-only LEAD
checkpoint were set up for exploration, but startup tests timed out before a
camera frame. No learned-policy CARLA rollout or standard CARLA score has been
obtained. The [report source and build guide](report-source/README.md) documents
the follow-up and its limitations.

For navigation, use the [documentation index](README.md),
[reproduction index](../reproduction/README.md), [training guide](../training/README.md),
[experiment index](../experiments/README.md) and
[source provenance](provenance/README.md).
