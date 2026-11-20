# LiDAR Racing

**Do better car detections make a better autonomous racer?** This CMPE249
project by Bradley Yong connects a LiDAR detector to a simulated racing system
and evaluates both detection and driving. It was developed with AI coding assistance.

The main finding is that better offline detection did not automatically produce
better driving. A detector change exposed a downstream driver halt, and the
conventional Reference policy remained ahead in the final comparison.

Start with the [plain-language guide](docs/START_HERE.md), the
[eight-page report](docs/report.pdf), or the
[course presentation (PowerPoint)](docs/presentation.pptx), its [PDF](docs/presentation.pdf), and the
[112-second demo](docs/LiDAR_Racing_Demo.mp4). The GitHub demo is an interim
reference; the course submission still requires an accessible YouTube or Google
Drive link.

## How the system works

```text
Webots LiDAR points → VoxelNeXt car boxes → tracks and geometric hazards
                   → neural lane/speed proposal → guard and controller
                   → steering and pedals → simulated vehicle → next observation
```

Webots runs vehicle physics on a Mac; NVIDIA GPUs run detector inference. A
supervised neural driver proposes a lane and speed, while conventional guards
and control decide how to execute them. The car receives its own localization
and an approximate reference map. Opponent ground truth is reserved for labels,
offline scoring and labeled replay; it is not a driving input. RGB is recorded
for viewing. See the [architecture](docs/architecture.md) for the full data flow.

## What the evidence says

| Question | Result and scope |
| --- | --- |
| Did detector fitting help? | On 6,144 known simulated development frames, custom center-match recall at score 0.35 rose from **51.98%** for the original model to **96.27%** for one full-model fit. The matched training comparison uses one seed. [Study](docs/final_study.md) |
| What does a score sweep show? | On the same known development population, BEV IoU 0.5 AP101 is **81.56%** for the original and **97.67%** for that full fit. Outputs use a common post-NMS score floor of 0.1. These are fixed-postprocessing development results, not official benchmark AP. [AP protocol and reproduction](reproduction/controlled-bev-ap/README.md) |
| Did it improve driving? | In a matched stopped-car case, the adapted detector pipeline exposed a driver halt. A frozen-state probe suggests sensitivity to an aged geometric fragment; it does not prove the cause of the entire trajectory. [Report](docs/report.pdf) |
| Which driver ranked first? | In the **73-case study**, the neural finalist and Reference each completed 128/128 clean paired-confirmation laps, but the neural finalist was **3.74% slower** by paired median. Reference ranked first among four tested policies across balanced heats. The final races used the **original detector**. [Study](docs/final_study.md) |

## Which video should I watch?

The [original 112-second demo](docs/LiDAR_Racing_Demo.mp4) matches the
[course presentation](docs/presentation.pptx), which ends after slide 10 with a thank-you slide. It shows an engineering chase from
P20 to sustained P1 at **325.5 simulated race seconds**. Its footage is edited
and accelerated; the video duration is not the lap time. The hero uses a
100 m/s planning ceiling versus 60 m/s for the rivals, so this is an
unequal-configuration demonstration. [Chase details](docs/rear_grid_demo.md)

The later [121.44-second engineering highlight](docs/LiDAR_Racing_Qualified_Highlight.mp4)
and [complete flying lap](docs/LiDAR_Racing_Full_Flying_Lap_1x.mp4) show a
different recording. The highlight is longer than the course's two-minute limit.
In this later chase, the hero has a 100 m/s software limit versus 60 m/s for
base rivals, with different policies. It is a demonstration, not a fair policy
ranking. See the [engineering results and limits](experiments/local-optimization/README.md),
[updated presentation](docs/Presentation_Update.pdf) and
[timed cues](docs/Presentation_Update_Notes.md).

## Reproduce the saved results

The CPU recipes operate on saved numeric predictions or measurements. They do
not run detector inference, train a model, or reproduce Webots physics. Use
Python 3.11 from the repository root:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test,reproduction]'
.venv/bin/python -m race_lidar doctor
.venv/bin/python -m pytest -q
.venv/bin/python -m race_lidar demo --output runs/cpu-demo --frames 20
.venv/bin/python scripts/evaluate_cached_qualification.py --output runs/qualification.json
.venv/bin/python -I reproduction/controlled-native/reproduce.py --output runs/native.json
.venv/bin/python reproduction/controlled-bev-ap/reproduce.py --output runs/bev-ap.json
.venv/bin/python -I reproduction/controlled-rotation/reproduce.py --output runs/rotation.json
.venv/bin/python -I reproduction/control-cadence/reproduce.py --output-dir runs/control-cadence
```

Use fresh output paths; the recipes preserve existing results. The synthetic
demo uses geometric clustering. The [reproduction index](reproduction/README.md)
explains each saved input, and [verification](docs/verification.md) describes
the CPU checks and CI.

Live inference, detector fitting and physical races require external weights,
reference data, Webots and compatible CUDA hardware. These assets are not
bundled here. See [setup](docs/setup.md) and the
[artifact catalog](artifacts/README.md).

## Where to look next

| Location | Contents |
| --- | --- |
| [docs](docs/README.md) | Report, presentations, study protocol and evidence. |
| [src/race_lidar](src/race_lidar) and [webots](webots) | Base-study perception, driving, control and simulation code. |
| [scripts](scripts/README.md) and [training](training/README.md) | Commands, data preparation, training and evaluation. |
| [tests](tests), [reproduction](reproduction/README.md) and [examples](examples) | Regression checks, saved evidence and fixtures. |
| [experiments](experiments/README.md) | Later engineering changes, source overlays and negative results. |
| [research](research/rich006/README.md) | Historical teacher and fitter source for audit. |

The installed base source and later experimental overlays are distinct. Use the
[apply-order recipe](experiments/local-optimization/APPLY_OVERLAY.md) to
reconstruct the latest version; the [local archive catalog](docs/archive.md)
lists superseded media. The [original study demo](docs/demo.mp4) is another
historical recording.

CARLA 0.9.16 and a released camera-based LEAD checkpoint have been prepared for
follow-up. Bounded startup tests did not produce a camera frame; there is **no
learned-policy CARLA rollout or score yet**. This is an approximate simulation
study with supplied pose/map and custom scoring, not a calibrated Formula 1 or
real-world safety result.

See [contributions](CONTRIBUTIONS.md), [source provenance](docs/provenance/README.md),
[third-party notices](NOTICE.md) and the project [MIT license](LICENSE).
