# Mach-Jesus — final 10-minute presentation cues

**Bradley Yong · CMPE249 · San José State University**

Canonical repository: [github.com/j1s3ng/Mach-Jesus](https://github.com/j1s3ng/Mach-Jesus).

**Planned slot: 10:00 = 478.56 s spoken + 121.44 s playback.** The real [qualified highlight](Mach_Jesus_Qualified_Highlight.mp4) plays after slide 8, from **5:10 to 7:11.44**. Keep the PDF and movie in the same folder. The video itself has no audio narration; speak briefly before/after it, within the spoken allocations. Do not add two minutes of commentary over the playback.

## 1 · Three evidence tracks · 0:00–0:25

- Introduce the project as LiDAR perception plus hybrid racing control, evaluated through actual closed-loop evidence.
- Separate the original 73-case study, later engineering chase/comparison, and saved solo cadence sensitivity. The map graphic is a real saved trajectory.

## 2 · Problem and constraints · 0:25–1:00

- Detection/localization and clean competitive driving are different questions, each with its own gate.
- Own pose and a known approximate map are supplied. Generic simulated vehicle/tyre dynamics, custom scoring and one matched training seed limit transfer claims.
- Opponent truth stays in offline labels/scoring/replay.

## 3 · Actual pipeline · 1:00–1:45

- Follow measured LiDAR → VoxelNeXt → tracks/raw hazards → supervised intent → route/traffic guards → steering/pedals → Webots physics.
- Raw-point hazards complement semantic boxes, while own pose/map support driving intent and planning. Conventional guards stay authoritative.
- Perception is 5 Hz per simulated second; control cadence varies by experiment. RGB records viewing, not detector input.

## 4 · Original detector development · 1:45–2:20

- At a common raw cutoff, full-model fits raise DEV recall from about 52% to 96%. Retain the false-positive counts shown inside the bars.
- Data/mining, initialization, batch order and 1,200 updates are matched. These are single-seed development observations, not fresh qualification or a driving winner.
- This training is historical. Today's engineering update adds no new fitting, weights or RL.

## 5 · Original 73-case study · 2:20–3:25

- Account for 15 baseline cases, 30 paired confirmations, four balanced heats and 24 fresh geometry cases. The historical baseline's DNF stays in the complete record.
- Both confirmation arms complete 128/128 clean laps, but the neural finalist is 3.74% slower by paired median and is not promoted. Final races use the original detector, separate from the adapted-detector DEV bars.
- Every policy occupies each grid slot once; all 80 championship entries finish clean. Reference ranks first among these four policies under the declared aggregate rules. A single heat win does not decide that ranking.

## 6 · Prior retained chase · 3:25–3:55

- Identify the screenshot as the earlier record: 15/18 planned targets, 212.419 s reference crossing, and the explicit 206-post scene cleanup.
- Hero 100 m/s and base-rival 60 m/s software limits/different policies make this an engineering chase. Its mixed motion metrics remain historical context, separate from the new matched comparison and recording.

## 7 · Closed matched A8/B9 comparison · 3:55–4:50

- All six fresh repeats qualify, three per arm, under the same detector/service/source; native safety is clean and all 20 cars remain active at the lead stop.
- B's mean crossing is 198.203 s versus 206.755 s for A: 4.14% lower. Target means are 14/18 for both, with 13–15 ranges in both arms.
- Motion is measured over each whole active episode, not a matched-first-circuit window. B raises TV 1.92%, XY jerk 6.73%, yaw >1 Hz RMS 9.40% and yaw 1–6 Hz power 18.16%.
- B is selected for an aggressive chase, accepting that tradeoff. Plan/profile and corrective budgets change together; these same-condition repeats establish no significance or model promotion.
- Metadata is transparent: the inherited six-case rules label says 20 ms, while actual worlds, all 20 actors, director audits and the optimization protocol verify 10 ms / 100 Hz. Closed inputs are preserved; the label is not used as physical timing evidence.

## 8 · Actual qualified recording · 4:50–5:10

- Introduce the real clip: a new selected-B record crosses the original reference twice, reaches first observed P1 at 85.4 s, and closes with all 20 active/native clean.
- First/second crossing scalars are 198.846 s and 371.236 s; their difference is 172.390 s. The separately verified full flying-lap video is 172.44 s because its native frame boundaries enclose both crossings. The extra 0.04954 s is boundary padding, not slower motion or retiming.
- The highlight preserves the opening through P1, then makes one explicit forward cut. It shows the first crossing plus a short flying excerpt; both complete circuits remain in the 373.24 s full native movie.

## Play the actual highlight · 5:10–7:11.44

Play [Mach_Jesus_Qualified_Highlight.mp4](Mach_Jesus_Qualified_Highlight.mp4) at its encoded native 1× speed. It is 121.44 s, 3036 frames at 25 fps; every source frame is retained in the separate 9331-frame full movie. Original learned weights, all 17 actor imports, 100 Hz control, 50 Hz saved trace and 5 Hz actual cloud/prediction pairing qualify. The offline GT map is labelled; unequal caps remain an engineering-chase limitation.

The optional [complete flying lap at native 1×](Mach_Jesus_Full_Flying_Lap_1x.mp4) is 172.44 s/4311 frames and fully decode verified. It is for extended review, outside the timed 10-minute presentation; do not play both videos in this slot.

## 9 · Solo cadence sensitivity · 7:11.44–8:06.44

- One clear-track case per 20/10/5 ms condition; damping OFF, budget 8, original 50 Hz recordings. All three pass native safety and hit 18/18 targets within 0.5 m.
- Smaller steps lower TV/XY jerk and pose-yaw >1 Hz RMS but raise compute cost and 1–6 Hz yaw power. 10 ms crosses at 183.283 s versus 188.787 s for 20 ms; 5 ms is slightly slower than 10 ms.
- Integration, control frequency and the unchanged five-sample filter horizon vary together. This is n=1 coupled sensitivity, not isolated physics causality or traffic generalization.

## 10 · Portable reproduction · 8:06.44–8:31.44

- Recompute 27,801 original native samples with hash/timestamp/metric-AST/frozen-value checks; no Webots, inference or GPU runs.
- Existing isolated CPU checks report 284 passed and 7 optional skips. Installed-wheel checks pass outside the checkout without Torch. Hosted CI is prepared, not claimed as executed.

## 11 · D05 retained negative · 8:31.44–9:01.44

- OFF is 3/3 clean, D05 only 2/3, D10 3/3. The failed D05 road departure remains; completion-only means cannot establish a robust gain.
- That failed attempt alone uses a fresh service lease, leaving cause unresolved. The later terminal-metadata audit failure is separate from the earlier physical departure. No damping variant is promoted.

## 12 · Instructor discussion and repository · 9:01.44–10:00

- Ask which next experiment is most valuable: hold control frequency/filter horizon fixed while changing integration; add independent traffic seeds/service repetitions; or define a clearer safety/pace/motion/compute acceptance rule.
- Keep the original Reference ranking, unequal-cap chase, same-service budget comparison and solo sensitivity separate. A documented pace/motion tradeoff is useful without claiming universal improvement.
- Point to the canonical repository, and distinguish existing repository access from these locally prepared updates. Give the instructor time to choose a direction. Do not claim any optional altered-grid result unless a later separately closed report is supplied.

## Evidence and playback bindings

- [Final deck](Presentation_Update.pdf) · [real poster](control-cadence-demo.png) · [source/media hashes](../experiments/local-optimization/media.json)
- Original study: `docs/final_study.md`; matched public citation: `experiments/local-optimization/evidence/matched-budget-report.json`.
- Qualification SHA256: `e2d61fbd7c48dadc9f70ca86baf07e74cf081b446af8967e74ba833ca72c3856`. Highlight SHA256: `ca7e8fa41531f469f2c24001c8c1b105508f4c30b45921092c2804fccfe7ed67`.
- Earlier working decks remain in the local project; the original course deck is preserved in this repository. No optional grid outputs or incomplete media are read by the final builder. The complete normal-speed movie/source bindings remain in the private project.

## Optional Q&A: later grid check

A separate [changed-grid pair](../experiments/local-optimization/evidence/altered-grid-pair.json) subsequently qualified: A: 205.022 s / 13 targets versus B: 198.842 s / 14 targets, all 20 active/native clean, with higher B motion measures. This is one additional pair on a separate service context, not pooled with the six-run comparison. It is outside the timed deck.

## Timing and capture context

Control, sensing and native video rates refer to simulation time. The two-circuit recording required 807.97 wall seconds for 373.24 simulated seconds including startup/shutdown (about 0.46× real time); rendering is separate. Both later altered-grid cases disable camera capture, while the six-case parent cohort captured R20. Comparisons across these contexts do not isolate the grid change.
