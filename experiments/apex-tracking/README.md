# Apex tracking — local review

**The selected 20-car demonstration reached 15/18 predeclared planned-map anchors within 0.5 m, including all 14 anchors where nominal-line tracking was requested.** It crossed the common reference threshold at **212.419 s** on the race clock from its rear-grid start and closed in sustained P1 at **213.42 s**, with all 20 cars active and zero native overlaps, wheel-rule violations, inward-margin violations or pose resets. This threshold crossing includes the rear-grid approach; it is not a flying-lap time or completion of the nominal 20-lap race.

The earlier apex videos and interactive replay are in the [local archive catalog](../../docs/archive.md#apex-tracking-media). The [measured comparison figure](figures/planned-anchors-and-motion.png) ([PDF](figures/planned-anchors-and-motion.pdf)) and [frozen metrics](evidence/selected_demo_metrics.json) retain definitions and limitations.

The main practical change was removing **206 non-collidable decorative trackside posts** from the perception scene. Their LiDAR returns could be treated as obstacles and keep a passing car away from its nominal line. Physical obstacles, road geometry, cars and vehicle physics remain. The selected run uses the original fixed/nominal control geometry: C2 reference, finite return and yaw-stability options are **off**. Historical improvement includes this scene change and is not attributed to those disabled features. Earlier unsuccessful controller variants remain negative development evidence.

A same-scene/source/camera bundle comparison also favored the selected original-geometry settings:

| Same clean scene; same source and 25 fps camera | Common-reference crossing from grid | Planned anchors within 0.5 m |
|---|---:|---:|
| Experimental option bundle enabled | 261.723 s | 3/18 |
| Selected option bundle disabled | 212.419 s | 15/18 |

This is a bundled option comparison, not an isolated C2 or yaw-controller experiment. The enabled run recorded zero yaw-feedback activations and zero steering changes from that feedback; no isolated yaw-effect claim follows. The frozen metrics retain this negative result alongside the selected take.

Against the prior recorded scene, planned-anchor hits rose from 2/18 to 15/18 and the first common-reference crossing fell from 235.903 to 212.419 s. Mean anchor error fell from 7.028 to 1.842 m, while the largest miss rose from 15.933 to 18.725 m. The three remaining misses occurred during measured passing intent; this does not establish that those misses were necessary or optimal. Planned anchors score map XY at the same original-reference progress; they are not surveyed physical apexes or a claim of perfect cornering.

On matched first-circuit 50 Hz recordings, steering-command variation/time fell **15.68%**, command rate RMS fell 12.55%, and command acceleration RMS fell 16.51%. Actual body yaw-rate RMS above 1 Hz fell **5.31%**. Body-motion results remain mixed: forward-difference yaw-acceleration RMS rose **31.97%** and lateral XY jerk rose **2.16%**. Actual commands and body-motion derivatives are separate measurements. This is a single-run benchmark, and it does not establish that all jitter is fixed.

The enhanced learned hero has a 100 m/s software maximum and a driving overlay; base rivals use different policies and lower limits. The field is not a fair best-model comparison. Independent sustained progress crossings counted 19 distinct active rivals, 16 on the left and three on the right; retirements were excluded. These sampled crossings do not certify every physical manoeuvre. All cars remained active at the stop.

Fresh selected-source fixtures retained native safety and command priorities. The [cut-in](evidence/cutin_safety.json) and [constrained rear](evidence/rear_safety.json) audits observed their declared stimuli with clean responses. A separate [controlled middle-pass test](evidence/controlled_middle_pass.json) recorded three whole-envelope-between samples at 8.0/8.2/8.4 s, both progress crossings at 8.4 s confirmed at 9.0 s, and full clearance at 8.6 s confirmed at 9.2 s. Minimum side clearances were about 1.051/1.141 m. Its two opponents were fixed-pace cars; that controlled proof is separate from the 20-car video.

The custom road rule requires two wheel centers on generated tarmac/white lines and excludes kerbs. It is a simulator approximation. Reference/corrective software lateral budgets remain 8/12.8 m/s², not measured tire forces. Own-car Webots pose/velocity and the known map are controller inputs; opponent ground truth is excluded from control. The offline replay shows current/past opponent observations for display only and never draws their future actual paths.

The native camera is 25 fps; LiDAR/model decisions are 5 Hz and commands/body telemetry are 50 Hz. All 5,360 camera frames are retained. The presentation is 190.667 s at 1.2× for its driving section because retaining every frame exceeds the original short target; the normal recording is 214.4 s at native 25 fps. Camera cadence is a viewing change, separate from control results. No motion interpolation, display command smoothing or future measurements are introduced.

The replay retains all 1,068 saved three-second measured forecasts. Its new-run raw mean/bounds audit found zero interval anomalies, while hard-veto and nonzero soft-cost activations were both zero. This does not establish calibrated forecasts or a forecast-based control benefit. Teal approved-return geometry is an unused optional display legend in driving frames; that feature never activates in this selected run. The final card omits the unused legend.

Seven exact source overlays, [selected settings](evidence/selected_settings.json), safety/measurement summaries and the [dependency recipe](APPLY_OVERLAY.md) are included. Source and measurements remain here; the earlier media are archived locally. The [historical media receipt](media.json) records their original public paths and hashes, not current availability. The original README bytes are preserved in the [documentation amendment ledger](../../docs/provenance/DOCUMENTATION_AMENDMENTS.json).
