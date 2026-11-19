# Engineering experiments

The base `src/` tree preserves the completed study. These folders preserve later
engineering extensions without replacing that snapshot. Their source overlays
are exact versions, not independent packages to combine in arbitrary order.

| Extension | Focus | Evidence to start with |
| --- | --- | --- |
| [Control cadence and pace](local-optimization/README.md) | Six matched 100 Hz traffic runs, qualified two-circuit demo, and explicit pace/motion tradeoff. | [All six results](local-optimization/evidence/matched-budget-report.json), [source recipe](local-optimization/APPLY_OVERLAY.md). |
| [Apex tracking](apex-tracking/README.md) | Selected scene-cleanup demo: 15/18 planned targets within 0.5 m; mixed body-motion results. | [Measured comparison](apex-tracking/evidence/selected_demo_metrics.json), [source recipe](apex-tracking/APPLY_OVERLAY.md). |
| [Trajectory and trail braking](trajectory-trail-brake/README.md) | Saved three-second traffic forecasts and coordinated brake release; soft penalty rejected. | [Results](trajectory-trail-brake/evidence/results.json), [failed combined case](trajectory-trail-brake/evidence/combined_failed_case.json). |
| [Smooth steering](smooth-steering/README.md) | Spatial feedforward averaging and actual steering/body trade-offs. | [Results](smooth-steering/evidence/results.json), [rejected dwell candidate](smooth-steering/evidence/rejected_dwell_candidate.json). |
| [Apex and gap handling](apex-and-gap/README.md) | Whole-road cornering, custom two-wheel limits and controlled middle passes. | [Clear-lap calibration](apex-and-gap/evidence/clear_lap_selection.json), [middle-pass proof](apex-and-gap/evidence/controlled_middle_pass.json). |
| [Smooth line and pass-first](smooth-line-pass-first/README.md) | Frozen map-line optimization and passing on both sides. | [Map-only analysis](smooth-line-pass-first/analysis/map_only_report.json), [manifest](smooth-line-pass-first/manifest.json). |
| [Rear-grid chase](rear-grid-chase/README.md) | Initial P20-to-P1 engineering chase and explicit controller overrides. | [Source/settings manifest](rear-grid-chase/manifest.json). |

For the latest selected source, apply the base study, `apex-and-gap/overlay`,
`smooth-steering/overlay`, `trajectory-trail-brake/overlay`, then
`apex-tracking/overlay`, then `local-optimization/overlay`, preserving relative paths. The
[latest reconstruction receipt](local-optimization/evidence/source-reconstruction.json)
checks all 17 actor runtime modules and two preparation generators. External
weights, reference/line artifacts and simulator assets are still required.

Chase gains do not supersede the [balanced 73-case study](../docs/final_study.md).
Observe each experiment's scene, vehicle, control and measurement definitions;
some historical comparisons change more than one factor. Original JSON/hash
receipts remain historical evidence. Large earlier videos are in the
[local media archive](../docs/archive.md), with original identities preserved.
The [documentation amendment ledger](../docs/provenance/DOCUMENTATION_AMENDMENTS.json)
accounts for archive-link edits to three READMEs while retaining their exact
manifest-pinned originals and all frozen manifest bytes.
