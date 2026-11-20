# Command catalog

Run scripts from the repository root with `PYTHONPATH=src:.` when needed.
The [setup guide](../docs/setup.md) gives complete environment/artifact recipes.
Executable paths stay stable because tests, imports and historical source
receipts bind them.

## Portable CPU entry points

| Entry point | Purpose |
| --- | --- |
| `python -m race_lidar doctor` | Inspect the local environment. |
| `python -m race_lidar demo` | Generate and replay a synthetic geometric-clustering fixture. |
| [evaluate_cached_qualification.py](evaluate_cached_qualification.py) | Recompute matching/counts from the included saved qualification inputs. |
| [Native cached reproduction](../reproduction/controlled-native/README.md) | Recompute the controlled native-development comparison. |
| [BEV AP reproduction](../reproduction/controlled-bev-ap/README.md) | Recompute score-ranked BEV AP on the known simulated development set. |
| [Rotation cached reproduction](../reproduction/controlled-rotation/README.md) | Recompute the frozen rotation diagnostic. |
| [Motion and cadence reproduction](../reproduction/control-cadence/README.md) | Recompute measurements from saved solo driving traces. |

## Simulator, races and viewing

| Entry point | Purpose |
| --- | --- |
| [run_webots.py](run_webots.py) / `./run sim` | Scripted sensor-capture fixture. |
| [run_hotlap.py](run_hotlap.py) / `./run hotlap` | Physical single-car run with external artifacts. |
| [run_improvement_heat.py](run_improvement_heat.py) | Physical multi-car development heat. |
| [run_tournament_series.py](run_tournament_series.py) / `./run tournament` | A tournament series. |
| [render_tournament.py](render_tournament.py) / `./run race-video` | Render recorded race results. |
| [camera_overlay.py](camera_overlay.py) / `./run overlay` | View recorded camera/point overlays. |
| [generate_hotlap_world.py](generate_hotlap_world.py), [generate_tournament_world.py](generate_tournament_world.py) | Prepare worlds; they do not establish a completed run. |

Later controller variants live in [experiment overlays](../experiments/README.md).
Use the selected source recipe and compatible external assets for that variant.

## Detector and driver development

- [serve_exported_detector.py](serve_exported_detector.py): checkpoint/hash-verified inference endpoint.
- [serve_race_vision.py](serve_race_vision.py): native vision service implementation.
- [build_sim_detector_dataset.py](build_sim_detector_dataset.py), [train_sim_detector.py](train_sim_detector.py), [qualify_sim_detector.py](qualify_sim_detector.py): data, fitting and development qualification.
- [train_rich_driver.py](train_rich_driver.py), [train_rich_commit.py](train_rich_commit.py), [train_rich_robust.py](train_rich_robust.py): supervised driver development.
- [training_sources.py](training_sources.py), [cv_training_sources.py](cv_training_sources.py), [validate_cv_sources.py](validate_cv_sources.py): source/boundary checks.
- [Training workflow](../training/README.md), [CUDA notes](cuda/requirements-kosmonaut.txt), [artifact catalog](../artifacts/README.md).

## Preparation and retained campaign tools

- [fetch_vehicle_assets.py](fetch_vehicle_assets.py), [fetch_f1_reference.py](fetch_f1_reference.py), [install_webots_macos.sh](install_webots_macos.sh): external setup.
- [verify_downloads.py](verify_downloads.py), [restore_staged_artifacts.py](restore_staged_artifacts.py): artifact validation/restoration.
- `audit_*`, `analyze_*`, `plot_*` and `report_*`: diagnostics and report generation.
- `final_*`, `prepare_final*`, `freeze_*`, `watch_*`, `race_reward_cem.py`, and queue/ledger tools: retained dated study/campaign logic. Their archived-input, identity and deadline assumptions remain explicit; they are not a single new-race launch recipe.

The default portable test suite and retained archived-context tests are separate.
See [setup](../docs/setup.md) before opting into historical campaign checks.
