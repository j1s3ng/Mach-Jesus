# Setup and reproduction

Run commands from the repository root into fresh output directories. The
[report](report.pdf), [116-second demo](demo.mp4) and [study summary](final_study.md)
describe completed experiments. These additions are local review changes;
this preparation step has not uploaded them.

## CPU installation and portable results

The portable recipes were verified with Python 3.11.6, NumPy 1.26.4 and Shapely
2.1.2. Other Python minor versions have not been verified for these recipes.
The separately recorded GPU environment used Python 3.12.14.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test,reproduction]'
export RACE_PYTHON="$PWD/.venv/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
./run doctor
.venv/bin/python -m pytest -q
./run demo --output runs/cpu-fixture --frames 20
.venv/bin/python scripts/evaluate_cached_qualification.py --output runs/qualification.json
.venv/bin/python -I reproduction/controlled-rotation/reproduce.py --output runs/rotation.json
.venv/bin/python -I reproduction/controlled-native/reproduce.py --output runs/native.json
.venv/bin/python -I reproduction/control-cadence/reproduce.py --output-dir runs/control-cadence
```

The default test selection is recorded in `PORTABLE_TESTS.json`; other retained
tests require explicit archived-context opt-in. Optional PyTorch tests are
skipped when PyTorch is absent. `demo` creates a synthetic
geometric-clustering fixture, which is separate from learned detector accuracy.
The three cache recipes query no model and need no simulator, GPU or full cloud
dataset. They verify saved inputs and recompute matching/counts from predictions
and labels: 8,955 qualification frames, 512 rotated label frames and 6,144 native
development frames. Native reproduction documents a historical 2.78e-17
summation-order difference on its common-center cohort. See
[reproduction details](../reproduction/README.md).

## External artifacts and scope

[The artifact catalog](../artifacts/downloads.json) records expected model,
reference and training-bundle hashes. Their download links remain unavailable.
Full detector training and physical races require those external artifacts;
they are not complete clean-clone reproductions today. Full point arrays and
private provenance are outside ordinary Git history. No CUDA runtime is bundled.

```sh
.venv/bin/python scripts/verify_downloads.py
# Only after obtaining a complete matching flat companion manifest:
.venv/bin/python scripts/restore_staged_artifacts.py --staging ../artifacts-v2
```

The strict restorer checks roles, paths and hashes and refuses conflicting
existing files. The private 72-file archive preserves `original_project/` paths;
it cannot be passed directly to this flat-manifest restorer. Verified extraction
and role mapping would be required first.

The system uses simulated measured LiDAR, supplied ego localization/map and a
supervised policy with tracking, a guard and conventional actuation. RGB is for
viewing; opponent truth is offline supervision/scoring only. All final races use
the original detector; adapted-detector qualification is a separate comparison.

## NVIDIA inference

The recorded Linux stack was Torch 2.13.0+cu130, spconv-cu126 2.3.8, cumm-cu126
0.7.11 and project-local CUDA 13.0 compiler components, on PRO 6000 and RTX 3090.
It extended an existing working runtime. `scripts/cuda/requirements-kosmonaut.txt`
is supplemental: it does not install Torch, a GPU driver or a complete CUDA
environment. A fresh standalone GPU build has not been verified. Use an isolated
compatible environment and the pinned upstream build instructions.

Acquire the original checkpoint independently from the upstream model zoo
(recorded Google Drive ID `1IV7e7G9X-61KXSjMGtQo579pzDNbhwvf`) and verify its hash.

```sh
git clone https://github.com/JIA-Lab-research/VoxelNeXt external/VoxelNeXt
git -C external/VoxelNeXt checkout b5b7d393cd1d0ecbbaeaca365b453b488791035d
export PCDET_ROOT="$PWD/external/VoxelNeXt"
export MODEL_CONFIG="$PCDET_ROOT/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
export CUDA_VISIBLE_DEVICES=0
# Build the upstream native extensions in your isolated CUDA environment.
# The recorded extension build used: python setup.py build_ext --inplace
export OFFICIAL_CHECKPOINT="$PWD/runs/improvement-12h/checkpoints/official_voxelnext.pth"
python scripts/serve_exported_detector.py --pcdet-root "$PCDET_ROOT" \
  --config "$MODEL_CONFIG" --checkpoint "$OFFICIAL_CHECKPOINT" \
  --expected-sha256 9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b \
  --port 8770
```

The server binds to loopback. In a separate Mac terminal, forward your GPU host:

```sh
export RACE_GPU_HOST=user@gpu-host
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -L 8770:127.0.0.1:8770 "$RACE_GPU_HOST"
```

Then check health and send the included measured TRAIN cloud from another terminal:

```sh
curl --fail http://127.0.0.1:8770/health
curl --fail -H 'Content-Type: application/x-npy' -H 'X-Car-Id: measured-smoke' \
  --data-binary @examples/measured/points.npy http://127.0.0.1:8770/predict
```

Inputs are single-sweep N×4 `[x,y,z,intensity]` arrays in metres, x forward/y
left/z up; the adapter adds a zero time-lag feature. Simulated reflectance is a
placeholder. The official ten-class nuScenes checkpoint supplies car output
within 50 m XY. `examples/measured/ground_truth.json` is a separate label and is
never sent to the endpoint. `metadata.json` records the cloud hash.

For the acquired calibrated pipeline, substitute `candidate_tau0425.pth`, hash
`0a68a778cb676eea56d1da60131ac55ab190a78e2ba14537fec13c2b996d63f6`
and a separate port such as 8772. The portable wrapper verifies tensor/source
bindings and enforces RAW 0.425. This wrapper has not been physically qualified
and is separate from the historical runtime; controller filtering is additional.

```sh
python scripts/cuda/smoke.py --pcdet-root "$PCDET_ROOT" --config "$MODEL_CONFIG" \
  --checkpoint "$OFFICIAL_CHECKPOINT" --points examples/measured/points.npy \
  --output runs/cuda-smoke
```

This gradient/save/reload smoke checks the GPU pipeline, not detector accuracy.

## Mac simulator and an engineering race

The recorded Webots R2025b nightly2026-11-12 installs into `.local/`; launchers
supporting `--webots` can use another installation. Verify that archived build's
availability/compatibility on a new machine. Fetched assets retain upstream notices.

```sh
bash scripts/install_webots_macos.sh
.venv/bin/python scripts/fetch_vehicle_assets.py
.venv/bin/python scripts/fetch_f1_reference.py --output .local/f1-reference
./run sim --output runs/sensor-capture --frames 100
```

The circular `sim` fixture uses scripted motion and rendered sensors; full-course
races use wheel physics. [Reference provenance](f1_reference.md) explains the
approximate OpenF1 geometry. Generic vehicle/tires do not establish F1 performance.

After acquiring the catalog's selected driver and exact reference, create a
relative-path roster and run a small development race against the original
detector server. The runner logs actions and separate evaluation truth.

```sh
mkdir -p configs
printf '%s\n' '[{"policy":"neural_rich005","label":"Rich005 seed 17","model_path":"runs/improvement-12h/rich-driver-005/seed-17/model.npz"}]' > configs/rich005.json
.venv/bin/python scripts/run_improvement_heat.py --cars 4 --seed 307 \
  --seconds 480 --spacing 16 --visibility 45 --grid-jitter .4 \
  --policies configs/rich005.json --reference .local/f1-reference/reference.npz \
  --vision-url http://127.0.0.1:8770 --secondary-cars 0 \
  --source-root "$PWD" --split development --output runs/rich005-four
.venv/bin/python scripts/evaluate_improvement_attempt.py runs/rich005-four \
  --expected-policy neural_rich005 \
  --expected-model-sha256 d9837d76d29f80368dae8ec8f611aa67b4c3e7885ed5bab51876fbff34afd9da \
  --expected-detector-sha256 9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b
```

This is a new engineering run, not historical scoring or formal promotion.
Measure workload before increasing cars. Recorded 20-car runs used 12 PRO 6000
and eight RTX 3090 streams, 5 Hz sensors and 50 Hz physics, slower than wall time.
Each GPU needs a separate detector process/port; preserve pipeline identities.

## Driver training

Portable Rich drivers use 48 features and six outputs with NumPy inference.
Training needs Torch and supports CPU fitting. Teachers consume measured or
synthetic hazard summaries rather than hidden opponent state; the reference
target-speed label is not a neural input. This is supervised imitation.

```sh
python scripts/train_rich_driver.py --reference .local/f1-reference/reference.npz \
  --output runs/rich-foundation --seeds 7 17 --cpu-threads 4 \
  --coverage critical --lane-summary-mode stopping_critical --cost-distillation
```

That trains a new foundation model. Historical Rich005 additionally requires
the retained Rich004 bundle and exact permitted engineering runs/source hashes:

```sh
python scripts/train_rich_commit.py --retained data/rich004 --initial data/rich004 \
  --failure-run data/engineering/rich004-s17-visibility-entry \
  --evaluation-run data/engineering/rich004-s17-field4-diagnostic \
  --reference .local/f1-reference/reference.npz --seeds 17 7 \
  --epochs 80 --cpu-threads 4 --output runs/rich005-reproduction
```

Use archived trainers for historical verification; current guards can differ.
Reserved bootstrap, holdout, confirmation and CV-qualification data and excluded
ancestors are rejected. Preserve those checks and sealed data roles. See the
[Rich005 model card](../reproduction/rich005_model_card.md) and
[Rich006 source notes](../research/rich006/README.md); their historical fitting
contexts are not portable raw-data rebuilding recipes.

## Detector training and evaluation

New datasets can be built from run-separated captures:

```sh
python scripts/build_sim_detector_dataset.py --train-runs data/engineering/train-a \
  data/engineering/train-b --val-runs data/engineering/development-c \
  --drop-duplicate-validation --output data/sim-detector-new
```

Captures require `config.json`, per-car `points_manifest.jsonl`, actual N×4 clouds
and timestamp-exact separate `gt_poses.jsonl`. Keep entire runs in one role and
remove exact TRAIN-cloud duplicates from validation. Labels include visible cars.
A same-run gradient smoke cannot establish independent validation.

For the matched fits, set `DATASET`, `MINER` and `RELEASES` to the acquired frozen
TRAIN/DEV dataset, mining checkpoint and source-release manifest. The original
initialization is restored after TRAIN-only mining. This command performs training:

```sh
python scripts/train_sim_detector.py --dataset "$DATASET" --pcdet-root "$PCDET_ROOT" \
  --config "$MODEL_CONFIG" --checkpoint "$OFFICIAL_CHECKPOINT" --mine-checkpoint "$MINER" \
  --mine-all-training --mine-allow-training-subset --mine-score .2 \
  --balance-scenes --negative-batch-fraction .5 --empty-loss-weight 3 \
  --freeze-batchnorm --seed 7 --batch-size 2 --max-steps 1200 \
  --training-releases "$RELEASES" --mode full --lr 1e-4 --backbone-lr 1e-5 \
  --output runs/matched-mixed-seed7
```

For other arms use `--mode head --lr 1e-4`, `--mode full --lr 1e-4`, or
`--mode full --lr 1e-5`, omitting `--backbone-lr` for uniform arms. The rotation
fit uses `scripts/train_sim_detector_rotaug.py` with mixed-rate arguments and
TRAIN-only independent rotations. This does not guarantee hardware determinism.

For TRAIN-only calibration, set `PROTOCOL` to the frozen protocol and `LEARNED`
to its bound checkpoint. Historical protocols bind the original candidate hash;
a new fitted artifact needs a new predeclared identity/protocol.

```sh
python scripts/calibrate_detector_score.py --dataset "$DATASET" \
  --training-releases "$RELEASES" --protocol "$PROTOCOL" \
  --official-checkpoint "$OFFICIAL_CHECKPOINT" --learned-checkpoint "$LEARNED" \
  --pcdet-root "$PCDET_ROOT" --config "$MODEL_CONFIG" --output runs/train-score-calibration
```

Evaluate with `train_sim_detector.py --evaluate-checkpoint ...`; sealed
qualification requires its frozen authority. `qualify_sim_detector.py` checks
frozen groups. The cached CPU evaluator is the immediately available path.

The [planned A2RL workflow](../training/README.md) requires approved gated access.
No real A2RL training or official AP result is claimed. See
[final evidence](final_evidence.md) for the separate archived-input 73-case audit.
Files under `docs/provenance/` describe earlier export snapshots, not an inventory
or validation of every subsequent edit in this repository.
