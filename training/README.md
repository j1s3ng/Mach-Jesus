# VoxelNeXt experiment workflow

This directory implements preparation and real CUDA execution for the proposal. **No A2RL fine-tuning or real-dataset validation result has been measured in this checkout.** The Mac demo uses the explicitly labeled clustering baseline; CUDA smoke status is recorded separately in its actual run artifacts. The A2RL release is gated, and the LAN host must be reachable for GPU execution.

Run the following commands from the `race-lidar/` project directory. Keep raw data, official pretrained weights and run artifacts out of Git. The official dataset access must belong to the user's account; these scripts do not register, accept terms, or bypass access controls.

1. Request access on the [official A2RL Vmax release](https://huggingface.co/datasets/a2rl-vmax/a2rl-vmax), then run `hf auth login` on the download machine. Do not put tokens in scripts or chat. Install the project's `data` optional dependencies to get the Hub client. Fetching defaults to code only; the full data download is explicit:

   ```sh
   python scripts/fetch_a2rl.py --metadata-only
   python scripts/fetch_a2rl.py
   python scripts/fetch_a2rl.py --data
   ```

   The helper pins release revision `4280b7f76e0e20e8aa0fc943a088a6f7051f6d8d`, records actual acquisition status and uses the local authenticated Hub session. The full release is approximately 67 GB before exports. The release's code stays under its respective licenses; dataset and extractor are CC BY-NC 4.0.

2. On the NVIDIA Linux host, use the release's bundled `code/vmax_OpenPCDet` and `code/vmax_nuscenes_devkit`, rather than stock OpenPCDet/devkit. Keep a dedicated project environment. The supplied Docker image uses CUDA 12.4/PyTorch 2.4 and is a sensible initial target for an RTX 3090 or T4. **Do not use that older CUDA runtime for the RTX PRO 6000 Blackwell.** That GPU needs a CUDA/PyTorch/spconv build supporting its architecture; start by inspecting the user's already working runtime. Do not replace a shared runtime or GPU driver. A preflight import is insufficient: the real pilot below is required to validate sparse convolution and CUDA NMS.

   Export the release with its own extractor:

   ```sh
   cd external/a2rl-vmax
   python extract.py --nuscenes --openpcdet --out code/vmax_OpenPCDet/data/nuscenes
   cd ../..
   python -m training.preflight \
     --openpcdet-root external/a2rl-vmax/code/vmax_OpenPCDet \
     --data-root external/a2rl-vmax/code/vmax_OpenPCDet/data/nuscenes \
     --checkpoint checkpoints/voxelnext_nuscenes.pth \
     --output artifacts/cuda_preflight.json
   ```

   Obtain the original nuScenes pretrained checkpoint from the [official VoxelNeXt repository](https://github.com/JIA-Lab-research/VoxelNeXt), together with its matching `cbgs_voxel0075_voxelnext.yaml` source configuration. The original ten-class network is retained so that every learned tensor can be reused. A checkpoint/config mismatch is a blocking error, rather than silently leaving randomly initialized model parameters.

3. Prepare immutable fit/development scenes and the four experiment configs. Adjust info filenames to the actual official export if needed; the listed one-sweep names are the release's documented layout. This reads only trusted local export pickles. It checks all cloud paths and GT layouts, keeps released validation scenes out of development, and records source/checkpoint/split hashes:

   ```sh
   python -m training.prepare \
     --openpcdet-root external/a2rl-vmax/code/vmax_OpenPCDet \
     --data-root external/a2rl-vmax/code/vmax_OpenPCDet/data/nuscenes \
     --train-info external/a2rl-vmax/code/vmax_OpenPCDet/data/nuscenes/v1.0-mini/nuscenes_infos_1sweeps_train.pkl \
     --val-info external/a2rl-vmax/code/vmax_OpenPCDet/data/nuscenes/v1.0-mini/nuscenes_infos_1sweeps_val.pkl \
     --source-config external/a2rl-vmax/code/vmax_OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml \
     --checkpoint checkpoints/voxelnext_nuscenes.pth \
     --output artifacts/study --epochs 20 --batch-size 1 --seed 42
   ```

   The release's racing config changes the network to one class and a much larger range. This workflow instead preserves the original source model's head structure, point-cloud range and voxel size. Original class order is retained, with **car output only** scored in the common **50 m XY-radius** region. All target comparisons use one sweep. Raw files contain `[x,y,z,intensity,ring]`; the loader discards ring and appends a **zero time-lag**, producing the five features the original checkpoint expects. No intensity normalization is invented. The runner checks this layout against the actual fork loader before any model execution. Inspect coordinate plots/intensity distributions before fitting. The ten-sweep pretraining versus one-sweep target difference remains a limitation shared by all comparisons.

   The three fits use the same geometric augmentation and scene budget. GT database sampling is disabled for all fits: a database generated from all released training scenes could paste development objects into fitting. Consequently this workflow does not need `create_gt_database.py`; the unmodified release recipe does require it. Class-balanced resampling is disabled because nine retained nuScenes classes have no positives in this dataset. Velocity output tensors remain compatible, while their loss weights are zero because trustworthy velocity supervision is outside this study.

4. Bind the intended available GPU with its existing UUID or logical index. First execute a real two-step pilot. It transfers all learned parameters/BN statistics, performs backward/optimizer steps, saves and reloads a checkpoint, and runs an inference pass. Frozen-backbone running statistics stay frozen in head-only mode:

   ```sh
   CUDA_VISIBLE_DEVICES=0 python -m training.run --plan artifacts/study/plan.json \
     --experiment head_1e-4 --phase pilot --pilot-steps 2 --output runs/cuda_pilot
   ```

   Read `status.json`, `input_layout_audit.json`, `checkpoint_transfer.json`, `trainable_parameters.json` and `losses.jsonl`. Pilot loss is a pipeline check. `training_executed` and `pilot_executed` remain false unless those operations actually finish.

5. Evaluate the original checkpoint on development, then run and evaluate each matched fit. The optimizer is constant-LR AdamW with source weight decay and gradient clipping; head-only starts at `1e-4`, full-network fits at `1e-4` and `1e-5`. Training uses all planned epochs without early stopping. Existing output directories are never overwritten:

   ```sh
   python -m training.run --plan artifacts/study/plan.json --experiment pretrained \
     --phase evaluate --output runs/pretrained_dev
   python -m training.run --plan artifacts/study/plan.json --experiment head_1e-4 \
     --phase train --output runs/head_1e-4
   python -m training.run --plan artifacts/study/plan.json --experiment head_1e-4 \
     --phase evaluate --checkpoint runs/head_1e-4/checkpoint_epoch_20.pth --output runs/head_1e-4_dev
   ```

   Repeat the last two commands for `full_1e-4` and `full_1e-5`. Run sequentially, and inspect existing GPU workloads before launching. No remote host, driver, service or unrelated workload is changed by these scripts.

   Evaluation removes GT boxes before inference and then joins predictions to labels by sample token. It exports predictions, GT for audit, measured warm-up/batch-one latency, torch GPU memory and **custom XY-center AP101 at 2 m**, recall and localization errors. These are **not official A2RL/nuScenes scores**. Official benchmark evaluation with the release's modified devkit and its arbitrary-sample-set interface remains to be integrated/verified after dataset access. Do not substitute stock nuScenes scoring: its default range can silently differ from the A2RL 300 m evaluation range. Report the exact scoring protocol, and never compare these custom diagnostics directly to published official AP.

6. Freeze selection using only completed development results, repeat the selected configuration with a second seed (`--seed 43` and fresh run directories), then evaluate held-out data only after that choice is fixed:

   ```sh
   python -m training.select --runs runs/head_1e-4_dev runs/full_1e-4_dev runs/full_1e-5_dev \
     --output artifacts/selection.json
   python -m training.run --plan artifacts/study/plan.json --experiment full_1e-4 \
     --phase train --seed 43 --output runs/full_1e-4_seed43
   # Replace the experiment/checkpoint below with the actual frozen selection.
   python -m training.run --plan artifacts/study/plan.json --experiment full_1e-4 \
     --phase evaluate --split heldout --selection artifacts/selection.json \
     --checkpoint runs/full_1e-4/checkpoint_epoch_20.pth --output runs/selected_heldout
   ```

   `training.select` verifies that all candidates used the same frozen plan and that the selected checkpoint has not changed. Held-out execution requires the matching frozen experiment and checkpoint. For the paired original-weight held-out evaluation, freeze the completed baseline development result separately with `training.select --runs runs/pretrained_dev --output artifacts/baseline_selection.json`, then evaluate `pretrained` with that selection. No fit may use held-out data.

The inference bridge API is `training.inference.load_predictor(pcdet_root, config_path, checkpoint_path, device='cuda:0')`. It returns a callable accepting finite `N×4` points and producing car-only boxes in metres with `[length,width,height]`, yaw radians and score. It uses the same preprocessing and strict weight transfer as the study. Live inference can use the public official VoxelNeXt/OpenPCDet source because `DatasetTemplate` does not read A2RL metadata; actual A2RL preparation/training still requires the gated fork. Real model timing and calibration must be measured on each host; simulated sensor intensities/density differ from the training data.

Sources: [official VoxelNeXt installation](https://github.com/JIA-Lab-research/VoxelNeXt/blob/master/docs/INSTALL.md), [OpenPCDet source model config](https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml), [OpenPCDet nuScenes loader](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/nuscenes/nuscenes_dataset.py), [A2RL release/toolchain/coordinate documentation](https://huggingface.co/datasets/a2rl-vmax/a2rl-vmax).
