#!/usr/bin/env python3
"""Behavior clone actual Webots track pace, augmented with synthetic traffic.

This is supervised imitation, not reinforcement learning. Held-out route sectors
diagnose interpolation; a final all-sector fit supplies the track-specific model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.training_sources import validate_training_source
from race_lidar.hotlap_driver import RacingReference, wrap_angle
from race_lidar.learned_driver import CURVATURE_DISTANCES_M, FEATURE_NAMES, LearnedDriver, feature_vector, save_numpy_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def traffic_teacher(state, base_speed):
    """Synthetic traffic labels; these are not observed racing outcomes."""
    distance, lateral, lead = (state[key] for key in ("front_distance_m", "front_lateral_m", "front_speed_mps"))
    previous = state["last_target_offset_m"]
    if abs(previous) > 1.5 and -12 < distance < 55:
        # Keep the selected passing lane until the associated car is behind.
        return base_speed, math.copysign(3.2, previous)
    if distance < -12 or distance >= 55 or abs(lateral) > 3.5 or lead >= base_speed - 2:
        return base_speed, 0.
    if distance < 0:
        return base_speed, 0.
    if state["left_clearance_m"] >= 4.5:
        return min(base_speed, lead + 9.), 3.2
    if state["right_clearance_m"] >= 4.5:
        return min(base_speed, lead + 9.), -3.2
    gap = 8. + .75 * state["body_speed_mps"]
    return min(base_speed, max(0., lead + .6 * (distance - gap))), 0.


def augment_traffic(state, mode, rng, base_speed):
    value = dict(state)
    # Traffic changes the actual speed at a given track position substantially.
    # Avoid teaching the policy that pace can be inferred from body speed alone.
    value["body_speed_mps"] = float(rng.uniform(0., max(15., base_speed + 10.)))
    value["heading_error_rad"] += rng.normal(0., .035)
    value["cross_track_m"] += rng.normal(0., .25)
    value.update(front_distance_m=float(rng.uniform(12, 45)), front_lateral_m=float(rng.uniform(-1, 1)),
                 front_speed_mps=float(max(2., min(base_speed - 4., rng.uniform(8, 25)))),
                 left_clearance_m=10., right_clearance_m=10., last_target_offset_m=0.)
    if mode == 0:  # no car
        value.update(front_distance_m=60., front_speed_mps=value["body_speed_mps"])
    elif mode == 1:  # right blocked, pass left
        value["right_clearance_m"] = float(rng.uniform(.5, 3.))
        value["front_lateral_m"] = float(rng.uniform(-3.3,-1.5))
    elif mode == 2:  # either side available, prefer left
        pass
    elif mode == 3:  # pass right when left unavailable
        value["left_clearance_m"] = float(rng.uniform(.5, 3.))
        value["front_lateral_m"] = float(rng.uniform(1.5,3.3))
    elif mode == 4:  # follow when both unavailable
        value.update(left_clearance_m=float(rng.uniform(.5, 3.)),right_clearance_m=float(rng.uniform(.5, 3.)))
    elif mode in (5, 6):  # committed pass, including alongside phase
        sign = 1. if mode == 5 else -1.
        value.update(front_distance_m=float(rng.uniform(-8, 30)),
                     last_target_offset_m=sign * float(rng.uniform(2., 3.3)),
                     cross_track_m=sign * float(rng.uniform(1.5, 3.3)),
                     front_lateral_m=-sign * float(rng.uniform(1.5,3.3)))
        value["right_clearance_m" if sign>0 else "left_clearance_m"]=float(np.clip(
            abs(value["front_distance_m"])-5+rng.normal(0,2),0,10))
    elif mode == 7:  # associated car now behind, return to line
        previous=float(rng.choice([-3.2,3.2]))
        value.update(front_distance_m=float(rng.uniform(-20,-13)),last_target_offset_m=previous,
                     cross_track_m=math.copysign(float(rng.uniform(1.5,3.3)),previous))
    elif mode == 8:  # close blocked car, request braking
        value.update(front_distance_m=float(rng.uniform(3,12)),front_speed_mps=float(rng.uniform(0,12)),
                     left_clearance_m=float(rng.uniform(.5,3.)),right_clearance_m=float(rng.uniform(.5,3.)))
    elif mode == 9:  # car outside the relevant track corridor
        value["front_lateral_m"] = float(rng.choice([-1,1])*rng.uniform(4.,7.))
    elif mode == 10:  # no lead, but side occupancy alone must not command a pass
        value.update(front_distance_m=60.,front_speed_mps=value["body_speed_mps"])
        value["left_clearance_m" if rng.random()<.5 else "right_clearance_m"]=float(rng.uniform(.5,3.))
    elif mode == 11:  # faster lead is pulling away
        value["front_speed_mps"]=base_speed+float(rng.uniform(3.,12.))
    elif mode == 12:  # actual right-lane backmarker, noisy occupancy estimate
        value.update(front_lateral_m=float(rng.uniform(-3.3,-1.5)),right_clearance_m=float(rng.uniform(.5,10.)))
    elif mode == 13:  # finish returning from a pass after the associated car clears
        previous=float(rng.choice([-3.2,3.2]))
        value.update(front_distance_m=60.,front_speed_mps=value["body_speed_mps"],
                     last_target_offset_m=previous,cross_track_m=math.copysign(float(rng.uniform(1.5,3.3)),previous))
    elif mode == 14:  # retain passing lane until the rear clearance reaches12m
        sign=float(rng.choice([-1.,1.]))
        value.update(front_distance_m=float(rng.uniform(-11.8,-8)),last_target_offset_m=sign*3.2,
                     cross_track_m=sign*float(rng.uniform(2.,3.3)),front_lateral_m=-sign*2.5)
        value["right_clearance_m" if sign>0 else "left_clearance_m"]=float(np.clip(
            abs(value["front_distance_m"])-5+rng.normal(0,2),0,10))
    else:  # farther lead, including right-lane occupancy, is not an active pass
        value.update(front_distance_m=float(rng.uniform(55.2,60)),front_lateral_m=float(rng.uniform(-3.3,-1.5)),
                     right_clearance_m=float(rng.uniform(.5,10.)))
    return value


def build_dataset(trace_path, reference_path, seed=7, stride=5):
    validate_training_source(trace_path)
    validate_training_source(reference_path)
    rng = np.random.default_rng(seed)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    summary_path = trace_path.parent / "lap_summary.json"
    validate_training_source(summary_path)
    summary = json.loads(summary_path.read_text())
    if not summary.get("valid_lap") or summary.get("position_resets_after_spawn") != 0:
        raise ValueError("Training requires an actually completed, valid physical lap without resets")
    rows = [row for row in rows if row["time_s"] <= summary["lap_time_s"]][::stride]
    reference = RacingReference.from_npz(reference_path)
    features, targets, sectors, kinds, progresses = [], [], [], [], []
    for row in rows:
        progress = float(row["progress_m"]) % reference.length
        _, yaw, _, _ = reference.sample(progress)
        state = {"progress_fraction": progress / reference.length, "body_speed_mps": row["speed_mps"],
                 "heading_error_rad": wrap_angle(yaw-row["yaw_rad"]), "cross_track_m": row["cross_track_error_m"],
                 "curvature_ahead": [reference.sample(progress+distance)[3] for distance in CURVATURE_DISTANCES_M],
                 "front_distance_m":60., "front_lateral_m":0., "front_speed_mps":row["speed_mps"],
                 "left_clearance_m":10., "right_clearance_m":10., "last_target_offset_m":0.}
        base = float(row["target_speed_mps"])
        variants = [(state,(base,0.),0)]
        for mode in range(16):
            augmented = augment_traffic(state,mode,rng,base)
            variants.append((augmented,traffic_teacher(augmented,base),1 if mode == 0 else 2))
        for observation, label, kind in variants:
            features.append(feature_vector(observation))
            targets.append((label[0]/60.,label[1]/3.5))
            sectors.append(min(15,int(progress/reference.length*16)))
            kinds.append(kind)
            progresses.append(progress)
    return np.asarray(features,np.float32),np.asarray(targets,np.float32),np.asarray(sectors),np.asarray(kinds),np.asarray(progresses),summary,len(rows),reference.length


def metrics(prediction, target, kinds):
    error = np.abs((prediction-target)*[60.,3.5])
    actual = kinds == 0
    passing = np.abs(target[:,1]) > .5
    return {"samples":len(target),"speed_mae_mps":float(error[:,0].mean()),
            "offset_mae_m":float(error[:,1].mean()),
            "actual_trace_speed_mae_mps":float(error[actual,0].mean()) if actual.any() else None,
            "synthetic_pass_direction_accuracy":float((np.sign(prediction[passing,1])==np.sign(target[passing,1])).mean()) if passing.any() else None}


def train_model(x,y,fit_mask,validation_mask,epochs,seed,output,label):
    import torch
    torch.manual_seed(seed)
    mean=x[fit_mask].mean(0);scale=np.maximum(x[fit_mask].std(0),.08)
    normalized=torch.from_numpy((x-mean)/scale)
    targets=torch.from_numpy(y)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.ReLU(),
                              torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-5)
    indices=np.flatnonzero(fit_mask)
    rng=np.random.default_rng(seed)
    best,best_epoch,best_weights=math.inf,0,None
    log=[]
    for epoch in range(1,epochs+1):
        model.train();rng.shuffle(indices)
        for start in range(0,len(indices),1024):
            batch=indices[start:start+1024]
            prediction=model(normalized[batch])
            loss=((prediction-targets[batch])**2*torch.tensor([1.5,1.])).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predicted=model(normalized).numpy()
            fit_loss=float(np.mean((predicted[fit_mask]-y[fit_mask])**2))
            val_loss=float(np.mean((predicted[validation_mask]-y[validation_mask])**2)) if validation_mask.any() else fit_loss
        log.append({"epoch":epoch,"training_mse_normalized":fit_loss,"selection_mse_normalized":val_loss})
        if val_loss < best:
            best,best_epoch=val_loss,epoch
            best_weights={key:value.detach().clone() for key,value in model.state_dict().items()}
        if epoch%30==0 or epoch==1:
            print(f'{label} epoch {epoch}/{epochs}: fit={fit_loss:.6f}, selection={val_loss:.6f}',flush=True)
    model.load_state_dict(best_weights)
    with torch.inference_mode():predicted=model(normalized).numpy()
    (output/f'{label}_epochs.json').write_text(json.dumps(log,indent=2)+'\n')
    return model,mean,scale,predicted,best_epoch


def export(model,mean,scale,path,metadata):
    import torch
    layers=[layer for layer in model if isinstance(layer,torch.nn.Linear)]
    return save_numpy_model(path,[layer.weight.detach().numpy() for layer in layers],
                            [layer.bias.detach().numpy() for layer in layers],mean,scale,metadata)


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,default=ROOT/'runs/hotlap-physics-pilot-5/control_trace.jsonl')
    parser.add_argument('--reference',type=Path,default=ROOT/'.local/f1-reference/reference.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/driver-training')
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    if args.epochs<1:parser.error('--epochs must be positive')
    if args.output.exists() and any(args.output.iterdir()):parser.error('Choose a fresh training output')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(6,torch.get_num_threads()))
    started=time.perf_counter()
    x,y,sectors,kinds,progress,summary,trace_rows,length=build_dataset(args.trace,args.reference,args.seed)
    validation=np.isin(sectors,[3,9,13]);fit=~validation
    # Leave a15m gap at held-out-sector boundaries; all augmentations of a source
    # row share its sector so adjacent augmented copies cannot leak across splits.
    for sector in [3,9,13]:
        for edge in [sector*length/16,(sector+1)*length/16]:
            distance=np.abs(progress-edge);distance=np.minimum(distance,length-distance)
            fit &= distance>15.
    model,mean,scale,predicted,best_epoch=train_model(x,y,fit,validation,args.epochs,args.seed,args.output,'sector_holdout')
    metadata={'method':'supervised_behavior_cloning_with_synthetic_traffic_teacher',
              'source_eligibility':[validate_training_source(path) for path in (args.trace,args.reference)],
              'source_trace_sha256':sha256(args.trace),'source_reference_sha256':sha256(args.reference),
              'reference_speed_is_model_input':False,'curvature_distances_m':list(CURVATURE_DISTANCES_M),
              'trained':True,'reinforcement_learning':False}
    export(model,mean,scale,args.output/'sector_holdout_model.npz',{**metadata,'fit':'sectors_except_3_9_13'})
    heldout_metrics=metrics(predicted[validation],y[validation],kinds[validation])
    actual_metrics=metrics(predicted[fit],y[fit],kinds[fit])
    final_epochs=max(best_epoch,80)
    final,final_mean,final_scale,final_prediction,final_epoch=train_model(x,y,np.ones(len(x),bool),np.zeros(len(x),bool),final_epochs,args.seed+1,args.output,'all_sector_refit')
    model_path=export(final,final_mean,final_scale,args.output/'model.npz',{**metadata,'fit':'all_available_track_sectors','epochs':final_epoch})
    # Independently compare the portable runtime with the trained Torch network.
    portable=LearnedDriver.from_npz(model_path)
    parity=float(np.max(np.abs(portable.forward_features(x[:256])-final_prediction[:256])))
    if parity>2e-5:raise RuntimeError(f'NumPy/Torch export mismatch: {parity}')
    report={'status':'completed','method':metadata['method'],'reinforcement_learning':False,
            'source_eligibility':metadata['source_eligibility'],
            'training_device':'cpu','torch':torch.__version__,'seed':args.seed,'source_trace':str(args.trace.resolve()),
            'source_trace_sha256':metadata['source_trace_sha256'],'source_reference_sha256':metadata['source_reference_sha256'],
            'source_valid_lap_time_s':summary['lap_time_s'],'actual_trace_rows_after_stride':trace_rows,
            'dataset_samples':len(x),'actual_trace_examples':int((kinds==0).sum()),
            'perturbed_no_traffic_examples':int((kinds==1).sum()),'synthetic_traffic_examples':int((kinds==2).sum()),
            'feature_names':list(FEATURE_NAMES),'reference_speed_is_model_input':False,
            'heldout_route_sectors':[3,9,13],'route_sector_count':16,'split_guard_distance_m':15.,
            'heldout_diagnostic_model':{'fit_samples':int(fit.sum()),'heldout_samples':int(validation.sum()),'selected_epoch':best_epoch,
                                        'fit_metrics':actual_metrics,'heldout_metrics':heldout_metrics},
            'deployment_model':{'path':str(model_path),'sha256':sha256(model_path),'fit':'all_available_track_sectors',
                                'epochs':final_epoch,'training_fit_metrics':metrics(final_prediction,y,kinds),
                                'independent_lap_evaluation_completed':False},
            'numpy_torch_max_abs_error':parity,'training_seconds':time.perf_counter()-started,
            'limitations':['One deterministic teacher lap; no independent real driving data.',
                           'Traffic labels are synthetic rules, not demonstrated passing outcomes.',
                           'Held-out sectors diagnose a separate fit; deployment refit uses all sectors.',
                           'Low-level steering/braking and feasibility guards remain conventional.']}
    (args.output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(args.output/'training_examples.npz',features=x,targets=y,sectors=sectors,kinds=kinds,
             holdout_fit_mask=fit,holdout_validation_mask=validation)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
