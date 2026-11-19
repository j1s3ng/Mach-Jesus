"""Prepare and run a reproducible protocol-v2 race or engineering scenario."""
from __future__ import annotations

import argparse
import colorsys
import fcntl
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
from improvement_ledger import Ledger,digest,environment_rules_hash,scenario_identity


def prepare(args):
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a fresh attempt directory')
    output.mkdir(parents=True,exist_ok=True)
    source=args.source_root.resolve()
    reference=args.reference.resolve()
    frozen_env=os.environ.copy();frozen_env['PYTHONPATH']=str(source/'src')
    reference_length=float(subprocess.check_output([sys.executable,'-c',
        'import sys;from race_lidar.hotlap_driver import RacingReference;print(RacingReference.from_npz(sys.argv[1]).length)',
        str(reference)],cwd=source,env=frozen_env,text=True))
    frozen_manifest=source/'manifest.json'
    frozen=json.loads(frozen_manifest.read_text()) if frozen_manifest.exists() else None
    frozen_visibility=frozen.get('metadata',{}).get('policy_parameters',{}).get('visible_distance_m') if frozen else None
    if frozen_visibility is not None:
        if args.visibility is not None and args.visibility!=frozen_visibility:
            raise ValueError('Controller visibility must match this frozen candidate parameter')
        args.visibility=float(frozen_visibility)
    elif args.visibility is None:args.visibility=45.
    policies=json.loads(args.policies.read_text()) if args.policies else [
        {'policy':'reference','label':'Reference','model_path':None},
        {'policy':'neural_v1','label':'Neural V1','model_path':str(ROOT/'runs/driver-training/model.npz')},
        {'policy':'neural_v4_seed7','label':'Neural V4 seed7','model_path':str(ROOT/'runs/improvement-12h/training/candidates-v4/v4-seed7/model.npz')},
        {'policy':'neural_v4_seed17','label':'Neural V4 seed17','model_path':str(ROOT/'runs/improvement-12h/training/candidates-v4/v4-seed17/model.npz')}]
    if frozen is not None and args.policies is None:
        policies=[{**p,'model_path':str(source/p['model_relpath']) if p.get('model_relpath') else None}
                  for p in frozen['metadata']['policies']]
    if args.single_policy:
        policies=[p for p in policies if p['policy']==args.single_policy]
        if len(policies)!=1:raise ValueError('The requested single-policy candidate is not in the frozen roster')
    import numpy as np
    rng=np.random.default_rng(args.seed)
    entrants=[]
    for grid in range(args.cars):
        spec=policies[(grid+args.rotation)%len(policies)]
        if spec.get('model_path') and not Path(spec['model_path']).is_file():raise FileNotFoundError(spec['model_path'])
        color=colorsys.hsv_to_rgb(grid/args.cars,.8,.95)
        progress=-args.spacing*grid
        progress-=float(rng.uniform(0.,args.grid_jitter))
        if args.scenario=='parking':progress=reference_length-130.-args.spacing*grid
        entry={**spec,'car_id':f'R{grid+1:02d}','grid_index':grid,'start_progress_m':progress,
               'start_lane_m':(-2.5 if grid%2==0 else 2.5)+float(rng.uniform(-args.grid_jitter,args.grid_jitter)),'seed':args.seed+grid,
               'color':' '.join(f'{v:.3f}' for v in color),'record_video':not args.no_video}
        if args.scenario=='stopped':
            if grid==0:entry.update(start_lane_m=args.ego_lane,start_progress_m=args.ego_progress)
            else:entry.update(policy='stationary_obstacle',label='Stopped obstacle',model_path=None,
                             start_progress_m=args.obstacle_progress,start_lane_m=0. if grid==1 else (-3.2 if args.blocked_side=='right' else 3.2),
                             controller='stationary_race_obstacle',sensors=False,record_video=False,is_static=True)
        if grid>=args.cars-args.secondary_cars:entry['vision_url']=args.secondary_url
        entrants.append(entry)
    rules={'protocol':'race-improvement-v2','reference_length_m':reference_length,'laps':1,'sensor_period_ms':200,
           'physics_step_ms':20,'max_speed_mps':60.,'lateral_accel_budget_mps2':8.,'friction_coefficient':1.5,
           'vehicle_model':'generic Webots Car 880kg electric','physics_threads':8,'field_size':args.cars,
           'control_version':'predictive_v2','visible_distance_m':args.visibility,'grid_spacing_m':args.spacing,
           'post_finish_mode':'physical_staggered_runoff_parking','race_cutoff_seconds':args.seconds}
    rules['grid_jitter_m']=args.grid_jitter
    rules.update(lidar_max_range_m=100.,brake_decel_mps2=8.,acceleration_mps2=5.,track_error_limit_m=6.,total_vehicle_mass_kg=880.)
    capacity=args.capacity or args.scenario!='race'
    config={'output_dir':str(output),'reference_path':str(reference),'reference_length_m':reference_length,
            'entrants':entrants,'cars':{e['car_id']:e for e in entrants},'start_time_s':1.,'laps':1,
            'max_sim_seconds':1.+args.seconds,'sensor_period_ms':200,'video_period_ms':200,
            'record_video':not args.no_video,'pointcloud_stride_frames':args.pointcloud_stride,
            'record_ground_truth':True,'perception_enabled':True,'vision_url':args.vision_url,
            'expected_detector_checkpoint_sha256':args.detector_sha,
            'mode':'capacity' if capacity else 'race','rules':rules,'physics_threads':8,'seed':args.seed,
            'max_speed_mps':60.,'lateral_accel_mps2':8.,'brake_decel_mps2':8.,'acceleration_mps2':5.,
            'control_version':'predictive_v2','visible_distance_m':args.visibility,'post_finish_mode':'park',
            'scenario':args.scenario,'grid_rotation':args.rotation,'source_root':str(source)}
    config['policy_parameters']={'visible_distance_m':args.visibility}
    config['dataset_split']=args.split
    config['excluded_from_training']=args.split in ('bootstrap','holdout','confirmation','cv_qualification',
                                                  'championship','geometric-confirmation')
    if frozen is not None:config['snapshot_sha256']=frozen['snapshot_sha256']
    if capacity:config['capacity_seconds']=args.seconds
    if args.scenario=='stopped':config['hazard']={'ego_car_id':'R01','progress_m':args.obstacle_progress}
    config_path=output/'config.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
    worlds=source/'webots/worlds';worlds.mkdir(parents=True,exist_ok=True)
    world=worlds/'improvement_active.wbt'
    subprocess.run([sys.executable,str(source/'scripts/generate_tournament_world.py'),
                    '--config',str(config_path),'--output',str(world)],cwd=source,env=frozen_env,check=True)
    for name in {'tournament_director',*[e.get('controller','tournament_racer') for e in entrants]}:
        directory=source/'webots/controllers'/name
        (directory/'runtime.ini').write_text(f'[python]\nCOMMAND = {sys.executable}\n')
    return config_path,world


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cars',type=int,default=4)
    p.add_argument('--rotation',type=int,default=0);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--seconds',type=float,default=480.);p.add_argument('--spacing',type=float,default=12.)
    p.add_argument('--visibility',type=float);p.add_argument('--secondary-cars',type=int,default=0)
    p.add_argument('--scenario',choices=['race','stopped','parking'],default='race')
    p.add_argument('--obstacle-progress',type=float,default=100.)
    p.add_argument('--ego-progress',type=float,default=0.);p.add_argument('--ego-lane',type=float,default=0.)
    p.add_argument('--blocked-side',choices=['right','left'],default='right')
    p.add_argument('--capacity',action='store_true');p.add_argument('--no-video',action='store_true')
    p.add_argument('--pointcloud-stride',type=int,default=25);p.add_argument('--policies',type=Path)
    p.add_argument('--single-policy');p.add_argument('--grid-jitter',type=float,default=0.)
    p.add_argument('--vision-url',default='http://127.0.0.1:8770')
    p.add_argument('--secondary-url',default='http://127.0.0.1:8771')
    p.add_argument('--detector-sha',default='9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b')
    p.add_argument('--source-root',type=Path,default=ROOT)
    p.add_argument('--reference',type=Path,default=ROOT/'.local/f1-reference/reference.npz')
    p.add_argument('--stage',choices=['engineering','bootstrap','promotion'],default='engineering')
    p.add_argument('--attempt-id');p.add_argument('--candidate-id',default='protocol-v2-development')
    p.add_argument('--split',default='development')
    a=p.parse_args()
    if (not 1<=a.cars<=64 or not 0<=a.secondary_cars<=a.cars
            or not math.isfinite(a.seconds) or a.seconds<=0):
        p.error('Invalid field/time; duration must be finite and positive')
    lock_path=ROOT/'runs/improvement-12h/simulator.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as simulator_lock:
        fcntl.flock(simulator_lock,fcntl.LOCK_EX)
        config,world=prepare(a)
        config_data=json.loads(config.read_text());ledger=Ledger(ROOT/'runs/improvement-12h/ledger.jsonl')
        attempt_id=a.attempt_id or a.output.name
        scenario=scenario_identity(a)
        identity={'attempt_id':attempt_id,'candidate_id':a.candidate_id,'scenario_id':digest(scenario)[:20],
                  'split':a.split,'stage':a.stage,'comparison_key':digest(scenario),
                  'scenario_sha256':digest(scenario),'rules_sha256':environment_rules_hash(config_data['rules']),
                  'snapshot_sha256':config_data.get('snapshot_sha256'),'expected_entrants':1 if a.scenario=='stopped' else a.cars,
                  'is_hazard':a.scenario=='stopped','tier':str(a.cars),'output_dir':str(a.output.resolve())}
        ledger.append('attempt_scheduled',identity)
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        command=[str(ROOT/'.local/Webots.app/Contents/MacOS/webots'),'--stdout','--stderr','--mode=fast',
                 '--batch','--minimize','--no-rendering',str(world)]
        print(f'Start {a.scenario} cars={a.cars} rotation={a.rotation}: {config}',flush=True)
        ledger.append('attempt_started',{'attempt_id':attempt_id})
        code=subprocess.call(command,env=env)
        outcome={'audit_verified':False,'snapshot_verified':False,'audit_issues':['engineering_unfrozen_or_no_evaluation']}
        if code==0:
            try:
                from evaluate_improvement_attempt import evaluate_attempt
                outcome=evaluate_attempt(a.output,a.source_root if config_data.get('snapshot_sha256') else None,
                                         config_data.get('snapshot_sha256'))
            except Exception as error:outcome['audit_issues']=[f'{type(error).__name__}: {error}']
            entrants=outcome.get('entrants',[])
            outcome['all_valid_finishes']=bool(entrants) and all(row.get('valid_finish') is True for row in entrants)
            outcome['all_clean_finishes']=bool(entrants) and all(row.get('clean_finish') is True for row in entrants)
            (a.output/'ledger_outcome.json').write_text(json.dumps(outcome,indent=2)+'\n')
        ledger.append('attempt_finished',{'attempt_id':attempt_id,'status':'completed' if code==0 else 'failed',
                                          'exit_code':code,'outcome':outcome})
        raise SystemExit(code)


if __name__=='__main__':main()
