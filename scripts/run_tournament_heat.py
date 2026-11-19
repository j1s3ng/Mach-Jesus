"""Prepare a reproducible racing grid and run one capacity trial or race heat."""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from race_lidar.hotlap_driver import RacingReference
from generate_tournament_world import generate


def prepare(output,count,reverse=False,capacity_seconds=None,perception=True,record_video=True,vision_url='http://127.0.0.1:8770',secondary_cars=0,physics_threads=1):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError(f'Choose a fresh heat output: {output}')
    output.mkdir(parents=True,exist_ok=True)
    reference=ROOT/'.local/f1-reference/reference.npz';path=RacingReference.from_npz(reference)
    policies=[('reference','Reference baseline',None),('neural_v1','Neural V1',ROOT/'runs/driver-training/model.npz'),
              ('neural_v2','Neural V2',ROOT/'runs/driver-training-v2/model.npz'),('neural_v3','Neural V3',ROOT/'runs/driver-training-v3/model.npz')]
    entrants=[]
    order=list(range(count))
    if reverse:order.reverse()
    for grid_index,identity in enumerate(order):
        policy,label,model=policies[identity%4]
        rgb=colorsys.hsv_to_rgb(identity/max(count,1),.8,.95)
        entrants.append({'car_id':f'R{identity+1:02d}','policy':policy,'label':label,'grid_index':grid_index,
                         'model_path':str(model) if model else None,'start_progress_m':-12.*grid_index,
                         'start_lane_m':-2.5 if grid_index%2==0 else 2.5,
                         'color':' '.join(f'{x:.3f}' for x in rgb),'record_video':record_video})
        if secondary_cars and identity>=count-secondary_cars:
            entrants[-1]['vision_url']='http://127.0.0.1:8771'
    rules={'vehicle':'same generic Webots Car','body_mass_kg':800,'wheel_mass_kg_each':20,'tire_friction':1.5,
           'max_speed_mps':60,'lateral_accel_mps2':8,'brake_decel_mps2':8,'acceleration_mps2':5,
           'sensor_period_ms':200,'physics_step_ms':20,'vision_checkpoint':'official nuScenes VoxelNeXt kernel1','laps':1,
           'lateral_accel_budget_mps2':8,'friction_coefficient':1.5,'vehicle_model':'generic Webots Car 880kg electric',
           'start':'common standing-start clock; 12m stagger; shared finish line','reference_length_m':path.length,
           'physics_threads':physics_threads}
    config={'output_dir':str(output),'reference_path':str(reference),'reference_length_m':path.length,
            'entrants':entrants,'cars':{e['car_id']:e for e in entrants},'start_time_s':1.,'laps':1,
            'max_sim_seconds':480.,'sensor_period_ms':200,'video_period_ms':200,'record_video':record_video,
            'perception_enabled':perception,'vision_url':vision_url if perception else '',
            'mode':'capacity' if capacity_seconds is not None else 'race','rules':rules,
            'physics_threads':physics_threads,
            'max_speed_mps':60.,'lateral_accel_mps2':8.,'brake_decel_mps2':8.,'acceleration_mps2':5.}
    if capacity_seconds is not None:config['capacity_seconds']=float(capacity_seconds)
    config_path=output/'config.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
    world=ROOT/'webots/worlds'/f'tournament_{count}.wbt';generate(config_path,world)
    return config_path,world


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cars',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reverse-grid',action='store_true');p.add_argument('--capacity-seconds',type=float)
    p.add_argument('--physics-only',action='store_true');p.add_argument('--no-video',action='store_true')
    p.add_argument('--vision-url',default='http://127.0.0.1:8770')
    p.add_argument('--secondary-cars',type=int,default=0,help='Send this many identities to the RTX3090 on port8771')
    p.add_argument('--physics-threads',type=int,default=1)
    p.add_argument('--visible',action='store_true')
    a=p.parse_args()
    if not 2<=a.cars<=32:p.error('--cars must be between 2 and 32')
    if not 0<=a.secondary_cars<=a.cars:p.error('--secondary-cars must fit within the field')
    if not 1<=a.physics_threads<=16:p.error('--physics-threads must be between 1 and 16')
    config,world=prepare(a.output,a.cars,a.reverse_grid,a.capacity_seconds,not a.physics_only,not a.no_video,a.vision_url,a.secondary_cars,a.physics_threads)
    for name in ('tournament_racer','tournament_director'):
        directory=ROOT/'webots/controllers'/name;directory.mkdir(parents=True,exist_ok=True)
        (directory/'runtime.ini').write_text(f'[python]\nCOMMAND = {sys.executable}\n')
    command=[str(ROOT/'.local/Webots.app/Contents/MacOS/webots'),'--stdout','--stderr',
             '--mode=realtime' if a.visible else '--mode=fast']
    if not a.visible:command+=['--batch','--minimize','--no-rendering']
    command.append(str(world))
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    raise SystemExit(subprocess.call(command,env=env))


if __name__=='__main__':main()
