"""Create identical physical cars with independent racing controllers/sensors."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from generate_hotlap_world import build,physical_car,reference_spawn
from race_lidar.hotlap_driver import RacingReference


def remove_node(source,marker):
    start=source.index(marker)
    opening=source.index('{',start)
    depth=0;quoted=False;escaped=False
    for i in range(opening,len(source)):
        ch=source[i]
        if escaped:escaped=False;continue
        if ch=='\\' and quoted:escaped=True;continue
        if ch=='"':quoted=not quoted;continue
        if quoted:continue
        if ch=='{':depth+=1
        elif ch=='}':
            depth-=1
            if depth==0:return source[:start]+source[i+1:]
    raise ValueError('Unclosed source vehicle node')


def generate(config_path,output):
    config_path=Path(config_path).resolve();output=Path(output).resolve()
    config=json.loads(config_path.read_text())
    reference_path=Path(config['reference_path'])
    base=output.with_name('.'+output.stem+'_track.wbt')
    build(reference_path,base,backmarkers=0, decorative_trackside_posts_enabled=
          config.get("decorative_trackside_posts_enabled", True))
    source=remove_node(base.read_text(),'DEF EGO Car')
    source=source.replace('basicTimeStep 20 coordinateSystem',
                          f'optimalThreadCount {int(config.get("physics_threads",1))} basicTimeStep 20 coordinateSystem')
    source=source.replace('basicTimeStep 20 coordinateSystem',
                          f'randomSeed {int(config.get("seed",0))} basicTimeStep 20 coordinateSystem')
    source=source.replace('follow "ego"','follow "'+config['entrants'][0]['car_id']+'"')
    reference=RacingReference.from_npz(reference_path)
    vehicles=[]
    for i,entrant in enumerate(config['entrants']):
        point,yaw=reference_spawn(reference,entrant['start_progress_m'],entrant['start_lane_m'])
        node=physical_car(entrant['car_id'],f'RACER_{i}',point,yaw,entrant['color'],
                          entrant.get('controller','tournament_racer'),[entrant['car_id'],str(config_path)],
                          sensors=entrant.get('sensors',True))
        node=node.replace('HOTLAP_LIDAR',f'RACER_{i}_LIDAR').replace('HOTLAP_CAMERA',f'RACER_{i}_CAMERA')
        vehicles.append(node)
    director=f'''Robot {{ name "race_director" controller "tournament_director" supervisor TRUE window "<none>"
      controllerArgs [ {json.dumps(str(config_path))} ] }}'''
    output.write_text(source+'\n'+'\n'.join(vehicles)+'\n'+director+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(generate(a.config,a.output))
