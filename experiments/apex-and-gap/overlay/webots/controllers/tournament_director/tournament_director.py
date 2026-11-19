"""Race timing, position, contact audit and finish supervision; never car control."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from controller import Supervisor
from race_lidar.hotlap_driver import RacingReference
from race_lidar.demo_chase import ChaseStop


def atomic_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def overlap(ca,ya,cb,yb):
    axes=[np.array([math.cos(a),math.sin(a)]) for a in (ya,ya+math.pi/2,yb,yb+math.pi/2)]
    delta=cb-ca
    a=[axes[0],axes[1]];b=[axes[2],axes[3]]
    for axis in axes:
        ra=2.5*abs(np.dot(a[0],axis))+1.05*abs(np.dot(a[1],axis))
        rb=2.5*abs(np.dot(b[0],axis))+1.05*abs(np.dot(b[1],axis))
        if abs(np.dot(delta,axis))>=ra+rb:return False
    return True


def project(reference,xy,previous):
    # Ordered local projection handles negative grid positions and lap wrapping.
    count=len(reference.xy);cycle=math.floor(previous/reference.length)
    wrapped=previous%reference.length
    base=max(0,int(np.searchsorted(reference.s,wrapped,side='right')-1))+cycle*count
    indices=np.arange(base-8,base+15)
    ids=indices%count
    segments=reference.segment[ids]
    frac=np.clip(np.sum((xy-reference.xy[ids])*segments,axis=1)/(reference.ds[ids]**2),0,1)
    projected=reference.xy[ids]+frac[:,None]*segments
    distance=np.linalg.norm(xy-projected,axis=1)
    s=(indices//count)*reference.length+reference.s[ids]+frac*reference.ds[ids]
    allowed=(s>=previous-4)&(s<=previous+6)
    if not allowed.any():return previous,float(np.min(distance))
    distance[~allowed]=np.inf;k=int(np.argmin(distance))
    return max(previous,float(s[k])),float(distance[k])


def main():
    config=json.loads(Path(sys.argv[1]).read_text());run=Path(config['output_dir'])
    supervisor=Supervisor();dt=int(supervisor.getBasicTimeStep())
    reference=RacingReference.from_npz(config['reference_path'])
    entrants=config['entrants'];nodes=[supervisor.getFromDef(f'RACER_{i}') for i in range(len(entrants))]
    progress={e['car_id']:e['start_progress_m'] for e in entrants}
    collisions={e['car_id']:0 for e in entrants};pair_counts={}
    max_errors={e['car_id']:0. for e in entrants}
    tyre_config = config.get('tyre_track_limits', {})
    tyre_road = None
    tyre_audit = {}
    open_tyre_clusters = {}
    if tyre_config.get('enabled'):
        from race_lidar.road_tyre_bounds import RoadTyreBounds
        if int(tyre_config.get('minimum_tyres_on_track', 2)) != 2:
            raise ValueError('This engineering rule requires two wheel contact centers on track')
        tyre_road = RoadTyreBounds.from_npz(tyre_config['geometry_path'], reference=reference,
            expected_sha256=tyre_config['geometry_sha256'])
        tyre_audit = {e['car_id']: dict(samples=0, minimum_on_track_count=4,
            two_tyres_on_track_steps=0, three_tyres_on_track_steps=0,
            tyre_limit_violation_steps=0, patch_margin_violation_steps=0,
            diagonal_only_two_center_steps=0, first_violation=None,
            violation_clusters=[]) for e in entrants}
    wall=time.perf_counter();tick=0;start=float(config.get('start_time_s',1.))
    limit=float(config.get('capacity_seconds',config.get('max_sim_seconds',480.)))
    status='time_limit';last_print=-1
    all_terminal_since = None
    chase = ChaseStop(config['stop_when_hero_leads'], config['cars']['R20']['start_progress_m']) if config.get('stop_when_hero_leads') else None
    chase_observation = None
    board=(run/'leaderboard.jsonl').open('x')
    gt=(run/'gt_poses.jsonl').open('x') if config.get('record_ground_truth',False) else None
    try:
        while supervisor.step(dt)!=-1:
            elapsed=max(0,supervisor.getTime()-start);tick+=1
            rows=[];footprints=[];gt_cars=[]
            for e,node in zip(entrants,nodes):
                xy=np.asarray(node.getPosition())[:2];r=np.asarray(node.getOrientation()).reshape(3,3)
                yaw=math.atan2(r[1,0],r[0,0]);center=xy+r[:2,:2]@np.array([1.6,0])
                car_id=e['car_id'];progress[car_id],error=project(reference,xy,progress[car_id])
                if gt is not None and tick%10==0:
                    gt_cars.append({'car_id':car_id,'position_m':list(node.getPosition()),'rotation':r.ravel().tolist(),
                                    'box_center_local_m':[1.6,0.,.1875],'size_m':[4.95,2.02,1.075]})
                max_errors[car_id]=max(max_errors[car_id],error)
                result_path=run/'cars'/car_id/'result.json'
                result=None
                if result_path.exists():
                    try:result=json.loads(result_path.read_text())
                    except (ValueError,OSError):pass
                rows.append({'car_id':car_id,'policy':e['policy'],'label':e['label'],'grid_index':e['grid_index'],
                             'progress_m':progress[car_id],'position_m':xy.tolist(),'color':e['color'],
                             'finished':bool(result and result.get('completed') and result.get('valid_lap')),
                             'terminal':bool(result and result.get('status') not in (None,'running','initializing')),
                             'result_status':result.get('status') if result else 'initializing',
                             'finish_time_s':result.get('finish_time_s') if result else None})
                footprints.append((car_id,center,yaw,not rows[-1]['terminal']))
                if tyre_road is not None and supervisor.getTime() >= start and not rows[-1]['terminal']:
                    contact = tyre_road.tyre_contact_status(xy, yaw, progress_m=progress[car_id])
                    audit = tyre_audit[car_id]
                    count = contact['on_track_count']
                    audit['samples'] += 1
                    audit['minimum_on_track_count'] = min(audit['minimum_on_track_count'], count)
                    audit['two_tyres_on_track_steps'] += int(count == 2)
                    audit['three_tyres_on_track_steps'] += int(count == 3)
                    audit['tyre_limit_violation_steps'] += int(count < 2)
                    audit['patch_margin_violation_steps'] += int(contact['patch_margin_on_track_count'] < 2)
                    audit['diagonal_only_two_center_steps'] += int(contact['diagonal_only_two_centers'])
                    if count < 2 and audit['first_violation'] is None:
                        audit['first_violation'] = dict(race_time_s=elapsed,
                            simulation_time_s=supervisor.getTime(), position_m=xy.tolist(),
                            yaw_rad=yaw, progress_m=progress[car_id], contact=contact)
                    if count < 2:
                        if car_id not in open_tyre_clusters:
                            open_tyre_clusters[car_id] = dict(first_race_time_s=elapsed,
                                first_progress_m=progress[car_id], samples=0,
                                minimum_on_track_count=count)
                        cluster = open_tyre_clusters[car_id]
                        cluster.update(last_race_time_s=elapsed, last_progress_m=progress[car_id])
                        cluster['samples'] += 1
                        cluster['minimum_on_track_count'] = min(cluster['minimum_on_track_count'], count)
                    elif car_id in open_tyre_clusters:
                        cluster = open_tyre_clusters.pop(car_id)
                        cluster['duration_s'] = cluster['samples'] * dt / 1000.
                        audit['violation_clusters'].append(cluster)
            # Audit contact-sized envelopes on every physical step, separate from drivers.
            if supervisor.getTime()>=start:
                for i,(ida,ca,ya,active_a) in enumerate(footprints):
                    for idb,cb,yb,active_b in footprints[i+1:]:
                        if np.dot(cb-ca,cb-ca)<49 and overlap(ca,ya,cb,yb):
                            # Freeze each entrant's audit at its own finish/failure.
                            # Active racers still have to avoid a cooldown/stopped car.
                            collisions[ida]+=int(active_a);collisions[idb]+=int(active_b)
                            key=f'{ida}:{idb}';pair_counts[key]=pair_counts.get(key,0)+1
            rows.sort(key=lambda r:(0 if r['finished'] else 2 if r['terminal'] else 1,
                                    r['finish_time_s'] if r['finished'] else -r['progress_m']))
            for rank,row in enumerate(rows,1):row['position']=rank
            if chase is not None:
                chase_observation = chase.update(rows, elapsed, supervisor.getTime())
            if tick%10==0:
                board.write(json.dumps({'time_s':elapsed,'simulation_time_s':supervisor.getTime(),'cars':rows})+'\n');board.flush()
                atomic_json(run/'live_leaderboard.json',{'time_s':elapsed,'cars':rows})
                if gt is not None:
                    gt.write(json.dumps({'timestamp_ns':round(supervisor.getTime()*1e9),'cars':gt_cars})+'\n');gt.flush()
            if int(elapsed)//20!=last_print:
                last_print=int(elapsed)//20
                leaders=', '.join('{}:{} {:.0f}m'.format(r['position'],r['car_id'],r['progress_m']) for r in rows[:4])
                print(f'Race {elapsed:.1f}s | '+leaders,flush=True)
            operator_stop = config.get('operator_stop_file')
            if operator_stop and Path(operator_stop).is_file():
                if tick%10!=0:
                    board.write(json.dumps({'time_s':elapsed,'simulation_time_s':supervisor.getTime(),'cars':rows})+'\n');board.flush()
                status = 'operator_stop'; break
            if chase_observation is not None and chase_observation['stop_reason']:
                if tick%10!=0:
                    board.write(json.dumps({'time_s':elapsed,'simulation_time_s':supervisor.getTime(),'cars':rows})+'\n');board.flush()
                    atomic_json(run/'live_leaderboard.json',{'time_s':elapsed,'simulation_time_s':supervisor.getTime(),'cars':rows})
                status = chase_observation['stop_reason']; break
            result_paths=[run/'cars'/e['car_id']/'result.json' for e in entrants]
            if config.get('mode')!='capacity' and all(p.exists() for p in result_paths):
                try:terminal=all(json.loads(p.read_text()).get('status') not in (None,'running','initializing') for p in result_paths)
                except (OSError,ValueError):terminal=False
                if terminal:
                    if all_terminal_since is None:all_terminal_since=supervisor.getTime()
                    if supervisor.getTime()-all_terminal_since>=.3:
                        status='all_cars_reported';break
            if elapsed>=limit:break
    finally:
        board.close()
        if gt is not None:gt.close()
    results=[]
    for e in entrants:
        p=run/'cars'/e['car_id']/'result.json'
        if p.exists():
            try:result=json.loads(p.read_text())
            except (ValueError,OSError):result={}
        else:result={}
        result.update(car_id=e['car_id'],policy=e['policy'],label=e['label'],grid_index=e['grid_index'],
                      progress_m=progress[e['car_id']],collision_samples=collisions[e['car_id']],
                      director_maximum_reference_error_m=max_errors[e['car_id']])
        result.setdefault('completed',False);result.setdefault('status','timeout' if config.get('mode')!='capacity' else 'capacity_only')
        result.setdefault('finish_time_s',None)
        results.append(result)
    elapsed=max(0,supervisor.getTime()-start);duration=time.perf_counter()-wall
    report={'status':status,'mode':config.get('mode','race'),'car_count':len(entrants),
            'simulation_seconds':elapsed,'wall_seconds':duration,'real_time_factor':elapsed/duration if duration else None,
            'cars':results,'collisions_by_car':collisions,'collision_pair_samples':pair_counts,'rules':config.get('rules',{})}
    if tyre_road is not None:
        for car_id, cluster in open_tyre_clusters.items():
            cluster['duration_s'] = cluster['samples'] * dt / 1000.
            tyre_audit[car_id]['violation_clusters'].append(cluster)
        report['tyre_track_limits_audit'] = dict(
            definition='At least two of four actual wheel contact centers inside the exact generated tarmac/white-line strip',
            kerbs_count_as_track=False, physics_step_ms=dt, minimum_required_count=2,
            planning_patch_margin_m=tyre_road.planning_patch_margin_m,
            geometry_sha256=tyre_road.geometry_sha256, cars=tyre_audit,
            reference_distance_is_not_the_track_limit=True,
            full_tyre_contact_physics_claimed=False)
    if chase_observation is not None:
        chase_observation = {**chase_observation, 'stop_reason': status,
            'lead_achieved': status == 'hero_took_lead', 'collisions_by_car': collisions,
            'collision_pair_samples': pair_counts}
        report['chase'] = chase_observation
        atomic_json(run/'chase_result.json', chase_observation)
    atomic_json(run/'heat_results.json',report)
    print(json.dumps({'status':status,'cars':len(entrants),'real_time_factor':report['real_time_factor']}),flush=True)
    supervisor.simulationQuit(0)


if __name__=='__main__':main()
