"""Audit full requested LiDAR-stream coverage in a closed capacity opener.

Short capacity attempts are not full laps. Every scheduled car must retain its
entire configured simulated-time stream; a reduced active field cannot pass.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from scripts.evaluate_improvement_attempt import evaluate_attempt


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_summary(records,*,start_s,duration_s,period_ms):
    period_ns=int(period_ms*1e6);expected=int(round(duration_s*1e9/period_ns));issues=[]
    times=np.asarray([r['timestamp_ns'] for r in records],dtype=np.int64)
    if len(records)!=expected:issues.append('incomplete_full_window_stream_count')
    if len({r['frame_id']for r in records})!=len(records):issues.append('duplicate_frame_id')
    if any(not isinstance(r.get('points'),int)or r['points']<=0 for r in records):issues.append('missing_measured_points')
    if any(not isinstance(r.get('inference_ms'),(int,float))or not math.isfinite(r['inference_ms'])or r['inference_ms']<0 for r in records):issues.append('invalid_RPC_clock')
    tolerance_ns=1000
    if len(times)>1 and not np.all(np.abs(np.diff(times)-period_ns)<=tolerance_ns):issues.append('cadence_gap_or_nonmonotonic_clock')
    if not len(times):issues.append('no_authored_sensor_frames')
    else:
        if times[0]>round(start_s*1e9)+period_ns+tolerance_ns:issues.append('late_stream_start')
        if times[-1]<round((start_s+duration_s)*1e9)-period_ns-tolerance_ns:issues.append('early_stream_termination')
    latencies=[r['inference_ms']for r in records if isinstance(r.get('inference_ms'),(int,float))and math.isfinite(r['inference_ms'])]
    return {'verified':not issues,'issues':issues,'expected_frames':expected,'observed_frames':len(records),
            'first_time_s':float(times[0]/1e9)if len(times)else None,'last_time_s':float(times[-1]/1e9)if len(times)else None,
            'rpc_roundtrip_ms_p50':float(np.median(latencies))if latencies else None,
            'rpc_roundtrip_ms_p95':float(np.quantile(latencies,.95))if latencies else None,
            'rpc_over_sensor_period':sum(v>period_ms for v in latencies)}


def audit(run,*,independent=True):
    run=Path(run).resolve();sources={}
    def read(path):sources[str(path)]=sha(path);return json.loads(path.read_text())
    config=read(run/'config.json');heat=read(run/'heat_results.json')
    if config['mode']!='capacity' or config['scenario']!='race':raise ValueError('Only an opening capacity race is supported')
    source=Path(config['source_root']);snapshot=config['snapshot_sha256']
    outcome=evaluate_attempt(run,source,snapshot)if independent else read(run/'ledger_outcome.json')
    issues=[]
    if not outcome.get('audit_verified')or not outcome.get('snapshot_verified'):issues.append('independent_physics_source_audit_failed')
    duration=float(config['capacity_seconds']);period=int(config['sensor_period_ms']);entrants=config['entrants']
    if len(entrants)!=int(heat['car_count'])or len({e['car_id']for e in entrants})!=len(entrants):issues.append('scheduled_actor_inventory_mismatch')
    if abs(heat['simulation_seconds']-duration)>.021:issues.append('supervisor_closed_before_capacity_window')
    cars={};latencies=[];urls={}
    for entry in entrants:
        identity=entry['car_id'];folder=run/'cars'/identity;result_path=folder/'result.json';pred_path=folder/'predictions.jsonl'
        if not result_path.is_file()or not pred_path.is_file():
            cars[identity]={'verified':False,'issues':['missing_authored_actor_files']};issues.append(identity+':missing_authored_actor_files');continue
        result=read(result_path);sources[str(pred_path)]=sha(pred_path)
        records=[json.loads(line)for line in pred_path.read_text().splitlines()if line.strip()]
        summary=stream_summary(records,start_s=float(config['start_time_s']),duration_s=duration,period_ms=period)
        if any(row.get('car_id')!=identity for row in records):summary['issues'].append('prediction_actor_identity_differs')
        if result.get('sensor_frames')!=len(records):summary['issues'].append('summary_frame_count_differs')
        if result.get('failure'):summary['issues'].append('controller_failure')
        if result.get('physics_step_ms')!=20 or result.get('sensor_period_ms')!=period:summary['issues'].append('physics_sensor_cadence_differs')
        expected_url=entry.get('vision_url',config['vision_url'])
        if result.get('vision_url')!=expected_url:summary['issues'].append('GPU_endpoint_assignment_differs')
        summary.update(verified=not summary['issues'],vision_url=result.get('vision_url'),controller_reason=result.get('reason'),
                       track_violation_steps=result.get('track_violation_steps'),position_resets_after_spawn=result.get('position_resets_after_spawn'))
        cars[identity]=summary
        issues.extend(identity+':'+item for item in summary['issues'])
        urls[expected_url]=urls.get(expected_url,0)+1
        latencies.extend(row['inference_ms']for row in records if isinstance(row.get('inference_ms'),(int,float))and math.isfinite(row['inference_ms']))
    active=[row for row in cars.values()if row.get('first_time_s')is not None]
    overlap_start=max((row['first_time_s']for row in active),default=None)
    overlap_end=min((row['last_time_s']for row in active),default=None)
    clean=(not heat.get('collision_pair_samples') and all(row.get('track_violation_steps')==0 and row.get('position_resets_after_spawn')==0 for row in cars.values()))
    sources[str(Path(__file__).resolve())]=sha(__file__)
    if any(sha(path)!=value for path,value in sources.items()):raise ValueError('Closed capacity evidence changed during audit')
    return {'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'role':'MEASURED_CAPACITY_OPENER_NOT_FULL_RACE',
            'full_requested_stream_window_verified':not issues,'issues':sorted(set(issues)),'scheduled_cars':len(entrants),
            'configured_sensor_period_ms':period,'configured_physics_step_ms':20,'simulation_seconds':duration,
            'observed_wall_seconds':heat['wall_seconds'],'real_time_factor':heat['real_time_factor'],
            'successful_frames':sum(row.get('observed_frames',0)for row in cars.values()),
            'common_all_actor_observed_interval_s':[overlap_start,overlap_end]if len(active)==len(entrants)else None,
            'configured_and_actual_endpoint_counts':urls,'clean_opening_window':clean,'per_car':cars,
            'rpc_roundtrip_ms_p50':float(np.median(latencies))if latencies else None,
            'rpc_roundtrip_ms_p95':float(np.quantile(latencies,.95))if latencies else None,
            'RPC_over_sensor_period_count':sum(value>period for value in latencies),'source_sha256':sources,
            'independent_audit_executed':independent,'all_valid_lap_claim':False,
            'limits':['Every configured car counts; early termination/dropped streams fail capacity coverage.',
                      '5Hz is simulated-time cadence. RTF below1 is not wall-time real-time performance.',
                      'RPC clocks include serialization/network/queue/compute; they are not GPU-only inference.',
                      'Traffic, sensing, physics and control share the workload; no global hardware maximum is established.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stored-audit',action='store_true',help='Descriptive legacy audit; skips a new independent physics audit')
    a=p.parse_args()
    if a.output.exists():p.error('Use a fresh audit output')
    result=audit(a.run,independent=not a.stored_audit);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k]for k in ['full_requested_stream_window_verified','scheduled_cars','successful_frames','real_time_factor','rpc_roundtrip_ms_p95','issues']},indent=2))


if __name__=='__main__':main()
