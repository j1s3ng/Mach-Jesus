"""Run a resumable physical-evaluation queue until the campaign deadline.

The parent agent adds frozen candidates and jobs as evidence arrives. Every job
is isolated in its own process group and has a bounded wall-clock timeout.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'runs/improvement-12h'
sys.path.insert(0,str(ROOT))
from scripts.queue_detector_preflight import DetectorPreflight


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def detector_guarded_trial(job, checker, deadline, run_trial):
    """The only trial-dispatch gate; refusal has no physical-output side effect."""
    report=checker.check(job,deadline=min(deadline-300,time.time()+45))
    if not report['ready']:
        return None,report
    if time.time()+job.get('estimated_wall_seconds',900)+300>deadline:
        report.update(ready=False,error='deadline_reserve_after_detector_preflight')
        return None,report
    return run_trial(report),report


def execute_trial(job,state,state_path,deadline,preflight):
    output=ROOT/job['output']
    command=[sys.executable,str(ROOT/'scripts/run_improvement_heat.py'),'--output',str(output),
             '--attempt-id',job['job_id'],*job['args']]
    log_path=CAMPAIGN/'logs'/f"{job['job_id']}.log";log_path.parent.mkdir(exist_ok=True)
    proof_path=CAMPAIGN/'detector_preflight'/f"{job['job_id']}.json"
    state.pop('pending_detector_job',None)
    state.update(status='running',current=job['job_id'],current_output=str(output),command=command,
                 detector_preflight=str(proof_path))
    write(state_path,state);print(f"START {job['job_id']}",flush=True)
    start=time.time();timeout=min(job.get('max_wall_seconds',3600),deadline-time.time()-300)
    with log_path.open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        preflight.update(physical_trial_launched=True,launched_epoch_s=time.time(),child_pid=process.pid)
        write(proof_path,preflight)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=20)
            code=124
    return {'status':'completed' if code==0 else 'failed','exit_code':code,
            'wall_seconds':time.time()-start,'output':str(output),'log':str(log_path),
            'detector_preflight':str(proof_path)}


def main():
    owner=(CAMPAIGN/'queue.lock').open('a+')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=json.loads((CAMPAIGN/'campaign.json').read_text())
    deadline=dt.datetime.fromisoformat(campaign['deadline_utc']).timestamp()
    state_path=CAMPAIGN/'queue_status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'completed':{},'current':None}
    if state.get('current'):
        state['interrupted_previous_job']=state['current'];state['current']=None
    write(CAMPAIGN/'queue_process.json',{'pid':os.getpid(),'started_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    checker=DetectorPreflight(ROOT,CAMPAIGN)
    detector_waits=0
    while time.time()<deadline-300:
        jobs=json.loads((CAMPAIGN/'queue.json').read_text()) if (CAMPAIGN/'queue.json').exists() else []
        pending=[j for j in jobs if j['job_id'] not in state['completed']]
        state['heartbeat_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        if not pending:
            state['status']='waiting_for_next_candidate';write(state_path,state);time.sleep(10);continue
        job=pending[0]
        failed_dependency=None
        for requirement in job.get('requires',[]):
            parent=next((j for j in jobs if j['job_id']==requirement['job_id']),None)
            artifact=ROOT/parent['output']/'ledger_outcome.json' if parent else None
            result=json.loads(artifact.read_text()) if artifact and artifact.exists() else {}
            if any(result.get(key) is not True for key in requirement.get('true_fields',['audit_verified'])):
                failed_dependency=requirement['job_id'];break
        if failed_dependency:
            state['completed'][job['job_id']]={'status':'skipped_failed_dependency','dependency':failed_dependency}
            write(state_path,state);continue
        if time.time()+job.get('estimated_wall_seconds',900)+300>deadline:
            state['status']='deadline_reserve';write(state_path,state);break
        output=ROOT/job['output']
        if output.exists() and (output/'heat_results.json').exists():
            state['completed'][job['job_id']]={'status':'recovered_completed_output','output':str(output)}
            write(state_path,state);continue
        if output.exists() and any(output.iterdir()):
            state['completed'][job['job_id']]={'status':'interrupted_output_requires_audit','output':str(output)}
            write(state_path,state);continue
        result,proof=detector_guarded_trial(job,checker,deadline,
            lambda evidence:execute_trial(job,state,state_path,deadline,evidence))
        if result is None:
            detector_waits+=1
            delay=min(30.,2.**min(detector_waits,5))
            state.update(status='waiting_for_detector',current=None,pending_detector_job=job['job_id'],
                         detector_preflight=proof,detector_wait_attempts=detector_waits,next_detector_check_s=delay)
            state.pop('current_output',None);state.pop('command',None)
            write(state_path,state)
            print(f"WAIT_DETECTOR {job['job_id']} attempt={detector_waits}",flush=True)
            time.sleep(min(delay,max(0.,deadline-time.time()-300)))
            continue
        detector_waits=0
        state['completed'][job['job_id']]=result
        state['current']=None;state.pop('pending_detector_job',None)
        write(state_path,state);print(f"END {job['job_id']} exit={result['exit_code']}",flush=True)
    state.update(status='deadline_reached',current=None);write(state_path,state)


if __name__=='__main__':main()
