#!/usr/bin/env python3
"""Re-audit closed campaign trials and apply complete-suite gates until deadline.

Never launches races, changes queues, changes frozen inputs, or overwrites a
recorded outcome. New evaluations and status reports live in evaluation/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_improvement_attempt import evaluate_attempt
from scripts.generate_improvement_cases import verify_case
from scripts.improvement_ledger import (Ledger,candidate_descriptor,compare_candidates,digest,
    entrant_descriptor,environment_rules_hash,validate_bootstrap,verify_snapshot)

CODE_FILES=(Path(__file__),ROOT/'scripts/improvement_ledger.py',ROOT/'scripts/evaluate_improvement_attempt.py',
            ROOT/'src/race_lidar/race_results.py')

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def instant(text):return dt.datetime.fromisoformat(text.replace('Z','+00:00'))
def read(path,default=None):return json.loads(Path(path).read_text()) if Path(path).exists() else default
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(value if isinstance(value,str) else json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
def fingerprints():return {str(p.relative_to(ROOT)):sha(p) for p in CODE_FILES}


def substantive(outcome):
    """Ignore convenience totals, presentation strings, and evaluator bookkeeping."""
    keys=('car_id','valid_finish','clean_finish','finish_time_s','progress_m','collision_samples',
          'track_violation_steps','position_resets_after_spawn')
    entries=[{**{key:row.get(key) for key in keys},'descriptor':entrant_descriptor(row)}
             for row in outcome.get('entrants',[])]
    return {**{key:outcome.get(key) for key in ('audit_verified','snapshot_verified','snapshot_sha256',
            'rules_sha256','homogeneous_field','hazard_pass')},
            'entrants':sorted(entries,key=lambda row:str(row['car_id']))}


def world_issues(config,case):
    """Independently check that recorded starts implement the canonical case."""
    import numpy as np
    scenario=case['scenario'];rules=config.get('rules',{});issues=[]
    checks={'scenario':config.get('scenario'),'cars':len(config['entrants']),
            'rotation':config.get('grid_rotation'),'seed':config.get('seed'),
            'spacing':rules.get('grid_spacing_m'),'seconds':rules.get('race_cutoff_seconds'),
            'grid_jitter':rules.get('grid_jitter_m')}
    for key,value in checks.items():
        if value!=scenario[key]:issues.append('world_mismatch:'+key)
    if config.get('dataset_split')!=case['split'] or config.get('excluded_from_training') is not True:
        issues.append('evaluation_training_exclusion_missing')
    rng=np.random.default_rng(scenario['seed'])
    for grid,entry in enumerate(config['entrants']):
        progress=-scenario['spacing']*grid-float(rng.uniform(0.,scenario['grid_jitter']))
        lane=(-2.5 if grid%2==0 else 2.5)+float(rng.uniform(-scenario['grid_jitter'],scenario['grid_jitter']))
        if scenario['scenario']=='stopped':
            progress=scenario['ego_progress'] if grid==0 else scenario['obstacle_progress']
            lane=scenario['ego_lane'] if grid==0 else (0. if grid==1 else (-3.2 if scenario['blocked_side']=='right' else 3.2))
        if entry.get('car_id')!=f'R{grid+1:02d}' or entry.get('grid_index')!=grid:
            issues.append('grid_identity_mismatch')
        if abs(entry['start_progress_m']-progress)>1e-8 or abs(entry['start_lane_m']-lane)>1e-8:
            issues.append('grid_pose_mismatch:'+entry['car_id'])
        if entry.get('seed')!=scenario['seed']+grid:issues.append('driver_seed_mismatch')
    if scenario['scenario']=='stopped' and config.get('hazard',{}).get('progress_m')!=scenario['obstacle_progress']:
        issues.append('static_obstacle_progress_mismatch')
    return sorted(set(issues))


def physics_binding(manifest):
    files=[{key:row[key] for key in ('path','sha256')} for row in manifest['files']
           if row['path'].startswith('webots/protos/') or row['path'] in {
               'scripts/generate_hotlap_world.py','scripts/generate_tournament_world.py','reference.npz'}]
    return digest(files)


class CampaignEvaluation:
    def __init__(self,campaign,cases_path,baseline='reference-safe45',candidate='reference-fast70',
                 output_directory=None,publish_progress=True,bootstrap_only=False):
        self.campaign=Path(campaign).resolve();self.ledger=Ledger(self.campaign/'ledger.jsonl')
        self.cases_path=Path(cases_path).resolve();self.cases=read(self.cases_path)
        self.cases_sha256=sha(self.cases_path)
        if len(self.cases)!=15 or sum(bool(c['is_hazard']) for c in self.cases)!=3:
            raise ValueError('Formal holdout must contain the15 canonical cases, including3 hazards')
        for case in self.cases:verify_case(case)
        self.case_by_id={c['scenario_id']:c for c in self.cases}
        if len(self.case_by_id)!=15:raise ValueError('Canonical case identities must be unique')
        self.baseline,self.candidate=baseline,None if bootstrap_only else candidate
        if not self.baseline or (self.candidate is not None and self.baseline==self.candidate):
            raise ValueError('Baseline and comparison candidate must be distinct; use bootstrap_only for one baseline')
        self.bootstrap_only=bootstrap_only
        self.output=Path(output_directory).resolve() if output_directory else self.campaign/'evaluation'
        self.output.mkdir(parents=True,exist_ok=True);self.publish_progress=publish_progress
        self.code=fingerprints()
        self._bundle_signatures={}

    def deadline(self,state):
        run=state.get('run') or read(self.campaign/'campaign.json',{})
        return run.get('deadline_utc',run.get('deadline'))

    def audit_trial(self,attempt,registration,case):
        folder=Path(attempt['output_dir']);config=read(folder/'config.json')
        source=Path(registration['snapshot_path'])
        source_key=str(source.resolve())
        if source_key not in self._bundle_signatures:
            manifest=read(source/'manifest.json')
            source_stats={row['path']:((source/row['path']).stat().st_size,(source/row['path']).stat().st_mtime_ns)
                          for row in manifest['files']}
            self._bundle_signatures[source_key]=(manifest,{'manifest_sha256':sha(source/'manifest.json'),'files':source_stats})
        manifest,source_signature=self._bundle_signatures[source_key]
        files=[folder/'config.json',folder/'heat_results.json']
        for entry in config['entrants']:
            if entry.get('is_static'):continue
            base=folder/'cars'/entry['car_id']
            files.extend(base/name for name in ('result.json','controller_summary.json','control_trace.jsonl',
                                                'predictions.jsonl','policy_actions.jsonl'))
        stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) if p.exists() else None for p in files}
        identity=digest({'files':stats,'registration':registration,'case':case,'evaluation_sources':self.code,
                         'frozen_source_signature':source_signature,
                         'stored_outcome':substantive(attempt.get('outcome') or {})})
        cache=self.output/'reaudits'/(hashlib.sha256(attempt['attempt_id'].encode()).hexdigest()[:20]+'.json')
        cached=read(cache,{})
        if cached.get('input_identity')==identity:return cached
        descriptor=candidate_descriptor(registration)
        outcome=evaluate_attempt(folder,source,manifest['snapshot_sha256'],
            expected_rules_sha256=attempt['rules_sha256'],expected_policy=descriptor['policy_key'],
            expected_model_sha256=descriptor['model_sha256'],expected_detector_checkpoint_sha256=descriptor['detector_sha256'],
            expected_control_parameters=descriptor.get('control_parameters'))
        issues=world_issues(config,case)
        if substantive(outcome)!=substantive(attempt.get('outcome') or {}):issues.append('stored_outcome_differs_from_independent_reaudit')
        if not outcome['audit_verified'] or not outcome['snapshot_verified']:issues.extend(outcome['audit_issues'])
        if outcome['snapshot_sha256']!=attempt['snapshot_sha256']:issues.append('scheduled_snapshot_mismatch')
        if environment_rules_hash(config['rules'])!=attempt['rules_sha256']:issues.append('scheduled_environment_mismatch')
        result={'attempt_id':attempt['attempt_id'],'input_identity':identity,'audited_at':utc(),
                'verified':not issues,'issues':sorted(set(issues)),'outcome':outcome,
                'physics_binding_sha256':physics_binding(manifest),'input_file_stats':stats,
                'input_file_sha256':{str(p):sha(p) for p in files if p.exists()},
                'source_files':self.code,'historical_outcome_rewritten':False}
        atomic(cache,result);return result

    def inspect(self,state):
        audits={};arms={}
        for cid in ((self.baseline,) if self.bootstrap_only else (self.baseline,self.candidate)):
            attempts=[a for a in state['attempts'] if a.get('candidate_id')==cid and a.get('split')=='holdout'
                      and a.get('stage') in {'bootstrap','promotion'}]
            issues=[];seen=Counter(a.get('scenario_id') for a in attempts)
            registration=state['candidates'].get(cid,{})
            for attempt in attempts:
                if attempt['scenario_id'] not in self.case_by_id:
                    issues.append(attempt['attempt_id']+':unexpected_case');continue
                if attempt['state']!='finished':continue
                try:
                    result=self.audit_trial(attempt,registration,self.case_by_id[attempt['scenario_id']])
                except Exception as error:
                    result={'attempt_id':attempt['attempt_id'],'verified':False,
                            'issues':[f'{type(error).__name__}: {error}']}
                audits[attempt['attempt_id']]=result
                if not result['verified']:issues.append(attempt['attempt_id']+':reaudit_failed')
            complete=(set(seen)==set(self.case_by_id) and all(count==1 for count in seen.values())
                      and len(attempts)==15 and all(a['state']=='finished' and a.get('status') in {'completed','finished'} for a in attempts))
            ready=complete and not issues and all(audits.get(a['attempt_id'],{}).get('verified') for a in attempts)
            arms[cid]={'attempts':attempts,'scheduled_cases':len(attempts),
                       'closed_cases':sum(a['state']=='finished' for a in attempts),'required_cases':15,
                       'complete':complete,'independently_verified_complete_suite':ready,'issues':issues}
        bindings={r.get('physics_binding_sha256') for r in audits.values() if r.get('verified')}
        common_world=len(bindings)<=1
        return arms,audits,common_world

    def decisions(self,state,arms,audits,common_world,act):
        history=read(self.output/'decisions.json',{})
        evidence=digest({'arms':{cid:arm['attempts'] for cid,arm in arms.items()},'code':self.code,
                         'case_manifest_sha256':sha(self.cases_path),'audits':{key:value.get('input_identity') for key,value in audits.items()}})
        decisions={};deadline=self.deadline(state)
        within=deadline is not None and dt.datetime.now(dt.timezone.utc)<instant(deadline)
        if state.get('incumbent_id') is None and arms[self.baseline]['independently_verified_complete_suite']:
            preview=validate_bootstrap(state,self.baseline,self.cases,'holdout');decisions['bootstrap']=preview
            if act and within and history.get('bootstrap_evidence')!=evidence:
                decisions['bootstrap']=self.ledger.bootstrap_incumbent(self.baseline,self.cases,split='holdout')
                history.update(bootstrap_evidence=evidence,bootstrap=decisions['bootstrap']);atomic(self.output/'decisions.json',history)
                state=self.ledger.state()
        if not self.bootstrap_only and all(arm['independently_verified_complete_suite'] for arm in arms.values()):
            comparison=compare_candidates(state['attempts'],self.baseline,self.candidate,self.cases,'holdout',
                                          snapshot_registry=state['candidates'])
            if not common_world:
                comparison['approved']=False;comparison['issues'].append('common_world_assets_or_reference_changed')
            decisions['comparison']=comparison
            if (act and within and comparison['approved'] and state.get('incumbent_id')==self.baseline
                    and history.get('promotion_evidence')!=evidence):
                decisions['promotion']=self.ledger.promote(self.candidate,self.cases,split='holdout')
                history.update(promotion_evidence=evidence,promotion=decisions['promotion']);atomic(self.output/'decisions.json',history)
        return decisions

    def metrics(self,arm,audits):
        strata={str(n):{'scheduled_entrants':0,'closed_entrants':0,'valid':0,'clean':0,'times':[]} for n in (4,8,20)}
        hazards=[]
        for attempt in arm['attempts']:
            audit=audits.get(attempt['attempt_id'],{});verified=audit.get('verified',False)
            out=audit.get('outcome') or attempt.get('outcome') or {}
            if attempt.get('is_hazard'):
                hazards.append({'attempt_id':attempt['attempt_id'],'state':attempt['state'],
                                'verified':verified,'passed':out.get('hazard_pass') if verified else None});continue
            tier=str(attempt.get('tier'));bucket=strata.get(tier)
            if bucket is None:continue
            bucket['scheduled_entrants']+=attempt['expected_entrants']
            if attempt['state']=='finished':bucket['closed_entrants']+=attempt['expected_entrants']
            if verified:
                bucket['valid']+=sum(e['valid_finish'] is True for e in out['entrants'])
                bucket['clean']+=sum(e['clean_finish'] is True for e in out['entrants'])
                bucket['times'].extend(e['finish_time_s'] for e in out['entrants'] if e['valid_finish'])
        for value in strata.values():
            times=value.pop('times');value['mean_valid_lap_s']=statistics.mean(times) if times else None
        return {'strata':strata,'hazards':hazards,'verified_hazard_passes':sum(h['passed'] is True for h in hazards),
                'closed_hazards':sum(h['state']=='finished' for h in hazards)}

    def timing(self,attempts):
        intervals={}
        for record in self.ledger.records():
            if record['event_type'] not in {'background_postprocessing_started','background_postprocessing_finished'}:continue
            value=record['payload'];key=value.get('job_id',value.get('pid',record['sequence']))
            intervals[key]={**intervals.get(key,{}),**value}
        output=[]
        for attempt in attempts:
            folder=Path(attempt['output_dir']);path=folder/'heat_results.json'
            if not path.exists():continue
            data=read(path);wall=data.get('wall_seconds');simulation=data.get('simulation_seconds')
            if not isinstance(wall,(int,float)) or wall<=0:continue
            end=dt.datetime.fromtimestamp(path.stat().st_mtime,dt.timezone.utc);start=end-dt.timedelta(seconds=wall)
            overlaps=[]
            for key,value in intervals.items():
                begin=value.get('started_utc');finish=value.get('ended_utc')
                if begin and instant(begin)<end and (not finish or instant(finish)>start):overlaps.append(str(key))
            output.append({'attempt_id':attempt['attempt_id'],'simulation_seconds':simulation,'wall_seconds':wall,
                'simulation_to_wall_ratio':simulation/wall if simulation is not None else None,
                'background_postprocessing_jobs':overlaps,
                'wall_clock_context':'includes recorded background postprocessing' if overlaps else 'no recorded heavy postprocessing overlap',
                'interval_method':'supervisor report mtime minus its recorded wall duration; formal wall rate is not an unloaded capacity benchmark'})
        return output

    def paired_pace(self,arms,audits):
        if self.bootstrap_only:
            return {str(n):{'paired_clean_finishes':0,'median_candidate_to_baseline_ratio':None,
                            'maximum_candidate_to_baseline_ratio':None} for n in (4,8,20)}
        old={a['scenario_id']:a for a in arms[self.baseline]['attempts'] if not a.get('is_hazard')}
        new={a['scenario_id']:a for a in arms[self.candidate]['attempts'] if not a.get('is_hazard')}
        ratios=defaultdict(list)
        for case_id in old.keys() & new.keys():
            a,b=audits.get(old[case_id]['attempt_id'],{}),audits.get(new[case_id]['attempt_id'],{})
            if not a.get('verified') or not b.get('verified'):continue
            baseline={r['car_id']:r for r in a['outcome']['entrants']}
            for row in b['outcome']['entrants']:
                previous=baseline.get(row['car_id'])
                if previous and previous['clean_finish'] and row['clean_finish']:
                    ratios[str(old[case_id]['tier'])].append(row['finish_time_s']/previous['finish_time_s'])
        return {str(n):{'paired_clean_finishes':len(ratios[str(n)]),
                        'median_candidate_to_baseline_ratio':statistics.median(ratios[str(n)]) if ratios[str(n)] else None,
                        'maximum_candidate_to_baseline_ratio':max(ratios[str(n)]) if ratios[str(n)] else None}
                for n in (4,8,20)}

    def once(self,act=False):
        if sha(self.cases_path)!=self.cases_sha256:
            raise ValueError('Frozen canonical case manifest changed; no decision may use revised cases')
        self._bundle_signatures={}
        state=self.ledger.state();arms,audits,common_world=self.inspect(state)
        decisions=self.decisions(state,arms,audits,common_world,act);state=self.ledger.state()
        counts=Counter(a['state'] for a in state['attempts']);deadline=self.deadline(state)
        expired=deadline is not None and dt.datetime.now(dt.timezone.utc)>=instant(deadline)
        metrics={cid:self.metrics(arm,audits) for cid,arm in arms.items()}
        queue=read(self.campaign/'queue_status.json',{})
        report={'schema_version':1,'updated_utc':utc(),'deadline_utc':deadline,
            'status':'deadline_reached' if expired else 'watching','baseline_id':self.baseline,'candidate_id':self.candidate,
            'evaluation_mode':'baseline_bootstrap_only' if self.bootstrap_only else 'matched_holdout_comparison',
            'incumbent_id':state.get('incumbent_id'),'decisions_enabled':act,'confirmation_required':True,
            'geometric_confirmation_required':True,
            'common_physical_assets_verified':common_world,'canonical_cases_path':str(self.cases_path),
            'canonical_cases_sha256':sha(self.cases_path),'evaluation_source_sha256':self.code,
            'ledger_attempt_counts':dict(counts),'total_ledger_attempts':len(state['attempts']),
            'formal':{cid:{**{k:v for k,v in arm.items() if k!='attempts'},**metrics[cid]} for cid,arm in arms.items()},
            'paired_clean_pace':self.paired_pace(arms,audits),
            'reaudits':audits,'decisions':decisions,'historical_failures_rewritten':False,
            'all_attempt_inventory':[{key:a.get(key) for key in ('attempt_id','candidate_id','stage','split','state','status','performance_status','output_dir')}
                                     for a in state['attempts']],
            'timing':self.timing([a for arm in arms.values() for a in arm['attempts']]),
            'queue':{'status':queue.get('status'),'current':queue.get('current'),'current_output':queue.get('current_output'),
                     'heartbeat_utc':queue.get('heartbeat_utc'),'completed_jobs':len(queue.get('completed',{}))},
            'scope':('Complete15-case baseline holdout bootstrap only; no comparison candidate or promotion is enabled. '
                     'A single finalist chosen from engineering/CEM only will require the unchanged full paired fresh confirmation suite.'
                     if self.bootstrap_only else
                     'Complete matched holdout evidence only. Fresh confirmation remains required; no claim of general racing superiority.')}
        atomic(self.output/'report.json',report)
        text=self.markdown(report);atomic(self.output/'REPORT.md',text)
        progress={key:value for key,value in report.items() if key not in {'reaudits','all_attempt_inventory','timing'}}
        progress['report_path']=str(self.output/'REPORT.md')
        progress['latest_video']=str(self.campaign/'milestones/clean20-video/race_video.mp4')
        if self.publish_progress:
            atomic(self.campaign/'progress.json',progress);atomic(self.campaign/'PROGRESS.md',text)
        return report

    def markdown(self,report):
        lines=['# Local racing improvement', '',f"Updated {report['updated_utc']}. Deadline: {report['deadline_utc']}.",
               '',f"Current queue job: `{report['queue']['current'] or 'none reported'}`.",
               f"Verified incumbent: `{report['incumbent_id'] or 'not initialized; complete baseline suite required'}`.",
               '', ('Bootstrap-only monitoring: no neural finalist or promotion is enabled. Fresh paired confirmation and new hazard layouts remain required.'
                    if self.bootstrap_only else
                    'Fresh confirmation and separately predeclared new hazard layouts remain required after holdout selection.'), '',
               '| Candidate | Closed cases /15 | Verified hazards | Field | Verified clean / closed cars | Valid | Mean valid lap (s) |',
               '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for cid,value in report['formal'].items():
            for tier,bucket in value['strata'].items():
                pace='—' if bucket['mean_valid_lap_s'] is None else f"{bucket['mean_valid_lap_s']:.3f}"
                lines.append(f"| {cid} | {value['closed_cases']} /15 | {value['verified_hazard_passes']} /3 | {tier} | "
                             f"{bucket['clean']} / {bucket['closed_entrants']} | {bucket['valid']} | {pace} |")
        lines += ['',f"All ledger attempts retained: {report['total_ledger_attempts']} ({report['ledger_attempt_counts']}).",'',
                  f"[Latest 20-car engineering video]({self.campaign/'milestones/clean20-video/race_video.mp4'}) · "
                  f"[Final standings image]({self.campaign/'milestones/clean20-video/final_standings.jpg'})", '',
                  'The video records 20 valid zero-overlap laps; its legacy parameter-attestation limitation excludes it from formal promotion.']
        pairs=[(tier,value) for tier,value in report['paired_clean_pace'].items() if value['paired_clean_finishes']]
        if pairs:
            lines += ['','Paired clean pace (candidate / baseline; below1 is faster):']
            lines += [f"- {tier} cars: median{value['median_candidate_to_baseline_ratio']:.4f}, "
                      f"slowest ratio{value['maximum_candidate_to_baseline_ratio']:.4f}, {value['paired_clean_finishes']} pairs."
                      for tier,value in pairs]
        for kind,decision in report['decisions'].items():
            lines += ['',f"{kind}: {'approved' if decision.get('approved') else 'not approved'}."]
            if decision.get('issues'):lines.append('Reasons: '+', '.join(decision['issues']))
        labels=[row for row in report['timing'] if row['background_postprocessing_jobs']]
        if labels:
            lines += ['','Formal wall-time measurements including recorded background postprocessing:']
            lines += [f"- {row['attempt_id']}: {', '.join(row['background_postprocessing_jobs'])}." for row in labels]
        lines += ['',report['scope'],'']
        return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=ROOT/'runs/improvement-12h')
    parser.add_argument('--cases',type=Path)
    parser.add_argument('--baseline',default='reference-safe45');parser.add_argument('--candidate',default='reference-fast70')
    parser.add_argument('--bootstrap-only',action='store_true',help='Audit/bootstrap only the complete baseline; disable all comparison/promotion paths')
    parser.add_argument('--output-directory',type=Path,help='Separate immutable-history suite reports, decision cache and watcher lock')
    parser.add_argument('--no-progress',action='store_true',help='Do not replace the campaign-wide progress documents')
    parser.add_argument('--interval',type=float,default=30.);parser.add_argument('--once',action='store_true')
    parser.add_argument('--act',action='store_true',help='Enable complete-suite bootstrap/promotion decisions')
    args=parser.parse_args()
    if not 1<=args.interval<=60:parser.error('interval must be between1 and60 seconds')
    directory=args.output_directory or args.campaign/'evaluation';directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'watcher.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another evaluation watcher owns the campaign lock')
    evaluator=CampaignEvaluation(args.campaign,args.cases or args.campaign/'formal-cases/holdout.json',args.baseline,args.candidate,
                                 output_directory=directory,publish_progress=not args.no_progress,bootstrap_only=args.bootstrap_only)
    atomic(directory/'watcher_process.json',{'pid':os.getpid(),'started_utc':utc(),'nice':os.nice(0),
        'baseline':evaluator.baseline,'candidate':evaluator.candidate,'bootstrap_only':args.bootstrap_only,
        'decisions_enabled':args.act,'source_sha256':evaluator.code})
    while True:
        if fingerprints()!=evaluator.code:
            atomic(directory/'watcher_restart.json',{'utc':utc(),'reason':'evaluation source changed; reload before any decision'})
            lock.close();os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
        try:
            report=evaluator.once(args.act)
            print(json.dumps({'utc':report['updated_utc'],'incumbent':report['incumbent_id'],
                'closed_cases':{cid:value['closed_cases'] for cid,value in report['formal'].items()},
                'status':report['status']}),flush=True)
        except Exception as error:
            atomic(directory/'watcher_error.json',{'utc':utc(),'error':f'{type(error).__name__}: {error}'})
            print(f'{utc()} watcher error: {type(error).__name__}: {error}',flush=True)
            if args.once:raise
            state=evaluator.ledger.state();deadline=evaluator.deadline(state)
            if deadline and dt.datetime.now(dt.timezone.utc)>=instant(deadline):break
        else:
            if args.once or report['status']=='deadline_reached':break
        time.sleep(args.interval)


if __name__=='__main__':main()
