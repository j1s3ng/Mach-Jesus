"""Fresh CPU-only completion audit for baseline15 + confirmation30 + final28.

No training, model queries, promotion, queue changes or old report/cache writes.
Actual auditing is admitted only after all73 authoritative cases have closed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.improvement_ledger import Ledger,digest,verify_snapshot,compare_candidates
from scripts.generate_improvement_cases import formal_cases
from scripts.exact_legacy_demo_scope import verified_legacy_demo_id
from scripts.final_roster_binding import verify_frozen_roster_chain
from scripts.final_scored_inputs import capture_scored_inputs,verify_scored_input_inventory
from scripts.analyze_final_championship import audit_final
from scripts.watch_improvement_evaluation import CampaignEvaluation,physics_binding,sha,substantive
from scripts.watch_final_confirmation import FinalConfirmationEvaluation


def read(path):return json.loads(Path(path).read_text())


def study_inventory(campaign,selection_receipt,confirmation_pack,final_pack,joint_receipt,joint_forecast,championship_results):
    """Cheap fail-closed metadata preflight, before any large fresh CPU audit."""
    campaign=Path(campaign).resolve();ledger=Ledger(campaign/'ledger.jsonl')
    state,records=ledger.state(),ledger.records();queue=read(campaign/'queue.json')
    queue_by_id={j['job_id']:j for j in queue}
    if len(queue_by_id)!=len(queue):raise ValueError('Duplicate queue IDs')
    completed=read(campaign/'queue_status.json').get('completed',{})
    selection=read(selection_receipt);cp,fp=read(confirmation_pack),read(final_pack)
    admission,forecast=read(joint_receipt),read(joint_forecast)
    chain=verify_frozen_roster_chain(campaign,fp['actual_roster_path'],selection_receipt)
    roster=chain['roster']
    if (fp.get('actual_roster_sha256')!=chain['roster_sha256'] or fp.get('policies')!=roster['policies']
        or fp.get('snapshot_sha256')!=roster['snapshot_sha256']
        or Path(fp['snapshot_path']).resolve()!=Path(roster['snapshot_path']).resolve()):
        raise ValueError('Final pack differs from the actual engineering-frozen roster')
    jobs=cp['jobs']+fp['jobs'];ids=[j['job_id'] for j in jobs]
    if len(cp['jobs'])!=30 or len(fp['jobs'])!=28 or len(set(ids))!=58:
        raise ValueError('Exactly the complete30+28 immutable jobs are required')
    if (admission.get('admitted') is not True or admission.get('confirmed_all58_fresh_jobs')!=ids
        or admission.get('forecast_file_sha256')!=sha(joint_forecast)
        or forecast.get('confirmation_pack_sha256')!=digest(cp) or forecast.get('final_pack_sha256')!=digest(fp)
        or forecast.get('selection_receipt_sha256')!=sha(selection_receipt)):
        raise ValueError('Exact joint58 admission, pack and selection hashes are required')
    authority=[r for r in records if r['event_type']=='joint_final_dispatch_forecast_approved'
        and r['payload'].get('file_sha256')==sha(joint_forecast)
        and r['payload'].get('all_mandatory_work_included') is True
        and r['payload'].get('physics_cases_or_safety_gates_changed') is False]
    if len(authority)!=1 or authority[0]['sha256']!=admission.get('forecast_approval_record_sha256'):
        raise ValueError('Original explicit joint forecast authority is required')
    queued=[r for r in records if r['event_type']=='joint_final58_jobs_enqueued']
    admitted=[r for r in records if r['event_type']=='joint_final_work_admitted_before_enqueue']
    if (len(queued)!=1 or len(admitted)!=1
        or any(r['payload'].get('receipt_sha256')!=sha(joint_receipt) or r['payload'].get('all58_job_ids')!=ids for r in (queued[0],admitted[0]))
        or admitted[0]['sequence']>=queued[0]['sequence']):
        raise ValueError('Unique original joint58 enqueue lifecycle is required')
    if any(queue_by_id.get(j['job_id'])!=j for j in jobs):
        raise ValueError('An immutable queued final job changed')
    legacy=verified_legacy_demo_id(campaign,records=records,jobs=queue)
    reserved=[j['job_id'] for j in queue if '--split' in j['args'] and
        j['args'][j['args'].index('--split')+1] in {'confirmation','championship','geometric-confirmation'}
        and not(legacy is not None and j['job_id']==legacy)]
    if Counter(reserved)!=Counter(ids):raise ValueError('Extra or missing reserved final queue attempts')
    baseline,finalist=selection['baseline_id'],selection['finalist_id']
    arms=[(baseline,'holdout'),(baseline,'confirmation'),(finalist,'confirmation')]
    selected=[]
    for cid,split in arms:
        trials=[a for a in state['attempts'] if a.get('candidate_id')==cid and a.get('split')==split]
        required=formal_cases(split)
        if (len(trials)!=15 or Counter(a.get('scenario_id') for a in trials)!=Counter(c['scenario_id'] for c in required)
            or any(a.get('stage') not in {'bootstrap','promotion'} for a in trials)):
            raise ValueError('Every exact baseline/confirmation case must occur once, with no elision')
        selected.extend(trials)
    final_trials=[a for a in state['attempts'] if a.get('split') in {'championship','geometric-confirmation'}]
    confirm_trials=[a for a in selected if a['split']=='confirmation']
    if Counter(a['attempt_id'] for a in confirm_trials+final_trials)!=Counter(ids):
        raise ValueError('Final ledger attempts do not equal the exact joint58 IDs')
    selected.extend(final_trials)
    if len(selected)!=73 or len({a['attempt_id'] for a in selected})!=73:
        raise ValueError('Exactly73 unique required source-audited attempts are required')
    for attempt in selected:
        aid=attempt['attempt_id'];ending=completed.get(aid,{})
        if (attempt.get('state')!='finished' or attempt.get('status') not in {'completed','finished'}
            or ending.get('status') not in {'completed','recovered_completed_output'}
            or ending.get('exit_code',0)!=0):
            raise ValueError('Required attempt is missing/infrastructure/unclosed:'+aid)
        if aid not in queue_by_id or Path(attempt['output_dir']).resolve()!=(ROOT/queue_by_id[aid]['output']).resolve():
            raise ValueError('Actual required output differs from the immutable queued job:'+aid)
    if any(r['event_type']=='attempt_scheduled' and r['payload'].get('attempt_id') in ids
           and r['sequence']<=admitted[0]['sequence'] for r in records):
        raise ValueError('Final attempt predates actual joint admission')
    summary=read(championship_results);audit=summary.get('final_championship_audit',{})
    if (audit.get('all_planned_cases_closed_and_verified') is not True
        or audit.get('complete_four_policy_six_geometry_inventory') is not True
        or len(audit.get('audits',[]))!=28 or any(row.get('verified') is not True for row in audit['audits'])):
        raise ValueError('Current strict complete28 championship/geometry audit is required')
    verify_scored_input_inventory(campaign,audit)
    return {'state':state,'records':records,'attempts':selected,'formal_arms':arms,'baseline_id':baseline,'finalist_id':finalist,
        'roster_chain':chain,'championship_summary':summary,'joint58_ids':ids,
        'bound_paths':[str(Path(p).resolve()) for p in (selection_receipt,confirmation_pack,final_pack,joint_receipt,joint_forecast,championship_results)],
        'bound_file_sha256':{str(Path(p).resolve()):sha(p) for p in (selection_receipt,confirmation_pack,final_pack,joint_receipt,joint_forecast,championship_results)}}


def verified_promotion_status(basis):
    """A current incumbent marker cannot replace the actual confirmation chain."""
    baseline,finalist=basis['baseline_id'],basis['finalist_id'];state=basis['state'];records=basis['records']
    promoted=[r for r in records if r['event_type']=='candidate_promoted' and r['payload'].get('candidate_id')==finalist]
    if not promoted:return False
    if len(promoted)!=1:raise ValueError('Promotion is not unique for the frozen finalist')
    computed=compare_candidates(state['attempts'],baseline,finalist,formal_cases('confirmation'),
        'confirmation',snapshot_registry=state['candidates'])
    computed['decision_sha256']=digest(computed)
    decisions=[r for r in records if r['event_type']=='promotion_decision' and r['payload']==computed]
    event=promoted[0]
    if (not computed['approved'] or len(decisions)!=1 or decisions[0]['sequence']>=event['sequence']
        or event['payload']!={k:computed[k] for k in ('incumbent_id','candidate_id','decision_sha256')}
        or state.get('incumbent_id')!=finalist):
        raise ValueError('Actual unique unchanged confirmation promotion chain is not verified')
    def instant(value):
        parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
        if parsed.tzinfo is None:raise ValueError('Authoritative promotion clocks must include UTC offsets')
        return parsed
    required=[a for a in state['attempts'] if a.get('candidate_id') in (baseline,finalist)
        and a.get('split')=='confirmation']
    run=state.get('run') or {}
    try:
        finished=[instant(a['ledger_finished_at_utc']) for a in required]
        decision_utc=instant(decisions[0]['utc']);promoted_utc=instant(event['utc'])
        deadline=instant(run.get('deadline_utc',run.get('deadline')))
        chronological=(len(finished)==30 and max(finished)<decision_utc<=promoted_utc<deadline)
    except (KeyError,AttributeError,TypeError,ValueError):
        chronological=False
    if not chronological:raise ValueError('Promotion clocks must follow all30 closed confirmations and precede the campaign deadline')
    return True


def derive_study_fields(basis,baseline_report,confirmation_report,*,inputs_stable=True):
    """Derive every public flag/metric from exact fresh audits, never markers."""
    state=basis['state'];baseline,finalist=basis['baseline_id'],basis['finalist_id'];metrics=[];all_verified=True
    for report,arms in ((baseline_report,[(baseline,'holdout')]),
                        (confirmation_report,[(baseline,'confirmation'),(finalist,'confirmation')])):
        if report.get('decisions_enabled') is not False:raise ValueError('Fresh final auditors must never act')
        if set(report.get('formal',{}))!={cid for cid,_ in arms}:raise ValueError('Fresh report arm inventory differs')
        expected_audits=set();derived={}
        for cid,split in arms:
            trials=[a for a in basis['attempts'] if a.get('candidate_id')==cid and a.get('split')==split]
            required=formal_cases(split)
            if len(trials)!=15 or Counter(a['scenario_id'] for a in trials)!=Counter(c['scenario_id'] for c in required):
                raise ValueError('Fresh audit must retain every exact canonical arm/case once')
            ids={a['attempt_id'] for a in trials};expected_audits|=ids;audits=report.get('reaudits',{})
            for trial in trials:
                audit=audits.get(trial['attempt_id'],{});out=audit.get('outcome',{})
                if (audit.get('attempt_id')!=trial['attempt_id'] or audit.get('verified') is not True
                    or audit.get('issues') or out.get('audit_verified') is not True
                    or out.get('snapshot_verified') is not True
                    or substantive(out)!=substantive(trial.get('outcome') or {})):
                    all_verified=False
            arm=report['formal'][cid]
            actual_metrics=CampaignEvaluation.metrics(None,{'attempts':trials},audits)
            if any(arm.get(k)!=v for k,v in actual_metrics.items()):raise ValueError('Fresh report metrics differ from its audited outcomes')
            if (arm.get('scheduled_cases')!=15 or arm.get('closed_cases')!=15
                or arm.get('complete') is not True or arm.get('independently_verified_complete_suite') is not True
                or arm.get('issues')):all_verified=False
            derived[cid]={**arm,**actual_metrics}
        if set(report.get('reaudits',{}))!=expected_audits:all_verified=False
        if report.get('common_physical_assets_verified') is not True:all_verified=False
        metrics.append(derived)
    assets={physics_binding(verify_snapshot(basis['state']['candidates'][cid]['snapshot_path'])) for cid in (baseline,finalist)}
    assets.add(physics_binding(verify_snapshot(basis['roster_chain']['roster']['snapshot_path'])))
    if len(assets)!=1:all_verified=False
    promoted=verified_promotion_status(basis)
    if confirmation_report.get('overall_formal_approval') is not promoted:
        raise ValueError('Fresh report promotion marker differs from actual ledger promotion chain')
    summary=basis['championship_summary'];champ=summary['final_championship_audit']
    if summary.get('best_policy_supported') is not champ.get('winner_released'):
        raise ValueError('Championship release marker differs from its strict audit')
    complete28=(champ.get('all_planned_cases_closed_and_verified') is True
        and champ.get('complete_four_policy_six_geometry_inventory') is True
        and len(champ.get('audits',[]))==28 and all(a.get('verified') is True for a in champ['audits']))
    verified=bool(inputs_stable and all_verified and complete28)
    return {'status':'COMPLETE_SOURCE_VERIFIED_STUDY' if verified else 'INCOMPLETE_REVIEW_ONLY',
        'complete_source_verified_study':verified,'required_case_counts':{'baseline':15,'confirmation':30,'championship':4,'fresh_geometry':24,'total':73},
        'formal_promoted':promoted,'formal_evaluation_mode':confirmation_report['evaluation_mode'],
        'qualified_champion_released':summary.get('best_policy_supported') is True,
        'champion_policies':summary.get('best_policies_among_tested',[]),
        'baseline_metrics':metrics[0],'confirmation_metrics':metrics[1],
        'decisions_enabled':False,'new_model_queries':False,'old_report_cache_history_or_progress_rewritten':False,
        'genuine_negative_outcomes_remain_results':True}


def fresh_championship_report(campaign,basis):
    """Re-audit the same28 cases; a stored green classification is insufficient."""
    old=basis['championship_summary'];audit=old['final_championship_audit']
    paths={Path(p).resolve() for p in audit['metadata_input_sha256']}
    tournament=[p for p in paths if p.name=='tournament.json']
    roster=Path(basis['roster_chain']['roster_path']).resolve()
    roster_arguments=[p for p in audit['metadata_input_sha256'] if Path(p).resolve()==roster]
    if len(roster_arguments)!=1:raise ValueError('Exactly one original metadata argument must resolve to the verified frozen roster')
    canonical={Path(campaign).resolve()/'final-evaluation-plan'/name for name in ('championship20.json','geometric-confirmation.json')}
    geometry=paths-{roster,*canonical,*tournament}
    if len(tournament)!=1 or len(geometry)!=1:raise ValueError('Exact original28 audit metadata paths are required')
    # The scorer records the supplied roster argument as a metadata key. Replay
    # its exact original spelling; retain full equality of every audited field.
    current=audit_final(campaign,tournament[0].parent,Path(roster_arguments[0]),next(iter(geometry)))
    if current!=old:raise ValueError('Fresh strict28 source/classification differs from the bound original report')
    return current


def audit_final_evidence(campaign,*,selection_receipt,confirmation_pack,final_pack,joint_receipt,joint_forecast,championship_results,output_directory):
    """Run only after capacity benchmarks and all73 cases have closed."""
    campaign=Path(campaign).resolve();output=Path(output_directory).resolve()
    if output.exists() or not output.is_relative_to(campaign):raise ValueError('Use a NEW empty audit/output directory inside the campaign')
    args=dict(selection_receipt=selection_receipt,confirmation_pack=confirmation_pack,final_pack=final_pack,
        joint_receipt=joint_receipt,joint_forecast=joint_forecast,championship_results=championship_results)
    basis=study_inventory(campaign,**args)
    state=basis['state'];chain=basis['roster_chain'];roster=chain['roster']
    metadata=[*basis['bound_paths'],campaign/'formal-cases/holdout.json',campaign/'formal-cases/confirmation.json',
        ROOT/'scripts/final_evidence.py',ROOT/'scripts/watch_final_confirmation.py',ROOT/'scripts/audit_engineering_finalist_selection.py']
    snapshots=[state['candidates'][cid]['snapshot_path'] for cid in (basis['baseline_id'],basis['finalist_id'])]
    for path in snapshots:verify_snapshot(path)
    capture=lambda:capture_scored_inputs(campaign,[a['output_dir'] for a in basis['attempts']],roster['snapshot_path'],
        metadata,chain,additional_snapshots=snapshots,formal_arms=basis['formal_arms'])
    before=capture();output.mkdir(parents=True)
    ledger_before=(campaign/'ledger.jsonl').read_bytes()
    base=CampaignEvaluation(campaign,campaign/'formal-cases/holdout.json',basis['baseline_id'],None,
        output_directory=output/'baseline-fresh',publish_progress=False,bootstrap_only=True)
    baseline_report=base.once(act=False)
    confirmation=FinalConfirmationEvaluation(campaign,selection_receipt,output_directory=output/'confirmation-fresh')
    confirmation_report=confirmation.once(act=False)
    current_championship=fresh_championship_report(campaign,basis)
    championship_fresh_path=output/'championship-fresh.json'
    championship_fresh_path.write_text(json.dumps(current_championship,indent=2,allow_nan=False)+'\n')
    after=capture()
    derived=derive_study_fields(basis,baseline_report,confirmation_report,inputs_stable=before==after)
    # No act=True call occurs. Unrelated later ledger notes are permitted and
    # separately fingerprinted by the selected-case inventory, not rewritten.
    if not (campaign/'ledger.jsonl').read_bytes().startswith(ledger_before):raise ValueError('Historical ledger bytes changed during final audit')
    result={'schema_version':1,'audited_utc':datetime.now(timezone.utc).isoformat(),
        **derived,
        'fresh_case_reports':{'baseline':str(output/'baseline-fresh/report.json'),'confirmation':str(output/'confirmation-fresh/report.json'),
            'championship':str(championship_fresh_path)},
        'fresh_case_report_sha256':{str(p):sha(p) for p in (output/'baseline-fresh/report.json',output/'confirmation-fresh/report.json',championship_fresh_path)},
        'bound_paths':basis['bound_paths'],'bound_file_sha256':basis['bound_file_sha256'],
        'joint58_ids':basis['joint58_ids'],'frozen_engineering_roster_binding':chain,
        'scored_input_inventory':after,'source_sha256':{'scripts/final_evidence.py':sha(__file__)},
        'decisions_enabled':False,'new_model_queries':False,'old_report_cache_history_or_progress_rewritten':False,
        'genuine_negative_outcomes_remain_results':True}
    result['evidence_semantic_sha256']=digest(result)
    path=output/'evidence_report.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');path.chmod(0o444)
    return result


def verify_final_evidence(campaign,evidence_report_path,*,require_complete=True):
    """Read-only packaging gate. Rehash all73 inputs; no rescoring or inference."""
    report=read(evidence_report_path)
    if report.get('evidence_semantic_sha256')!=digest({k:v for k,v in report.items() if k!='evidence_semantic_sha256'}):
        raise ValueError('Final evidence report semantic hash changed')
    if (set(report.get('fresh_case_reports',{}))!={'baseline','confirmation','championship'}
        or set(report.get('fresh_case_report_sha256',{}))!=set(report['fresh_case_reports'].values())
        or len(report.get('bound_paths',[]))!=6 or len(set(report['bound_paths']))!=6
        or set(report.get('bound_file_sha256',{}))!=set(report['bound_paths'])):
        raise ValueError('Every exact fresh report and immutable input hash is required')
    if require_complete and report.get('complete_source_verified_study') is not True:raise ValueError('All73 required cases are not independently source verified')
    for p,expected in {**report['bound_file_sha256'],**report['fresh_case_report_sha256']}.items():
        if sha(p)!=expected:raise ValueError('Frozen final evidence file changed:'+p)
    keys=('selection_receipt','confirmation_pack','final_pack','joint_receipt','joint_forecast','championship_results')
    basis=study_inventory(campaign,**dict(zip(keys,report['bound_paths'])))
    if basis['joint58_ids']!=report['joint58_ids'] or basis['roster_chain']!=report['frozen_engineering_roster_binding']:
        raise ValueError('Actual final frozen source/roster/enqueue evidence changed')
    verify_scored_input_inventory(campaign,report)
    baseline=read(report['fresh_case_reports']['baseline']);confirmation=read(report['fresh_case_reports']['confirmation'])
    if read(report['fresh_case_reports']['championship'])!=basis['championship_summary']:
        raise ValueError('Bound fresh28 report differs from actual strict championship evidence')
    derived=derive_study_fields(basis,baseline,confirmation)
    if any(report.get(k)!=v for k,v in derived.items()):
        raise ValueError('Final flags/metrics differ from exact fresh73 audit and promotion evidence')
    if report['source_sha256']!={'scripts/final_evidence.py':sha(__file__)}:raise ValueError('Final evidence verifier source changed; fresh audit required')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--verify',type=Path)
    for name in ('selection-receipt','confirmation-pack','final-pack','joint-receipt','joint-forecast','championship-results','output-directory'):
        p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.verify:r=verify_final_evidence(a.campaign,a.verify)
    else:
        names=('selection_receipt','confirmation_pack','final_pack','joint_receipt','joint_forecast','championship_results','output_directory')
        if any(getattr(a,k) is None for k in names):p.error('All concrete closed-study paths and a new output directory are required')
        r=audit_final_evidence(a.campaign,**{k:getattr(a,k) for k in names})
    print(json.dumps({k:r[k] for k in ('status','complete_source_verified_study','formal_promoted','qualified_champion_released')},indent=2))
