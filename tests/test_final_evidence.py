"""Final packaging derives73-case completeness, never trusting saved green flags."""
import copy
import json
from pathlib import Path

import pytest

from scripts import final_evidence as final
from scripts.final_scored_inputs import capture_scored_inputs,verify_scored_input_inventory,ledger_binding
from scripts.improvement_ledger import Ledger,digest,freeze_snapshot
from scripts.watch_improvement_evaluation import CampaignEvaluation


@pytest.fixture
def original_championship_metadata(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    campaign=tmp_path/'campaign';roster=campaign/'selection/roster.json'
    tournament=campaign/'championship/tournament.json';geometry=campaign/'championship/geometry_manifest.json'
    canonical=[campaign/'final-evaluation-plan'/name for name in ('championship20.json','geometric-confirmation.json')]
    old={'qualification_issues':[],'final_championship_audit':{
        'metadata_input_sha256':{str(p):'bound-source-sha' for p in (roster,tournament,geometry,*canonical)},
        'auditor_source_sha256':'same-current-scorer','audits':[{'outcome':{'finish_time_s':123.5}}]}}
    basis={'championship_summary':old,'roster_chain':{'roster_path':str(roster)}}
    return campaign,roster,old,basis


@pytest.mark.parametrize('spelling',['relative','absolute'])
def test_fresh28_replays_exact_original_verified_roster_argument(original_championship_metadata,monkeypatch,spelling):
    campaign,roster,old,basis=original_championship_metadata
    argument=str(roster.relative_to(Path.cwd())) if spelling=='relative' else str(roster)
    metadata=old['final_championship_audit']['metadata_input_sha256']
    metadata[argument]=metadata.pop(str(roster))
    calls=[]
    def audited(c,d,r,g):
        calls.append(str(r));assert r.resolve()==roster.resolve();return copy.deepcopy(old)
    monkeypatch.setattr(final,'audit_final',audited)
    assert final.fresh_championship_report(campaign,basis)==old
    assert calls==[argument]


@pytest.mark.parametrize('defect',['missing','duplicate-alias','mismatched-resolved-roster'])
def test_fresh28_requires_unique_original_argument_for_verified_roster(original_championship_metadata,monkeypatch,defect):
    campaign,roster,old,basis=original_championship_metadata
    metadata=old['final_championship_audit']['metadata_input_sha256']
    if defect=='missing':metadata.pop(str(roster))
    elif defect=='duplicate-alias':metadata[str(roster.relative_to(Path.cwd()))]=metadata[str(roster)]
    else:metadata[str(roster.with_name('different-roster.json'))]=metadata.pop(str(roster))
    def forbidden(*a):raise AssertionError('Unbound or ambiguous roster must be rejected before scoring')
    monkeypatch.setattr(final,'audit_final',forbidden)
    with pytest.raises(ValueError,match='Exactly one original metadata argument'):
        final.fresh_championship_report(campaign,basis)


@pytest.mark.parametrize('mutation',['substantive-outcome','source','qualification'])
def test_fresh28_argument_replay_preserves_full_strict_report_equality(original_championship_metadata,monkeypatch,mutation):
    campaign,roster,old,basis=original_championship_metadata
    metadata=old['final_championship_audit']['metadata_input_sha256']
    metadata[str(roster.relative_to(Path.cwd()))]=metadata.pop(str(roster))
    current=copy.deepcopy(old)
    if mutation=='substantive-outcome':current['final_championship_audit']['audits'][0]['outcome']['finish_time_s']+=.1
    elif mutation=='source':current['final_championship_audit']['auditor_source_sha256']='changed-source'
    else:current['qualification_issues'].append('new-required-failure')
    monkeypatch.setattr(final,'audit_final',lambda *a:current)
    with pytest.raises(ValueError,match='Fresh strict28 source/classification differs'):
        final.fresh_championship_report(campaign,basis)


def basis_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(final,'verify_snapshot',lambda p:{'physical':'same'})
    monkeypatch.setattr(final,'physics_binding',lambda manifest:'one-common-physical-stack')
    attempts=[];audits={}
    for cid,split in [('baseline','holdout'),('baseline','confirmation'),('finalist','confirmation')]:
        for index,case in enumerate(final.formal_cases(split)):
            aid=cid+'-'+case['case_id'];hazard=case['is_hazard'];entrants=[]
            for i in range(case['expected_entrants']):
                # One retained baseline DNF is a complete/source-verified result.
                valid=not hazard and not(cid=='baseline' and split=='holdout' and index==0 and i==0)
                entrants.append({'car_id':f'R{i+1:02d}','valid_finish':valid,'clean_finish':valid,
                    'finish_time_s':100.+i if valid else None,'progress_m':5300. if valid else 200.,
                    'collision_samples':0,'track_violation_steps':0,'position_resets_after_spawn':0,
                    'policy_key':'reference' if cid=='baseline' else 'neural','model_sha256':None if cid=='baseline' else 'a'*64})
            out={'audit_verified':True,'snapshot_verified':True,'snapshot_sha256':cid,
                'rules_sha256':case['scenario_id'],'homogeneous_field':True,'hazard_pass':True if hazard else None,'entrants':entrants}
            trial={'attempt_id':aid,'candidate_id':cid,'split':split,'stage':'promotion','scenario_id':case['scenario_id'],
                'state':'finished','status':'completed','is_hazard':hazard,'tier':case['tier'],
                'expected_entrants':case['expected_entrants'],'outcome':copy.deepcopy(out)}
            attempts.append(trial);audits[aid]={'attempt_id':aid,'verified':True,'issues':[],'outcome':out}
    state={'attempts':attempts,'candidates':{c:{'snapshot_path':str(tmp_path/c)} for c in ('baseline','finalist')},'incumbent_id':'baseline'}
    champion={'best_policy_supported':False,'best_policies_among_tested':[],'final_championship_audit':{
        'winner_released':False,'all_planned_cases_closed_and_verified':True,
        'complete_four_policy_six_geometry_inventory':True,'audits':[{'verified':True} for _ in range(28)]}}
    basis={'state':state,'records':[],'attempts':attempts,'baseline_id':'baseline','finalist_id':'finalist',
        'roster_chain':{'roster':{'snapshot_path':str(tmp_path/'common')}},'championship_summary':champion,'joint58_ids':['job'+str(i) for i in range(58)]}
    reports=[]
    for arms in [[('baseline','holdout')],[('baseline','confirmation'),('finalist','confirmation')]]:
        report={'decisions_enabled':False,'common_physical_assets_verified':True,'formal':{},'reaudits':{},
            'overall_formal_approval':False,'evaluation_mode':'frozen_finalist_paired_fresh_confirmation'}
        for cid,split in arms:
            rows=[a for a in attempts if a['candidate_id']==cid and a['split']==split]
            mine={a['attempt_id']:copy.deepcopy(audits[a['attempt_id']]) for a in rows};report['reaudits'].update(mine)
            report['formal'][cid]={'scheduled_cases':15,'closed_cases':15,'required_cases':15,
                'complete':True,'independently_verified_complete_suite':True,'issues':[],
                **CampaignEvaluation.metrics(None,{'attempts':rows},mine)}
        reports.append(report)
    return basis,*reports


def test_complete73_can_preserve_DNF_and_no_promotion_or_qualified_champion(tmp_path,monkeypatch):
    basis,baseline,confirmation=basis_fixture(tmp_path,monkeypatch)
    result=final.derive_study_fields(basis,baseline,confirmation)
    assert result['complete_source_verified_study'] and result['required_case_counts']['total']==73
    assert not result['formal_promoted'] and not result['qualified_champion_released']
    assert sum(s['clean'] for s in result['baseline_metrics']['baseline']['strata'].values())==127


@pytest.mark.parametrize('defect',['missing_audit','false_verified','missing_case','wrong_metrics','extra_audit'])
def test_missing_false_or_elided_fresh_case_never_counts_as_complete(tmp_path,monkeypatch,defect):
    basis,baseline,confirmation=basis_fixture(tmp_path,monkeypatch)
    aid=next(iter(confirmation['reaudits']))
    if defect=='missing_audit':confirmation['reaudits'].pop(aid)
    elif defect=='false_verified':confirmation['reaudits'][aid]['verified']=False
    elif defect=='missing_case':basis['attempts'].pop()
    elif defect=='wrong_metrics':confirmation['formal']['baseline']['strata']['20']['clean']+=1
    else:confirmation['reaudits']['unlisted']={'verified':True}
    try:r=final.derive_study_fields(basis,baseline,confirmation)
    except ValueError:return
    assert not r['complete_source_verified_study']


def test_incumbent_and_saved_promoted_marker_do_not_replace_actual_unique_promotion_chain(tmp_path,monkeypatch):
    basis,baseline,confirmation=basis_fixture(tmp_path,monkeypatch)
    basis['state']['incumbent_id']='finalist';confirmation['overall_formal_approval']=True
    with pytest.raises(ValueError,match='actual ledger promotion chain'):
        final.derive_study_fields(basis,baseline,confirmation)


def test_promotion_requires_exact_unchanged_confirmation_decision_then_unique_event(tmp_path,monkeypatch):
    basis,_,_=basis_fixture(tmp_path,monkeypatch)
    computed={'approved':True,'candidate_id':'finalist','incumbent_id':'baseline','split':'confirmation'}
    monkeypatch.setattr(final,'compare_candidates',lambda *a,**kw:copy.deepcopy(computed))
    decision={**computed,'decision_sha256':digest(computed)}
    basis['state']['incumbent_id']='finalist';basis['state']['run']={'deadline':'2026-11-14T21:50:47Z'}
    for a in basis['state']['attempts']:a['ledger_finished_at_utc']='2026-11-14T18:00:00Z'
    basis['records']=[
        {'event_type':'promotion_decision','sequence':10,'utc':'2026-11-14T18:01:00Z','payload':decision},
        {'event_type':'candidate_promoted','sequence':11,'utc':'2026-11-14T18:02:00Z','payload':{k:decision[k] for k in ('incumbent_id','candidate_id','decision_sha256')}}]
    assert final.verified_promotion_status(basis)
    basis['records'][0]['payload']['split']='holdout'
    with pytest.raises(ValueError,match='unchanged confirmation promotion chain'):final.verified_promotion_status(basis)


@pytest.mark.parametrize('defect',['early_decision','reversed_promotion_clock','late_promotion','missing_finish_clock','naive_clock'])
def test_real_promotion_requires_authoritative_full_confirmation_chronology_and_deadline(tmp_path,monkeypatch,defect):
    basis,_,_=basis_fixture(tmp_path,monkeypatch)
    computed={'approved':True,'candidate_id':'finalist','incumbent_id':'baseline','split':'confirmation'}
    monkeypatch.setattr(final,'compare_candidates',lambda *a,**kw:copy.deepcopy(computed))
    decision={**computed,'decision_sha256':digest(computed)}
    basis['state'].update(incumbent_id='finalist',run={'deadline':'2026-11-14T21:50:47Z'})
    for a in basis['state']['attempts']:a['ledger_finished_at_utc']='2026-11-14T18:00:00Z'
    basis['records']=[
        {'event_type':'promotion_decision','sequence':10,'utc':'2026-11-14T18:01:00Z','payload':decision},
        {'event_type':'candidate_promoted','sequence':11,'utc':'2026-11-14T18:02:00Z','payload':{k:decision[k] for k in ('incumbent_id','candidate_id','decision_sha256')}}]
    if defect=='early_decision':basis['records'][0]['utc']='2026-11-14T17:59:00Z'
    elif defect=='reversed_promotion_clock':basis['records'][1]['utc']='2026-11-14T18:00:30Z'
    elif defect=='late_promotion':basis['records'][1]['utc']='2026-11-14T21:50:48Z'
    elif defect=='missing_finish_clock':basis['state']['attempts'][-1].pop('ledger_finished_at_utc')
    else:basis['records'][1]['utc']='2026-11-14T18:02:00'
    with pytest.raises(ValueError,match='Promotion clocks'):final.verified_promotion_status(basis)


def saved_fixture(tmp_path,monkeypatch):
    basis,baseline,confirmation=basis_fixture(tmp_path,monkeypatch)
    bound=[]
    for i in range(6):
        p=tmp_path/f'bound{i}.json';p.write_text('{}');bound.append(str(p))
    reports={}
    for name,data in [('baseline',baseline),('confirmation',confirmation),('championship',basis['championship_summary'])]:
        p=tmp_path/(name+'.json');p.write_text(json.dumps(data));reports[name]=str(p)
    basis['bound_paths']=bound
    result={'schema_version':1,**final.derive_study_fields(basis,baseline,confirmation),
        'fresh_case_reports':reports,'fresh_case_report_sha256':{p:final.sha(p) for p in reports.values()},
        'bound_paths':bound,'bound_file_sha256':{p:final.sha(p) for p in bound},
        'joint58_ids':basis['joint58_ids'],'frozen_engineering_roster_binding':basis['roster_chain'],
        'scored_input_inventory':{'test-boundary':True},'source_sha256':{'scripts/final_evidence.py':final.sha(final.__file__)}}
    monkeypatch.setattr(final,'study_inventory',lambda *a,**kw:copy.deepcopy(basis))
    monkeypatch.setattr(final,'verify_scored_input_inventory',lambda *a,**kw:True)
    return result


@pytest.mark.parametrize('tamper',['completion','metrics','promoted','champion','counts'])
def test_saved_report_recalculated_semantic_digest_cannot_forge_flags_or_metrics(tmp_path,monkeypatch,tamper):
    result=saved_fixture(tmp_path,monkeypatch)
    if tamper=='completion':result['status']='SOMETHING_ELSE'
    elif tamper=='metrics':result['baseline_metrics']['baseline']['strata']['20']['clean']+=1
    elif tamper=='promoted':result['formal_promoted']=True
    elif tamper=='champion':result['qualified_champion_released']=True
    else:result['required_case_counts']['total']=58
    result['evidence_semantic_sha256']=digest(result)
    path=tmp_path/'tampered.json';path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='flags/metrics differ'):
        final.verify_final_evidence(tmp_path,path)


@pytest.mark.parametrize('name',['gt_poses.jsonl','contact_samples.jsonl'])
def test_changed_nonleading_formal_GT_or_contact_bytes_invalidate_complete_input_gate(tmp_path,name):
    campaign=tmp_path/'campaign';campaign.mkdir();ledger=Ledger(campaign/'ledger.jsonl')
    ledger.append('attempt_scheduled',{'attempt_id':'nonleading-formal','candidate_id':'finalist','split':'confirmation'})
    run=tmp_path/'formal-run';run.mkdir();(run/'config.json').write_text('{"entrants":[]}')
    path=run/name;path.write_text('{"position_m":[1,2,3],"overlap_samples":0}\n')
    source=tmp_path/'source.py';source.write_text('# fixedsource');snapshot=tmp_path/'snapshot';freeze_snapshot(snapshot,{'source.py':source})
    inventory=capture_scored_inputs(campaign,[run],snapshot,[],{'verified':False},formal_arms=[('finalist','confirmation')])
    report={'scored_input_inventory':inventory}
    assert verify_scored_input_inventory(campaign,report)
    path.write_text('{"position_m":[8,2,3],"overlap_samples":1}\n')
    with pytest.raises(ValueError,match='scored input bytes/presence changed'):
        verify_scored_input_inventory(campaign,report)


def test_formal_ledger_binding_detects_extra_same_arm_but_ignores_later_capacity(tmp_path):
    ledger=Ledger(tmp_path/'ledger.jsonl')
    ledger.append('attempt_scheduled',{'attempt_id':'fresh-case','candidate_id':'finalist','split':'confirmation'})
    before=ledger_binding(ledger.records(),[('finalist','confirmation')])
    ledger.append('attempt_scheduled',{'attempt_id':'capacity64','candidate_id':'reference-fast70','split':'development'})
    assert ledger_binding(ledger.records(),[('finalist','confirmation')])==before
    ledger.append('attempt_scheduled',{'attempt_id':'hidden-confirmation-retry','candidate_id':'finalist','split':'confirmation'})
    assert ledger_binding(ledger.records(),[('finalist','confirmation')])!=before


def metadata_study(tmp_path,monkeypatch,*,omit=None,early_final=False):
    basis,_,_=basis_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(final,'ROOT',tmp_path)
    campaign=tmp_path/'campaign';campaign.mkdir();ledger=Ledger(campaign/'ledger.jsonl')
    trials=copy.deepcopy(basis['attempts'])
    for i in range(28):trials.append({'attempt_id':'final-'+str(i),'candidate_id':'mixed',
        'split':'championship' if i<4 else 'geometric-confirmation','stage':'engineering',
        'scenario_id':'final-scene-'+str(i),'state':'finished','status':'completed','outcome':{}})
    def job(a):return {'job_id':a['attempt_id'],'output':'runs/'+a['attempt_id'],'args':['--split',a['split']]}
    for a in trials:a['output_dir']=str(tmp_path/'runs'/a['attempt_id'])
    base=[a for a in trials if a['split']=='holdout'];conf=[a for a in trials if a['split']=='confirmation'];last=trials[-28:]
    def record(a):
        if a['attempt_id']==omit:return
        ledger.append('attempt_scheduled',{k:v for k,v in a.items() if k not in {'state','status','outcome'}})
        ledger.append('attempt_finished',{'attempt_id':a['attempt_id'],'status':'completed','outcome':a['outcome']})
    for a in base:record(a)
    roster={'snapshot_path':str(tmp_path/'common'),'snapshot_sha256':'snapshot','policies':['reference','n1','n2','n3']}
    roster_path=campaign/'roster.json';roster_path.write_text(json.dumps(roster));roster_sha=digest(roster)
    chain={'verified':True,'roster':roster,'roster_sha256':roster_sha,'roster_path':str(roster_path)}
    monkeypatch.setattr(final,'verify_frozen_roster_chain',lambda *a,**kw:copy.deepcopy(chain))
    monkeypatch.setattr(final,'verify_scored_input_inventory',lambda *a,**kw:True)
    selection=campaign/'selection.json';selection.write_text(json.dumps({'baseline_id':'baseline','finalist_id':'finalist'}))
    cp={'jobs':[job(a) for a in conf]};fp={'jobs':[job(a) for a in last],
        'actual_roster_path':str(roster_path),'actual_roster_sha256':roster_sha,**roster}
    paths={'selection_receipt':selection}
    for key,value in [('confirmation_pack',cp),('final_pack',fp)]:
        p=campaign/(key+'.json');p.write_text(json.dumps(value));paths[key]=p
    forecast={'confirmation_pack_sha256':digest(cp),'final_pack_sha256':digest(fp),'selection_receipt_sha256':final.sha(selection)}
    forecast_path=campaign/'forecast.json';forecast_path.write_text(json.dumps(forecast));paths['joint_forecast']=forecast_path
    authority=ledger.append('joint_final_dispatch_forecast_approved',{'file_sha256':final.sha(forecast_path),
        'all_mandatory_work_included':True,'physics_cases_or_safety_gates_changed':False})
    ids=[a['attempt_id'] for a in conf+last]
    if early_final:record(conf[0])
    admission={'admitted':True,'confirmed_all58_fresh_jobs':ids,'forecast_file_sha256':final.sha(forecast_path),
               'forecast_approval_record_sha256':authority['sha256']}
    admission_path=campaign/'admission.json';admission_path.write_text(json.dumps(admission));paths['joint_receipt']=admission_path
    event={'receipt_sha256':final.sha(admission_path),'all58_job_ids':ids}
    ledger.append('joint_final_work_admitted_before_enqueue',event);ledger.append('joint_final58_jobs_enqueued',event)
    for a in conf+last:
        if early_final and a['attempt_id']==conf[0]['attempt_id']:continue
        record(a)
    (campaign/'queue.json').write_text(json.dumps([job(a) for a in trials]))
    (campaign/'queue_status.json').write_text(json.dumps({'completed':{a['attempt_id']:{'status':'completed','exit_code':0} for a in trials}}))
    champ=campaign/'championship.json';champ.write_text(json.dumps(basis['championship_summary']));paths['championship_results']=champ
    return campaign,ledger,paths,conf[0]['attempt_id']


def test_real_metadata_front_door_accepts_exact73_without_reading_physical_log_boundaries(tmp_path,monkeypatch):
    campaign,_,paths,_=metadata_study(tmp_path,monkeypatch)
    study=final.study_inventory(campaign,**paths)
    assert len(study['attempts'])==73 and len(study['joint58_ids'])==58


@pytest.mark.parametrize('defect',['missing_case','extra_case','wrong_queue_output','wrong_actual_output','early_final'])
def test_real_metadata_front_door_rejects_elision_substitution_or_early_final(tmp_path,monkeypatch,defect):
    missing='baseline-'+final.formal_cases('holdout')[0]['case_id']
    campaign,ledger,paths,first=metadata_study(tmp_path,monkeypatch,
        omit=missing if defect=='missing_case' else None,early_final=defect=='early_final')
    if defect=='extra_case':ledger.append('attempt_scheduled',{'attempt_id':'hidden-holdout-retry',
        'candidate_id':'baseline','split':'holdout','stage':'promotion','scenario_id':final.formal_cases('holdout')[0]['scenario_id']})
    elif defect=='wrong_queue_output':
        p=campaign/'queue.json';q=json.loads(p.read_text());next(j for j in q if j['job_id']==first)['output']='a-different-run';p.write_text(json.dumps(q))
    elif defect=='wrong_actual_output':
        # Stored ledger output is an independent part of the identity, not just
        # whatever path appears in the immutable job pack.
        records=ledger.records()
        for r in records:
            if r['event_type']=='attempt_scheduled' and r['payload'].get('attempt_id')==first:r['payload']['output_dir']=str(tmp_path/'wrong')
        original=final.Ledger
        class ChangedLedger:
            def __init__(self,p):self.inner=original(p)
            def records(self):return records
            def state(self):
                s=self.inner.state();next(a for a in s['attempts'] if a['attempt_id']==first)['output_dir']=str(tmp_path/'wrong');return s
        monkeypatch.setattr(final,'Ledger',ChangedLedger)
    with pytest.raises(ValueError):final.study_inventory(campaign,**paths)
