import copy
import datetime as dt
import json

from scripts.generate_improvement_cases import formal_cases
from scripts.improvement_ledger import OFFICIAL_DETECTOR_SHA256,Ledger,freeze_snapshot
from scripts.watch_improvement_evaluation import CampaignEvaluation,substantive


def campaign(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir();(root/'formal-cases').mkdir()
    cases=formal_cases('holdout');case_path=root/'formal-cases/holdout.json'
    case_path.write_text(json.dumps(cases))
    ledger=Ledger(root/'ledger.jsonl')
    now=dt.datetime.now(dt.timezone.utc)
    ledger.append('run_initialized',{'started_at_utc':(now-dt.timedelta(seconds=10)).isoformat(),
                                    'deadline_utc':(now+dt.timedelta(hours=1)).isoformat()})
    source=tmp_path/'source.py';source.write_text('# frozen controller\n')
    snapshots={}
    for cid,distance in (('reference-safe45',45.),('reference-fast70',70.)):
        bundle=tmp_path/cid
        manifest=freeze_snapshot(bundle,{'source.py':source},metadata={'policy_parameters':{'visible_distance_m':distance}})
        ledger.register_candidate(cid,bundle,{'policy_key':'reference','model_sha256':None,
                                             'control_parameters':{'visible_distance_m':distance}})
        snapshots[cid]=manifest['snapshot_sha256']
    evaluator=CampaignEvaluation(root,case_path)
    def independent(attempt,registration,case):
        return {'attempt_id':attempt['attempt_id'],'verified':attempt.get('outcome',{}).get('audit_verified',False),
                'issues':[],'outcome':attempt.get('outcome',{}),'physics_binding_sha256':'same-physical-environment',
                'input_identity':attempt['attempt_id']+str(attempt.get('outcome',{}).get('audit_verified'))}
    monkeypatch.setattr(evaluator,'audit_trial',independent)
    return evaluator,ledger,cases,snapshots


def schedule(ledger,cases,cid,snapshot,finish_count=15,time=100.,verified=True):
    payloads=[]
    distance=45. if cid=='reference-safe45' else 70.
    for i,case in enumerate(cases):
        aid=cid+'-'+case['case_id'];count=case['expected_entrants'];hazard=case['is_hazard']
        outcome={'audit_verified':verified,'snapshot_verified':True,'homogeneous_field':True,
                 'snapshot_sha256':snapshot,'rules_sha256':'world-'+case['scenario_id'],'hazard_pass':True if hazard else None,
                 'entrants':[{'car_id':f'R{j+1:02d}','policy_key':'reference','model_sha256':None,
                   'runtime_detector_checkpoint_sha256':OFFICIAL_DETECTOR_SHA256,
                   'runtime_policy_parameters':{'visible_distance_m':distance},
                   'valid_finish':not hazard,'clean_finish':not hazard,'finish_time_s':None if hazard else time,
                   'progress_m':case['scenario']['obstacle_progress']+25. if hazard else 5300.,
                   'collision_samples':0,'track_violation_steps':0,'position_resets_after_spawn':0} for j in range(count)]}
        spec={'attempt_id':aid,'candidate_id':cid,'scenario_id':case['scenario_id'],'split':'holdout',
              'stage':'bootstrap' if cid=='reference-safe45' else 'promotion','comparison_key':case['comparison_key'],
              'scenario_sha256':case['scenario_sha256'],'rules_sha256':outcome['rules_sha256'],
              'snapshot_sha256':snapshot,'expected_entrants':count,'is_hazard':hazard,'tier':case['tier'],
              'output_dir':str(ledger.path.parent/'unused'/aid)}
        ledger.append('attempt_scheduled',spec)
        terminal={'attempt_id':aid,'status':'completed','outcome':outcome};payloads.append(terminal)
        if i<finish_count:ledger.append('attempt_finished',terminal)
    return payloads


def test_watcher_does_not_bootstrap_before_every_canonical_case_is_closed(tmp_path,monkeypatch):
    evaluator,ledger,cases,snapshots=campaign(tmp_path,monkeypatch)
    terminal=schedule(ledger,cases,evaluator.baseline,snapshots[evaluator.baseline],finish_count=14)
    report=evaluator.once(act=True)
    assert report['incumbent_id'] is None
    assert not any(r['event_type']=='bootstrap_decision' for r in ledger.records())
    ledger.append('attempt_finished',terminal[-1])
    report=evaluator.once(act=True)
    assert report['incumbent_id']==evaluator.baseline
    assert report['formal'][evaluator.baseline]['verified_hazard_passes']==3
    assert report['confirmation_required']


def test_complete_matched_improvement_promotes_once_without_changing_history(tmp_path,monkeypatch):
    evaluator,ledger,cases,snapshots=campaign(tmp_path,monkeypatch)
    schedule(ledger,cases,evaluator.baseline,snapshots[evaluator.baseline],time=100.)
    schedule(ledger,cases,evaluator.candidate,snapshots[evaluator.candidate],time=95.)
    original=ledger.records()
    report=evaluator.once(act=True)
    assert report['incumbent_id']==evaluator.candidate
    assert report['paired_clean_pace']['20']['median_candidate_to_baseline_ratio']==.95
    assert ledger.records()[:len(original)]==original
    before=len(ledger.records());evaluator.once(act=True)
    assert len(ledger.records())==before
    assert (evaluator.campaign/'progress.json').exists() and (evaluator.campaign/'PROGRESS.md').exists()


def test_failed_audit_and_pending_trial_never_disappear_to_make_suite_pass(tmp_path,monkeypatch):
    evaluator,ledger,cases,snapshots=campaign(tmp_path,monkeypatch)
    schedule(ledger,cases,evaluator.baseline,snapshots[evaluator.baseline],verified=False)
    before=ledger.records();report=evaluator.once(act=True)
    assert report['incumbent_id'] is None
    assert report['formal'][evaluator.baseline]['closed_cases']==15
    assert not report['formal'][evaluator.baseline]['independently_verified_complete_suite']
    assert ledger.records()==before


def test_derived_convenience_keys_do_not_invalidate_substantive_reaudit():
    first={'audit_verified':True,'snapshot_verified':True,'entrants':[], 'all_clean_finishes':True,
           'evaluation_source_sha256':'old-validator'}
    second=copy.deepcopy(first);second.pop('all_clean_finishes');second['evaluation_source_sha256']='new-validator'
    assert substantive(first)==substantive(second)
    second['audit_verified']=False
    assert substantive(first)!=substantive(second)


def test_case_manifest_cannot_change_after_watcher_initializes(tmp_path,monkeypatch):
    evaluator,ledger,cases,snapshots=campaign(tmp_path,monkeypatch)
    evaluator.cases_path.write_text(json.dumps(cases)+'\n')
    import pytest
    with pytest.raises(ValueError,match='manifest changed'):
        evaluator.once(act=True)


def test_unchanged_closed_trials_use_cache_but_changed_trace_is_reaudited(tmp_path,monkeypatch):
    import scripts.watch_improvement_evaluation as watcher
    original,ledger,cases,snapshots=campaign(tmp_path,monkeypatch)
    evaluator=CampaignEvaluation(original.campaign,original.cases_path)
    folder=tmp_path/'closed';base=folder/'cars/R01';base.mkdir(parents=True)
    config={'entrants':[{'car_id':'R01'}],'rules':{'max_speed_mps':60.}}
    (folder/'config.json').write_text(json.dumps(config))
    for name in ('heat_results.json','cars/R01/result.json','cars/R01/controller_summary.json',
                 'cars/R01/control_trace.jsonl','cars/R01/predictions.jsonl','cars/R01/policy_actions.jsonl'):
        (folder/name).write_text('{}\n')
    outcome={'audit_verified':True,'snapshot_verified':True,'audit_issues':[],
             'snapshot_sha256':snapshots[evaluator.baseline],
             'rules_sha256':watcher.environment_rules_hash(config['rules']),'entrants':[]}
    attempt={'attempt_id':'cache-regression','output_dir':str(folder),'outcome':outcome,
             'rules_sha256':outcome['rules_sha256'],'snapshot_sha256':outcome['snapshot_sha256']}
    registration=ledger.state()['candidates'][evaluator.baseline]
    calls=[]
    def independent(*args,**kwargs):
        calls.append(True)
        return copy.deepcopy(outcome)
    monkeypatch.setattr(watcher,'evaluate_attempt',independent)
    monkeypatch.setattr(watcher,'world_issues',lambda config,case:[])
    first=evaluator.audit_trial(attempt,registration,cases[0])
    evaluator._bundle_signatures={}  # A new poll still uses the durable cache.
    second=evaluator.audit_trial(attempt,registration,cases[0])
    assert first['input_identity']==second['input_identity'] and len(calls)==1
    assert first['audited_at']==second['audited_at']
    (base/'control_trace.jsonl').write_text('{}\n{}\n')
    third=evaluator.audit_trial(attempt,registration,cases[0])
    assert len(calls)==2 and third['input_identity']!=first['input_identity']


def test_recovery_suite_has_separate_decisions_and_can_preserve_global_progress(tmp_path,monkeypatch):
    original,ledger,cases,_=campaign(tmp_path,monkeypatch)
    marker=original.campaign/'progress.json';marker.write_text('{"old_suite_preserved":true}')
    directory=original.campaign/'evaluation-suite2'
    evaluator=CampaignEvaluation(original.campaign,original.cases_path,'reference-safe45-suite2','reference-fast70-suite2',
                                 output_directory=directory,publish_progress=False)
    report=evaluator.once(act=True)
    assert report['baseline_id']=='reference-safe45-suite2'
    assert report['formal']['reference-safe45-suite2']['scheduled_cases']==0
    assert report['incumbent_id']is None
    assert json.loads(marker.read_text())=={'old_suite_preserved':True}
    assert (directory/'report.json').exists()
    assert not (original.output/'report.json').exists()


def test_bootstrap_only_rejects_partial_fast_baseline_then_bootstraps_without_self_promotion(tmp_path,monkeypatch):
    original,ledger,cases,snapshots=campaign(tmp_path,monkeypatch)
    evaluator=CampaignEvaluation(original.campaign,original.cases_path,baseline='reference-fast70',
                                 candidate='reference-fast70',bootstrap_only=True)
    monkeypatch.setattr(evaluator,'audit_trial',original.audit_trial)
    terminal=schedule(ledger,cases,'reference-fast70',snapshots['reference-fast70'],finish_count=14)
    report=evaluator.once(act=True)
    assert report['candidate_id'] is None and report['evaluation_mode']=='baseline_bootstrap_only'
    assert report['incumbent_id'] is None
    assert set(report['formal'])=={'reference-fast70'}
    ledger.append('attempt_finished',terminal[-1])
    report=evaluator.once(act=True)
    assert report['incumbent_id']=='reference-fast70'
    assert report['decisions']['bootstrap']['approved']
    assert not any(record['event_type'] in {'promotion_decision','candidate_promoted'} for record in ledger.records())
    assert all(row['paired_clean_finishes']==0 for row in report['paired_clean_pace'].values())
    before=ledger.records();evaluator.once(act=True)
    assert ledger.records()==before


def test_same_baseline_and_comparison_candidate_fail_closed(tmp_path,monkeypatch):
    import pytest
    original,_,_,_=campaign(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='must be distinct'):
        CampaignEvaluation(original.campaign,original.cases_path,baseline='reference-fast70',candidate='reference-fast70')
