import copy
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.analyze_final_championship as final
from scripts.improvement_ledger import OFFICIAL_DETECTOR_SHA256,Ledger,digest,environment_rules_hash,freeze_snapshot
from scripts.prepare_final_evaluation_protocols import prepare

RULES={'reference_length_m':1000.,'laps':1,'sensor_period_ms':200,'physics_step_ms':20,
       'max_speed_mps':60.,'lateral_accel_budget_mps2':8.,'friction_coefficient':1.5,
       'vehicle_model':'identical physical Car'}


def result(time,**extra):
    return {'status':'completed','completed':True,'finish_time_s':time,'valid_lap':True,
            'collision_samples':0,'track_violation_steps':0,'position_resets_after_spawn':0,**extra}


def setup_championship(tmp_path,monkeypatch,*,geometry_policy_indices=(0,1,2,3),failed_geometry_policy=None):
    campaign=tmp_path/'campaign';campaign.mkdir();prepare(campaign)
    ledger=Ledger(campaign/'ledger.jsonl');files={}
    for index in range(3):
        source=tmp_path/f'model{index}.npz';source.write_bytes(f'unique weights {index}'.encode());files[source.name]=source
    source=tmp_path/'source.py';source.write_text('# fixed driver source');files[source.name]=source
    bundle=tmp_path/'bundle';manifest=freeze_snapshot(bundle,files,{'policy_parameters':{'visible_distance_m':70.}})
    hashes={row['path']:row['sha256'] for row in manifest['files']}
    policies=[{'policy_key':'reference','model_sha256':None,'detector_sha256':OFFICIAL_DETECTOR_SHA256,
               'control_parameters':{'visible_distance_m':70.}}]
    policies.extend({**policies[0],'policy_key':f'neural{index}','model_sha256':hashes[f'model{index}.npz']} for index in range(3))
    declared=final.declare_roster(campaign,bundle,policies)
    # This fixture tests race/geometry scoring. The real unique frozen chain is
    # exercised separately against actual synthetic freezer output.
    monkeypatch.setattr(final,'verify_frozen_roster_chain',lambda *_a,**_k:{'verified':True,'test_boundary':True})
    capture=final.capture_scored_inputs
    monkeypatch.setattr(final,'capture_scored_inputs',lambda campaign,runs,snapshot,metadata,chain:
        capture(campaign,runs,snapshot,metadata,{'verified':False}))
    directory=campaign/'championship';directory.mkdir();outcomes={};calls=[]
    def add(case,hazard=False,selected_policy=None,series=0):
        scenario=case['scenario'];folder=((campaign/'geometry'/str(series)/selected_policy['policy_key']) if hazard else directory)/case['case_id'];folder.mkdir(parents=True)
        rng=np.random.default_rng(scenario['seed']);entries=[];records={};rows=[]
        for slot in range(scenario['cars']):
            descriptor=policies[(slot+scenario['rotation'])%4] if not hazard else selected_policy
            progress=-scenario['spacing']*slot-float(rng.uniform(0.,scenario['grid_jitter']))
            lane=(-2.5 if slot%2==0 else 2.5)+float(rng.uniform(-scenario['grid_jitter'],scenario['grid_jitter']))
            if hazard:
                progress=scenario['ego_progress'] if slot==0 else scenario['obstacle_progress']
                lane=scenario['ego_lane'] if slot==0 else (0. if slot==1 else (-3.2 if scenario['blocked_side']=='right' else 3.2))
            cid=f'R{slot+1:02d}'
            entries.append({'car_id':cid,'grid_index':slot,'seed':scenario['seed']+slot,'start_progress_m':progress,
                            'start_lane_m':lane,'policy':descriptor['policy_key'],'is_static':hazard and slot>0})
            if hazard and slot>0:continue
            policy_index=policies.index(descriptor)
            records[cid]=result(50.+policy_index+slot/100.,model_sha256=descriptor['model_sha256'],
                                progress_m=1000.2 if not hazard else scenario['obstacle_progress']+25.,
                                runtime_detector_checkpoint_sha256=descriptor['detector_sha256'],
                                runtime_policy_parameters=descriptor['control_parameters'])
            rows.append({'car_id':cid,**descriptor,'runtime_detector_checkpoint_sha256':descriptor['detector_sha256'],
                         'runtime_policy_parameters':descriptor['control_parameters'],'valid_finish':not hazard,
                         'clean_finish':not hazard,'finish_time_s':records[cid]['finish_time_s'] if not hazard else None,
                         'progress_m':records[cid]['progress_m'],'collision_samples':0,'track_violation_steps':0,
                         'position_resets_after_spawn':0})
            base=folder/'cars'/cid;base.mkdir(parents=True);(base/'result.json').write_text(json.dumps(records[cid]))
        rules={**RULES,'field_size':scenario['cars'],'grid_spacing_m':scenario['spacing'],
               'race_cutoff_seconds':scenario['seconds'],'grid_jitter_m':scenario['grid_jitter']}
        config={'entrants':entries,'rules':rules,'seed':scenario['seed'],'grid_rotation':scenario['rotation'],
                'scenario':scenario['scenario'],'dataset_split':case['split'],'excluded_from_training':True,
                'source_root':str(bundle)}
        if hazard:config['hazard']={'ego_car_id':'R01','progress_m':scenario['obstacle_progress']}
        (folder/'config.json').write_text(json.dumps(config))
        (folder/'heat_results.json').write_text(json.dumps({'status':'all_cars_reported','cars':records,
                                                          'collisions_by_car':{cid:0 for cid in records}}))
        genuine_hazard_failure=hazard and selected_policy['policy_key']==failed_geometry_policy
        if genuine_hazard_failure:rows[0]['progress_m']=scenario['obstacle_progress']-10.
        outcome={'audit_verified':True,'snapshot_verified':True,'audit_issues':[],
                 'snapshot_sha256':manifest['snapshot_sha256'],'rules_sha256':environment_rules_hash(rules),
                 'homogeneous_field':hazard,'entrants':rows,'hazard_pass':not genuine_hazard_failure if hazard else None}
        attempt={'attempt_id':case['case_id'] if not hazard else str(series)+'-'+selected_policy['policy_key']+'-'+case['case_id'],
                 'candidate_id':'mixed' if not hazard else 'final-geometry-'+selected_policy['policy_key'],
                 'output_dir':str(folder),'split':case['split'],'stage':'championship' if not hazard else 'confirmation',
                 'expected_entrants':1 if hazard else 20,'snapshot_sha256':manifest['snapshot_sha256'],
                 'rules_sha256':outcome['rules_sha256'],**{key:case[key] for key in ('scenario_id','scenario_sha256','comparison_key')}}
        ledger.append('attempt_scheduled',attempt);ledger.append('attempt_finished',{'attempt_id':attempt['attempt_id'],'status':'completed','outcome':outcome})
        outcomes[str(folder.resolve())]=outcome
        return folder
    races=[add(case) for case in final.read(campaign/'final-evaluation-plan/championship20.json')]
    hazards=[add(case,True,policies[index],series) for series,index in enumerate(geometry_policy_indices)
             for case in final.read(campaign/'final-evaluation-plan/geometric-confirmation.json')]
    (directory/'tournament.json').write_text(json.dumps({'heats':[{'path':str(path)} for path in races]}))
    geometry_manifest=campaign/'geometry-manifest.json';geometry_manifest.write_text(json.dumps([{'path':str(path)} for path in hazards]))
    def audit(folder,*args,**kwargs):
        calls.append(str(folder));return copy.deepcopy(outcomes[str(Path(folder).resolve())])
    monkeypatch.setattr(final,'evaluate_attempt',audit)
    return campaign,directory,Path(declared['roster_path']),geometry_manifest,outcomes,races,hazards,calls


def test_complete_exact_schedule_and_six_fresh_passes_release_tested_winner(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,*_=setup_championship(tmp_path,monkeypatch)
    report=final.audit_final(campaign,directory,roster,geometry)
    assert report['best_policy_supported'] and report['best_policies_among_tested']==['reference']
    assert report['identical_full_grid_distributions']
    assert report['final_championship_audit']['all_attempts_inventoried']
    assert report['final_championship_audit']['historical_records_modified'] is False
    assert report['final_championship_audit']['all_planned_cases_closed_and_verified']
    assert len(report['final_championship_audit']['audits'])==28


def test_omitted_race_attempt_and_geometry_failure_cannot_release_winner(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,outcomes,races,hazards,_=setup_championship(tmp_path,monkeypatch)
    manifest=final.read(directory/'tournament.json');manifest['heats'].pop()
    (directory/'tournament.json').write_text(json.dumps(manifest))
    report=final.audit_final(campaign,directory,roster,geometry)
    assert not report['best_policy_supported']
    assert 'ledger_attempt_inventory_mismatch:championship' in report['qualification_issues']
    assert not report['final_championship_audit']['all_attempts_inventoried']
    (directory/'tournament.json').write_text(json.dumps({'heats':[{'path':str(path)} for path in races]}))
    outcomes[str(hazards[0].resolve())]['hazard_pass']=False
    report=final.audit_final(campaign,directory,roster,geometry)
    assert not report['best_policy_supported']
    assert 'six_fresh_geometry_passes_required:reference' in report['qualification_issues']


@pytest.mark.parametrize('change',['checkpoint','policy_slot','detector','parameters','spawn','seed','physics','crossing_audit'])
def test_actual_checkpoint_perception_parameters_spawns_and_crossings_are_bound(tmp_path,monkeypatch,change):
    campaign,directory,roster,geometry,outcomes,races,_,_=setup_championship(tmp_path,monkeypatch)
    outcome=outcomes[str(races[0].resolve())];row=outcome['entrants'][0]
    if change=='checkpoint':row['model_sha256']='a'*64
    elif change=='policy_slot':row['policy_key']='neural0'
    elif change=='detector':row['runtime_detector_checkpoint_sha256']='b'*64
    elif change=='parameters':row['runtime_policy_parameters']={'visible_distance_m':90.}
    elif change=='crossing_audit':outcome.update(audit_verified=False,audit_issues=['no_independent_physical_finish_crossing'])
    else:
        config=final.read(races[0]/'config.json')
        if change=='seed':config['seed']+=1
        elif change=='physics':config['rules']['max_speed_mps']=90.
        else:config['entrants'][0]['start_progress_m']+=1.
        (races[0]/'config.json').write_text(json.dumps(config))
    report=final.audit_final(campaign,directory,roster,geometry)
    assert not report['best_policy_supported']
    assert not report['final_championship_audit']['verified']


def test_missing_geometry_and_hidden_extra_attempts_remain_disqualifying(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,_,_,hazards,_=setup_championship(tmp_path,monkeypatch)
    rows=final.read(geometry);rows.pop(0);geometry.write_text(json.dumps(rows))
    report=final.audit_final(campaign,directory,roster,geometry)
    assert not report['best_policy_supported']
    assert 'ledger_attempt_inventory_mismatch:geometric-confirmation' in report['qualification_issues']
    assert 'six_fresh_geometry_passes_required:reference' in report['qualification_issues']
    geometry.write_text(json.dumps([{'path':str(path)} for path in hazards]))
    ledger=Ledger(campaign/'ledger.jsonl');attempt=copy.deepcopy(ledger.state()['attempts'][-1])
    attempt.update(attempt_id='hidden-failed-rerun',output_dir=str(campaign/'not-in-manifest'))
    ledger.append('attempt_scheduled',attempt)
    report=final.audit_final(campaign,directory,roster,geometry)
    assert not report['best_policy_supported']
    assert 'ledger_attempt_inventory_mismatch:geometric-confirmation' in report['qualification_issues']


def test_roster_cannot_be_redeclared_after_attempts_are_scheduled(tmp_path,monkeypatch):
    campaign,_,roster,_,*_=setup_championship(tmp_path,monkeypatch)
    value=final.read(roster)
    with pytest.raises(ValueError,match='before scheduling'):
        final.declare_roster(campaign,value['snapshot_path'],value['policies'])


def test_winner_six_only_cannot_release_without_other18_planned_geometry_runs(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,*_=setup_championship(tmp_path,monkeypatch,geometry_policy_indices=(0,))
    report=final.audit_final(campaign,directory,roster,geometry)
    assert not report['best_policy_supported']
    assert not report['final_championship_audit']['verified']
    assert not report['final_championship_audit']['all_planned_cases_closed_and_verified']
    assert 'exact24_geometry_paths_required' in report['qualification_issues']
    assert 'incomplete_or_duplicate_four_policy_six_geometry_inventory' in report['qualification_issues']
    assert report['final_championship_audit']['all_attempts_inventoried']  # Every attempted case remains visible.


def test_duplicate_actual_policy_case_cannot_replace_missing_policy_even_with24_paths(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,*_=setup_championship(tmp_path,monkeypatch,geometry_policy_indices=(0,0,2,3))
    report=final.audit_final(campaign,directory,roster,geometry)
    assert not report['best_policy_supported']
    assert not report['final_championship_audit']['complete_four_policy_six_geometry_inventory']
    assert 'duplicate_geometry_attempts_for_policy_case' in report['qualification_issues']


def test_all24_audited_nonwinner_hazard_failures_are_preserved_without_hiding_or_reselecting(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,*_=setup_championship(tmp_path,monkeypatch,failed_geometry_policy='neural0')
    report=final.audit_final(campaign,directory,roster,geometry)
    audit=report['final_championship_audit']
    assert audit['verified'] and audit['all_planned_cases_closed_and_verified']
    assert report['best_policy_supported'] and report['best_policies_among_tested']==['reference']
    assert all(v is False for v in audit['geometry_passes']['neural0'].values())
    assert sum(a['outcome'].get('hazard_pass') is False for a in audit['audits'])==6


def test_genuine_winner_hazard_failure_keeps_full_audit_inventory_but_no_release_or_reselection(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,*_=setup_championship(tmp_path,monkeypatch,failed_geometry_policy='reference')
    report=final.audit_final(campaign,directory,roster,geometry)
    audit=report['final_championship_audit']
    assert audit['all_planned_cases_closed_and_verified']
    assert report['generic_best_policy_supported']
    assert report['observed_leading_policies']==['reference']
    assert not audit['winner_released'] and not report['best_policy_supported']
    assert report['best_policies_among_tested']==[]
    assert 'six_fresh_geometry_passes_required:reference' in report['qualification_issues']


def test_geometry_policy_counter_is_bound_to_actual_config_not_manifest_path_label(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,_,_,hazards,_=setup_championship(tmp_path,monkeypatch)
    config=final.read(hazards[6]/'config.json')
    config['entrants'][0]['policy']='reference'
    (hazards[6]/'config.json').write_text(json.dumps(config))
    report=final.audit_final(campaign,directory,roster,geometry)
    assert not report['best_policy_supported']
    assert any('geometry_actual_config_policy_mismatch' in x for x in report['qualification_issues'])


def test_later_self_consistent_predeclaration_cannot_replace_first_case_manifest(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,*_=setup_championship(tmp_path,monkeypatch)
    path=campaign/'final-evaluation-plan/geometric-confirmation.json'
    changed=list(reversed(final.read(path)))
    path.chmod(0o644)  # Adversarial mutation is confined to this temporary fixture.
    path.write_text(json.dumps(changed))
    Ledger(campaign/'ledger.jsonl').append('final_evaluation_predeclared', {
        'plan':'geometric-confirmation','definition_sha256':digest(changed)})
    report=final.audit_final(campaign,directory,roster,geometry)
    assert not report['best_policy_supported']
    assert 'case_definitions_differ_from_first_immutable_predeclaration:geometric-confirmation' in report['qualification_issues']


def test_independently_declared_roster_without_frozen_engineering_chain_cannot_release(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,*_=setup_championship(tmp_path,monkeypatch)
    def unbound(*a,**kw):raise ValueError('No actual engineering freeze receipt')
    monkeypatch.setattr(final,'verify_frozen_roster_chain',unbound)
    report=final.audit_final(campaign,directory,roster,geometry)
    assert report['final_championship_audit']['all_planned_cases_closed_and_verified']
    assert not report['best_policy_supported']
    assert 'frozen_engineering_roster_chain_not_verified' in report['qualification_issues']


def test_export_rehash_rejects_changed_nonleading_geometry_result_even_if_report_flags_stay_true(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,_,_,hazards,_=setup_championship(tmp_path,monkeypatch)
    report=final.audit_final(campaign,directory,roster,geometry)
    audit=report['final_championship_audit']
    assert audit['verified'] and final.verify_scored_input_inventory(campaign,audit)
    path=hazards[6]/'cars/R01/result.json'
    row=final.read(path);row['progress_m']-=15.
    path.write_text(json.dumps(row))
    assert audit['verified']  # A saved marker alone cannot detect changed bytes.
    with pytest.raises(ValueError,match='scored input bytes/presence changed'):
        final.verify_scored_input_inventory(campaign,audit)


def test_derived_cache_outputs_and_unrelated_ledger_growth_are_not_scoring_inputs(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,_,_,hazards,_=setup_championship(tmp_path,monkeypatch)
    report=final.audit_final(campaign,directory,roster,geometry)
    cache=hazards[6]/'derived-reaudit-cache.json';cache.write_text('{"fresh":true}')
    Ledger(campaign/'ledger.jsonl').append('later_capacity_or_media_note',{'not_used_for_scoring':True})
    assert final.verify_scored_input_inventory(campaign,report['final_championship_audit'])


def test_extra_final_attempt_invalidates_saved_ledger_fingerprint_even_without_new_files(tmp_path,monkeypatch):
    campaign,directory,roster,geometry,*_=setup_championship(tmp_path,monkeypatch)
    report=final.audit_final(campaign,directory,roster,geometry)
    ledger=Ledger(campaign/'ledger.jsonl')
    ledger.append('attempt_scheduled',{'attempt_id':'unlisted-final-rerun','split':'geometric-confirmation'})
    with pytest.raises(ValueError,match='ledger input changed'):
        final.verify_scored_input_inventory(campaign,report['final_championship_audit'])
