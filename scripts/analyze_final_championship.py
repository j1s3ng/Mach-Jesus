#!/usr/bin/env python3
"""Declare a frozen roster or audit the complete final championship and hazards.

This scorer never launches races, trains, changes queues, or edits race records.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from race_lidar.race_results import analyze_directory,write_analysis
from scripts.evaluate_improvement_attempt import evaluate_attempt
from scripts.generate_improvement_cases import verify_case
from scripts.final_roster_binding import verify_frozen_roster_chain
from scripts.final_scored_inputs import capture_scored_inputs,verify_scored_input_inventory
from scripts.improvement_ledger import (Ledger,_bundle_contains_descriptor,candidate_descriptor,
    digest,entrant_descriptor,environment_rules,environment_rules_hash,verify_snapshot)
from scripts.watch_improvement_evaluation import physics_binding,substantive,world_issues,sha


def read(path):return json.loads(Path(path).read_text())
def instant(value):return dt.datetime.fromisoformat(value.replace('Z','+00:00'))
def path_at(base,value):return (Path(base)/value).resolve()


def normalize_roster(value,manifest):
    rows=value.get('policies',value) if isinstance(value,dict) else value
    if not isinstance(rows,list) or len(rows)!=4:raise ValueError('Exactly four ordered policy descriptors are required')
    descriptors=[candidate_descriptor({'descriptor':row}) for row in rows]
    if len({row['policy_key'] for row in descriptors})!=4:raise ValueError('The four policy keys must be distinct')
    if any(not _bundle_contains_descriptor(manifest,row) for row in descriptors):
        raise ValueError('Every selected weight, detector and control parameter must be frozen in the common bundle')
    if any('control_parameters' not in row for row in descriptors):raise ValueError('Explicit common controller parameters are required')
    if len({digest([row['detector_sha256'],row['control_parameters']]) for row in descriptors})!=1:
        raise ValueError('Driver comparison requires identical detector and controller parameters')
    return descriptors


def declare_roster(campaign,snapshot,policies,output=None):
    campaign=Path(campaign).resolve();snapshot=Path(snapshot).resolve();manifest=verify_snapshot(snapshot)
    descriptors=normalize_roster(policies,manifest)
    roster={'schema_version':1,'snapshot_path':str(snapshot),'snapshot_sha256':manifest['snapshot_sha256'],
            'policies':descriptors,'excluded_from_training':True}
    output=Path(output or campaign/'final-evaluation-plan/roster.json').resolve()
    ledger=Ledger(campaign/'ledger.jsonl')
    if any(a.get('split') in {'championship','geometric-confirmation'} for a in ledger.state()['attempts']):
        raise ValueError('Declare the final roster before scheduling final evaluation attempts')
    if output.exists() and read(output)!=roster:raise ValueError('A declared roster cannot be rewritten')
    if not output.exists():
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(roster,indent=2)+'\n');output.chmod(0o444)
    identity=digest(roster)
    if not any(r['event_type']=='championship_roster_predeclared' and r['payload'].get('roster_sha256')==identity for r in ledger.records()):
        ledger.append('championship_roster_predeclared',{'roster_path':str(output),'roster_sha256':identity,
                                                       'snapshot_sha256':roster['snapshot_sha256']})
    return {'roster_path':str(output),'roster_sha256':identity,'policies':[row['policy_key'] for row in descriptors]}


def physical_core(config):
    # Scenario-dependent field size, starts and duration legitimately differ in
    # the stopped-hazard extension. All physical/actuation/sensor rules remain.
    return {key:value for key,value in environment_rules(config['rules']).items()
            if key not in {'field_size','grid_spacing_m','race_cutoff_seconds','grid_jitter_m'}}


def audit_final(campaign,directory,roster_path,geometry_manifest):
    campaign=Path(campaign).resolve();directory=Path(directory).resolve()
    ledger=Ledger(campaign/'ledger.jsonl');records=ledger.records();state=ledger.state();issues=[]
    try:engineering_roster_binding=verify_frozen_roster_chain(campaign,roster_path)
    except (ValueError,KeyError,OSError) as error:
        engineering_roster_binding={'verified':False,'issues':[f'{type(error).__name__}: {error}']}
        issues.append('frozen_engineering_roster_chain_not_verified')
    plan_dir=campaign/'final-evaluation-plan';cases=read(plan_dir/'championship20.json')
    geometry=read(plan_dir/'geometric-confirmation.json');roster=read(roster_path)
    manifest=verify_snapshot(roster['snapshot_path']);descriptors=normalize_roster(roster,manifest)
    if roster.get('snapshot_sha256')!=manifest['snapshot_sha256']:issues.append('roster_snapshot_mismatch')
    for case in cases+geometry:verify_case(case)
    if len(cases)!=4 or {c['scenario']['rotation'] for c in cases}!={0,1,2,3} or any(c['cars']!=20 for c in cases):
        issues.append('invalid_four_heat_plan')
    if len(geometry)!=6 or len({c['scenario_id'] for c in geometry})!=6:issues.append('invalid_six_geometry_plan')
    declarations={}
    for name,definitions in (('championship20',cases),('geometric-confirmation',geometry)):
        first=next((r for r in records if r['event_type']=='final_evaluation_predeclared'
                    and r['payload'].get('plan')==name),None)
        if first is None:issues.append('missing_immutable_plan_predeclaration:'+name)
        elif first['payload'].get('definition_sha256')!=digest(definitions):
            issues.append('case_definitions_differ_from_first_immutable_predeclaration:'+name)
        else:declarations[name]=first['utc']
    declared=[r for r in records if r['event_type']=='championship_roster_predeclared'
              and r['payload'].get('roster_sha256')==digest(roster)]
    if not declared:issues.append('missing_roster_predeclaration')
    roster_time=declared[0]['utc'] if declared else None
    specs=read(directory/'tournament.json').get('heats',[])
    race_paths=[path_at(directory,spec if isinstance(spec,str) else spec.get('path',spec.get('directory','.'))) for spec in specs]
    geo_path=Path(geometry_manifest).resolve();geo_data=read(geo_path)
    geo_specs=geo_data.get('heats',[]) if isinstance(geo_data,dict) else geo_data
    geo_paths=[path_at(geo_path.parent,spec if isinstance(spec,str) else spec['path']) for spec in geo_specs]
    if len(race_paths)!=4 or len(set(race_paths))!=4:issues.append('exact_four_race_paths_required')
    if len(geo_paths)!=24 or len(set(geo_paths))!=24:issues.append('exact24_geometry_paths_required')
    if len(set(geo_paths))!=len(geo_paths):issues.append('duplicate_geometry_paths')
    race_attempts=[a for a in state['attempts'] if a.get('split')=='championship']
    geo_attempts=[a for a in state['attempts'] if a.get('split')=='geometric-confirmation']
    for split,paths,attempts in (('championship',race_paths,race_attempts),('geometric-confirmation',geo_paths,geo_attempts)):
        recorded=[Path(a['output_dir']).resolve() for a in attempts]
        if Counter(recorded)!=Counter(paths):issues.append('ledger_attempt_inventory_mismatch:'+split)
    by_path={Path(a['output_dir']).resolve():a for a in race_attempts+geo_attempts}
    by_rotation={c['scenario']['rotation']:c for c in cases};geo_by_id={c['scenario_id']:c for c in geometry}
    audits=[];geometry_passes={row['policy_key']:{} for row in descriptors};seen_race=[];seen_geo=Counter();cores=[]
    input_metadata=(Path(roster_path),geo_path,directory/'tournament.json',plan_dir/'championship20.json',plan_dir/'geometric-confirmation.json')
    input_inventory=None
    try:input_inventory=capture_scored_inputs(campaign,race_paths+geo_paths,roster['snapshot_path'],input_metadata,engineering_roster_binding)
    except (ValueError,KeyError,OSError) as error:issues.append('cannot_fingerprint_scored_inputs:'+str(error))
    for split,paths in (('championship',race_paths),('geometric-confirmation',geo_paths)):
        for folder in paths:
            label=str(folder);own=[];attempt=by_path.get(folder);geometry_policy=None
            try:
                config=read(folder/'config.json');heat=read(folder/'heat_results.json')
                if attempt is None:raise ValueError('No corresponding scheduled attempt')
                if attempt['state']!='finished' or attempt.get('status') not in {'completed','finished'}:own.append('attempt_not_closed_successfully')
                if heat.get('status') not in {'all_cars_reported','time_limit'}:own.append('supervisor_not_closed')
                scheduled=attempt.get('ledger_scheduled_at_utc')
                plan_time=declarations.get('championship20' if split=='championship' else 'geometric-confirmation')
                if not scheduled or not roster_time or instant(scheduled)<instant(roster_time) or not plan_time or instant(scheduled)<instant(plan_time):
                    own.append('attempt_predates_roster_or_case_predeclaration')
                if split=='championship':
                    case=by_rotation.get(config.get('grid_rotation'))
                    if case is None:raise ValueError('Unplanned roster rotation')
                    seen_race.append(case['scenario_id'])
                else:
                    case=geo_by_id.get(attempt.get('scenario_id'))
                    if case is None:raise ValueError('Unplanned geometry scenario')
                own.extend(world_issues(config,case))
                for key in ('scenario_id','scenario_sha256','comparison_key'):
                    if attempt.get(key)!=case[key]:own.append('scheduled_case_mismatch:'+key)
                if attempt.get('snapshot_sha256')!=roster['snapshot_sha256'] or Path(config.get('source_root','')).resolve()!=Path(roster['snapshot_path']).resolve():
                    own.append('common_frozen_bundle_mismatch')
                if attempt.get('rules_sha256')!=environment_rules_hash(config['rules']):own.append('scheduled_rules_mismatch')
                expected_count=20 if split=='championship' else 1
                if attempt.get('expected_entrants')!=expected_count:own.append('scheduled_entrant_count_mismatch')
                out=evaluate_attempt(folder,roster['snapshot_path'],roster['snapshot_sha256'],
                    expected_rules_sha256=attempt.get('rules_sha256'),
                    expected_detector_checkpoint_sha256=descriptors[0]['detector_sha256'],
                    expected_control_parameters=descriptors[0]['control_parameters'])
                if not out.get('audit_verified') or not out.get('snapshot_verified'):own.extend(out.get('audit_issues') or ['independent_audit_failed'])
                if substantive(out)!=substantive(attempt.get('outcome') or {}):own.append('stored_outcome_differs_from_independent_audit')
                rows=out.get('entrants',[])
                if len(rows)!=expected_count or len({r['car_id'] for r in rows})!=expected_count:own.append('missing_or_duplicate_evaluated_cars')
                if split=='championship':
                    entries={e['car_id']:e for e in config['entrants']}
                    for row in rows:
                        grid=entries[row['car_id']]['grid_index'];expected=descriptors[(grid+case['scenario']['rotation'])%4]
                        if entrant_descriptor(row)!=expected:own.append('roster_slot_or_checkpoint_mismatch:'+row['car_id'])
                elif len(rows)==1:
                    actual=entrant_descriptor(rows[0]);selected=next((d for d in descriptors if d==actual),None)
                    if selected is None:own.append('geometry_descriptor_outside_frozen_roster')
                    else:
                        moving=[e for e in config['entrants'] if not e.get('is_static')]
                        if len(moving)!=1 or moving[0].get('car_id')!=rows[0]['car_id'] or moving[0].get('policy')!=selected['policy_key']:
                            own.append('geometry_actual_config_policy_mismatch')
                        geometry_policy=selected['policy_key']
                        key=(selected['policy_key'],case['scenario_id']);seen_geo[key]+=1
                        geometry_passes[selected['policy_key']][case['scenario_id']]=not own and out.get('hazard_pass') is True
                cores.append(digest(physical_core(config)))
                audits.append({'path':label,'split':split,'case_id':case['case_id'],'verified':not own,
                               'scenario_id':case['scenario_id'],'policy_key':geometry_policy,
                               'issues':sorted(set(own)),'outcome':out})
            except Exception as error:
                own.append(f'{type(error).__name__}: {error}')
                audits.append({'path':label,'split':split,'verified':False,'issues':own})
            issues.extend(label+':'+item for item in own)
    if Counter(seen_race)!=Counter(c['scenario_id'] for c in cases):issues.append('incomplete_or_duplicate_planned_race_cases')
    expected_geo=Counter((row['policy_key'],case['scenario_id']) for row in descriptors for case in geometry)
    if seen_geo!=expected_geo:issues.append('incomplete_or_duplicate_four_policy_six_geometry_inventory')
    if any(n!=1 for n in seen_geo.values()):issues.append('duplicate_geometry_attempts_for_policy_case')
    if len(set(cores))>1:issues.append('physical_or_sensor_rules_differ')
    if input_inventory is not None:
        try:
            if capture_scored_inputs(campaign,race_paths+geo_paths,roster['snapshot_path'],input_metadata,engineering_roster_binding)!=input_inventory:
                issues.append('scored_inputs_changed_during_independent_audit')
        except (ValueError,KeyError,OSError) as error:issues.append('scored_inputs_changed_during_independent_audit:'+str(error))
    summary=analyze_directory(directory)
    for policy in summary['observed_leading_policies']:
        passed=geometry_passes.get(policy,{})
        if set(passed)!=set(geo_by_id) or not all(passed.values()):issues.append('six_fresh_geometry_passes_required:'+policy)
    issues=sorted(set(issues));supported=summary['best_policy_supported'] and not issues
    summary['generic_best_policy_supported']=summary['best_policy_supported']
    summary['best_policy_supported']=supported
    summary['best_policies_among_tested']=summary['observed_leading_policies'] if supported else []
    summary['qualification_issues']=sorted(set(summary['qualification_issues']+issues))
    summary['final_championship_audit']={'verified':not issues,'winner_released':supported,'issues':issues,
        'roster_sha256':digest(roster),'snapshot_sha256':roster['snapshot_sha256'],
        'common_physics_assets_sha256':physics_binding(manifest),
        'frozen_engineering_roster_binding':engineering_roster_binding,
        'all_attempts_inventoried':not any('ledger_attempt_inventory_mismatch' in item for item in issues),
        'planned_case_counts':{'championship':4,'geometric-confirmation':24},
        'independently_verified_case_counts':{split:sum(a['split']==split and a.get('verified') is True for a in audits)
            for split in ('championship','geometric-confirmation')},
        'complete_four_policy_six_geometry_inventory':len(geo_paths)==24 and len(set(geo_paths))==24 and seen_geo==expected_geo,
        'all_planned_cases_closed_and_verified':len(audits)==28 and all(a.get('verified') is True for a in audits)
            and len(race_paths)==4 and len(set(race_paths))==4 and Counter(seen_race)==Counter(c['scenario_id'] for c in cases)
            and len(geo_paths)==24 and len(set(geo_paths))==24 and seen_geo==expected_geo
            and not any('ledger_attempt_inventory_mismatch' in item for item in issues),
        'geometry_policy_case_counts':[{'policy_key':policy,'scenario_id':scenario,'count':seen_geo[(policy,scenario)]}
            for policy,scenario in sorted(expected_geo)],
        'auditor_source_sha256':sha(__file__),
        'roster_binding_source_sha256':sha(Path(__file__).with_name('final_roster_binding.py')),
        'scored_input_inventory':input_inventory,
        'metadata_input_sha256':{str(p):sha(p) for p in (roster_path,geo_path,directory/'tournament.json',
            plan_dir/'championship20.json',plan_dir/'geometric-confirmation.json')},
        'geometry_passes':geometry_passes,'audits':audits,'historical_records_modified':False}
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    declare=commands.add_parser('declare-roster');declare.add_argument('--campaign',type=Path,required=True)
    declare.add_argument('--snapshot',type=Path,required=True);declare.add_argument('--policies',type=Path,required=True)
    declare.add_argument('--output',type=Path)
    audit=commands.add_parser('audit');audit.add_argument('--campaign',type=Path,required=True)
    audit.add_argument('--directory',type=Path,required=True);audit.add_argument('--roster',type=Path,required=True)
    audit.add_argument('--geometry-manifest',type=Path,required=True);audit.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='declare-roster':result=declare_roster(args.campaign,args.snapshot,read(args.policies),args.output)
    else:
        result=audit_final(args.campaign,args.directory,args.roster,args.geometry_manifest)
        write_analysis(args.output or args.directory/'final-analysis',result)
        result={key:result[key] for key in ('best_policy_supported','best_policies_among_tested','qualification_issues')}
    print(json.dumps(result,indent=2));return 0 if args.command=='declare-roster' or result['best_policy_supported'] else 1


if __name__=='__main__':raise SystemExit(main())
