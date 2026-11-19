#!/usr/bin/env python3
"""Predeclare final mixed-driver races and fresh-layout hazard confirmation.

Writes plans only. It never edits queues, runs a simulator, trains, or promotes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.improvement_ledger import Ledger,digest,scenario_identity
except ModuleNotFoundError:
    from improvement_ledger import Ledger,digest,scenario_identity


def case_definition(name,split,values,kind):
    scenario=scenario_identity(values);identity=digest(scenario);hazard=scenario['scenario']=='stopped'
    count=1 if hazard else scenario['cars']
    return {'schema_version':1,'case_id':name,'split':split,'scenario_id':identity[:20],
            'scenario_sha256':identity,'comparison_key':identity,'kind':kind,'is_hazard':hazard,
            'tier':str(scenario['cars']),'cars':scenario['cars'],'expected_entrants':count,'evaluated_entrants':count,
            'excluded_from_training':True,'scenario':scenario,
            'cli_args':[item for key,value in scenario.items() for item in ('--'+key.replace('_','-'),str(value))]}


def championship_cases():
    return [case_definition(f'championship20-roster-shift{rotation}','championship',{
        'scenario':'race','cars':20,'rotation':rotation,'seed':1150001+211*rotation,
        'spacing':14.,'seconds':480.,'grid_jitter':.3},'mixed_driver_championship') for rotation in range(4)]


def geometric_confirmation_cases():
    # Chosen from the shared map, not from these cases' unobserved outcomes.
    layouts=[(480.,340.,-1.,'right',2),(1380.,1295.,1.1,'right',3),
             (1960.,1770.,-1.7,'left',3),(2470.,2350.,.6,'right',2),
             (3680.,3540.,-.8,'right',3),(4320.,4160.,1.9,'left',3)]
    cases=[]
    for index,(target,ego,lane,blocked,cars) in enumerate(layouts):
        cases.append(case_definition(f'geometric-confirmation-{index+1}-target{target:g}',
            'geometric-confirmation',{'scenario':'stopped','cars':cars,'rotation':0,
             'seed':1300001+137*index,'spacing':14.,'seconds':120.,'grid_jitter':0.,
             'obstacle_progress':target,'ego_progress':ego,'ego_lane':lane,'blocked_side':blocked},
            'fresh_layout_hazard_confirmation'))
    return cases


def known_hazard_layouts(campaign):
    layouts=[]
    for path in Path(campaign).rglob('config.json'):
        config=json.loads(path.read_text());hazard=config.get('hazard')
        if not isinstance(hazard,dict) or 'progress_m' not in hazard:continue
        entrants=config.get('entrants',[])
        if not entrants:continue
        ego=next((e for e in entrants if e['car_id']==hazard.get('ego_car_id','R01')),entrants[0])
        layouts.append({'config':str(path),'target_progress_m':hazard['progress_m'],
                        'ego_progress_m':ego.get('start_progress_m'),'ego_lane_m':ego.get('start_lane_m')})
    return layouts


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=content:raise ValueError(f'Refusing to change an existing predeclaration: {path}')
        return
    path.write_text(content);path.chmod(0o444)


def prepare(campaign):
    campaign=Path(campaign).resolve();destination=campaign/'final-evaluation-plan'
    mixed=championship_cases();geometry=geometric_confirmation_cases()
    if (destination/'protocol.json').exists():
        if json.loads((destination/'championship20.json').read_text())!=mixed or json.loads((destination/'geometric-confirmation.json').read_text())!=geometry:
            raise ValueError('Existing predeclared cases differ; a revised plan needs a new version')
        return {'directory':str(destination),'championship_cases':len(mixed),'fresh_geometry_cases':len(geometry),
                'championship_sha256':digest(mixed),'fresh_geometry_sha256':digest(geometry),
                'already_predeclared':True,'queues_changed':False}
    known=known_hazard_layouts(campaign)
    old_targets={float(row['target_progress_m']) for row in known}
    if any(abs(case['scenario']['obstacle_progress']-old)<1. for case in geometry for old in old_targets):
        raise ValueError('A proposed fresh target has already been run; preserve this plan and choose a new version prospectively')
    protocol={'schema_version':1,'status':'prospective_plan_not_queued','source':'shared reference map and predefined starts, no outcomes from these cases',
        'championship':{'policy_count':4,'copies_per_policy':5,'field_size':20,'heats':4,
          'roster':'Choose and freeze exactly4 driver descriptors/weight hashes before any championship launch; keep roster order fixed.',
          'control_and_perception':'Every entrant uses one common selected guard-parameter set and one common detector checkpoint. Evaluate detector alternatives separately.',
          'grid_assignment':'policy_at_slot=(slot+rotation)%4; each policy occupies every one of20 slots exactly once across4 heats.',
          'replication':'Distinct reserved seeds and0.3m jitter; no driver/source/model changes between heats.',
          'scoring':'Per-heat winner is lowest valid common-clock finish. Policy rank: clean rate, valid rate, mean position, mean valid time. All scheduled entrants and DNFs count.',
          'qualification':'Require all4 closed audited heats, identical exact full slot distributions, and at least one clean finish by leader. Scope is these tested drivers/schedule, not statistical superiority.',
          'format':'Identical vehicle/tires/actuation, physical sensors, track/reference and physical staggered runoff parking; no opponent ground truth in control.'},
        'fresh_geometry':{'case_count':6,'objective':'Ego passes static target+20m with zero scored overlap samples, track violations and pose resets; source/model/physical bindings verified.',
          'freshness':'New target locations, approach distances and lane starts compared with recorded campaign configs at predeclaration.',
          'canonical_hazard_limit':'Canonical holdout/confirmation hazards repeat300/1000/3000m layouts; distinct seeds alone are weak generalization evidence.',
          'qualification':'Additional final qualification evidence; it does not replace or rewrite the canonical15-case holdout gates.'},
        'data_policy':'All planned championship and geometric-confirmation outputs are excluded from training and tuning.',
        'launcher_note':'Before queuing, include championship/geometric-confirmation in dataset exclusion flags; this script does not edit the launcher or queues.',
        'previously_recorded_hazard_targets_m':sorted(old_targets)}
    write_new(destination/'championship20.json',mixed);write_new(destination/'geometric-confirmation.json',geometry)
    write_new(destination/'protocol.json',protocol)
    ledger=Ledger(campaign/'ledger.jsonl')
    for name,cases in (('championship20',mixed),('geometric-confirmation',geometry)):
        payload={'plan':name,'path':str(destination/(name+'.json')),'case_count':len(cases),
                 'definition_sha256':digest(cases),'excluded_from_training':True,'queue_mutations':False,
                 'protocol_path':str(destination/'protocol.json')}
        if not any(r['event_type']=='final_evaluation_predeclared' and r['payload'].get('definition_sha256')==payload['definition_sha256'] for r in ledger.records()):
            ledger.append('final_evaluation_predeclared',payload)
    return {'directory':str(destination),'championship_cases':len(mixed),'fresh_geometry_cases':len(geometry),
            'championship_sha256':digest(mixed),'fresh_geometry_sha256':digest(geometry),'queues_changed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.campaign),indent=2))


if __name__=='__main__':main()
