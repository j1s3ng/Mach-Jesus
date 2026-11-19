import json

import pytest

from scripts.generate_improvement_cases import formal_cases,verify_case
from scripts.improvement_ledger import Ledger
from scripts.prepare_final_evaluation_protocols import championship_cases,geometric_confirmation_cases,prepare


def test_championship_gives_each_policy_every_slot_once_with_fixed_roster():
    cases=championship_cases();slots={index:[] for index in range(4)}
    assert len(cases)==4 and len({case['scenario']['seed'] for case in cases})==4
    for case in cases:
        verify_case(case)
        assert case['cars']==20 and case['excluded_from_training']
        for slot in range(20):slots[(slot+case['scenario']['rotation'])%4].append(slot)
    assert all(sorted(positions)==list(range(20)) for positions in slots.values())


def test_geometry_extension_has_new_layouts_and_reserved_seeds():
    cases=geometric_confirmation_cases();canonical=formal_cases('holdout')+formal_cases('confirmation')
    old_targets={c['scenario']['obstacle_progress'] for c in canonical if c['is_hazard']}
    assert len(cases)==6 and len({c['scenario']['obstacle_progress'] for c in cases})==6
    assert all(c['scenario']['obstacle_progress'] not in old_targets for c in cases)
    assert len({c['scenario']['obstacle_progress']-c['scenario']['ego_progress'] for c in cases})>=4
    assert len({c['scenario']['ego_lane'] for c in cases})==6
    assert {c['scenario']['seed'] for c in cases}.isdisjoint({c['scenario']['seed'] for c in canonical})
    for case in cases:verify_case(case)


def test_predeclaration_is_durable_and_never_changes_queues(tmp_path):
    campaign=tmp_path/'campaign';campaign.mkdir();ledger=Ledger(campaign/'ledger.jsonl')
    ledger.append('run_initialized',{})
    queue=campaign/'queue.json';queue.write_text('["owned-by-root"]\n');original=queue.read_bytes()
    result=prepare(campaign)
    assert result['championship_cases']==4 and result['fresh_geometry_cases']==6
    assert queue.read_bytes()==original
    assert sum(r['event_type']=='final_evaluation_predeclared' for r in ledger.records())==2
    old=campaign/'later-run';old.mkdir();(old/'config.json').write_text(json.dumps({
        'hazard':{'progress_m':480.,'ego_car_id':'R01'},'entrants':[{'car_id':'R01','start_progress_m':340.,'start_lane_m':-1.}]}))
    assert prepare(campaign)['already_predeclared']
    assert sum(r['event_type']=='final_evaluation_predeclared' for r in ledger.records())==2


def test_already_seen_geometry_cannot_be_declared_fresh(tmp_path):
    campaign=tmp_path/'campaign';old=campaign/'old';old.mkdir(parents=True)
    (old/'config.json').write_text(json.dumps({'hazard':{'progress_m':480.},'entrants':[{'car_id':'R01'}]}))
    with pytest.raises(ValueError,match='already been run'):
        prepare(campaign)
