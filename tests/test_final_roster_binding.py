"""Frozen evidence/order binds both final packs; a second roster is not a substitute."""
import copy
import json

import pytest

from scripts import final_roster_binding as binding
from scripts.improvement_ledger import Ledger, OFFICIAL_DETECTOR_SHA256, candidate_descriptor, digest, freeze_snapshot


def fixture(tmp_path, *, late_binding=False):
    campaign=tmp_path/'campaign';campaign.mkdir()
    ledger=Ledger(campaign/'ledger.jsonl');ledger.append('run_initialized',{})
    files={}
    for i in range(3):
        path=tmp_path/f'model{i}.npz';path.write_bytes(('unique model'+str(i)).encode());files[path.name]=path
    snapshot=tmp_path/'common-source'
    manifest=freeze_snapshot(snapshot,files,{'policy_parameters':{'visible_distance_m':70.}})
    policies=[{'policy_key':'reference','model_sha256':None,'detector_sha256':OFFICIAL_DETECTOR_SHA256,
               'control_parameters':{'visible_distance_m':70.}}]
    policies.extend({**policies[0],'policy_key':'neural'+str(i),'model_sha256':binding.sha(files[f'model{i}.npz'])} for i in range(3))
    ids=['baseline','n0','n1','n2']
    for cid,desc in zip(ids,policies):ledger.register_candidate(cid,snapshot,desc)
    registrations={cid:{'snapshot_sha256':reg['snapshot_sha256'],'descriptor':candidate_descriptor(reg)}
        for cid,reg in ledger.state()['candidates'].items()}
    rank=[{'candidate_id':cid,'policy_key':desc['policy_key'],'model_sha256':desc['model_sha256']} for cid,desc in zip(ids[1:],policies[1:])]
    report={'status':'VERIFIED_ENGINEERING_SELECTION','formal_outcomes_used':False,
        'selected_finalist_id':'n0','selected_roster_learned_ids':ids[1:],
        'provisional_eligible_rank':rank,'selection_protocol_sha256':binding.PROTOCOL_SHA256,
        'prospective_draft_sha256':binding.DRAFT_SHA256}
    report['selection_semantic_sha256']=digest(report)
    report_path=campaign/'engineering-report.json';report_path.write_text(json.dumps(report))
    evidence={'path':str(report_path),'sha256':binding.sha(report_path),'selection_semantic_sha256':report['selection_semantic_sha256']}
    roster={'schema_version':1,'snapshot_path':str(snapshot),'snapshot_sha256':manifest['snapshot_sha256'],
        'policies':policies,'excluded_from_training':True}
    roster_path=campaign/'roster.json';roster_path.write_text(json.dumps(roster))
    bound={'status':'ACTUAL_ROSTER_BOUND_TO_ENGINEERING_BEFORE_DECLARATION','roster':roster,
        'roster_canonical_sha256':digest(roster),'formal_outcomes_used_for_selection':False,
        'engineering_selection_report':evidence,'selection_protocol_sha256':binding.PROTOCOL_SHA256,
        'prospective_draft_sha256':binding.DRAFT_SHA256,'all_eligible_rank':rank,
        'registered_bindings':registrations,'finalist_id':'n0','source_sha256':{'test-source':'immutable'},
        'learned_ids_in_rank_order':ids[1:]}
    bound_path=campaign/'engineering-roster.json';bound_path.write_text(json.dumps(bound))
    bp={'receipt_path':str(bound_path),'receipt_sha256':binding.sha(bound_path),'roster_sha256':digest(roster),
        'engineering_selection_report_sha256':evidence['sha256'],'engineering_rank_sha256':digest(rank)}
    dp={'roster_path':str(roster_path),'roster_sha256':digest(roster),'snapshot_sha256':manifest['snapshot_sha256']}
    if late_binding:ledger.append('championship_roster_predeclared',dp)
    ledger.append('final_roster_engineering_selection_bound',bp)
    if not late_binding:ledger.append('championship_roster_predeclared',dp)
    receipt={**bound,'status':'FINALIST_FROZEN_FROM_ENGINEERING_ONLY','baseline_id':'baseline',
        'selected_roster_learned_ids':ids[1:],
        'actual_roster':{'path':str(roster_path),'file_sha256':binding.sha(roster_path),'canonical_sha256':digest(roster)},
        'roster_selection_receipt':{'path':str(bound_path),'sha256':binding.sha(bound_path)}}
    receipt_path=campaign/'selection.json';receipt_path.write_text(json.dumps(receipt))
    ledger.append('finalist_frozen_for_confirmation',{'selection_receipt_path':str(receipt_path),
        'selection_receipt_sha256':binding.sha(receipt_path),'roster_sha256':digest(roster),
        'candidate_id':'n0','baseline_id':'baseline'})
    return campaign,ledger,roster_path,receipt_path,report_path


def test_exact_frozen_roster_chain_is_verified_without_test_result_selection(tmp_path):
    campaign,ledger,roster,receipt,_=fixture(tmp_path)
    before=ledger.path.read_bytes()
    proof=binding.verify_frozen_roster_chain(campaign,roster,receipt)
    assert proof['verified'] and proof['finalist_id']=='n0'
    assert not proof['selection_recomputed_from_test_results']
    assert ledger.path.read_bytes()==before


def test_even_identical_bytes_at_a_second_roster_path_cannot_substitute(tmp_path):
    campaign,_,roster,receipt,_=fixture(tmp_path)
    second=campaign/'second-roster.json';second.write_bytes(roster.read_bytes())
    with pytest.raises(ValueError,match='exact engineering-frozen file/order'):
        binding.verify_frozen_roster_chain(campaign,second,receipt)


def test_retroactive_engineering_binding_and_second_roster_declaration_reject(tmp_path):
    campaign,ledger,roster,receipt,_=fixture(tmp_path,late_binding=True)
    with pytest.raises(ValueError,match='must precede'):
        binding.verify_frozen_roster_chain(campaign,roster,receipt)
    ledger.append('championship_roster_predeclared',{'roster_sha256':'unrelated'})
    with pytest.raises(ValueError,match='Exactly one'):
        binding.verify_frozen_roster_chain(campaign,roster,receipt)


@pytest.mark.parametrize('changed', ['roster','selection','engineering_report'])
def test_changed_bound_file_bytes_invalidate_chain(tmp_path,changed):
    campaign,_,roster,receipt,report=fixture(tmp_path)
    path={'roster':roster,'selection':receipt,'engineering_report':report}[changed]
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError):binding.verify_frozen_roster_chain(campaign,roster,receipt)


def test_prior_reserved_schedule_cannot_be_hidden_by_later_declarations(tmp_path):
    campaign,ledger,roster,receipt,_=fixture(tmp_path)
    records=ledger.records()
    # Only the read-only input boundary is perturbed; no real ledger is rewritten.
    records.insert(-1,{'event_type':'attempt_scheduled','sequence':0,
        'payload':{'attempt_id':'too-early','split':'championship'}})
    original=binding.Ledger
    class EvidenceLedger:
        def __init__(self,path):self.inner=original(path)
        def state(self):return self.inner.state()
        def records(self):return copy.deepcopy(records)
    from unittest.mock import patch
    with patch.object(binding,'Ledger',EvidenceLedger),pytest.raises(ValueError,match='preceded'):
        binding.verify_frozen_roster_chain(campaign,roster,receipt)
