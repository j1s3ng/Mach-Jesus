"""Verify the already-frozen engineering roster chain without reselecting it."""
from pathlib import Path
import json

from scripts.exact_legacy_demo_scope import verified_legacy_demo_id
from scripts.improvement_ledger import Ledger, OFFICIAL_DETECTOR_SHA256, candidate_descriptor, digest, verify_snapshot, _bundle_contains_descriptor
from scripts.watch_improvement_evaluation import sha

PROTOCOL_SHA256 = '6f45474e2b227741a2c3fa10354f4999f0636b0123f7712c5203e58ce8f91faa'
DRAFT_SHA256 = 'cb49428cef009166fa485e6d1a0b8a74eb86d183caa653e443f7db6022e592f6'


def read(path):
    return json.loads(Path(path).read_text())


def verify_frozen_roster_chain(campaign, roster_path, selection_receipt=None):
    campaign, roster_path = Path(campaign).resolve(), Path(roster_path).resolve()
    ledger = Ledger(campaign / 'ledger.jsonl')
    records, state = ledger.records(), ledger.state()
    unique = lambda kind: [r for r in records if r['event_type'] == kind]
    finals, declarations, bindings = map(unique, ('finalist_frozen_for_confirmation',
        'championship_roster_predeclared', 'final_roster_engineering_selection_bound'))
    if any(len(rows) != 1 for rows in (finals, declarations, bindings)):
        raise ValueError('Exactly one frozen engineering finalist/roster chain is required')
    final, declaration, binding_event = finals[0], declarations[0], bindings[0]
    path = Path(final['payload']['selection_receipt_path']).resolve()
    if selection_receipt is not None and Path(selection_receipt).resolve() != path:
        raise ValueError('A different finalist receipt cannot supply this roster')
    if sha(path) != final['payload'].get('selection_receipt_sha256'):
        raise ValueError('Frozen finalist receipt changed')
    receipt, roster = read(path), read(roster_path)
    identity = digest(roster)
    actual = receipt.get('actual_roster', {})
    if (receipt.get('status') != 'FINALIST_FROZEN_FROM_ENGINEERING_ONLY'
            or receipt.get('formal_outcomes_used_for_selection') is not False
            or receipt.get('selection_protocol_sha256') != PROTOCOL_SHA256
            or receipt.get('prospective_draft_sha256') != DRAFT_SHA256
            or Path(actual.get('path', '')).resolve() != roster_path
            or actual.get('file_sha256') != sha(roster_path)
            or actual.get('canonical_sha256') != identity):
        raise ValueError('Final roster is not the exact engineering-frozen file/order')
    binding_path = Path(receipt['roster_selection_receipt']['path']).resolve()
    if sha(binding_path) != receipt['roster_selection_receipt']['sha256']:
        raise ValueError('Pre-declare engineering roster receipt changed')
    bound = read(binding_path)
    if (bound.get('status') != 'ACTUAL_ROSTER_BOUND_TO_ENGINEERING_BEFORE_DECLARATION'
            or bound.get('roster') != roster or bound.get('roster_canonical_sha256') != identity
            or bound.get('formal_outcomes_used_for_selection') is not False):
        raise ValueError('Pre-declare engineering receipt does not bind this roster')
    for key in ('engineering_selection_report', 'selection_protocol_sha256', 'prospective_draft_sha256',
                'all_eligible_rank', 'registered_bindings', 'finalist_id', 'source_sha256'):
        if bound.get(key) != receipt.get(key):
            raise ValueError('Frozen engineering roster chain differs:' + key)
    learned = receipt.get('selected_roster_learned_ids', [])
    if (len(learned) != 3 or len(set(learned)) != 3
            or bound.get('learned_ids_in_rank_order') != learned
            or receipt.get('finalist_id') != learned[0]):
        raise ValueError('Frozen learned roster/finalist order differs')
    ids = [receipt['baseline_id'], *learned]
    registration = receipt['registered_bindings']
    expected = []
    for cid in ids:
        current = state['candidates'].get(cid, {})
        actual_registration = {'snapshot_sha256': current.get('snapshot_sha256'),
                               'descriptor': candidate_descriptor(current)}
        if actual_registration != registration.get(cid):
            raise ValueError('Frozen registered driver binding changed:' + cid)
        expected.append(actual_registration['descriptor'])
    if (roster.get('policies') != expected or len({r['policy_key'] for r in expected}) != 4
            or expected[0]['policy_key'] != 'reference' or expected[0]['model_sha256'] is not None
            or any(r['model_sha256'] is None for r in expected[1:])
            or any(r['detector_sha256'] != OFFICIAL_DETECTOR_SHA256
                   or r.get('control_parameters') != {'visible_distance_m': 70.} for r in expected)
            or expected.count(registration[receipt['finalist_id']]['descriptor']) != 1):
        raise ValueError('Exact reference-first roster and finalist descriptor are required')
    manifest = verify_snapshot(roster['snapshot_path'])
    if (manifest['snapshot_sha256'] != roster.get('snapshot_sha256')
            or any(not _bundle_contains_descriptor(manifest, d) for d in expected)):
        raise ValueError('Common frozen roster snapshot changed')
    evidence = receipt['engineering_selection_report']
    if sha(evidence['path']) != evidence['sha256']:
        raise ValueError('Frozen engineering evidence bytes changed')
    report = read(evidence['path'])
    if (report.get('selection_semantic_sha256') != evidence.get('selection_semantic_sha256')
            or report.get('selection_semantic_sha256') != digest({k: v for k, v in report.items() if k != 'selection_semantic_sha256'})
            or report.get('status') != 'VERIFIED_ENGINEERING_SELECTION'
            or report.get('formal_outcomes_used') is not False
            or report.get('selection_protocol_sha256') != PROTOCOL_SHA256
            or report.get('prospective_draft_sha256') != DRAFT_SHA256
            or report.get('selected_finalist_id') != receipt['finalist_id']
            or report.get('selected_roster_learned_ids') != learned
            or report.get('provisional_eligible_rank') != receipt['all_eligible_rank']):
        raise ValueError('Frozen evidence does not verify the selected engineering rank')
    bp, dp, fp = binding_event['payload'], declaration['payload'], final['payload']
    if (not binding_event['sequence'] < declaration['sequence'] < final['sequence']
            or Path(bp.get('receipt_path', '')).resolve() != binding_path
            or bp.get('receipt_sha256') != sha(binding_path) or bp.get('roster_sha256') != identity
            or bp.get('engineering_selection_report_sha256') != evidence['sha256']
            or bp.get('engineering_rank_sha256') != digest(receipt['all_eligible_rank'])
            or Path(dp.get('roster_path', '')).resolve() != roster_path
            or dp.get('roster_sha256') != identity or dp.get('snapshot_sha256') != roster['snapshot_sha256']
            or fp.get('roster_sha256') != identity or fp.get('candidate_id') != receipt['finalist_id']
            or fp.get('baseline_id') != receipt['baseline_id']):
        raise ValueError('Engineering binding must precede the unique roster/finalist declarations')
    legacy = verified_legacy_demo_id(campaign, records=records)
    if any(r['event_type'] == 'attempt_scheduled' and r['payload'].get('split') in
           {'confirmation', 'championship', 'geometric-confirmation'}
           and not (legacy is not None and r['payload'].get('attempt_id') == legacy)
           and r['sequence'] <= final['sequence'] for r in records):
        raise ValueError('Reserved evaluation preceded the actual finalist/roster freeze')
    return {'verified': True, 'roster_path': str(roster_path), 'roster_file_sha256': sha(roster_path),
        'roster_sha256': identity, 'roster': roster, 'selection_receipt_path': str(path),
        'selection_receipt_sha256': sha(path), 'engineering_roster_receipt_sha256': sha(binding_path),
        'engineering_report_sha256': evidence['sha256'], 'finalist_id': receipt['finalist_id'],
        'finalist_descriptor': registration[receipt['finalist_id']]['descriptor'],
        'binding_ledger_sha256': binding_event['sha256'], 'roster_declaration_sha256': declaration['sha256'],
        'finalist_declaration_sha256': final['sha256'], 'selection_recomputed_from_test_results': False}
