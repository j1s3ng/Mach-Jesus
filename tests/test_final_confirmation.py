import copy
import datetime as dt
import json

import pytest

from scripts.generate_improvement_cases import formal_cases
from scripts.improvement_ledger import Ledger, OFFICIAL_DETECTOR_SHA256, candidate_descriptor, freeze_snapshot
from scripts.improvement_queue_edit import validate_jobs
from scripts.watch_final_confirmation import (FinalConfirmationEvaluation, confirmation_jobs,
    require_baseline_bootstrap, require_confirmation_baseline, require_confirmation_dispatch,
    DESCRIPTIVE_MODE, sha, validate_selection)
from scripts.watch_final_confirmation import verify_engineering_selection


def setup_campaign(tmp_path, monkeypatch, *, fallback=False, holdout_closed=15, holdout_audit=True):
    root = tmp_path / 'campaign'
    (root / 'formal-cases').mkdir(parents=True)
    cases = formal_cases('confirmation')
    case_path = root / 'formal-cases/confirmation.json'
    case_path.write_text(json.dumps(cases))
    ledger = Ledger(root / 'ledger.jsonl')
    now = dt.datetime.now(dt.timezone.utc)
    ledger.append('run_initialized', {
        'deadline_utc': (now + dt.timedelta(hours=3)).isoformat()})
    model = tmp_path / 'weights.npz'
    model.write_bytes(b'test-only weight content')
    model_sha = sha(model)
    for cid, policy, model_hash in (('baseline', 'reference', None), ('finalist', 'learned', model_sha)):
        snapshot = tmp_path / cid
        freeze_snapshot(snapshot, {'models/weights.npz': model},
            {'policy_parameters': {'visible_distance_m': 70.}})
        ledger.register_candidate(cid, snapshot, {'policy_key': policy, 'model_sha256': model_hash,
            'detector_sha256': OFFICIAL_DETECTOR_SHA256, 'control_parameters': {'visible_distance_m': 70.}})
    holdout = formal_cases('holdout')
    holdout_path = root / 'formal-cases/holdout.json'
    holdout_path.write_text(json.dumps(holdout))
    if fallback:
        import scripts.watch_final_confirmation as module
        authority = json.loads((module.ROOT / 'runs/improvement-12h' / module.FALLBACK_PLAN).read_text())
        authority['baseline_id'] = 'baseline'
        authority['source_file_sha256'].update(holdout=sha(holdout_path), confirmation=sha(case_path))
        path = root / module.FALLBACK_PLAN
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(authority))
        monkeypatch.setattr(module, 'FALLBACK_SHA256', sha(path))
        ledger.append('baseline_bootstrap_descriptive_fallback_predeclared', {
            'file_sha256': sha(path), 'baseline_id': 'baseline',
            'baseline_attempts_at_declaration': [], 'formal_gate_or_approval_relaxed': False,
            'fresh_case_count': 30})
    schedule(ledger, holdout, 'baseline', closed=holdout_closed,
             failed_audit=not holdout_audit, hazard_pass=not fallback)
    assert ledger.bootstrap_incumbent('baseline', holdout, split='holdout')['approved'] is (not fallback)
    evidence = root / 'engineering_selection.json'
    evidence.write_text(json.dumps({'status': 'VERIFIED_ENGINEERING_SELECTION',
        'selected_finalist_id': 'finalist', 'formal_outcomes_used': False}))
    state = ledger.state()
    receipt = root / 'selection.json'
    receipt.write_text(json.dumps({'status': 'FINALIST_FROZEN_FROM_ENGINEERING_ONLY',
        'frozen_utc': now.isoformat(), 'formal_outcomes_used_for_selection': False,
        'baseline_id': 'baseline', 'finalist_id': 'finalist', 'confirmation_cases_sha256': sha(case_path),
        'registered_bindings': {cid: {'snapshot_sha256': reg['snapshot_sha256'],
            'descriptor': candidate_descriptor(reg)} for cid, reg in state['candidates'].items()},
        'engineering_selection_report': {'path': str(evidence), 'sha256': sha(evidence)}}))
    ledger.append('finalist_frozen_for_confirmation', {'selection_receipt_sha256': sha(receipt)})
    # These fixtures exercise confirmation gating; the real engineering auditor
    # has its separate source/scene/ranking tests and is not synthesized here.
    monkeypatch.setattr('scripts.watch_final_confirmation.verify_engineering_selection',
                        lambda report_path, campaign: json.load(open(report_path)))
    def audit(self, attempt, registration, case):
        return {'attempt_id': attempt['attempt_id'], 'verified': attempt['outcome']['audit_verified'],
            'issues': [], 'outcome': attempt['outcome'], 'physics_binding_sha256': 'common',
            'input_identity': attempt['attempt_id']}
    monkeypatch.setattr(FinalConfirmationEvaluation, 'audit_trial', audit)
    evaluator = FinalConfirmationEvaluation(root, receipt)
    return root, ledger, cases, receipt, evaluator


def schedule(ledger, cases, cid, *, closed=15, pace=100., failed_audit=False, hazard_pass=True):
    registration = ledger.state()['candidates'][cid]
    descriptor = candidate_descriptor(registration)
    endings = []
    for index, case in enumerate(cases):
        aid = cid + '-' + case['case_id']
        hazard = case['is_hazard']
        outcome = {'audit_verified': not failed_audit, 'snapshot_verified': True,
            'homogeneous_field': True, 'snapshot_sha256': registration['snapshot_sha256'],
            'rules_sha256': case['scenario_id'], 'hazard_pass': hazard_pass if hazard else None,
            'entrants': [{'car_id': f'R{i+1:02d}', 'policy_key': descriptor['policy_key'],
                'model_sha256': descriptor['model_sha256'],
                'runtime_detector_checkpoint_sha256': OFFICIAL_DETECTOR_SHA256,
                'runtime_policy_parameters': {'visible_distance_m': 70.},
                'valid_finish': not hazard, 'clean_finish': not hazard,
                'finish_time_s': None if hazard else pace, 'progress_m': 5300.,
                'collision_samples': 0, 'track_violation_steps': 0, 'position_resets_after_spawn': 0}
                for i in range(case['expected_entrants'])]}
        ledger.append('attempt_scheduled', {'attempt_id': aid, 'candidate_id': cid,
            'scenario_id': case['scenario_id'], 'split': case['split'], 'stage': 'promotion',
            'comparison_key': case['comparison_key'], 'scenario_sha256': case['scenario_sha256'],
            'rules_sha256': outcome['rules_sha256'], 'snapshot_sha256': registration['snapshot_sha256'],
            'expected_entrants': case['expected_entrants'], 'is_hazard': hazard, 'tier': case['tier'],
            'output_dir': str(ledger.path.parent / 'unused' / aid)})
        ending = {'attempt_id': aid, 'status': 'completed', 'outcome': outcome}
        endings.append(ending)
        if index < closed:
            ledger.append('attempt_finished', ending)
    return endings


def test_complete30_promotes_through_existing_confirmation_gate_once(tmp_path, monkeypatch):
    root, ledger, cases, receipt, evaluator = setup_campaign(tmp_path, monkeypatch)
    schedule(ledger, cases, 'baseline')
    schedule(ledger, cases, 'finalist', pace=95.)
    before = ledger.records()
    report = evaluator.once(act=True)
    assert report['incumbent_id'] == 'finalist'
    assert report['decisions']['promotion']['split'] == 'confirmation'
    assert ledger.records()[:len(before)] == before
    after = ledger.records()
    evaluator.once(act=True)
    assert ledger.records() == after
    assert not (root / 'progress.json').exists()
    assert not report['bootstrap_performed']


def test_incomplete_or_failed_audit_never_counts_as_complete_suite(tmp_path, monkeypatch):
    _, ledger, cases, _, evaluator = setup_campaign(tmp_path, monkeypatch)
    schedule(ledger, cases, 'baseline')
    endings = schedule(ledger, cases, 'finalist', closed=14, pace=95.)
    before = ledger.records()
    report = evaluator.once(act=True)
    assert not report['formal']['finalist']['complete']
    assert ledger.records() == before
    bad = copy.deepcopy(endings[-1]); bad['outcome']['audit_verified'] = False
    ledger.append('attempt_finished', bad)
    report = evaluator.once(act=True)
    assert not report['formal']['finalist']['independently_verified_complete_suite']
    assert report['incumbent_id'] == 'baseline'


def test_case_replacement_or_changed_selection_is_rejected(tmp_path, monkeypatch):
    root, _, cases, receipt, evaluator = setup_campaign(tmp_path, monkeypatch)
    cases[0]['scenario']['seed'] += 1
    (root / 'formal-cases/confirmation.json').write_text(json.dumps(cases))
    with pytest.raises(ValueError, match='changed'):
        evaluator.once(act=True)


def test_full30_job_pack_preserves_every_case_without_failure_dependencies(tmp_path, monkeypatch):
    root, _, cases, receipt, _ = setup_campaign(tmp_path, monkeypatch)
    pack = confirmation_jobs(root, receipt)
    validate_jobs(pack['jobs'])
    assert len(pack['jobs']) == 30
    assert len({j['job_id'] for j in pack['jobs']}) == 30
    assert all('--split' in j['args'] and 'confirmation' in j['args'] for j in pack['jobs'])
    assert all(not j.get('requires') for j in pack['jobs'])
    assert sum('--secondary-cars' in j['args'] and j['args'][j['args'].index('--secondary-cars')+1] == '8'
               for j in pack['jobs']) == 8
    assert not pack['enqueue_authorized_by_this_function']


def test_unfrozen_candidate_or_extra_confirmation_arm_is_rejected(tmp_path, monkeypatch):
    root, ledger, cases, receipt, _ = setup_campaign(tmp_path, monkeypatch)
    ledger.append('attempt_scheduled', {'attempt_id': 'another-arm', 'candidate_id': 'another',
        'split': 'confirmation', 'stage': 'promotion'})
    with pytest.raises(ValueError, match='Another candidate'):
        validate_selection(root, receipt, root / 'formal-cases/confirmation.json')


def test_incumbent_seed_alone_does_not_substitute_for_complete_bootstrap(tmp_path, monkeypatch):
    root, ledger, _, _, _ = setup_campaign(tmp_path, monkeypatch)
    unverified = Ledger(tmp_path / 'seeded/ledger.jsonl')
    unverified.append('run_initialized', {'incumbent_id': 'baseline'})
    registration = ledger.state()['candidates']['baseline']
    unverified.register_candidate('baseline', registration['snapshot_path'], registration['descriptor'])
    with pytest.raises(ValueError, match='Complete verified'):
        require_baseline_bootstrap(unverified, 'baseline')


def test_whole30_dispatch_gate_preserves_other_final_work_and_rejects_modified_pack(tmp_path, monkeypatch):
    root, _, _, receipt, _ = setup_campaign(tmp_path, monkeypatch)
    pack = confirmation_jobs(root, receipt)
    proof = require_confirmation_dispatch(pack, root, receipt, remaining_other_required_seconds=300)
    assert proof['admitted']
    with pytest.raises(ValueError, match='whole paired30'):
        require_confirmation_dispatch(pack, root, receipt, remaining_other_required_seconds=3000)
    changed = copy.deepcopy(pack); changed['jobs'].pop()
    with pytest.raises(ValueError, match='plan changed'):
        require_confirmation_dispatch(changed, root, receipt, remaining_other_required_seconds=300)


def test_bare_status_marker_is_not_verified_engineering_selection(tmp_path):
    marker = tmp_path / 'unproved.json'
    marker.write_text(json.dumps({'status': 'VERIFIED_ENGINEERING_SELECTION',
        'selected_finalist_id': 'finalist', 'formal_outcomes_used': False}))
    with pytest.raises(ValueError, match='semantic hash'):
        verify_engineering_selection(marker, tmp_path)


def test_rejected_complete_hazard_bootstrap_runs_all30_as_descriptive_only(tmp_path, monkeypatch):
    root, ledger, cases, receipt, evaluator = setup_campaign(tmp_path, monkeypatch, fallback=True)
    pack = confirmation_jobs(root, receipt)
    assert len(pack['jobs']) == 30
    assert pack['baseline_bootstrap_proof']['evaluation_mode'] == DESCRIPTIVE_MODE
    assert not pack['baseline_bootstrap_proof']['formal_promotion_allowed']
    schedule(ledger, cases, 'baseline')
    schedule(ledger, cases, 'finalist', pace=95.)
    before = ledger.records()
    monkeypatch.setattr(ledger.__class__, 'promote', lambda *a, **kw: pytest.fail('Descriptive fallback promoted'))
    report = evaluator.once(act=True)
    assert report['evaluation_mode'] == DESCRIPTIVE_MODE
    assert report['decisions']['stateless_paired_gate_preview']['approved']
    assert not report['formal_promotion_allowed']
    assert not report['overall_formal_approval'] and not report['formal_release']
    assert not report['decisions_enabled'] and report['decisions_requested']
    assert report['incumbent_id'] is None
    assert ledger.records() == before
    assert 'DESCRIPTIVE ONLY' in (evaluator.output / 'REPORT.md').read_text()


@pytest.mark.parametrize('closed,audit', [(14, True), (15, False)])
def test_descriptive_fallback_rejects_missing_or_infrastructure_holdout(tmp_path, monkeypatch, closed, audit):
    root, ledger, _, receipt, evaluator = setup_campaign(tmp_path, monkeypatch,
        fallback=True, holdout_closed=closed, holdout_audit=audit)
    with pytest.raises(ValueError, match='exclusively genuine hazard rejection'):
        confirmation_jobs(root, receipt)
    with pytest.raises(ValueError, match='exclusively genuine hazard rejection'):
        evaluator.once(act=True)
    assert ledger.state()['incumbent_id'] is None


def test_descriptive_fallback_requires_fresh_independent_reaudit(tmp_path, monkeypatch):
    root, ledger, _, _, evaluator = setup_campaign(tmp_path, monkeypatch, fallback=True)
    original = evaluator.audit_trial
    def changed(attempt, registration, case):
        result = original(attempt, registration, case)
        result['verified'] = False
        return result
    with pytest.raises(ValueError, match='independent verification'):
        require_confirmation_baseline(ledger, 'baseline', root, changed)


def test_descriptive_fallback_requires_prior_declaration_and_recorded_rejection(tmp_path, monkeypatch):
    root, ledger, _, _, evaluator = setup_campaign(tmp_path, monkeypatch, fallback=True)
    records = ledger.records()
    monkeypatch.setattr(ledger, 'records', lambda: [r for r in records if r['event_type'] != 'bootstrap_decision'])
    with pytest.raises(ValueError, match='rejected bootstrap decision'):
        require_confirmation_baseline(ledger, 'baseline', root, evaluator.audit_trial)
    altered = copy.deepcopy(records)
    next(r for r in altered if r['event_type'] == 'baseline_bootstrap_descriptive_fallback_predeclared')['sequence'] = 100000
    monkeypatch.setattr(ledger, 'records', lambda: altered)
    with pytest.raises(ValueError, match='before all baseline attempts'):
        require_confirmation_baseline(ledger, 'baseline', root, evaluator.audit_trial)


def test_only_exact_verified_legacy_id_is_ignored_not_other_engineering_confirmation(tmp_path, monkeypatch):
    root, ledger, _, receipt, _ = setup_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr('scripts.watch_final_confirmation.verified_legacy_demo_id',
                        lambda *a, **kw: 'presentation-00517-original-20')
    ledger.append('attempt_scheduled', {'attempt_id': 'presentation-00517-original-20',
        'candidate_id': 'presentation-00517-original-20', 'split': 'confirmation', 'stage': 'engineering'})
    validate_selection(root, receipt, root / 'formal-cases/confirmation.json')
    ledger.append('attempt_scheduled', {'attempt_id': 'another-engineering-demo',
        'candidate_id': 'another-engineering-demo', 'split': 'confirmation', 'stage': 'engineering'})
    with pytest.raises(ValueError, match='Another candidate'):
        validate_selection(root, receipt, root / 'formal-cases/confirmation.json')


def test_missing_attempt_id_cannot_equal_absent_legacy_exception(tmp_path, monkeypatch):
    root, ledger, _, receipt, _ = setup_campaign(tmp_path, monkeypatch)
    state, records = ledger.state(), ledger.records()
    records.append({'sequence': records[-1]['sequence'] + 1, 'event_type': 'attempt_scheduled',
                    'payload': {'candidate_id': 'another', 'split': 'confirmation'}})
    monkeypatch.setattr(Ledger, 'records', lambda self: records)
    with pytest.raises(ValueError, match='Another candidate'):
        validate_selection(root, receipt, root / 'formal-cases/confirmation.json', state=state)
