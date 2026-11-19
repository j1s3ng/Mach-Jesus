"""The complete final schedule cannot pass by timing each pack separately."""
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import json

import pytest

from scripts import final_schedule_admission as admission
from scripts.improvement_ledger import Ledger


def fixture(tmp_path, monkeypatch, *, hours=4):
    monkeypatch.setattr(admission, 'ROOT', tmp_path)
    monkeypatch.setattr(admission, 'sources', lambda: {'test-source': 'immutable'})
    campaign = tmp_path / 'campaign'; campaign.mkdir()
    ledger = Ledger(campaign / 'ledger.jsonl')
    ledger.append('run_initialized', {'deadline_utc': (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()})
    def job(identity, cars, stopped=False):
        return {'job_id': identity, 'output': 'outputs/' + identity,
            'estimated_wall_seconds': 90 if stopped else {4: 150, 8: 300, 20: 600}[cars],
            'max_wall_seconds': 1800,
            'args': ['--cars', str(cars), '--scenario', 'stopped' if stopped else 'race', '--seconds', '120' if stopped else '480']}
    pending, closed = job('current-field4', 4), job('closed-field20', 20)
    (campaign / 'queue.json').write_text(json.dumps([closed, pending]))
    (campaign / 'queue_status.json').write_text(json.dumps({'completed': {'closed-field20': {'status': 'completed'}},
        'current': pending['job_id'], 'status': 'running'}))
    conf = {'jobs': [job(f'confirm-{arm}-{size}-{i}', size) for arm in range(2) for size in (4, 8, 20) for i in range(4)]
        + [job(f'confirm-{arm}-hazard-{i}', 3, True) for arm in range(2) for i in range(3)],
        'estimated_wall_seconds': 8940}
    final = {'jobs': [job('champ-' + str(i), 20) for i in range(4)] + [job('geometry-' + str(i), 3, True) for i in range(24)],
             'estimated_complete_pack_seconds': 6480, 'actual_roster_path': str(campaign / 'roster.json'),
             'actual_roster_sha256': 'the-frozen-roster', 'policies': ['reference', 'n1', 'n2', 'n3'],
             'snapshot_sha256': 'frozen-snapshot', 'snapshot_path': str(campaign / 'frozen-source')}
    selection = campaign / 'selection.json'; selection.write_text('{}')
    def frozen_binding(_campaign, path, receipt=None):
        assert str(path) == str(campaign / 'roster.json')
        return {'verified': True, 'roster_sha256': 'the-frozen-roster',
            'roster': {'policies': ['reference', 'n1', 'n2', 'n3'],
                'snapshot_sha256': 'frozen-snapshot', 'snapshot_path': str(campaign / 'frozen-source')}}
    monkeypatch.setattr(admission, 'verify_frozen_roster_chain', frozen_binding)
    phases = [{'phase': 'report_video_export_review', 'seconds': 1200, 'reason': 'Preserve media/report review'},
              {'phase': 'cleanup', 'seconds': 300, 'reason': 'Hard deadline process cleanup'}]
    monkeypatch.setattr(admission, 'rate_authority', lambda _c: {'path': 'immutable-authority', 'sha256': 'rates',
        'rates_seconds': {'known_hazard': 45, 'field4': 170, 'field8': 270, 'field20': 560, 'capacity_opener': 180},
        'closed_run_measurements': []})
    calls = []
    def conf_guard(*a, **kw):
        calls.append(('confirmation', kw))
        return {'admitted': True}
    def final_guard(*a, **kw): calls.append(('final28', kw)); return True
    monkeypatch.setattr(admission, 'require_confirmation_dispatch', conf_guard)
    monkeypatch.setattr(admission, 'require_execution_bindings', final_guard)
    return campaign, ledger, conf, final, selection, phases, calls


def approved(campaign, ledger, conf, final, selection, phases):
    forecast = admission.prepare_joint_forecast(campaign, conf, final, selection, phases)
    forecast['status'] = admission.APPROVED
    path = campaign / 'approved-joint-forecast.json'; path.write_text(json.dumps(forecast))
    ledger.append(admission.APPROVAL_EVENT, {'file_sha256': admission.sha(path),
        'all_mandatory_work_included': True, 'physics_cases_or_safety_gates_changed': False})
    return path


def test_measured_both_packs_plus_every_unique_pending_job_and_reserve_are_counted(tmp_path, monkeypatch):
    campaign, ledger, conf, final, selection, phases, calls = fixture(tmp_path, monkeypatch)
    path = approved(campaign, ledger, conf, final, selection, phases)
    proof = admission.require_joint_final_admission(campaign, conf, final, selection, path)
    assert proof['seconds'] == {'pending_queue': 170, 'confirmation_measured': 8270, 'final28_measured': 3320,
        'other_required': 1500, 'whole_required_measured': 13260, 'confirmation_nominal_pack': 8940,
        'final28_nominal_pack': 6480, 'nominal_both_packs': 15420}
    assert [name for name, _ in calls] == ['confirmation', 'final28']
    assert calls[0][1]['remaining_other_required_seconds'] == 4690
    assert calls[0][1]['reserve_seconds'] == 300
    assert proof['counts']['pending_queue'] == 1


def test_cpu_preparation_neither_approves_nor_enqueues(tmp_path, monkeypatch):
    campaign, ledger, conf, final, selection, phases, _ = fixture(tmp_path, monkeypatch)
    before = ledger.path.read_bytes(), (campaign / 'queue.json').read_bytes()
    draft = admission.prepare_joint_forecast(campaign, conf, final, selection, phases)
    assert draft['status'].startswith('DRAFT')
    assert before == (ledger.path.read_bytes(), (campaign / 'queue.json').read_bytes())
    with pytest.raises(ValueError, match='execute=True'):
        admission.append_joint_final_jobs(campaign, conf, final, selection, 'unused', campaign / 'receipt.json')


@pytest.mark.parametrize('defect', ['missing_case', 'duplicate', 'drop_cleanup', 'drop_review', 'unapproved'])
def test_missing_work_duplicate_identity_or_unapproved_plan_rejects(tmp_path, monkeypatch, defect):
    campaign, ledger, conf, final, selection, phases, _ = fixture(tmp_path, monkeypatch)
    path = approved(campaign, ledger, conf, final, selection, phases)
    if defect == 'missing_case': conf['jobs'].pop()
    elif defect == 'duplicate': final['jobs'][0] = copy.deepcopy(conf['jobs'][0])
    elif defect in ('drop_cleanup', 'drop_review'):
        phases[-1 if defect == 'drop_cleanup' else 0]['seconds'] = 1
        with pytest.raises(ValueError, match='Retain at least'):
            admission.prepare_joint_forecast(campaign, conf, final, selection, phases)
        return
    elif defect == 'unapproved':
        r = json.loads(path.read_text()); r['status'] = 'DRAFT'; path.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        admission.require_joint_final_admission(campaign, conf, final, selection, path)


@pytest.mark.parametrize('change', ['queue', 'completion', 'pack', 'rates'])
def test_forecast_is_invalidated_by_new_pending_work_closed_status_or_pack_changes(tmp_path, monkeypatch, change):
    campaign, ledger, conf, final, selection, phases, _ = fixture(tmp_path, monkeypatch)
    path = approved(campaign, ledger, conf, final, selection, phases)
    if change == 'queue':
        q = json.loads((campaign / 'queue.json').read_text()); q.append({**q[-1], 'job_id': 'another', 'output': 'another'})
        (campaign / 'queue.json').write_text(json.dumps(q))
    elif change == 'completion':
        p = campaign / 'queue_status.json'; r = json.loads(p.read_text()); r['completed']['current-field4'] = {'status': 'completed'}; r['current'] = None; p.write_text(json.dumps(r))
    elif change == 'pack': conf['jobs'][0]['args'] += ['--pointcloud-stride', '5']
    elif change == 'rates':
        original = admission.rate_authority(campaign)
        original['rates_seconds']['field20'] = 559
        monkeypatch.setattr(admission, 'rate_authority', lambda _c: original)
    with pytest.raises(ValueError, match='stale or changed'):
        admission.require_joint_final_admission(campaign, conf, final, selection, path)


def test_both_packs_cannot_be_admitted_when_combined_required_work_exceeds_deadline(tmp_path, monkeypatch):
    campaign, ledger, conf, final, selection, phases, calls = fixture(tmp_path, monkeypatch, hours=3)
    path = approved(campaign, ledger, conf, final, selection, phases)
    with pytest.raises(ValueError, match=r'full30\+28'):
        admission.require_joint_final_admission(campaign, conf, final, selection, path)
    assert not calls


def test_expired_approval_and_advance_during_deep_guards_reject(tmp_path, monkeypatch):
    campaign, ledger, conf, final, selection, phases, _ = fixture(tmp_path, monkeypatch)
    path = approved(campaign, ledger, conf, final, selection, phases)
    with pytest.raises(ValueError, match='fresh at actual append'):
        admission.require_joint_final_admission(campaign, conf, final, selection, path,
            now=datetime.now(timezone.utc) + timedelta(seconds=301))
    def advanced(*a, **kw):
        p = campaign / 'queue_status.json'; r = json.loads(p.read_text()); r['completed']['current-field4'] = {'status': 'completed'}; p.write_text(json.dumps(r)); return True
    monkeypatch.setattr(admission, 'require_execution_bindings', advanced)
    with pytest.raises(ValueError, match='Queue advanced'):
        admission.require_joint_final_admission(campaign, conf, final, selection, path)


def test_explicit_temporary_append_holds_writer_lock_checks_both_guards_and_adds_exact58(tmp_path, monkeypatch):
    campaign, ledger, conf, final, selection, phases, calls = fixture(tmp_path, monkeypatch)
    path = approved(campaign, ledger, conf, final, selection, phases)
    guard = admission.require_execution_bindings
    def check_locked(*a, **kw):
        with (campaign / 'queue_edit.lock').open('a+') as lock:
            with pytest.raises(BlockingIOError): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return guard(*a, **kw)
    monkeypatch.setattr(admission, 'require_execution_bindings', check_locked)
    proof = admission.append_joint_final_jobs(campaign, conf, final, selection, path, campaign / 'receipt.json', execute=True)
    assert proof['enqueued_jobs'] == 58
    assert len(json.loads((campaign / 'queue.json').read_text())) == 60
    assert [n for n, _ in calls] == ['confirmation', 'final28']
    assert [r['event_type'] for r in ledger.records()][-2:] == ['joint_final_work_admitted_before_enqueue', 'joint_final58_jobs_enqueued']


def test_orphan_active_current_cannot_disappear_from_full_pending_cost(tmp_path, monkeypatch):
    campaign, ledger, conf, final, selection, phases, _ = fixture(tmp_path, monkeypatch)
    p = campaign / 'queue_status.json'; r = json.loads(p.read_text()); r['current'] = 'not-in-queue'; p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='exact pending queue job'):
        admission.prepare_joint_forecast(campaign, conf, final, selection, phases)


@pytest.mark.parametrize('scenario,cars,seconds,identity', [
    ('stopped', 20, 480, 'unsafe-hazard'), ('stopped', 3, 480, 'long-hazard'),
    ('race', 20, 120, 'short-field'), ('race', 64, 60, 'unknown-capacity'),
    ('race', 24, 60, 'renamed-capacity'), ('race', 4, 960, 'two-lap-field'),
    ('unknown', 4, 480, 'unknown-scenario')])
def test_unknown_scenario_shapes_have_no_cheap_rate(scenario, cars, seconds, identity):
    with pytest.raises(ValueError, match='No measured approved planning stratum'):
        admission.stratum({'job_id': identity, 'args': ['--scenario', scenario, '--cars', str(cars), '--seconds', str(seconds)]})


def test_only_the_three_exact_approved_capacity_openers_receive_allowance():
    for identity, cars in admission.CAPACITY_OPENERS.items():
        assert admission.stratum({'job_id': identity, 'args': ['--cars', str(cars), '--seconds', '60']}) == 'capacity_opener'


def test_approval_expiring_during_deep_guards_never_reaches_append(tmp_path, monkeypatch):
    campaign, ledger, conf, final, selection, phases, _ = fixture(tmp_path, monkeypatch)
    path = approved(campaign, ledger, conf, final, selection, phases)
    clock = [datetime.now(timezone.utc)]
    monkeypatch.setattr(admission, 'utc_now', lambda: clock[0])
    def slow_guard(*a, **kw): clock[0] += timedelta(seconds=301); return True
    monkeypatch.setattr(admission, 'require_execution_bindings', slow_guard)
    before = (campaign / 'queue.json').read_bytes()
    with pytest.raises(ValueError, match='expired during deep audit'):
        admission.append_joint_final_jobs(campaign, conf, final, selection, path, campaign / 'receipt.json', execute=True)
    assert (campaign / 'queue.json').read_bytes() == before
    assert not (campaign / 'receipt.json').exists()


@pytest.mark.parametrize('key,value', [('actual_roster_sha256', 'another-declared-roster'),
    ('policies', ['reference', 'n2', 'n1', 'n3']), ('snapshot_sha256', 'other-weights'),
    ('snapshot_path', 'another-source')])
def test_separately_valid_pack_cannot_replace_the_engineering_frozen_roster_order_or_source(tmp_path, monkeypatch, key, value):
    campaign, _, conf, final, selection, phases, _ = fixture(tmp_path, monkeypatch)
    final[key] = value
    with pytest.raises(ValueError, match='exact engineering-frozen roster/source/order'):
        admission.prepare_joint_forecast(campaign, conf, final, selection, phases)
