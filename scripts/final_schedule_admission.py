"""Whole final-work admission and an explicit, locked root enqueue front door.

Default use prepares a CPU-only forecast. Nothing is appended unless root calls
append_joint_final_jobs(..., execute=True) after a new immutable forecast and its
matching ledger approval. Existing case, safety and single-pack gates remain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.improvement_ledger import Ledger, digest
from scripts.improvement_queue_edit import edit_queue, validate_jobs
from scripts.final_roster_binding import verify_frozen_roster_chain
from scripts.prepare_final_race_jobs import require_execution_bindings
from scripts.watch_final_confirmation import require_confirmation_dispatch
from scripts.watch_improvement_evaluation import instant, sha

RATE_AUTHORITY = 'plans/remaining_required_work_forecast_20261114T1457Z.json'
RATE_AUTHORITY_SHA256 = '2c446ee43a22cc256413615420341b7b59afe67036bbacc80280f61ba1467e62'
APPROVAL_EVENT = 'joint_final_dispatch_forecast_approved'
APPROVED = 'ROOT_APPROVED_MEASURED_JOINT_FINAL_DISPATCH_FORECAST'
SOURCE_FILES = ('scripts/final_schedule_admission.py', 'scripts/watch_final_confirmation.py',
    'scripts/prepare_final_race_jobs.py', 'scripts/exact_legacy_demo_scope.py',
    'scripts/improvement_queue_edit.py', 'scripts/final_roster_binding.py')
CAPACITY_OPENERS = {'capacity-20-v3-matched-control': 20,
    'capacity-24-v3-recovery1': 24, 'capacity-32-v3-recovery1': 32}


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return {p: sha(ROOT / p) for p in SOURCE_FILES}


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def utc_now():
    return datetime.now(timezone.utc)


def flag(job, name, default=None):
    args = job['args']
    if args.count(name) > 1:
        raise ValueError('Duplicate scheduling flag:' + name)
    return args[args.index(name) + 1] if name in args else default


def stratum(job):
    scenario = flag(job, '--scenario', 'race')
    cars = int(flag(job, '--cars', '20'))
    seconds = float(flag(job, '--seconds', '480'))
    if scenario == 'stopped' and cars in (2, 3) and seconds == 120:
        return 'known_hazard'
    if (scenario == 'race' and job['job_id'] in CAPACITY_OPENERS
            and cars == CAPACITY_OPENERS[job['job_id']] and seconds == 60):
        return 'capacity_opener'
    if scenario != 'race' or cars not in (4, 8, 20) or seconds != 480 or int(flag(job, '--laps', '1')) != 1:
        raise ValueError('No measured approved planning stratum for job:' + job['job_id'])
    return 'field' + str(cars)


def rate_authority(campaign):
    path = Path(campaign) / RATE_AUTHORITY
    if sha(path) != RATE_AUTHORITY_SHA256:
        raise ValueError('Prospective measured rate authority changed')
    authority = read(path)
    rates = authority['planning_rates_seconds']
    if authority.get('status') != 'APPROVED_MEASURED_WHOLE_REQUIRED_WORK_FORECAST' or not all(positive(v) for v in rates.values()):
        raise ValueError('Measured planning rate authority is invalid')
    measurements = []
    for row in authority['closed_run_measurements']:
        path = Path(row['path'])
        heat = read(path)
        if (sha(path) != row['sha256'] or heat.get('wall_seconds') != row['wall_seconds']
                or heat.get('simulation_seconds') != row['simulation_seconds']
                or heat.get('car_count') != row['cars'] or not positive(row['wall_seconds'])
                or heat.get('status') not in {'all_cars_reported', 'time_limit'}):
            raise ValueError('Closed timing measurement changed:' + str(path))
        key = 'known_hazard' if row['cars'] in (2, 3) else 'field' + str(row['cars'])
        if key not in rates or row['wall_seconds'] > rates[key]:
            raise ValueError('Planning rate is below its bound closed measurement')
        measurements.append(row)
    if not {'known_hazard', 'field4', 'field8', 'field20', 'capacity_opener'} <= rates.keys():
        raise ValueError('All required planning rates must be explicit')
    return {'path': str((Path(campaign) / RATE_AUTHORITY).resolve()), 'sha256': RATE_AUTHORITY_SHA256,
        'rates_seconds': rates, 'closed_run_measurements': measurements,
        'scope': 'Measured scheduling forecast, not a worst-case guarantee or isolated hardware benchmark. Capacity-opener rate is the explicit prospective planning allowance, not a verified24/32 result.'}


def forecast_body(campaign, confirmation_pack, final_pack, selection_receipt, other_required_work, *, jobs=None):
    campaign = Path(campaign).resolve()
    jobs = read(campaign / 'queue.json') if jobs is None else jobs
    validate_jobs(jobs)
    status = read(campaign / 'queue_status.json')
    ledger = Ledger(campaign / 'ledger.jsonl')
    state = ledger.state()
    confirmation, final = confirmation_pack['jobs'], final_pack['jobs']
    roster_binding = verify_frozen_roster_chain(campaign, final_pack['actual_roster_path'], selection_receipt)
    roster = roster_binding['roster']
    if (final_pack.get('actual_roster_sha256') != roster_binding['roster_sha256']
            or final_pack.get('policies') != roster['policies']
            or final_pack.get('snapshot_sha256') != roster['snapshot_sha256']
            or Path(final_pack.get('snapshot_path', '')).resolve() != Path(roster['snapshot_path']).resolve()):
        raise ValueError('Both final packs must use the exact engineering-frozen roster/source/order')
    if len(confirmation) != 30 or len(final) != 28:
        raise ValueError('Both full30 and full28 packs are mandatory')
    combined = confirmation + final
    ids = [j['job_id'] for j in combined]
    if len(set(ids)) != 58 or len({j['output'] for j in combined}) != 58:
        raise ValueError('Final58 identities and output paths must be unique')
    scheduled = {a['attempt_id'] for a in state['attempts']}
    if scheduled.intersection(ids) or {j['job_id'] for j in jobs}.intersection(ids):
        raise ValueError('All58 final jobs must be fresh and not already queued/scheduled')
    if any((ROOT / j['output']).exists() for j in combined):
        raise ValueError('Every final output path must be fresh')
    validate_jobs(jobs + combined)
    authority = rate_authority(campaign)
    rates = authority['rates_seconds']
    estimate = lambda rows: sum(rates[stratum(j)] for j in rows)
    completed = status.get('completed', {})
    pending = [j for j in jobs if j['job_id'] not in completed]
    pending_ids = {j['job_id'] for j in pending}
    if any(status.get(key) and status[key] not in pending_ids for key in ('current', 'pending_detector_job')):
        raise ValueError('Active/waiting current job is not bound to an exact pending queue job')
    phase_names = [p.get('phase') for p in other_required_work]
    if (len(set(phase_names)) != len(phase_names) or any(not p.get('phase') or not positive(p.get('seconds'))
            or not p.get('reason') for p in other_required_work)):
        raise ValueError('Every additional required phase needs unique identity, positive time and reason')
    phases = {p['phase']: p for p in other_required_work}
    if phases.get('report_video_export_review', {}).get('seconds', 0) < 1200 or phases.get('cleanup', {}).get('seconds', 0) < 300:
        raise ValueError('Retain at least1200s report/media/review and300s cleanup')
    pending_seconds, confirmation_seconds, final_seconds = map(estimate, (pending, confirmation, final))
    other_seconds = sum(p['seconds'] for p in other_required_work)
    return {'schema_version': 1, 'campaign': str(campaign),
        'deadline_utc': state['run']['deadline_utc'], 'planning_rate_authority': authority,
        'confirmation_pack_sha256': digest(confirmation_pack), 'final_pack_sha256': digest(final_pack),
        'selection_receipt_sha256': sha(selection_receipt),
        'frozen_engineering_roster_binding': roster_binding,
        'queue_canonical_sha256': digest(jobs), 'queue_completed_canonical_sha256': digest(completed),
        'queue_activity_sha256': digest({k: status.get(k) for k in ('current', 'pending_detector_job', 'status')}),
        'scheduled_attempt_ids_sha256': digest(sorted(scheduled)), 'current_job_at_full_forecast_cost': status.get('current'),
        'unique_pending_queue_job_ids': [j['job_id'] for j in pending],
        'pending_queue_job_bindings': {j['job_id']: digest(j) for j in pending},
        'all58_final_job_ids': ids, 'other_required_work': other_required_work,
        'counts': {'pending_queue': len(pending), 'confirmation': 30, 'championship': 4, 'fresh_geometry': 24},
        'seconds': {'pending_queue': pending_seconds, 'confirmation_measured': confirmation_seconds,
            'final28_measured': final_seconds, 'other_required': other_seconds,
            'whole_required_measured': pending_seconds + confirmation_seconds + final_seconds + other_seconds,
            'confirmation_nominal_pack': confirmation_pack['estimated_wall_seconds'],
            'final28_nominal_pack': final_pack['estimated_complete_pack_seconds'],
            'nominal_both_packs': confirmation_pack['estimated_wall_seconds'] + final_pack['estimated_complete_pack_seconds']},
        'source_sha256': sources(), 'all_pending_queue_jobs_counted_once': True,
        'current_job_remaining_discount_seconds': 0, 'physics_cases_or_safety_gates_changed': False,
        'interpretation': 'Explicit approved planning rates replace no immutable pack metadata. Both original single-pack guards remain separately required. Full current job charged; no elapsed-progress credit. Uncreated mandatory phases must be named in other_required_work. Estimates are not a deadline or hardware guarantee.'}


def prepare_joint_forecast(campaign, confirmation_pack, final_pack, selection_receipt, other_required_work):
    return {**forecast_body(campaign, confirmation_pack, final_pack, selection_receipt, other_required_work),
        'prepared_utc': utc_now().isoformat(),
        'status': 'DRAFT_REQUIRES_NEW_EXPLICIT_ROOT_FORECAST_APPROVAL', 'enqueue_performed': False}


def require_joint_final_admission(campaign, confirmation_pack, final_pack, selection_receipt, forecast_path, *, jobs=None, now=None):
    """Recheck under the queue writer lock immediately before the atomic append."""
    campaign, forecast_path = Path(campaign).resolve(), Path(forecast_path).resolve()
    forecast = read(forecast_path)
    if forecast.get('status') != APPROVED:
        raise ValueError('A new explicitly approved joint final forecast is required')
    ledger = Ledger(campaign / 'ledger.jsonl')
    approvals = [r for r in ledger.records() if r['event_type'] == APPROVAL_EVENT
        and r['payload'].get('file_sha256') == sha(forecast_path)
        and r['payload'].get('all_mandatory_work_included') is True
        and r['payload'].get('physics_cases_or_safety_gates_changed') is False]
    if len(approvals) != 1:
        raise ValueError('Exactly one matching root joint forecast approval is required')
    fixed_now = now
    now = now or utc_now()
    if (instant(forecast['prepared_utc']) > instant(approvals[0]['utc'])
            or instant(approvals[0]['utc']) > now
            or (now - instant(approvals[0]['utc'])).total_seconds() > 300):
        raise ValueError('Joint approval must be fresh at actual append')
    current = forecast_body(campaign, confirmation_pack, final_pack, selection_receipt,
                            forecast['other_required_work'], jobs=jobs)
    if any(forecast.get(k) != value for k, value in current.items()):
        raise ValueError('Joint forecast is stale or changed; prepare a new explicit approval')
    required = current['seconds']['whole_required_measured']
    deadline = instant(current['deadline_utc'])
    if (deadline - now).total_seconds() < required:
        raise ValueError('All pending work plus full30+28 and exports/cleanup do not fit')
    phases = {p['phase']: p['seconds'] for p in current['other_required_work']}
    cleanup = phases['cleanup']
    other = current['seconds']['pending_queue'] + current['seconds']['final28_measured'] + current['seconds']['other_required'] - cleanup
    confirmation_proof = require_confirmation_dispatch(confirmation_pack, campaign, selection_receipt,
        remaining_other_required_seconds=other, reserve_seconds=cleanup, now=now)
    require_execution_bindings(final_pack, campaign, now=now)
    # A closing child can change the queue status during deep independent case
    # checks. Never append against a snapshot that ceased to be the approved one.
    after = read(campaign / 'queue_status.json')
    if (digest(after.get('completed', {})) != current['queue_completed_canonical_sha256']
            or digest({k: after.get(k) for k in ('current', 'pending_detector_job', 'status')}) != current['queue_activity_sha256']
            or digest(read(campaign / 'queue.json')) != current['queue_canonical_sha256']):
        raise ValueError('Queue advanced during admission; refresh explicit forecast')
    checked = fixed_now or utc_now()
    if (checked - instant(approvals[0]['utc'])).total_seconds() > 300:
        raise ValueError('Joint approval expired during deep audit; refresh approval before append')
    strict_required = max(required, confirmation_proof.get('whole_required_seconds', required),
                          final_pack['estimated_complete_pack_seconds'] + final_pack.get('reserve_seconds', 300))
    if (deadline - checked).total_seconds() < strict_required:
        raise ValueError('Deadline no longer fits after deep admission audit')
    return {'schema_version': 1, 'checked_utc': checked.isoformat(), 'admitted': True,
        'forecast_approval_utc': approvals[0]['utc'], 'actual_append_required_seconds': strict_required,
        'forecast_file_sha256': sha(forecast_path), 'forecast_approval_record_sha256': approvals[0]['sha256'],
        'whole_required_seconds': required, 'deadline_utc': deadline.isoformat(),
        'planning_slack_seconds': (deadline - checked).total_seconds() - strict_required,
        'confirmed_all58_fresh_jobs': current['all58_final_job_ids'],
        'queue_canonical_sha256_before_append': current['queue_canonical_sha256'],
        'queue_completed_canonical_sha256': current['queue_completed_canonical_sha256'],
        'queue_activity_sha256': current['queue_activity_sha256'],
        'counts': current['counts'], 'seconds': current['seconds'],
        'confirmation_original_single_pack_guard': confirmation_proof,
        'final28_original_single_pack_guard': True, 'sources': current['source_sha256'],
        'case_or_safety_gate_changes': False, 'forecast_is_not_a_guarantee': True}


def append_joint_final_jobs(campaign, confirmation_pack, final_pack, selection_receipt, forecast_path, receipt_path, *, execute=False):
    """The explicit root front door; tests use temporary campaigns only."""
    if not execute:
        raise ValueError('Explicit root execute=True is required; preparation cannot enqueue')
    campaign, receipt_path = Path(campaign).resolve(), Path(receipt_path).resolve()
    if not receipt_path.is_relative_to(campaign) or receipt_path.exists():
        raise ValueError('A fresh receipt path inside the campaign is required')
    ledger = Ledger(campaign / 'ledger.jsonl')
    with edit_queue(campaign / 'queue.json') as jobs:
        proof = require_joint_final_admission(campaign, confirmation_pack, final_pack,
            selection_receipt, forecast_path, jobs=jobs)
        # Explicit source/time freshness after independent guards, immediately
        # before committing the locked list; no process is signaled or launched.
        append_time = utc_now()
        if (sources() != proof['sources']
                or (append_time - instant(proof['forecast_approval_utc'])).total_seconds() > 300
                or (instant(proof['deadline_utc']) - append_time).total_seconds() < proof['actual_append_required_seconds']):
            raise ValueError('Source or deadline changed before atomic final append')
        live_status = read(campaign / 'queue_status.json')
        if (digest(live_status.get('completed', {})) != proof['queue_completed_canonical_sha256']
                or digest({k: live_status.get(k) for k in ('current', 'pending_detector_job', 'status')}) != proof['queue_activity_sha256']):
            raise ValueError('Queue status changed immediately before final append')
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with receipt_path.open('x') as stream:
            json.dump(proof, stream, indent=2, allow_nan=False); stream.write('\n')
        receipt_path.chmod(0o444)
        ledger.append('joint_final_work_admitted_before_enqueue', {'receipt_path': str(receipt_path),
            'receipt_sha256': sha(receipt_path), 'all58_job_ids': proof['confirmed_all58_fresh_jobs'],
            'whole_required_seconds': proof['whole_required_seconds'],
            'actual_append_required_seconds': proof['actual_append_required_seconds']})
        jobs.extend(confirmation_pack['jobs'] + final_pack['jobs'])
    event = ledger.append('joint_final58_jobs_enqueued', {'receipt_sha256': sha(receipt_path),
        'all58_job_ids': proof['confirmed_all58_fresh_jobs'], 'queue_canonical_sha256_after_append': digest(jobs),
        'root_explicit_execute': True, 'processes_started_by_helper': False})
    return {'receipt_path': str(receipt_path), 'enqueued_jobs': 58, 'ledger_sequence': event['sequence']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--confirmation-pack', type=Path, required=True)
    p.add_argument('--final-pack', type=Path, required=True)
    p.add_argument('--selection-receipt', type=Path, required=True)
    p.add_argument('--other-required-work', type=Path, required=True)
    p.add_argument('--draft-output', type=Path, required=True)
    args = p.parse_args()
    if args.draft_output.exists():
        p.error('Preserve the old forecast; use a fresh output')
    result = prepare_joint_forecast(args.campaign, read(args.confirmation_pack), read(args.final_pack),
        args.selection_receipt, read(args.other_required_work))
    with args.draft_output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'seconds': result['seconds']}))
