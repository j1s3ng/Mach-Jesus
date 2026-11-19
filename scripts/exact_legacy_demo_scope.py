"""Verify the one immutable class recording excluded from fresh-test guards.

This does not reclassify its split or outcome. It grants no training, selection,
bootstrap, promotion or championship eligibility and never changes any file.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_improvement_cases import formal_cases
from scripts.improvement_ledger import Ledger, digest, verify_snapshot
from scripts.watch_improvement_evaluation import instant, sha

LEGACY_ID = 'presentation-00517-original-20'
PLAN = 'plans/exact_legacy_presentation_demo_exception_20261114T1522Z.json'
PLAN_SHA256 = '506eba357be5e1d62c506db4f3f7a3e4dc4d1581ff0ffcd9bd684d3593df5def'
APPROVAL_EVENT = 'exact_legacy_presentation_demo_exception_approved'
APPROVAL_RECORD_SHA256 = '7052458ffa624e4d2274de3ba7d2cffb87e3952e6e1a98e209dce3e97f9f746d'


def read(path):
    return json.loads(Path(path).read_text())


def verified_legacy_demo_id(campaign, records=None, jobs=None):
    """Return only LEGACY_ID after exact proof, or None in campaigns without it.

    Supplied records/jobs must be the current complete ledger/queue snapshots.
    Any changed, duplicate or extra legacy trial event fails closed. Callers
    still reject every other reserved schedule/job, regardless of its stage.
    """
    campaign = Path(campaign).resolve()
    records = Ledger(campaign / 'ledger.jsonl').records() if records is None else records
    jobs = (read(campaign / 'queue.json') if (campaign / 'queue.json').exists() else []) if jobs is None else jobs
    plan_path = campaign / PLAN
    actual_trials = [r for r in records if r['event_type'].startswith('attempt_')
                     and r['payload'].get('attempt_id') == LEGACY_ID]
    actual_jobs = [j for j in jobs if j.get('job_id') == LEGACY_ID]
    if not actual_trials and not actual_jobs and not plan_path.exists():
        return None
    if not plan_path.exists() or sha(plan_path) != PLAN_SHA256:
        raise ValueError('The exact approved legacy demonstration authority is required')
    plan = read(plan_path)
    if (plan.get('status') != 'ROOT_APPROVED_EXACT_LEGACY_PRESENTATION_DEMO_SCOPE_EXCEPTION'
            or plan.get('attempt_id') != LEGACY_ID or plan.get('candidate_id') != LEGACY_ID
            or plan.get('stage') != 'engineering' or plan.get('split') != 'confirmation'
            or plan.get('excluded_from_training') is not True
            or plan.get('fresh_canonical_confirmation_attempts_at_approval') != []
            or plan.get('not_one_of_canonical15_cases') is not True):
        raise ValueError('Legacy demonstration exception is not exact')
    if (len(actual_jobs) != 1 or actual_jobs[0] != plan['queue_job']
            or digest(actual_jobs[0]) != plan['queue_job_canonical_sha256']):
        raise ValueError('Legacy demonstration whole queue job changed or duplicated')
    keys = ('sequence', 'event_type', 'utc', 'sha256', 'payload')
    projected = [{k: r[k] for k in keys} for r in actual_trials]
    if projected != plan['ledger_records']:
        raise ValueError('Legacy demonstration trial ledger records changed or duplicated')
    for record in actual_trials:
        if digest({k: v for k, v in record.items() if k != 'sha256'}) != record['sha256']:
            raise ValueError('Legacy demonstration ledger record hash is invalid')
    scheduled = actual_trials[0]
    if (scheduled['event_type'] != 'attempt_scheduled'
            or scheduled['payload']['scenario_sha256'] != plan['scenario_sha256']
            or scheduled['payload']['snapshot_sha256'] != plan['snapshot_sha256']):
        raise ValueError('Legacy demonstration scheduled identity differs')
    events = [r for r in records if r['event_type'] == APPROVAL_EVENT]
    if len(events) != 1:
        raise ValueError('Exactly one legacy scope approval is required')
    event, payload = events[0], events[0]['payload']
    if (event['sha256'] != APPROVAL_RECORD_SHA256
            or digest({k: v for k, v in event.items() if k != 'sha256'}) != event['sha256']
            or payload.get('file_sha256') != PLAN_SHA256 or payload.get('attempt_id') != LEGACY_ID
            or payload.get('queue_job_canonical_sha256') != plan['queue_job_canonical_sha256']
            or payload.get('scheduled_record_sha256') != scheduled['sha256']
            or payload.get('config_file_sha256') != plan['config_file_sha256']
            or payload.get('fresh_canonical_confirmation_attempts_at_approval') != []
            or payload.get('no_formal_gate_relaxation') is not True
            or event['sequence'] <= actual_trials[-1]['sequence']
            or instant(event['utc']) < instant(plan['approved_utc'])):
        raise ValueError('Legacy scope approval does not bind the immutable demonstration')
    cases_path = campaign / 'formal-cases/confirmation.json'
    cases = read(cases_path)
    if (sha(cases_path) != plan['canonical15_confirmation_file_sha256']
            or cases != formal_cases('confirmation')
            or plan['scenario_sha256'] in {c['scenario_sha256'] for c in cases}):
        raise ValueError('Legacy demonstration overlaps or changes canonical confirmation')
    canonical = {c['scenario_id'] for c in cases}
    if any(r['event_type'] == 'attempt_scheduled' and r['sequence'] <= event['sequence']
           and r['payload'].get('split') == 'confirmation'
           and (r['payload'].get('scenario_id') in canonical
                or r['payload'].get('stage') in {'bootstrap', 'promotion'}) for r in records):
        raise ValueError('Fresh confirmation preceded the legacy scope approval')
    config_path, snapshot = Path(plan['config_path']), Path(plan['snapshot_path'])
    if sha(config_path) != plan['config_file_sha256']:
        raise ValueError('Legacy demonstration config bytes changed')
    config = read(config_path)
    if config.get('dataset_split') != 'confirmation' or config.get('excluded_from_training') is not True:
        raise ValueError('Legacy demonstration training exclusion changed')
    manifest = verify_snapshot(snapshot)
    if (manifest['snapshot_sha256'] != plan['snapshot_sha256']
            or sha(snapshot / 'manifest.json') != plan['snapshot_manifest_file_sha256']):
        raise ValueError('Legacy demonstration frozen source changed')
    entries = config['entrants']
    if len(entries) != 20:
        raise ValueError('Legacy demonstration entrant identity changed')
    for entrant in entries:
        model = Path(entrant['model_path']).resolve()
        if (entrant['policy'] != 'neural_rich005_seed17' or not model.is_relative_to(snapshot.resolve())
                or sha(model) != plan['model_file_sha256']):
            raise ValueError('Legacy demonstration model binding changed')
    return LEGACY_ID
