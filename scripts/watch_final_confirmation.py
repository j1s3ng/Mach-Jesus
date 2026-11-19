#!/usr/bin/env python3
"""Audit the frozen finalist's complete paired confirmation suite; never choose it.

This separate watcher leaves the live holdout/bootstrap watcher unchanged. Only
the existing Ledger.promote(split='confirmation') gate can promote a driver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import datetime as dt
import fcntl
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'src')]
from scripts.improvement_ledger import (Ledger, OFFICIAL_DETECTOR_SHA256,
    candidate_descriptor, compare_candidates, digest, validate_bootstrap)
from scripts.generate_improvement_cases import formal_cases
from scripts.improvement_queue_edit import validate_jobs
from scripts.exact_legacy_demo_scope import verified_legacy_demo_id
from scripts.watch_improvement_evaluation import (CampaignEvaluation, atomic,
    fingerprints, instant, read, sha, utc)

FALLBACK_PLAN = 'plans/baseline_bootstrap_descriptive_fallback_20261114T1508Z.json'
FALLBACK_SHA256 = '31e93c8dd62a1fd1c36729ac8a29e2b3bdccc72f04a3414785e3acb399688b31'
DESCRIPTIVE_MODE = 'DESCRIPTIVE_ONLY_NO_BOOTSTRAPPED_INCUMBENT'


def verify_engineering_selection(report_path, campaign):
    from scripts.audit_engineering_finalist_selection import verify_selection_report
    return verify_selection_report(report_path, campaign, require_complete=True)


def validate_selection(campaign, receipt_path, cases_path, state=None):
    """Require a concrete engineering-only selection frozen before any test job."""
    campaign, receipt_path, cases_path = map(Path, (campaign, receipt_path, cases_path))
    ledger = Ledger(campaign / 'ledger.jsonl')
    records = ledger.records()
    legacy_demo_id = verified_legacy_demo_id(campaign, records=records)
    state = state or ledger.state()
    receipt = read(receipt_path)
    if receipt.get('status') != 'FINALIST_FROZEN_FROM_ENGINEERING_ONLY':
        raise ValueError('An actual frozen finalist receipt is required')
    if receipt.get('formal_outcomes_used_for_selection') is not False:
        raise ValueError('Finalist selection must exclude formal test outcomes')
    if sha(cases_path) != receipt.get('confirmation_cases_sha256'):
        raise ValueError('Frozen confirmation cases changed')
    if read(cases_path) != formal_cases('confirmation'):
        raise ValueError('The unchanged canonical15 confirmation cases are required')
    baseline, candidate = receipt.get('baseline_id'), receipt.get('finalist_id')
    if not baseline or not candidate or baseline == candidate:
        raise ValueError('Distinct frozen baseline and learned finalist required')
    for cid in (baseline, candidate):
        registration = state['candidates'].get(cid, {})
        actual = {'snapshot_sha256': registration.get('snapshot_sha256'),
                  'descriptor': candidate_descriptor(registration)}
        if actual != receipt.get('registered_bindings', {}).get(cid):
            raise ValueError('Frozen candidate registration changed:' + cid)
        descriptor = actual['descriptor']
        if (descriptor.get('detector_sha256') != OFFICIAL_DETECTOR_SHA256
                or descriptor.get('control_parameters') != {'visible_distance_m': 70.}):
            raise ValueError('Confirmation requires the common official detector and70m')
    if candidate_descriptor(state['candidates'][baseline])['policy_key'] != 'reference':
        raise ValueError('Baseline must be the verified reference driver')
    if candidate_descriptor(state['candidates'][candidate])['model_sha256'] is None:
        raise ValueError('Finalist must bind actual learned weights')
    evidence = receipt['engineering_selection_report']
    if sha(evidence['path']) != evidence['sha256']:
        raise ValueError('Engineering selection evidence changed')
    report = verify_engineering_selection(evidence['path'], campaign)
    if (report.get('status') != 'VERIFIED_ENGINEERING_SELECTION'
            or report.get('selected_finalist_id') != candidate
            or report.get('formal_outcomes_used') is not False):
        raise ValueError('Engineering evidence does not verify this finalist')
    declarations = [r for r in records if r['event_type'] == 'finalist_frozen_for_confirmation']
    if len(declarations) != 1 or declarations[0]['payload'].get('selection_receipt_sha256') != sha(receipt_path):
        raise ValueError('Exactly one matching actual finalist declaration is required')
    declaration = declarations[0]
    if instant(receipt['frozen_utc']) > instant(declaration['utc']):
        raise ValueError('Receipt postdates its declaration')
    for record in records:
        if record['event_type'] != 'attempt_scheduled' or record['payload'].get('split') != 'confirmation':
            continue
        if legacy_demo_id is not None and record['payload'].get('attempt_id') == legacy_demo_id:
            continue
        if record['sequence'] <= declaration['sequence']:
            raise ValueError('Confirmation was scheduled before finalist freeze')
        if record['payload'].get('candidate_id') not in (baseline, candidate):
            raise ValueError('Another candidate entered the reserved confirmation suite')
    return receipt


def require_baseline_bootstrap(ledger, baseline):
    """Verify the real complete holdout initialization and its ledger chain."""
    state = ledger.state()
    initial = copy.deepcopy(state)
    initial['incumbent_id'] = None
    actual = validate_bootstrap(initial, baseline, formal_cases('holdout'), 'holdout')
    if not actual['approved']:
        raise ValueError('Complete verified baseline holdout bootstrap is required')
    actual_sha = digest(actual)
    decisions = [r for r in ledger.records() if r['event_type'] == 'bootstrap_decision'
        and r['payload'].get('decision_sha256') == actual_sha
        and r['payload'] == {**actual, 'decision_sha256': actual_sha}]
    initialized = [r for r in ledger.records() if r['event_type'] == 'incumbent_bootstrapped'
        and r['payload'].get('candidate_id') == baseline
        and r['payload'].get('decision_sha256') == actual_sha]
    if len(decisions) != 1 or len(initialized) != 1 or decisions[0]['sequence'] >= initialized[0]['sequence']:
        raise ValueError('A verified baseline bootstrap decision and initialization chain are required')
    return {'bootstrap_decision_sha256': actual_sha, 'baseline_id': baseline,
            'verified_complete_holdout_cases': len(actual['required_scenario_ids']),
            'evaluation_mode': 'frozen_finalist_paired_fresh_confirmation',
            'formal_promotion_allowed': True}


def require_confirmation_baseline(ledger, baseline, campaign, audit_trial):
    """A rejected hazard bootstrap permits diagnostics only under prior authority.

    This never initializes an incumbent or changes any bootstrap/promotion gate.
    Missing data, infrastructure failures and unverified sources fail closed.
    """
    state, records = ledger.state(), ledger.records()
    if state.get('incumbent_id') is not None:
        return require_baseline_bootstrap(ledger, baseline)
    cases = formal_cases('holdout')
    decision = validate_bootstrap(state, baseline, cases, 'holdout')
    issues = decision.get('issues') or []
    if decision['approved'] or not issues or any(
            not issue.endswith(':stopped_hazard_not_passed') for issue in issues):
        raise ValueError('Complete verified bootstrap, or exclusively genuine hazard rejection, is required')
    if any(r['event_type'] == 'incumbent_bootstrapped' for r in records):
        raise ValueError('Descriptive fallback requires no initialized incumbent')
    plan_path = Path(campaign) / FALLBACK_PLAN
    if sha(plan_path) != FALLBACK_SHA256:
        raise ValueError('Approved prospective descriptive fallback plan changed')
    plan = read(plan_path)
    if (plan.get('status') != 'ROOT_APPROVED_PROSPECTIVE_DESCRIPTIVE_FALLBACK_NO_GATE_CHANGE'
            or plan.get('baseline_id') != baseline
            or plan.get('baseline_attempts_at_declaration') != []):
        raise ValueError('Descriptive fallback authority does not bind this baseline')
    events = [r for r in records if r['event_type'] == 'baseline_bootstrap_descriptive_fallback_predeclared'
        and r['payload'].get('file_sha256') == FALLBACK_SHA256
        and r['payload'].get('baseline_id') == baseline
        and r['payload'].get('baseline_attempts_at_declaration') == []
        and r['payload'].get('formal_gate_or_approval_relaxed') is False
        and r['payload'].get('fresh_case_count') == 30]
    if len(events) != 1:
        raise ValueError('A unique prospective fallback ledger declaration is required')
    event = events[0]
    for split in ('holdout', 'confirmation'):
        path = Path(campaign) / ('formal-cases/' + split + '.json')
        if (sha(path) != plan['source_file_sha256'][split]
                or read(path) != formal_cases(split)):
            raise ValueError('Fallback requires unchanged canonical cases:' + split)
    selected = [a for a in state['attempts'] if a.get('candidate_id') == baseline
                and a.get('split') == 'holdout']
    counts = Counter(a.get('scenario_id') for a in selected)
    if (len(selected) != 15 or set(counts) != {c['scenario_id'] for c in cases}
            or any(n != 1 for n in counts.values())
            or any(a.get('stage') not in {'bootstrap', 'promotion'} or a.get('state') != 'finished'
                   or a.get('status') not in {'completed', 'finished'} for a in selected)):
        raise ValueError('Fallback requires every exact closed baseline case once')
    scheduled = [r for r in records if r['event_type'] == 'attempt_scheduled'
                 and r['payload'].get('candidate_id') == baseline
                 and r['payload'].get('split') == 'holdout']
    if len(scheduled) != 15 or any(r['sequence'] <= event['sequence'] for r in scheduled):
        raise ValueError('Fallback was not declared before all baseline attempts')
    decision_sha = digest(decision)
    decisions = [r for r in records if r['event_type'] == 'bootstrap_decision'
        and r['payload'] == {**decision, 'decision_sha256': decision_sha}]
    if len(decisions) != 1 or decisions[0]['sequence'] <= max(r['sequence'] for r in scheduled):
        raise ValueError('The exact rejected bootstrap decision must be recorded')
    by_case = {c['scenario_id']: c for c in cases}
    audits = {a['attempt_id']: audit_trial(a, state['candidates'][baseline], by_case[a['scenario_id']])
              for a in selected}
    if any(r.get('verified') is not True for r in audits.values()):
        raise ValueError('Fallback requires independent verification of every baseline case')
    bindings = {r.get('physics_binding_sha256') for r in audits.values()}
    if len(bindings) != 1 or None in bindings:
        raise ValueError('Fallback baseline physical assets are not verified common')
    # Recheck on independently recovered outcomes rather than trusting the ledger
    # marker. The real auditor also binds measured perception, source and clocks.
    independently_audited = copy.deepcopy(state)
    for attempt in independently_audited['attempts']:
        if attempt['attempt_id'] in audits:
            attempt['outcome'] = audits[attempt['attempt_id']]['outcome']
    if validate_bootstrap(independently_audited, baseline, cases, 'holdout') != decision:
        raise ValueError('Independent baseline evidence differs from the rejected bootstrap')
    return {'baseline_id': baseline, 'bootstrap_decision_sha256': decision_sha,
        'verified_complete_holdout_cases': 15, 'bootstrap_rejection_issues': issues,
        'fallback_plan_sha256': FALLBACK_SHA256, 'fallback_declaration_sha256': event['sha256'],
        'evaluation_mode': DESCRIPTIVE_MODE, 'formal_promotion_allowed': False,
        'independent_holdout_audit_identities': {k: v['input_identity'] for k, v in audits.items()}}


def confirmation_jobs(campaign, receipt_path, *, output_root='runs/improvement-12h/final-confirmation-001'):
    """Prepare all30 jobs without enqueueing, with no survivor dependencies."""
    campaign = Path(campaign).resolve()
    case_path = campaign / 'formal-cases/confirmation.json'
    receipt = validate_selection(campaign, receipt_path, case_path)
    state = Ledger(campaign / 'ledger.jsonl').state()
    evaluator = FinalConfirmationEvaluation(campaign, receipt_path)
    baseline_proof = require_confirmation_baseline(evaluator.ledger, receipt['baseline_id'], campaign,
                                                  evaluator.audit_trial)
    if baseline_proof['formal_promotion_allowed'] and state.get('incumbent_id') != receipt['baseline_id']:
        raise ValueError('Verified baseline must be the incumbent before confirmation scheduling')
    jobs = []
    for case in evaluator.cases:
        for cid in (receipt['baseline_id'], receipt['finalist_id']):
            registration = state['candidates'][cid]
            source = Path(registration['snapshot_path'])
            policy = candidate_descriptor(registration)['policy_key']
            identity = 'confirmation-001-' + cid + '-' + case['case_id']
            args = ['--candidate-id', cid, '--source-root', str(source),
                '--reference', str(source / 'reference.npz'), '--single-policy', policy,
                '--vision-url', 'http://127.0.0.1:8770', '--secondary-url', 'http://127.0.0.1:8771',
                '--secondary-cars', '8' if case['tier'] == '20' else '0',
                '--visibility', '70', '--detector-sha', OFFICIAL_DETECTOR_SHA256,
                '--stage', 'promotion', '--split', 'confirmation', '--no-video',
                '--pointcloud-stride', '0'] + case['cli_args']
            jobs.append({'job_id': identity, 'output': output_root + '/' + cid + '/' + case['case_id'],
                'estimated_wall_seconds': 90 if case['is_hazard'] else {'4': 150, '8': 300, '20': 600}[case['tier']],
                'max_wall_seconds': 600 if case['is_hazard'] else 1800,
                'args': args, 'final_confirmation_binding': {'selection_receipt_sha256': sha(receipt_path),
                    'snapshot_sha256': registration['snapshot_sha256'], 'case_sha256': case['scenario_sha256']}})
    if len(jobs) != 30:
        raise ValueError('Complete paired15+15 confirmation required')
    return {'schema_version': 1, 'prepared_utc': utc(), 'jobs': jobs,
        'output_root': output_root,
        'selection_receipt_sha256': sha(receipt_path), 'confirmation_cases_sha256': sha(case_path),
        'baseline_bootstrap_proof': baseline_proof,
        'estimated_wall_seconds': sum(j['estimated_wall_seconds'] for j in jobs),
        'enqueue_authorized_by_this_function': False, 'failure_dependent_case_skipping': False}


def require_confirmation_dispatch(pack, campaign, receipt_path, *, remaining_other_required_seconds,
                                  reserve_seconds=1200, now=None):
    """Admit the WHOLE30 plan while reserving the other required final work.

    The caller records a measured forecast for the remaining championship,
    geometry, audits and media work. Estimates are planning values; every job
    retains its separate finite maximum and the campaign's hard deadline.
    """
    campaign = Path(campaign).resolve()
    expected = confirmation_jobs(campaign, receipt_path, output_root=pack['output_root'])
    for key in ('jobs', 'selection_receipt_sha256', 'confirmation_cases_sha256',
                'baseline_bootstrap_proof', 'estimated_wall_seconds', 'output_root'):
        if pack.get(key) != expected[key]:
            raise ValueError('The complete frozen confirmation plan changed:' + key)
    validate_jobs(pack['jobs'])
    for value in (remaining_other_required_seconds, reserve_seconds):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite reserves for all other final work are required')
    ledger = Ledger(campaign / 'ledger.jsonl')
    state = ledger.state()
    planned_ids = {j['job_id'] for j in pack['jobs']}
    if any(a['attempt_id'] in planned_ids for a in state['attempts']):
        raise ValueError('A confirmation attempt ID is already scheduled')
    if any((ROOT / j['output']).exists() for j in pack['jobs']):
        raise ValueError('Confirmation output directories must be fresh')
    deadline = instant(state['run']['deadline_utc'])
    now = now or dt.datetime.now(dt.timezone.utc)
    required = pack['estimated_wall_seconds'] + remaining_other_required_seconds + reserve_seconds
    if (deadline - now).total_seconds() < required:
        raise ValueError('The whole paired30 suite and remaining final work do not fit the campaign deadline')
    return {'checked_utc': now.isoformat(), 'deadline_utc': deadline.isoformat(),
        'confirmation_estimate_seconds': pack['estimated_wall_seconds'],
        'remaining_other_required_seconds': remaining_other_required_seconds,
        'cleanup_report_reserve_seconds': reserve_seconds,
        'whole_required_seconds': required, 'selection_receipt_sha256': sha(receipt_path),
        'job_plan_sha256': digest(pack), 'admitted': True}


class FinalConfirmationEvaluation(CampaignEvaluation):
    def __init__(self, campaign, selection_receipt, output_directory=None):
        campaign = Path(campaign).resolve()
        cases_path = campaign / 'formal-cases/confirmation.json'
        self.selection_path = Path(selection_receipt).resolve()
        receipt = validate_selection(campaign, self.selection_path, cases_path)
        self.selection_sha256 = sha(self.selection_path)
        super().__init__(campaign, cases_path, receipt['baseline_id'], receipt['finalist_id'],
            output_directory=output_directory or campaign / 'evaluation-final-confirmation', publish_progress=False)
        if any(case.get('split') != 'confirmation' for case in self.cases):
            raise ValueError('Only unchanged canonical confirmation cases are accepted')
        self.code = self.current_sources()

    @staticmethod
    def current_sources():
        return {**fingerprints(), str(Path(__file__).relative_to(ROOT)): sha(__file__),
            'scripts/generate_improvement_cases.py': sha(ROOT / 'scripts/generate_improvement_cases.py'),
            'scripts/exact_legacy_demo_scope.py': sha(ROOT / 'scripts/exact_legacy_demo_scope.py'),
            'scripts/audit_engineering_finalist_selection.py': sha(ROOT / 'scripts/audit_engineering_finalist_selection.py')}

    def inspect(self, state):
        audits, arms = {}, {}
        for cid in (self.baseline, self.candidate):
            attempts = [a for a in state['attempts'] if a.get('candidate_id') == cid
                        and a.get('split') == 'confirmation']
            issues = []
            seen = Counter(a.get('scenario_id') for a in attempts)
            for attempt in attempts:
                if attempt.get('stage') not in {'bootstrap', 'promotion'}:
                    issues.append(attempt['attempt_id'] + ':wrong_stage')
                if attempt['scenario_id'] not in self.case_by_id:
                    issues.append(attempt['attempt_id'] + ':unexpected_case')
                    continue
                if attempt['state'] != 'finished':
                    continue
                try:
                    result = self.audit_trial(attempt, state['candidates'][cid], self.case_by_id[attempt['scenario_id']])
                except Exception as error:
                    result = {'attempt_id': attempt['attempt_id'], 'verified': False,
                              'issues': [f'{type(error).__name__}: {error}']}
                audits[attempt['attempt_id']] = result
                if not result['verified']:
                    issues.append(attempt['attempt_id'] + ':reaudit_failed')
            complete = (set(seen) == set(self.case_by_id) and all(n == 1 for n in seen.values())
                and len(attempts) == 15 and all(a['state'] == 'finished'
                    and a.get('status') in {'completed', 'finished'} for a in attempts))
            ready = complete and not issues and all(audits.get(a['attempt_id'], {}).get('verified') for a in attempts)
            arms[cid] = {'attempts': attempts, 'scheduled_cases': len(attempts),
                'closed_cases': sum(a['state'] == 'finished' for a in attempts), 'required_cases': 15,
                'complete': complete, 'independently_verified_complete_suite': ready, 'issues': issues}
        bindings = {r.get('physics_binding_sha256') for r in audits.values() if r.get('verified')}
        return arms, audits, len(bindings) <= 1

    def once(self, act=False):
        if self.current_sources() != self.code:
            raise ValueError('Frozen confirmation evaluator sources changed')
        if sha(self.selection_path) != self.selection_sha256 or sha(self.cases_path) != self.cases_sha256:
            raise ValueError('Frozen finalist receipt or confirmation cases changed')
        state = self.ledger.state()
        validate_selection(self.campaign, self.selection_path, self.cases_path, state)
        self._bundle_signatures = {}
        baseline_proof = require_confirmation_baseline(self.ledger, self.baseline, self.campaign,
                                                      self.audit_trial)
        formal_allowed = baseline_proof['formal_promotion_allowed']
        arms, audits, common_world = self.inspect(state)
        decisions = {}
        deadline = self.deadline(state)
        within = deadline is not None and dt.datetime.now(dt.timezone.utc) < instant(deadline)
        if all(arm['independently_verified_complete_suite'] for arm in arms.values()):
            comparison = compare_candidates(state['attempts'], self.baseline, self.candidate,
                self.cases, 'confirmation', snapshot_registry=state['candidates'])
            if not common_world:
                comparison['approved'] = False
                comparison['issues'].append('common_world_assets_or_reference_changed')
            decisions['comparison' if formal_allowed else 'stateless_paired_gate_preview'] = comparison
            if formal_allowed and act and within and comparison['approved'] and state.get('incumbent_id') == self.baseline:
                decisions['promotion'] = self.ledger.promote(self.candidate, self.cases, split='confirmation')
                state = self.ledger.state()
        report = {'schema_version': 1, 'updated_utc': utc(), 'deadline_utc': deadline,
            'evaluation_mode': baseline_proof['evaluation_mode'], 'decisions_enabled': act and formal_allowed,
            'decisions_requested': act, 'formal_promotion_allowed': formal_allowed,
            'overall_formal_approval': formal_allowed and state.get('incumbent_id') == self.candidate,
            'formal_release': formal_allowed and state.get('incumbent_id') == self.candidate,
            'baseline_id': self.baseline, 'candidate_id': self.candidate, 'incumbent_id': state.get('incumbent_id'),
            'selection_receipt_sha256': self.selection_sha256, 'canonical_cases_sha256': self.cases_sha256,
            'evaluation_source_sha256': self.code, 'common_physical_assets_verified': common_world,
            'baseline_bootstrap_proof': baseline_proof,
            'formal': {cid: {**{k: v for k, v in arm.items() if k != 'attempts'}, **self.metrics(arm, audits)}
                       for cid, arm in arms.items()},
            'paired_clean_pace': self.paired_pace(arms, audits), 'reaudits': audits, 'decisions': decisions,
            'all_attempt_inventory': state['attempts'], 'historical_outcomes_rewritten': False,
            'engineering_selected_finalist_changed': False, 'bootstrap_performed': False,
            'scope': 'Complete paired15+15 fresh confirmation only. Mixed-race ranking and six new hazard layouts per driver are separate evaluations.'}
        atomic(self.output / 'report.json', report)
        lines = ['# Frozen finalist confirmation', '', f"Updated {report['updated_utc']}.",
            f"Incumbent: `{report['incumbent_id']}`; selected finalist: `{self.candidate}`.", '',
            '| Arm | Closed /15 | Independent complete suite | Verified hazards /3 |',
            '| --- | ---: | --- | ---: |']
        lines += [f"| {cid} | {arm['closed_cases']} | {arm['independently_verified_complete_suite']} | {arm['verified_hazard_passes']} |"
                  for cid, arm in report['formal'].items()]
        lines += ['', report['scope'], '', ('The existing unchanged safety and pace gates control promotion.'
            if formal_allowed else 'DESCRIPTIVE ONLY: the complete baseline failed a stopped hazard. No incumbent was initialized; no formal approval, promotion or release is permitted, even if the paired diagnostic preview passes.'), '']
        atomic(self.output / 'REPORT.md', '\n'.join(lines))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'runs/improvement-12h')
    parser.add_argument('--selection-receipt', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--act', action='store_true')
    parser.add_argument('--jobs-output', type=Path)
    args = parser.parse_args()
    if args.jobs_output:
        if args.jobs_output.exists():
            parser.error('Retain existing job plan and use a new path')
        atomic(args.jobs_output, confirmation_jobs(args.campaign, args.selection_receipt))
        return
    evaluator = FinalConfirmationEvaluation(args.campaign, args.selection_receipt, args.output_directory)
    with (evaluator.output / 'watcher.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            report = evaluator.once(act=args.act)
            if args.once or (report['deadline_utc'] and dt.datetime.now(dt.timezone.utc) >= instant(report['deadline_utc'])):
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
