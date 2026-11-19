"""Full byte fingerprints of final scoring inputs, excluding derived caches."""
import hashlib
import json
from pathlib import Path

from scripts.improvement_ledger import Ledger, digest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('scripts/analyze_final_championship.py', 'scripts/final_scored_inputs.py',
    'scripts/final_roster_binding.py', 'scripts/exact_legacy_demo_scope.py', 'scripts/evaluate_improvement_attempt.py',
    'scripts/improvement_ledger.py', 'scripts/generate_improvement_cases.py',
    'scripts/watch_improvement_evaluation.py', 'src/race_lidar/race_results.py')


def read(path):
    return json.loads(Path(path).read_text())


def file_row(path):
    path = Path(path).resolve()
    if not path.exists():
        return {'path': str(path), 'exists': False}
    before = path.stat()
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Scored input changed while hashing:' + str(path))
    return {'path': str(path), 'exists': True, 'bytes': after.st_size, 'sha256': hasher.hexdigest()}


def ledger_binding(records, formal_arms=()):
    """Later unrelated events are harmless; new/changed final attempts are not."""
    arms = {tuple(value) for value in formal_arms}
    ids = {r['payload']['attempt_id'] for r in records if r['event_type'] == 'attempt_scheduled'
           and (r['payload'].get('split') in {'championship', 'geometric-confirmation'}
                or (r['payload'].get('candidate_id'),r['payload'].get('split')) in arms)}
    declarations = {'final_evaluation_predeclared', 'final_roster_engineering_selection_bound',
        'championship_roster_predeclared', 'finalist_frozen_for_confirmation', 'final_selection_freeze_started'}
    selected = [r for r in records if r['event_type'] in declarations
                or (r['event_type'].startswith('attempt_') and r['payload'].get('attempt_id') in ids)]
    return {'record_count': len(selected), 'sha256': digest(selected),'formal_arms':[list(a) for a in sorted(arms)],
            'final_attempt_ids': sorted(ids), 'records': [{'sequence': r['sequence'], 'sha256': r['sha256']} for r in selected]}


def capture_scored_inputs(campaign, run_paths, snapshot, metadata_files, roster_chain, *, additional_snapshots=(), formal_arms=()):
    paths = {Path(p).resolve() for p in metadata_files}
    for run in run_paths:
        run = Path(run).resolve()
        # Heat results contain the authoritative physics-step overlap totals and
        # raw pair sample counters. The current director has no separate full
        # per-contact event stream. Optional additional streams retain presence.
        paths.update(run / name for name in ('config.json', 'heat_results.json', 'gt_poses.jsonl',
            'leaderboard.jsonl', 'ledger_outcome.json', 'contact_samples.jsonl', 'collision_samples.jsonl'))
        config = read(run / 'config.json') if (run / 'config.json').exists() else {}
        if config.get('reference_path'):
            paths.add(Path(config['reference_path']))
        for entry in config.get('entrants', []):
            folder = (run / 'cars' / entry['car_id']).resolve()
            if not folder.is_relative_to(run):
                raise ValueError('Unsafe scored entrant path')
            paths.add(folder / 'result.json')
            model = entry.get('model_path')
            if isinstance(entry.get('policy'), dict):
                model = entry['policy'].get('model_path', model)
            if model:
                paths.add(Path(model))
            if not entry.get('is_static'):
                paths.update(folder / name for name in ('controller_summary.json', 'control_trace.jsonl',
                    'predictions.jsonl', 'policy_actions.jsonl'))
    for root in (snapshot,*additional_snapshots):
        root=Path(root).resolve()
        manifest = read(root / 'manifest.json')
        paths.add(root / 'manifest.json')
        for entry in manifest['files']:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe frozen scored source path')
            paths.add(root / relative)
    paths.update(ROOT / source for source in SOURCES)
    records = Ledger(Path(campaign) / 'ledger.jsonl').records()
    if roster_chain.get('verified'):
        selection = Path(roster_chain['selection_receipt_path'])
        paths.add(selection)
        receipt = read(selection)
        paths.add(Path(receipt['roster_selection_receipt']['path']))
        paths.add(Path(receipt['engineering_selection_report']['path']))
        for record in records:
            if record['event_type'] == 'final_selection_freeze_started':
                paths.add(Path(record['payload']['selection_blueprint_path']))
    body = {'schema_version': 1, 'files': [file_row(p) for p in sorted(paths, key=str)],
        'ledger_input_binding': ledger_binding(records,formal_arms),
        'scope': 'All explicit evaluator/ranking config, heat/authoritative overlap counters, per-car results, closure/trace/prediction/action files; GT poses and leaderboard provenance; frozen source/driver weights/reference; roster/selection/engineering receipts and used canonical manifests; current CPU evaluator support sources. Missing optional contact streams are recorded as absent. No derived audit cache, raw LiDAR NPY or camera video is a scoring input; media bytes require their own provenance.'}
    body['inventory_sha256'] = digest(body)
    return body


def verify_scored_input_inventory(campaign, audit):
    """Read-only video/export gate; no inference, scoring or decision rewrite."""
    inventory = audit.get('scored_input_inventory')
    if not inventory or inventory.get('schema_version') != 1:
        raise ValueError('Full scored input inventory is required')
    if inventory.get('inventory_sha256') != digest({k: v for k, v in inventory.items() if k != 'inventory_sha256'}):
        raise ValueError('Scored input inventory semantic hash changed')
    for expected in inventory['files']:
        if file_row(expected['path']) != expected:
            raise ValueError('Closed scored input bytes/presence changed:' + expected['path'])
    if ledger_binding(Ledger(Path(campaign) / 'ledger.jsonl').records(),inventory['ledger_input_binding'].get('formal_arms',())) != inventory['ledger_input_binding']:
        raise ValueError('Final attempt/declaration ledger input changed')
    return True
