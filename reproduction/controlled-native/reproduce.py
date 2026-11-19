#!/usr/bin/env python3
"""Offline native-view reproduction from frozen labels and cached detections."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE))
from metric import assign_boxes, geometry, matched_errors, paired_errors
import numpy as np
import shapely

CORE = ('head-lr1e-4-s7', 'full-uniform-lr1e-4-s7', 'full-uniform-lr1e-5-s7',
        'full-mixed-head1e-4-backbone1e-5-s7')
ORIGINAL = 'original_pretrained'
AUG = 'full-mixed-rotaug4-s7'
MODELS = (ORIGINAL, *CORE, AUG)
PAIR = (CORE[-1], AUG)
CUTOFFS = (.35, .425)
INPUT_FILES = {'README.md', 'requirements.txt', 'reproduce.py', 'metric.py',
    'test_reproduction.py', 'frames.jsonl.gz', 'predictions.jsonl.gz',
    'expected_reports.json', 'expected_cohorts.json.gz', 'sealed_comparisons.json', 'provenance.json'}


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def identity(row): return row['scene_id'], row['frame_id']


def read_cache(path, expected_count):
    raw = gzip.decompress(Path(path).read_bytes())
    if len(raw) > 100_000_000: raise ValueError('Unexpected decompressed cache size')
    values = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(values) != expected_count: raise ValueError('Missing or extra cached row: '+Path(path).name)
    return values


def validate_box(box, prediction=False):
    expected = {'center', 'size', 'yaw', 'score'} if prediction else {'center', 'size', 'yaw', 'visible_points'}
    if set(box) != expected or len(box['center']) != 3 or len(box['size']) != 3:
        raise ValueError('Unexpected numeric box fields/shape')
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
               for x in [*box['center'], *box['size'], box['yaw']]) or min(box['size']) <= 0:
        raise ValueError('Invalid or nonfinite box numbers')
    if prediction:
        if not .35 <= box['score'] <= 1: raise ValueError('Prediction below exported cache floor')
    elif not isinstance(box['visible_points'], int) or box['visible_points'] < 3:
        raise ValueError('Frozen visible target needs at least three measured returns')


def load_bundle(folder=BUNDLE, cutoff=None):
    folder = Path(folder); manifest = read(folder/'manifest.json')
    if manifest.get('manifest_sha256') != digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}):
        raise ValueError('Manifest semantic hash changed')
    if cutoff is not None and cutoff < manifest['exported_prediction_floor']:
        raise ValueError('Cutoff below cache floor cannot reconstruct omitted detections')
    if cutoff is not None and cutoff not in CUTOFFS: raise ValueError('Only predeclared .35 and .425 cutoffs are supported')
    if manifest.get('schema_version') != 1 or manifest.get('models') != list(MODELS) or manifest.get('cutoffs') != list(CUTOFFS):
        raise ValueError('Frozen model/cutoff contract changed')
    if manifest['exported_prediction_floor'] != .35: raise ValueError('Exported floor must remain .35')
    if set(manifest['files']) != INPUT_FILES: raise ValueError('Complete immutable checksum inventory required')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or not (folder/name).is_file(): raise ValueError('Missing or unsafe bundled file')
        if (folder/name).stat().st_size != expected['bytes'] or sha(folder/name) != expected['sha256']:
            raise ValueError('Bundled file checksum changed: '+name)
    if np.__version__ != '1.26.4' or shapely.__version__ != '2.1.2':
        raise ValueError('Use pinned tested NumPy/Shapely versions from requirements.txt')
    frames = read_cache(folder/'frames.jsonl.gz', 6144)
    predictions = read_cache(folder/'predictions.jsonl.gz', 36864)
    for row in frames:
        if set(row) != {'scene_id', 'frame_id', 'gt_boxes'} or not re.fullmatch('[0-9a-f]{24}', row['frame_id']):
            raise ValueError('Unexpected frozen GT frame identity/fields')
        for box in row['gt_boxes']: validate_box(box)
    keys = [identity(row) for row in frames]
    if len(set(keys)) != 6144 or digest(sorted(keys)) != manifest['frame_identity_sha256']:
        raise ValueError('Duplicate or changed frozen GT population')
    if sum(len(row['gt_boxes']) for row in frames) != 8155: raise ValueError('Frozen 8155 visible-target population changed')
    pred_keys = []
    for row in predictions:
        if set(row) != {'pipeline', 'scene_id', 'frame_id', 'boxes'} or row['pipeline'] not in MODELS:
            raise ValueError('Unexpected prediction model/fields')
        for box in row['boxes']: validate_box(box, True)
        pred_keys.append((row['pipeline'], *identity(row)))
    if len(set(pred_keys)) != 36864 or set(pred_keys) != {(model, *key) for model in MODELS for key in keys}:
        raise ValueError('Duplicate or missing six-model/frame prediction coverage')
    return frames, predictions, read(folder/'expected_reports.json'), json.loads(gzip.decompress((folder/'expected_cohorts.json.gz').read_bytes())), read(folder/'sealed_comparisons.json')


def compare(actual, expected, where='result'):
    if type(actual) is not type(expected): raise ValueError('Parity type mismatch: '+where)
    if isinstance(expected, dict):
        if set(actual) != set(expected): raise ValueError('Parity field mismatch: '+where)
        for key in expected: compare(actual[key], expected[key], where+'.'+key)
    elif isinstance(expected, list):
        if len(actual) != len(expected): raise ValueError('Parity list mismatch: '+where)
        for i, (a, b) in enumerate(zip(actual, expected)): compare(a, b, where+'['+str(i)+']')
    elif isinstance(expected, float):
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Numeric parity mismatch: '+where)
    elif actual != expected: raise ValueError('Parity value mismatch: '+where)


def score(frames, predictions):
    by_model = {name: [r for r in predictions if r['pipeline'] == name] for name in MODELS}
    truth = {identity(row): row for row in frames}; reports = []; cohorts = {}
    for cutoff in CUTOFFS:
        errors, metrics = {}, {}
        for name in MODELS: errors[name], metrics[name] = matched_errors(by_model[name], frames, cutoff)
        common = set.intersection(*(set(errors[name]) for name in (ORIGINAL, *CORE)))
        extended = common & set(errors[AUG])
        common_sorted = sorted(common); extended_sorted = sorted(extended)
        means = {name: float(np.mean([errors[name][key] for key in common_sorted])) for name in (ORIGINAL, *CORE)}
        extended_means = {name: float(np.mean([errors[name][key] for key in extended_sorted])) for name in MODELS}
        assigned = {}
        for name in PAIR:
            assigned[name], fp = assign_boxes(by_model[name], truth, cutoff)
            if set(errors[name]) != set(assigned[name]) or fp != metrics[name]['false_positives']:
                raise ValueError('Published matcher identity/FP parity failed')
        pair = set(assigned[PAIR[0]]) & set(assigned[PAIR[1]])
        reports.append({'raw_score_cutoff': cutoff, 'per_model_all_targets': metrics,
            'original_plus_core4_common': {'models': [ORIGINAL, *CORE], 'matches': len(common), 'xy_center_mae_m': means},
            'separate_all6_common': {'models': list(MODELS), 'matches': len(extended), 'xy_center_mae_m': extended_means},
            'pair_common_target_count': len(pair),
            'pair_common_target_geometry': {name: geometry(assigned[name], pair) for name in PAIR},
            'paired_errors_vs_original': {name: paired_errors(errors[ORIGINAL], errors[name]) for name in CORE}})
        cohorts[str(cutoff)] = {'original_plus_core4': common_sorted, 'separate_all6': extended_sorted, 'mixed_aug_pair': sorted(pair)}
    return {'models': list(MODELS), 'cutoffs': list(CUTOFFS), 'frames': 6144, 'visible_targets': 8155, 'reports': reports}, json.loads(json.dumps(cohorts))


def sealed_projection(actual):
    low, high = actual['reports']
    return {'all_original_core_metrics_at_0p35': {name: low['per_model_all_targets'][name] for name in (ORIGINAL, *CORE)},
        'original_plus_core4_common_at_0p35': low['original_plus_core4_common'],
        'paired_errors_vs_original_at_0p35': low['paired_errors_vs_original'],
        'mixed_aug_pair_both_cutoffs': [{'raw_score_cutoff': row['raw_score_cutoff'],
            'per_model_all_targets': {name: row['per_model_all_targets'][name] for name in PAIR},
            'pair_common_target_count': row['pair_common_target_count'],
            'pair_common_target_geometry': row['pair_common_target_geometry']} for row in (low, high)]}


def maximum_float_delta(a, b):
    if isinstance(a, dict): return max((maximum_float_delta(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list): return max((maximum_float_delta(x, y) for x, y in zip(a, b)), default=0.)
    return abs(a-b) if isinstance(a, float) else 0.


def reproduce(folder=BUNDLE, cutoff=None):
    frames, predictions, expected, expected_cohorts, sealed = load_bundle(folder, cutoff)
    actual, cohorts = score(frames, predictions)
    if cohorts != expected_cohorts: raise ValueError('Exact common-target cohort identities changed')
    compare(actual, expected); projection = sealed_projection(actual); compare(projection, sealed)
    return {'status': 'VERIFIED_CACHED_NATIVE_NUMERIC_REPRODUCTION', 'numeric_export_values_exactly_equal': actual == expected,
        'all_sealed_metric_fields_within_1e_minus_12': True, 'sealed_values_exactly_equal': projection == sealed,
        'archived_mixed_aug_pair_metrics_and_geometry_exactly_equal': projection['mixed_aug_pair_both_cutoffs'] == sealed['mixed_aug_pair_both_cutoffs'],
        'archived_original_core_all_target_metrics_exactly_equal': projection['all_original_core_metrics_at_0p35'] == sealed['all_original_core_metrics_at_0p35'],
        'archived_original_paired_center_summaries_exactly_equal': projection['paired_errors_vs_original_at_0p35'] == sealed['paired_errors_vs_original_at_0p35'],
        'maximum_sealed_float_absolute_delta': maximum_float_delta(projection, sealed),
        'counts_and_cohort_IDs_exact': True, 'cached_GT_frame_rows': 6144, 'cached_prediction_rows': 36864,
        'model_cutoff_summary_count': 12, 'requirements': {'numpy': np.__version__, 'shapely': shapely.__version__},
        'raw_clouds_opened': False, 'weights_opened': False, 'GPU_or_network_queries': False,
        'role': 'KNOWN_DEVELOPMENT_CACHED_DIAGNOSTIC_ONLY', 'metrics': actual}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path)
    p.add_argument('--cutoff', type=float, help='Validate cache support; both predeclared cutoffs are always reported')
    args = p.parse_args()
    if args.output and args.output.exists(): p.error('Choose a fresh output; preserve earlier reproductions')
    result = reproduce(cutoff=args.cutoff)
    if args.output: args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'metrics'}, indent=2))


if __name__ == '__main__': main()
