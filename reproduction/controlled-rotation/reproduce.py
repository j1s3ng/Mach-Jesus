#!/usr/bin/env python3
"""Recompute the published cached rotation diagnostics; never run a model."""
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

# Explicit local import also works with Python's isolated (-I) mode.
sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE))
from metric import assign_boxes, geometry
import numpy as np
import shapely

ANGLES = (-90, 0, 90, 180)
CUTOFF = .425
CORE = ('head-lr1e-4-s7', 'full-uniform-lr1e-4-s7',
        'full-uniform-lr1e-5-s7', 'full-mixed-head1e-4-backbone1e-5-s7')
MODELS = ('original', *CORE, 'full-mixed-rotaug4-s7')
INPUT_FILES = {'README.md', 'requirements.txt', 'reproduce.py', 'metric.py',
               'test_reproduction.py', 'frames.jsonl.gz', 'predictions.jsonl.gz',
               'expected_reports.json', 'expected_cohorts.json', 'provenance.json'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(row):
    return row['scene_id'], row['frame_id']


def read_cache(path, expected_count):
    raw = gzip.decompress(Path(path).read_bytes())
    if len(raw) > 10_000_000:
        raise ValueError('Unexpected decompressed cache size')
    values = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(values) != expected_count:
        raise ValueError('Missing or extra cached row: ' + Path(path).name)
    return values


def validate_box(box, *, prediction):
    required = {'center', 'size', 'yaw', 'score'} if prediction else {'center', 'size', 'yaw', 'visible_points'}
    if set(box) != required:
        raise ValueError('Unexpected numeric box fields')
    if len(box['center']) != 3 or len(box['size']) != 3:
        raise ValueError('Three-dimensional box shape required')
    numbers = [*box['center'], *box['size'], box['yaw']]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in numbers):
        raise ValueError('Nonfinite or nonnumeric box')
    if min(box['size']) <= 0:
        raise ValueError('Positive box dimensions required')
    if prediction:
        if not CUTOFF <= box['score'] <= 1:
            raise ValueError('Prediction below the exported cache floor')
    elif not isinstance(box['visible_points'], int) or box['visible_points'] < 3:
        raise ValueError('Frozen visible-target population requires at least three measured returns')


def load_bundle(folder=BUNDLE, *, cutoff=CUTOFF):
    folder = Path(folder)
    manifest = read(folder / 'manifest.json')
    if manifest.get('manifest_sha256') != digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}):
        raise ValueError('Manifest semantic hash changed')
    if cutoff < manifest['exported_prediction_floor']:
        raise ValueError('Cutoff is below the cache floor; omitted predictions cannot be recovered')
    if cutoff != CUTOFF or manifest['exported_prediction_floor'] != CUTOFF:
        raise ValueError('Only the predeclared RAW .425 reproduction is supported')
    if manifest.get('schema_version') != 1 or manifest.get('models') != list(MODELS) or manifest.get('angles') != list(ANGLES):
        raise ValueError('Frozen model/angle contract changed')
    if set(manifest['files']) != INPUT_FILES:
        raise ValueError('Complete immutable input checksum inventory is required')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or not (folder / name).is_file():
            raise ValueError('Missing or unsafe bundled file: ' + name)
        if (folder / name).stat().st_size != expected['bytes'] or sha(folder / name) != expected['sha256']:
            raise ValueError('Bundled file checksum changed: ' + name)
    if np.__version__ != '1.26.4' or shapely.__version__ != '2.1.2':
        raise ValueError('Use the pinned tested NumPy/Shapely versions in requirements.txt')
    frames = read_cache(folder / 'frames.jsonl.gz', 512)
    predictions = read_cache(folder / 'predictions.jsonl.gz', 3072)
    frame_fields = {'scene_id', 'source_frame_id', 'frame_id', 'degrees', 'point_count', 'gt_boxes'}
    for row in frames:
        if set(row) != frame_fields or row['degrees'] not in ANGLES:
            raise ValueError('Unexpected frozen frame fields or angle')
        if not re.fullmatch('[0-9a-f]{24}', row['source_frame_id']) or row['frame_id'] != row['source_frame_id'] + ':rotation' + str(row['degrees']):
            raise ValueError('Frozen frame identity differs')
        if not isinstance(row['point_count'], int) or row['point_count'] < 0:
            raise ValueError('Invalid measured return count')
        for box in row['gt_boxes']:
            validate_box(box, prediction=False)
    frame_keys = [identity(row) for row in frames]
    if len(set(frame_keys)) != 512 or Counter(row['degrees'] for row in frames) != Counter({a: 128 for a in ANGLES}):
        raise ValueError('Duplicate or incomplete four-angle GT coverage')
    source_groups = Counter((row['scene_id'], row['source_frame_id']) for row in frames)
    if len(source_groups) != 128 or set(source_groups.values()) != {4}:
        raise ValueError('Every one of the 128 source frames needs all four rotations')
    if digest(sorted(frame_keys)) != manifest['frame_identity_sha256']:
        raise ValueError('Frozen frame identity population changed')
    if sum(len(row['gt_boxes']) for row in frames) != manifest['rotated_visible_target_count']:
        raise ValueError('Frozen visible-target count changed')
    index = {identity(row): row for row in frames}
    pred_fields = {'pipeline', 'scene_id', 'source_frame_id', 'frame_id', 'degrees', 'boxes'}
    pred_keys = []
    for row in predictions:
        if set(row) != pred_fields or row['pipeline'] not in MODELS or identity(row) not in index:
            raise ValueError('Prediction outside exact frozen model/frame population')
        truth = index[identity(row)]
        if any(row[k] != truth[k] for k in ('scene_id', 'source_frame_id', 'degrees')):
            raise ValueError('Prediction source frame or rotation differs')
        for box in row['boxes']:
            validate_box(box, prediction=True)
        pred_keys.append((row['pipeline'], *identity(row)))
    expected = {(model, *key) for model in MODELS for key in frame_keys}
    if len(set(pred_keys)) != 3072 or set(pred_keys) != expected:
        raise ValueError('Duplicate or missing model/frame prediction coverage')
    return frames, predictions, read(folder / 'expected_reports.json'), read(folder / 'expected_cohorts.json')


def shared(assignments, models):
    groups = [set(assignments[(model, angle)]) for model in models for angle in ANGLES]
    return set.intersection(*groups)


def score(frames, predictions):
    """Same analysis loop and common-target definitions as the archived report."""
    indexed = {(row['pipeline'], row['frame_id']): row for row in predictions}
    assignments, reports = {}, []
    for model in MODELS:
        for angle in ANGLES:
            expected = [row for row in frames if row['degrees'] == angle]
            selected = [row for row in expected if (model, row['frame_id']) in indexed]
            truth = {(row['scene_id'], row['frame_id']): row for row in selected}
            observed = [indexed[(model, row['frame_id'])] for row in selected]
            assigned, fp = assign_boxes(observed, truth, CUTOFF)
            source_ids = {row['frame_id']: row['source_frame_id'] for row in selected}
            normalized = {(key[0], source_ids[key[1]], key[2]): value for key, value in assigned.items()}
            assignments[(model, angle)] = normalized
            targets = sum(len(row['gt_boxes']) for row in selected)
            empty = {row['frame_id'] for row in selected if not row['gt_boxes']}
            reports.append({'model': model, 'degrees': angle, 'planned_frames': len(expected), 'scored_frames': len(selected),
                'planned_targets': sum(len(row['gt_boxes']) for row in expected), 'scored_targets': targets,
                'complete_coverage': len(selected) == len(expected), 'missing_frame_ids': [row['frame_id'] for row in expected if row not in selected],
                'true_positives': len(assigned), 'false_positives': fp, 'false_negatives_on_scored_inputs': targets-len(assigned),
                'recall_on_scored_inputs': len(assigned)/targets if targets else None,
                'empty_frames_scored': len(empty), 'empty_frame_false_positives': sum(len(row['boxes']) for row in observed if row['frame_id'] in empty)})
    core_shared = shared(assignments, CORE)
    with_original = shared(assignments, ('original', *CORE))
    with_optional = shared(assignments, MODELS)
    for report in reports:
        model, angle = report['model'], report['degrees']
        current = assignments[(model, angle)]
        common_original = set(current) & set(assignments[('original', angle)])
        def summary(cohort):
            keys = set(cohort) & set(current)
            return {'matched_common_targets': len(keys), 'geometry': geometry(current, keys) if keys else None}
        report['paired_same_angle_vs_original'] = summary(common_original)
        report['fixed_core4_all_angles_cohort'] = summary(core_shared) if model in CORE else None
        report['fixed_core4_plus_original_all_angles_cohort'] = summary(with_original) if model in ('original', *CORE) else None
        report['separate_optional_plus_core_and_original_cohort'] = summary(with_optional)
    cohorts = {'fixed_core4_all_angles': sorted(core_shared),
               'fixed_core4_plus_original_all_angles': sorted(with_original),
               'optional_extended_all_angles': sorted(with_optional)}
    cohorts = json.loads(json.dumps(cohorts))
    return {'core_models': list(CORE), 'all_models': list(MODELS[1:]), 'same_fixed_DEV_frames': 128,
        'angles': list(ANGLES), 'raw_score_cutoff': CUTOFF,
        'core4_fixed_all_angles_common_targets': len(core_shared),
        'core4_plus_original_fixed_all_angles_common_targets': len(with_original),
        'optional_extended_common_targets': len(with_optional), 'reports': reports}, cohorts


def compare(actual, expected, where='result'):
    """Counts/IDs exact; all floating metric fields use 1e-12 tolerance."""
    if type(actual) is not type(expected):
        raise ValueError('Parity type mismatch: ' + where)
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Parity field mismatch: ' + where)
        for key in expected:
            compare(actual[key], expected[key], where + '.' + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Parity list mismatch: ' + where)
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, where + '[' + str(i) + ']')
    elif isinstance(expected, float):
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Numeric parity mismatch: ' + where)
    elif actual != expected:
        raise ValueError('Parity value mismatch: ' + where)


def reproduce(folder=BUNDLE, *, cutoff=CUTOFF):
    frames, predictions, expected, expected_cohorts = load_bundle(folder, cutoff=cutoff)
    actual, cohorts = score(frames, predictions)
    if cohorts != expected_cohorts:
        raise ValueError('Exact common-target cohort identities changed')
    compare(actual, expected)
    return {'status': 'VERIFIED_CACHED_NUMERIC_REPRODUCTION',
        'numeric_json_values_exactly_equal': actual == expected,
        'numeric_float_tolerance': 1e-12, 'report_count': len(actual['reports']),
        'cached_GT_frame_rows': len(frames), 'cached_prediction_rows': len(predictions),
        'frozen_cohort_identity_parity': True, 'requirements': {'numpy': np.__version__, 'shapely': shapely.__version__},
        'raw_clouds_opened': False, 'weights_opened': False, 'GPU_or_network_queries': False,
        'role': 'KNOWN_DEVELOPMENT_CACHED_DIAGNOSTIC_ONLY', 'metrics': actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cutoff', type=float, default=CUTOFF)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Choose a fresh output; preserve previous reproductions')
    result = reproduce(cutoff=args.cutoff)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'metrics'}, indent=2))


if __name__ == '__main__':
    main()
