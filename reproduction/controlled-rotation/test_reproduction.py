#!/usr/bin/env python3
"""Self-contained regressions for coverage, integrity, cohorts and cache floors."""
import gzip
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reproduce as r
from metric import assign_boxes, geometry


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cached-rotation-test-')
        self.folder = Path(self.temp.name) / 'moved-bundle'
        shutil.copytree(HERE, self.folder, ignore=shutil.ignore_patterns('__pycache__', 'reproduced*.json'))

    def tearDown(self): self.temp.cleanup()

    def rebind(self, name):
        path = self.folder / name
        manifest = r.read(self.folder / 'manifest.json')
        manifest['files'][name] = {'sha256': r.sha(path), 'bytes': path.stat().st_size}
        manifest['manifest_sha256'] = r.digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        (self.folder / 'manifest.json').write_text(json.dumps(manifest))

    def write_rows(self, name, values):
        (self.folder / name).write_bytes(gzip.compress(b''.join(r.canonical(row)+b'\n' for row in values), mtime=0))
        self.rebind(name)

    def test_full_report_and_cohort_parity_without_network(self):
        original = socket.socket
        def forbidden(*args, **kwargs): raise AssertionError('Network access is forbidden')
        socket.socket = forbidden
        try: result = r.reproduce(self.folder)
        finally: socket.socket = original
        self.assertTrue(result['numeric_json_values_exactly_equal'])
        self.assertEqual(result['report_count'], 24)
        metrics = result['metrics']
        self.assertEqual(metrics['core4_fixed_all_angles_common_targets'], 102)
        self.assertEqual(metrics['core4_plus_original_fixed_all_angles_common_targets'], 41)
        self.assertEqual(metrics['optional_extended_common_targets'], 41)

    def test_changed_cache_bytes_are_rejected(self):
        path = self.folder / 'predictions.jsonl.gz'
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'): r.reproduce(self.folder)

    def test_missing_cache_checksum_cannot_silently_disable_integrity(self):
        manifest = r.read(self.folder / 'manifest.json')
        manifest['files'].pop('predictions.jsonl.gz')
        manifest['manifest_sha256'] = r.digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        (self.folder / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'checksum inventory'): r.reproduce(self.folder)

    def test_missing_row_is_rejected_even_with_updated_file_checksum(self):
        values = r.read_cache(self.folder / 'predictions.jsonl.gz', 3072)
        self.write_rows('predictions.jsonl.gz', values[:-1])
        with self.assertRaisesRegex(ValueError, 'Missing or extra'): r.reproduce(self.folder)

    def test_duplicate_row_cannot_replace_another_model_frame(self):
        values = r.read_cache(self.folder / 'predictions.jsonl.gz', 3072)
        values[-1] = values[0]
        self.write_rows('predictions.jsonl.gz', values)
        with self.assertRaisesRegex(ValueError, 'Duplicate or missing'): r.reproduce(self.folder)

    def test_common_cohort_identity_change_is_rejected(self):
        name = 'expected_cohorts.json'; value = r.read(self.folder / name)
        value['fixed_core4_all_angles'][0][2] += 1
        (self.folder / name).write_text(json.dumps(value)); self.rebind(name)
        with self.assertRaisesRegex(ValueError, 'cohort identities'): r.reproduce(self.folder)

    def test_cutoff_below_exported_floor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'below the cache floor'): r.reproduce(self.folder, cutoff=.35)

    def test_declared_higher_floor_cannot_support_lower_cutoff(self):
        manifest = r.read(self.folder / 'manifest.json'); manifest['exported_prediction_floor'] = .5
        manifest['manifest_sha256'] = r.digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        (self.folder / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'below the cache floor'): r.reproduce(self.folder)

    def test_below_floor_cached_box_is_not_silently_used(self):
        values = r.read_cache(self.folder / 'predictions.jsonl.gz', 3072)
        next(row for row in values if row['boxes'])['boxes'][0]['score'] = .4
        self.write_rows('predictions.jsonl.gz', values)
        with self.assertRaisesRegex(ValueError, 'below the exported cache floor'): r.reproduce(self.folder)

    def test_score_descending_unique_nearest_match_is_preserved(self):
        def box(x, score=1.): return {'center': [x, 0., 0.], 'size': [1., 1., 1.], 'yaw': 0., 'score': score}
        truth = {('scene', 'frame'): {'gt_boxes': [box(0.), box(3.)]}}
        # The lower-score ambiguous prediction comes first in the input file.
        assigned, fp = assign_boxes([{'scene_id': 'scene', 'frame_id': 'frame', 'boxes': [box(1.5, .5), box(.1, .9)]}], truth, .425)
        self.assertEqual(len(assigned), 2); self.assertEqual(fp, 0)
        self.assertEqual(assigned[('scene', 'frame', 0)]['prediction']['score'], .9)

    def test_pi_symmetric_heading_and_upright_iou_are_unchanged(self):
        import math
        gt = {'center': [0., 0., 0.], 'size': [4., 2., 1.], 'yaw': 0.}
        pred = dict(gt, yaw=math.pi)
        result = geometry({('scene', 'source', 0): {'truth': gt, 'prediction': pred, 'center_xy_m': 0.}}, {('scene', 'source', 0)})
        self.assertAlmostEqual(result['directed_heading_rad']['mean'], math.pi)
        self.assertAlmostEqual(result['undirected_box_heading_rad']['mean'], 0.)
        self.assertAlmostEqual(result['bev_iou']['mean'], 1.)
        self.assertAlmostEqual(result['upright_3d_iou']['mean'], 1.)

    def test_isolated_fresh_interpreter_runs_from_moved_folder(self):
        output = self.folder / 'reproduced-isolated.json'
        env = {'PATH': os.defpath, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        run = subprocess.run([sys.executable, '-I', str(self.folder / 'reproduce.py'), '--output', str(output)],
                             cwd=self.temp.name, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(json.loads(output.read_text())['numeric_json_values_exactly_equal'])


if __name__ == '__main__': unittest.main(verbosity=2)
