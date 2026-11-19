#!/usr/bin/env python3
"""Portable integrity, population, cutoff, cohort and isolated-run regressions."""
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


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='native-cache-test-')
        self.folder = Path(self.temp.name)/'moved bundle'
        shutil.copytree(HERE, self.folder, ignore=shutil.ignore_patterns('__pycache__', 'reproduced*.json'))

    def tearDown(self): self.temp.cleanup()

    def rebind(self, name):
        manifest = r.read(self.folder/'manifest.json'); path = self.folder/name
        manifest['files'][name] = {'sha256': r.sha(path), 'bytes': path.stat().st_size}
        manifest['manifest_sha256'] = r.digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        (self.folder/'manifest.json').write_text(json.dumps(manifest))

    def write_rows(self, name, values):
        (self.folder/name).write_bytes(gzip.compress(b''.join(r.canonical(v)+b'\n' for v in values), mtime=0)); self.rebind(name)

    def test_all_numeric_reports_and_named_cohorts_without_network(self):
        original = socket.socket
        def forbidden(*args, **kwargs): raise AssertionError('Network is forbidden')
        socket.socket = forbidden
        try: result = r.reproduce(self.folder)
        finally: socket.socket = original
        self.assertTrue(result['numeric_export_values_exactly_equal'])
        self.assertTrue(result['all_sealed_metric_fields_within_1e_minus_12'])
        self.assertTrue(result['archived_mixed_aug_pair_metrics_and_geometry_exactly_equal'])
        self.assertTrue(result['archived_original_core_all_target_metrics_exactly_equal'])
        self.assertEqual(result['model_cutoff_summary_count'], 12)
        low, high = result['metrics']['reports']
        self.assertEqual(low['original_plus_core4_common']['matches'], 3953)
        self.assertEqual([low['pair_common_target_count'], high['pair_common_target_count']], [7671, 7431])
        self.assertNotIn(r.AUG, low['original_plus_core4_common']['models'])
        self.assertIn(r.AUG, low['separate_all6_common']['models'])

    def test_changed_numeric_cache_bytes_fail_before_scoring(self):
        path = self.folder/'predictions.jsonl.gz'; path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'): r.reproduce(self.folder)

    def test_missing_prediction_row_is_not_an_empty_output(self):
        values = r.read_cache(self.folder/'predictions.jsonl.gz', 36864)
        self.write_rows('predictions.jsonl.gz', values[:-1])
        with self.assertRaisesRegex(ValueError, 'Missing or extra'): r.reproduce(self.folder)

    def test_duplicate_model_frame_cannot_replace_a_missing_one(self):
        values = r.read_cache(self.folder/'predictions.jsonl.gz', 36864); values[-1] = values[0]
        self.write_rows('predictions.jsonl.gz', values)
        with self.assertRaisesRegex(ValueError, 'Duplicate or missing'): r.reproduce(self.folder)

    def test_duplicate_GT_frame_is_rejected_after_checksum_rebind(self):
        values = r.read_cache(self.folder/'frames.jsonl.gz', 6144); values[-1] = values[0]
        self.write_rows('frames.jsonl.gz', values)
        with self.assertRaisesRegex(ValueError, 'Duplicate or changed'): r.reproduce(self.folder)

    def test_common_cohort_IDs_cannot_change_with_matching_counts(self):
        name = 'expected_cohorts.json.gz'; value = json.loads(gzip.decompress((self.folder/name).read_bytes()))
        value['0.35']['original_plus_core4'][0][2] += 1
        (self.folder/name).write_bytes(gzip.compress(r.canonical(value), mtime=0)); self.rebind(name)
        with self.assertRaisesRegex(ValueError, 'cohort identities'): r.reproduce(self.folder)

    def test_cutoff_below_floor_is_impossible(self):
        with self.assertRaisesRegex(ValueError, 'below cache floor'): r.reproduce(self.folder, cutoff=.25)

    def test_unpredeclared_cutoff_is_not_a_hidden_search(self):
        with self.assertRaisesRegex(ValueError, 'predeclared'): r.reproduce(self.folder, cutoff=.4)

    def test_below_floor_box_cannot_be_introduced(self):
        values = r.read_cache(self.folder/'predictions.jsonl.gz', 36864)
        next(row for row in values if row['boxes'])['boxes'][0]['score'] = .3
        self.write_rows('predictions.jsonl.gz', values)
        with self.assertRaisesRegex(ValueError, 'below exported cache floor'): r.reproduce(self.folder)

    def test_manifest_cannot_drop_a_numeric_cache_checksum(self):
        manifest = r.read(self.folder/'manifest.json'); manifest['files'].pop('frames.jsonl.gz')
        manifest['manifest_sha256'] = r.digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        (self.folder/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'checksum inventory'): r.reproduce(self.folder)

    def test_historical_reference_change_is_not_silently_accepted(self):
        name = 'sealed_comparisons.json'; value = r.read(self.folder/name)
        value['all_original_core_metrics_at_0p35'][r.ORIGINAL]['true_positives'] += 1
        (self.folder/name).write_text(json.dumps(value)); self.rebind(name)
        with self.assertRaisesRegex(ValueError, 'Parity value mismatch'): r.reproduce(self.folder)

    def test_moved_folder_isolated_fresh_interpreter(self):
        output = self.folder/'reproduced-moved.json'
        env = {'PATH': os.defpath, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        run = subprocess.run([sys.executable, '-I', str(self.folder/'reproduce.py'), '--output', str(output)],
            cwd=self.temp.name, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(r.read(output)['counts_and_cohort_IDs_exact'])


if __name__ == '__main__': unittest.main(verbosity=2)
