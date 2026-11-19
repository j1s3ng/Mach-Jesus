"""Summarize actual per-car detector latency and simulator throughput."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    run = args.directory.resolve()
    manifest = json.loads((run / 'tournament.json').read_text())
    heats = []
    for entry in manifest['heats']:
        heat = run / entry['path']
        config = json.loads((heat / 'config.json').read_text())
        result = json.loads((heat / 'heat_results.json').read_text())
        period = config['sensor_period_ms']
        cars, all_latencies = [], []
        for entrant in config['entrants']:
            car = heat / 'cars' / entrant['car_id']
            prediction_file = car / 'predictions.jsonl'
            latencies = []
            if prediction_file.exists():
                for line in prediction_file.open():
                    row = json.loads(line)
                    if row.get('inference_ms') is not None:
                        latencies.append(float(row['inference_ms']))
            final = json.loads((car / 'result.json').read_text()) if (car / 'result.json').exists() else {}
            all_latencies.extend(latencies)
            cars.append({'car_id': entrant['car_id'], 'policy': entrant['policy'],
                         'vision_url': entrant.get('vision_url', config['vision_url']),
                         'requests': len(latencies), 'p95_ms': float(np.percentile(latencies, 95)) if latencies else None,
                         'over_period_requests': sum(value > period for value in latencies),
                         'status': final.get('status', 'missing'), 'failure': final.get('failure')})
        heats.append({'heat_id': entry['heat_id'], 'car_count': len(cars),
                      'wall_clock_context': entry.get('wall_clock_context', 'race only'),
                      'sensor_period_ms': period, 'requests': len(all_latencies),
                      'median_ms': float(np.median(all_latencies)) if all_latencies else None,
                      'p95_ms': float(np.percentile(all_latencies, 95)) if all_latencies else None,
                      'p99_ms': float(np.percentile(all_latencies, 99)) if all_latencies else None,
                      'over_period_requests': sum(value > period for value in all_latencies),
                      'wall_seconds': result['wall_seconds'], 'simulation_seconds': result['simulation_seconds'],
                      'real_time_factor': result['real_time_factor'],
                      'statuses': dict(Counter(car['status'] for car in cars)), 'cars': cars})
    output = run / 'runtime_summary.json'
    output.write_text(json.dumps({'heats': heats}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
