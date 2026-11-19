"""Run two equal-rules, reversed-grid heats without changing the policies."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cars', type=int, default=20)
    parser.add_argument('--secondary-cars', type=int, default=8)
    parser.add_argument('--physics-threads', type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'tournament.json'
    if manifest.exists():
        parser.error('Choose a fresh tournament output directory')
    manifest.write_text(json.dumps({
        'heats': [{'heat_id': f'heat-{i}', 'path': f'heat-{i}'} for i in (1, 2)],
        'purpose': 'Compare four frozen policies with equal entries and reversed grids',
        'cars_per_heat': args.cars,
        'gpu_allocation': {'pro6000': args.cars - args.secondary_cars, 'rtx3090': args.secondary_cars},
        'physics_threads': args.physics_threads,
    }, indent=2) + '\n')
    for number in (1, 2):
        heat = output / f'heat-{number}'
        command = [sys.executable, str(ROOT / 'scripts/run_tournament_heat.py'),
                   '--cars', str(args.cars), '--secondary-cars', str(args.secondary_cars),
                   '--physics-threads', str(args.physics_threads),
                   '--output', str(heat)]
        if number == 2:
            command.append('--reverse-grid')
        print(f'Starting heat {number}: {args.cars} racers; log={output / f"heat-{number}.log"}', flush=True)
        with (output / f'heat-{number}.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(result.returncode)
        print(f'Heat {number} complete', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_tournament.py'), str(output)], check=True)


if __name__ == '__main__':
    main()
