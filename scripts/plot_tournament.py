"""Create a standalone figure of actual race progress and policy outcomes."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.local/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {'reference': '#64748b', 'neural_v1': '#2563eb',
          'neural_v2': '#ea580c', 'neural_v3': '#059669'}
LABELS = {'reference': 'Reference', 'neural_v1': 'Neural V1',
          'neural_v2': 'Neural V2', 'neural_v3': 'Neural V3'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    run = args.directory
    summary = json.loads((run / 'results.json').read_text())
    manifest = json.loads((run / 'tournament.json').read_text())
    figure = plt.figure(figsize=(14, 9), layout='constrained')
    grid = figure.add_gridspec(2, 2, height_ratios=[1.4, 1])
    for index, (heat, spec) in enumerate(zip(summary['heats'], manifest['heats'])):
        axis = figure.add_subplot(grid[0, index])
        standings = {row['car_id']: row for row in heat['standings']}
        length = heat['rules']['reference_length_m']
        traces, frozen = {}, {}
        for line in (run / spec['path'] / 'leaderboard.jsonl').open():
            board = json.loads(line)
            for row in board['cars']:
                car_id = row['car_id']
                progress = min(length, row['progress_m'])
                if car_id in frozen:
                    progress = frozen[car_id]
                elif row.get('terminal'):
                    frozen[car_id] = progress
                traces.setdefault(car_id, [[], []])
                traces[car_id][0].append(board['time_s'])
                traces[car_id][1].append(progress / 1000.)
        for car_id, (times, progress) in traces.items():
            result = standings[car_id]
            axis.plot(times, progress, color=COLORS[result['policy']],
                      lw=2.3 if result['race_winner'] else .9,
                      alpha=1. if result['race_winner'] else .55,
                      ls='-' if result['valid_finish'] else '--')
        for row in heat['standings']:
            if row['race_winner']:
                axis.annotate(f"{row['car_id']} / {LABELS[row['policy']]}\n{row['finish_time_s']:.3f} s",
                              (row['finish_time_s'], length / 1000.), xytext=(8, -40),
                              textcoords='offset points', color=COLORS[row['policy']], weight='bold', fontsize=10)
        axis.axhline(length / 1000., lw=.6, color='#334155', alpha=.5)
        minimum = min(values[1][0] for values in traces.values()) - .025
        axis.set(title=f"Heat {index + 1}: {'original' if index == 0 else 'reversed'} grid",
                 xlabel='Race time (seconds)', ylabel='Ordered progress (km)', ylim=(minimum, length / 1000. + .22))
        axis.grid(alpha=.15)
    legend = [Line2D([0], [0], color=COLORS[key], lw=2, label=LABELS[key]) for key in COLORS]
    figure.legend(handles=legend, loc='outside upper right', ncols=4, frameon=False)
    axis = figure.add_subplot(grid[1, :])
    policies = list(COLORS)
    counts = {policy: Counter() for policy in policies}
    for heat in summary['heats']:
        for row in heat['standings']:
            category = ('Clean valid finish' if row['clean_finish'] else 'Valid finish with overlaps') if row['valid_finish'] else (
                'Invalid full-lap crossing' if row['status'] == 'invalid_finish' else 'Did not finish')
            counts[row['policy']][category] += 1
    left = [0] * len(policies)
    for category, color in [('Clean valid finish', '#059669'), ('Valid finish with overlaps', '#2563eb'),
                            ('Invalid full-lap crossing', '#f59e0b'), ('Did not finish', '#cbd5e1')]:
        values = [counts[policy][category] for policy in policies]
        axis.barh([LABELS[policy] for policy in policies], values, left=left, color=color, label=category)
        for index, value in enumerate(values):
            if value:
                axis.text(left[index] + value / 2, index, str(value), ha='center', va='center',
                          color='#0f172a' if color in ('#f59e0b', '#cbd5e1') else 'white', weight='bold')
        left = [start + value for start, value in zip(left, values)]
    axis.set(title='All entries count, including failures and timeouts', xlabel='Entries across both heats')
    axis.invert_yaxis()
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.24), ncols=4, frameon=False, fontsize=9)
    count = len(summary['heats'][0]['standings'])
    figure.suptitle(f'{count}-car AI racing comparison', x=.02, ha='left', weight='bold', fontsize=17)
    output = run / 'comparison.png'
    figure.savefig(output, dpi=160, facecolor='white')
    print(output)


if __name__ == '__main__':
    main()
