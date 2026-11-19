"""Short display labels for the already frozen four-arm report; no new metrics."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.report_controlled_ablation import save_figures


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous vector figures')
    report=copy.deepcopy(json.loads(a.report.read_text()))
    labels=['Head\n1e−4','Full\n1e−4','Full\n1e−5','Mixed\n1e−4 / 1e−5']
    for row,label in zip(report['rows'],labels):row['label']=label
    a.output.mkdir(parents=True);save_figures(report,a.output)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output/'plot_provenance.json').write_text(json.dumps({'numeric_report_sha256':digest(a.report),'plot_source_sha256':digest(Path(__file__)),'renderer_source_sha256':digest(ROOT/'scripts/report_controlled_ablation.py'),'display_label_change_only':True,'figures':{f.name:digest(f) for f in a.output.iterdir() if f.is_file()}},indent=2)+'\n')


if __name__=='__main__':main()
