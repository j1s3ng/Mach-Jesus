"""Freeze runnable source, the reference, and the current candidate roster."""
import argparse
import hashlib
import json
from pathlib import Path
from improvement_ledger import freeze_snapshot,Ledger

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate_id')
    p.add_argument('--roster',type=Path);p.add_argument('--visibility',type=float,default=45.)
    p.add_argument('--detector',type=Path,help='Additional selected detector artifact; official baseline is retained')
    a=p.parse_args()
    files={}
    for folder in ('src','webots/controllers','webots/protos'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc' and path.name not in ('runtime.ini','.DS_Store'):
                files[str(path.relative_to(ROOT))]=path
    for name in ('generate_hotlap_world.py','generate_tournament_world.py'):
        files[f'scripts/{name}']=ROOT/'scripts'/name
    for name in ('serve_race_vision.py','serve_leased_candidate.py','watch_owned_gpu_services.py',
                 'package_scored_detector.py','start_sim_detector_candidate.sh'):
        path=ROOT/'scripts'/name
        if path.is_file():files[f'scripts/{name}']=path
    files['reference.npz']=ROOT/'.local/f1-reference/reference.npz'
    defaults=[
        ('neural_v1','Neural V1','runs/driver-training/model.npz'),
        ('neural_v4_seed7','Neural V4 seed7','runs/improvement-12h/training/candidates-v4/v4-seed7/model.npz'),
        ('neural_v4_seed17','Neural V4 seed17','runs/improvement-12h/training/candidates-v4/v4-seed17/model.npz')]
    roster=json.loads(a.roster.read_text()) if a.roster else [
        {'policy':'reference','label':'Reference','model_path':None},
        *[{'policy':policy,'label':label,'model_path':str(ROOT/model)} for policy,label,model in defaults]]
    policies=[]
    for spec in roster:
        entry={'policy':spec['policy'],'label':spec['label']}
        if spec.get('model_path'):
            key=f"models/{spec['policy']}.npz";files[key]=Path(spec['model_path']);entry['model_relpath']=key
        policies.append(entry)
    official=ROOT/'runs/improvement-12h/checkpoints/official_voxelnext.pth'
    if official.exists():files['detectors/official_voxelnext.pth']=official
    selected_detector=None
    if a.detector:
        files['detectors/selected_detector.pth']=a.detector
        selected_detector={'path':'detectors/selected_detector.pth',
                           'checkpoint_sha256':hashlib.sha256(a.detector.read_bytes()).hexdigest()}
    destination=ROOT/'runs/improvement-12h/snapshots'/a.candidate_id
    manifest=freeze_snapshot(destination,files,metadata={'candidate_id':a.candidate_id,'policies':policies,
                              'policy_parameters':{'visible_distance_m':a.visibility},
                              'selected_detector':selected_detector,
                              'protocol':'race-improvement-v2','training':'supervised imitation; simulated evaluation'})
    Ledger(ROOT/'runs/improvement-12h/ledger.jsonl').append('candidate_frozen',{
        'candidate_id':a.candidate_id,'snapshot_path':str(destination),'snapshot_sha256':manifest['snapshot_sha256']})
    print(destination);print(manifest['snapshot_sha256'])


if __name__=='__main__':main()
