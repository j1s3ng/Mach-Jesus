#!/usr/bin/env python3
"""Recompute simulated development BEV AP101 from pinned numeric caches."""
import argparse, gzip, hashlib, json, math
from pathlib import Path
import metric

ROOT=Path(__file__).resolve().parent
MODELS=('original','head_lr1e-4','full_lr1e-4','full_lr1e-5','full_mixed','full_mixed_rotaug')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(d):return json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def rows(p):
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def validate():
 manifest=json.loads((ROOT/'manifest.json').read_text())
 assert hashlib.sha256(canonical({k:v for k,v in manifest.items() if k!='manifest_sha256'})).hexdigest()==manifest['manifest_sha256']
 for name,pin in manifest['files'].items():
  if Path(name).name!=name:raise ValueError('invalid file path')
  p=ROOT/name
  if not p.is_file() or p.stat().st_size!=pin['bytes'] or sha(p)!=pin['sha256']:raise ValueError('file mismatch: '+name)
 truth={}
 for row in rows(ROOT/'frames.jsonl.gz'):
  k=metric.key(row)
  if k in truth:raise ValueError('duplicate frame')
  truth[k]={'eligible':[metric.polygon(b) for b in row['gt_boxes']],'ignored':[metric.polygon(b) for b in row['excluded_gt_boxes']]}
 if len(truth)!=6144 or sum(len(v['eligible']) for v in truth.values())!=8155 or sum(len(v['ignored']) for v in truth.values())!=10277:raise ValueError('unexpected population')
 predictions={name:[] for name in MODELS}
 for row in rows(ROOT/'predictions.jsonl.gz'):
  name=row.pop('model')
  if name not in predictions:raise ValueError('unexpected model')
  predictions[name].append(row)
 expected=json.loads((ROOT/'expected.json').read_text());result={}
 for name in MODELS:
  data=predictions[name];keys=[metric.key(r) for r in data]
  if len(keys)!=6144 or len(set(keys))!=6144 or set(keys)!=set(truth):raise ValueError('incomplete prediction frames')
  if any(not .1<=box['score']<=1 for r in data for box in r['boxes']):raise ValueError('prediction score outside floor')
  result[name]=metric.score(truth,data)
  for k,v in expected[name].items():
   actual=result[name][k]
   if isinstance(v,list):
    if len(v)!=len(actual) or any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(actual,v)):raise ValueError('mismatch '+name+' '+k)
   elif isinstance(v,float):
    if not math.isclose(actual,v,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('mismatch '+name+' '+k)
   elif actual!=v:raise ValueError('mismatch '+name+' '+k)
  print(name,'BEV AP101 %',round(result[name]['ap101']*100,3),flush=True)
 return {'status':'PASS','scope':'known simulated DEV, fixed score floor 0.1; not an official benchmark','models':result}
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 if args.output.exists():raise ValueError('choose a new output')
 out=validate();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
