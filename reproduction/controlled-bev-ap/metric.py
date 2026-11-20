import collections, math
import numpy as np
from shapely.geometry import Polygon

def key(r):return (r['scene_id'],r['frame_id'])

def polygon(b):
 x,y=b['center'][:2]; a,c=b['size'][:2]; yaw=b['yaw']
 if not all(math.isfinite(v) for v in (x,y,a,c,yaw)) or min(a,c)<=0:raise ValueError('invalid box geometry')
 cs,sn=math.cos(yaw),math.sin(yaw)
 return Polygon([(x+u*cs-v*sn,y+u*sn+v*cs) for u,v in ((-a/2,-c/2),(a/2,-c/2),(a/2,c/2),(-a/2,c/2))])

def iou(a,b):
 if not a.intersects(b):return 0.
 inter=a.intersection(b).area
 return inter/(a.area+b.area-inter)

def score(truth, predictions):
 denominator=sum(len(v['eligible']) for v in truth.values())
 if not denominator:raise ValueError('no eligible targets')
 ranked=[]
 for frame in predictions:
  k=key(frame)
  if k not in truth:raise ValueError('extra frame')
  for index,box in enumerate(frame['boxes']):
   value=float(box['score'])
   if not math.isfinite(value) or not 0<=value<=1:raise ValueError('invalid score')
   if value>=.1:ranked.append((-value,k[0],k[1],index,polygon(box)))
 ranked.sort(key=lambda x:x[:4]);used=collections.defaultdict(set);tp=[];fp=[];ignored=0
 for _,scene,frame,index,poly in ranked:
  k=(scene,frame);g=truth[k];eligible=g['eligible']
  choices=[(iou(poly,p),i) for i,p in enumerate(eligible) if i not in used[k]]
  overlap,index_gt=max(choices,default=(0.,-1))
  if overlap>=.5:
   used[k].add(index_gt);tp.append(1);fp.append(0)
  elif any(iou(poly,p)>=.5 for p in g['ignored']):ignored+=1
  else:tp.append(0);fp.append(1)
 t=np.cumsum(tp,dtype=np.int64);f=np.cumsum(fp,dtype=np.int64)
 recall=t/denominator;precision=t/np.maximum(t+f,1)
 envelope=np.maximum.accumulate(precision[::-1])[::-1]
 vals=[]
 for level in np.linspace(0.,1.,101):
  j=int(np.searchsorted(recall,level,side='left'))
  vals.append(float(envelope[j]) if j<len(envelope) else 0.)
 return {'ap101':float(np.mean(vals)),'max_recall':float(recall[-1]) if len(recall) else 0.,'tp':int(t[-1]) if len(t) else 0,'fp':int(f[-1]) if len(f) else 0,'ignored_predictions':ignored,'eligible_targets':denominator,'scored_predictions':len(ranked),'recall_samples':vals}
