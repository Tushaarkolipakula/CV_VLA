"""Fail closed on source leakage, malformed actions, or inadequate recovery coverage."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import h5py
import numpy as np
from .data import ChunkData
from .generate_all import SUITES

def audit(root,output):
    source=Path('/home/tushaar/data/LIBERO-datasets')
    clean=[str(source/s) for s in SUITES];aug=[str(Path(root)/s) for s in SUITES]
    if not (Path(root)/'COMPLETE.json').exists():raise ValueError('Generation is not complete')
    ds=ChunkData(clean+aug);baseline=ChunkData(clean)
    try:
        if ds.stats()!=baseline.stats() or ds.val_sources!=baseline.val_sources:raise ValueError('Clean and augmented stats/splits differ')
        tr={(ds.demos[i]['task'],ds.demos[i]['source']) for i in ds.train}
        va={(ds.demos[i]['task'],ds.demos[i]['source']) for i in ds.val}
        if tr&va:raise ValueError('Source-demo leakage')
        coverage=defaultdict(Counter);count=0
        for i,d in enumerate(ds.demos):
            if d['kind']!='recovery':continue
            g=ds.group(d);length=len(g['actions']);r=d['recovery_steps']
            if not 0<r<=length or length>r+49:raise ValueError('Bad compact recovery length')
            actions=g['actions'][:]
            if actions.shape[1]!=7 or not np.isfinite(actions).all() or np.abs(actions).max()>1.00001:raise ValueError('Invalid recovery actions')
            for key in ['agentview_rgb','eye_in_hand_rgb','gripper_states','ee_pos','ee_ori']:
                if len(g['obs'][key])!=length:raise ValueError('Observation/action length mismatch')
            if i in ds.train:coverage[d['task']][d['level']]+=1
            count+=1
        missing=[]
        for task in sorted({d['task'] for d in ds.demos}):
            for level in range(1,6):
                if coverage[task][level]<3:missing.append([task,level,coverage[task][level]])
        if missing:raise ValueError(f'Need at least 3 train recoveries per task per level; insufficient: {missing}')
        rng=np.random.default_rng(114)
        for level in range(1,6):
            batch=ds.batch(rng,16,level,1.)
            if not batch['is_recovery'].all() or not batch['valid'][:,0].all():raise ValueError('Recovery sampling test failed')
        result={'passed':True,'recovery_episodes':count,'train_source_demos':len(tr),'validation_source_demos':len(va),
            'train_coverage':{k:dict(v) for k,v in coverage.items()},'files':ds.files,'stats':ds.stats(),
            'checks':['disjoint source demos','shared clean quantile statistics','aligned pre-action recovery lengths',
                      'finite bounded actions','all five levels covered for every task','recovery-only sampling']}
        Path(output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['files','stats','train_coverage']}),flush=True)
        return result
    finally:ds.close();baseline.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit(a.root,a.out)
