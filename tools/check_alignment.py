import json
from pathlib import Path
import h5py
import numpy as np
from smolvla_pipeline.generate import restore_demo, make_env, find_bddl
from training.common import CAMERAS, detect_flip, proprio_from_env_obs, proprio_from_dataset

rows=[]
for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
    path=sorted((Path('/home/tushaar/data/LIBERO-datasets')/suite).glob('*.hdf5'))[0]
    with h5py.File(path,'r') as f:
        g=f['data/demo_0'];h,w=g['obs/agentview_rgb'].shape[1:3]
        env=make_env(find_bddl(path.stem.removesuffix('_demo'),suite),h,w)
        try:
            for t in [10,30,50]:
                if t+1>=len(g['states']):continue
                observations=[restore_demo(env,g,j) for j in [t,t+1]]
                row={'suite':suite,'file':str(path),'t':t,'state_errors':[], 'images':[]}
                target=proprio_from_dataset(g['obs'],t)
                for obs in observations:
                    row['state_errors'].append(float(np.abs(proprio_from_env_obs(obs)-target).mean()))
                    row['images'].append({ds:detect_flip(obs[live],g['obs'][ds][t]) for ds,live in CAMERAS})
                rows.append(row);print(json.dumps(row),flush=True)
        finally:env.close()
Path('results/full_run/alignment.json').write_text(json.dumps(rows,indent=2))
