"""Read existing HDF5 directly, preserving language and episode action chunks."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch


CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")


def state_array(group, index):
    return np.concatenate([group[k][index] for k in ("gripper_states", "ee_pos", "ee_ori")], axis=-1).astype(np.float32)


class ChunkData:
    def __init__(self, roots, chunk_size=50, seed=0, val_fraction=0.1, clean_offset=1):
        if clean_offset not in (0, 1) or not 0 < val_fraction < 1:
            raise ValueError("Use clean_offset 0/1 and a validation fraction strictly between 0 and 1")
        self.chunk_size, self.clean_offset = chunk_size, clean_offset
        self.handles, self.demos, self.files = {}, [], []
        self.flips = None
        for root in roots:
            for path in sorted(Path(root).expanduser().glob("*.hdf5")):
                with h5py.File(path, "r") as handle:
                    data = handle["data"]
                    task = str(data.attrs.get("task_name", path.stem.removesuffix("_demo")))
                    info = json.loads(data.attrs.get("problem_info", "{}"))
                    language = info.get("language_instruction", task.replace("_", " "))
                    flips = json.loads(data.attrs.get("image_flips", '{"agentview_rgb":"none","eye_in_hand_rgb":"none"}'))
                    if self.flips is not None and self.flips != flips:
                        raise ValueError("Mixed image conventions in input files")
                    self.flips = flips
                    self.files.append({"path":str(path.resolve()),"bytes":path.stat().st_size,
                                       "mtime_ns":path.stat().st_mtime_ns})
                    for name, g in data.items():
                        kind = str(g.attrs.get("kind", "clean"))
                        if kind not in ("clean", "recovery"):
                            raise ValueError(f"Unknown sample kind: {kind}")
                        offset = clean_offset if kind == "clean" else 0
                        length = len(g["actions"]) - offset
                        if length < 1:
                            continue
                        # Only task-success-verified initial-state recoveries.
                        if kind == "recovery" and not (g.attrs.get("task_success", False) and
                                                        g.attrs.get("trigger_type", "") == "initial_joint_state"):
                            continue
                        self.demos.append(dict(path=str(path.resolve()),key=name,kind=kind,
                            level=int(g.attrs.get("noise_level",0)),source=str(g.attrs.get("source_demo",name)),
                            task=task,language=language,length=length,offset=offset,
                            recovery_steps=int(g.attrs.get('recovery_steps',0))))
        if not self.demos:
            raise ValueError("No usable HDF5 demonstrations")
        clean = [(d['task'],d['source']) for d in self.demos if d['kind']=='clean']
        if len(set(clean)) != len(clean):
            raise ValueError("Duplicate clean source demos: supply originals once, plus recovery-only files")
        clean_sources = set(clean)
        if any((d['task'],d['source']) not in clean_sources for d in self.demos):
            raise ValueError("Recovery source has no matching clean demonstration")
        rng = np.random.default_rng(seed)
        val = set()
        for task in sorted({d['task'] for d in self.demos}):
            groups = sorted(s for t,s in clean_sources if t==task)
            if len(groups)<2:
                raise ValueError("At least two source demos per task are required")
            rng.shuffle(groups)
            n = min(len(groups)-1,max(1,round(len(groups)*val_fraction)))
            val.update((task,s) for s in groups[:n])
        self.train = [i for i,d in enumerate(self.demos) if (d['task'],d['source']) not in val]
        self.val = [i for i,d in enumerate(self.demos) if (d['task'],d['source']) in val]
        self.val_sources = [list(x) for x in sorted(val)]
        self.max_level = max(d['level'] for d in self.demos)

    def group(self, d):
        if d['path'] not in self.handles:
            self.handles[d['path']] = h5py.File(d['path'],'r')
        return self.handles[d['path']]['data'][d['key']]

    def stats(self):
        states, actions = [], []
        for i in self.train:
            d=self.demos[i]
            if d['kind']!='clean':continue
            g=self.group(d)
            states.append(state_array(g['obs'],slice(0,d['length'])))
            actions.append(g['actions'][d['offset']:d['offset']+d['length']].astype(np.float32))
        result={}
        for key,values in [('state',states),('action',actions)]:
            a=np.concatenate(values)
            if not np.isfinite(a).all(): raise ValueError(f'Non-finite {key}')
            result[key]={'q01':np.quantile(a,.01,axis=0).tolist(),'q99':np.quantile(a,.99,axis=0).tolist()}
        return result

    def sample(self, demo_id, t):
        d=self.demos[demo_id];g=self.group(d)
        if d['kind']=='recovery' and not 0<=t<d['recovery_steps']:
            raise ValueError('Recovery observations must be sampled inside the corrective window')
        start=t+d['offset'];stop=min(start+self.chunk_size,len(g['actions']))
        actions=np.zeros((self.chunk_size,7),dtype=np.float32)
        actions[:stop-start]=g['actions'][start:stop]
        valid=np.arange(self.chunk_size)<stop-start
        return {'images':np.stack([g['obs'][k][t].transpose(2,0,1) for k in CAMERAS]),
                'state':state_array(g['obs'],t),'action':actions,'valid':valid,'task':d['language']}

    def batch(self,rng,batch_size,max_level,recovery_ratio,validation=False):
        if not 0<=recovery_ratio<=1:raise ValueError('recovery_ratio must be in [0,1]')
        ids=self.val if validation else self.train
        pools={}
        for i in ids:
            d=self.demos[i]
            if d['kind']=='recovery' and d['level']>max_level:continue
            pools.setdefault(d['kind'],{}).setdefault(d['task'],[]).append(i)
        if recovery_ratio>0 and 'recovery' not in pools:
            raise ValueError('Curriculum requested but no verified initial-state recovery data is available at this level')
        items=[]
        for _ in range(batch_size):
            # Select the task first so tasks with more accepted recoveries do
            # not silently dominate training. Missing recovery coverage falls
            # back to clean data for that task and is visible in logged counts.
            tasks=sorted(pools['clean']);task=tasks[int(rng.integers(len(tasks)))]
            kind='recovery' if rng.random()<recovery_ratio and task in pools.get('recovery',{}) else 'clean'
            candidates=pools[kind][task]
            if kind=='recovery':
                # Equal level weight and then equal trajectory weight: acceptance
                # rate and trajectory length must not determine the noise mix.
                levels=sorted({self.demos[i]['level'] for i in candidates})
                level=int(rng.choice(levels));candidates=[i for i in candidates if self.demos[i]['level']==level]
                i=int(rng.choice(candidates));length=self.demos[i]['recovery_steps']
                if length<1:raise ValueError('Empty corrective window')
            else:
                lengths=np.array([self.demos[i]['length'] for i in candidates],dtype=float)
                i=int(rng.choice(candidates,p=lengths/lengths.sum()));length=self.demos[i]['length']
            item=self.sample(i,int(rng.integers(length)))
            item['is_recovery']=kind=='recovery'
            items.append(item)
        return {key:[x[key] for x in items] if key=='task' else torch.from_numpy(np.stack([x[key] for x in items]))
                for key in items[0]}

    def close(self):
        for h in self.handles.values():h.close()
        self.handles.clear()
