"""Evaluate SmolVLA with its saved preprocessing on standard LIBERO or LIBERO Plus."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .model import prepare,unnormalize_action
from .sim_worker import read_message,write_message
from training.common import apply_flip

ROOT=Path(__file__).resolve().parents[1]
HORIZONS={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True,help='step_N folder containing policy/ and manifest.json')
    p.add_argument('--benchmark',choices=['standard','plus'],default='standard')
    p.add_argument('--suite',default='libero_goal',choices=list(HORIZONS))
    p.add_argument('--episodes',type=int,default=None)
    p.add_argument('--task-start',type=int,default=0)
    p.add_argument('--task-limit',type=int,default=0,help='0 evaluates the full suite')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--action-steps',type=int,default=5)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists; use a fresh evaluation directory')
    episodes=a.episodes if a.episodes is not None else (1 if a.benchmark=='plus' else 50)
    if episodes<1 or a.action_steps<1 or a.task_start<0 or a.task_limit<0:p.error('Invalid evaluation counts')
    if a.benchmark=='plus' and episodes!=1:p.error('LIBERO Plus uses one trial per variant')
    manifest=json.loads((a.checkpoint/'manifest.json').read_text())
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig
    cfg=PreTrainedConfig.from_pretrained(a.checkpoint/'policy')
    cfg.device='cuda';cfg.load_vlm_weights=False
    if a.action_steps>cfg.chunk_size:p.error('Action steps exceed chunk length')
    cfg.n_action_steps=a.action_steps
    model=SmolVLAPolicy.from_pretrained(a.checkpoint/'policy',config=cfg,strict=True)
    model.eval();torch.set_num_threads(2)
    np.random.seed(a.seed);torch.manual_seed(a.seed)
    simulator=ROOT/'robo_env/bin/python' if a.benchmark=='standard' else ROOT.parent/'robo_plus/bin/python'
    repo=ROOT.parent/'external/LIBERO' if a.benchmark=='standard' else ROOT.parent/'LIBERO-plus'
    config=ROOT/'configs'/('libero_'+a.benchmark)
    a.out.mkdir(parents=True)
    metadata={'checkpoint':str(a.checkpoint.resolve()),'benchmark':a.benchmark,'suite':a.suite,
        'episodes_per_task':episodes,'seed':a.seed,'action_steps':a.action_steps,
        'checkpoint_sha256':hashlib.sha256((a.checkpoint/'policy/model.safetensors').read_bytes()).hexdigest(),
        'benchmark_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'task_start':a.task_start,'task_limit':a.task_limit,'results':[]}
    classes={}
    if a.benchmark=='plus':
        classification=json.loads((repo/'libero/libero/benchmark/task_classification.json').read_text())
        classes={x['name']:x for x in classification[a.suite]}
    def persist():
        rows=metadata['results'];metadata['completed_episodes']=len(rows)
        metadata['success_rate']=sum(x['success'] for x in rows)/len(rows) if rows else None
        groups=defaultdict(list)
        for row in rows:groups[row.get('category','standard')].append(int(row['success']))
        metadata['per_category']={k:{'successes':sum(v),'episodes':len(v),'rate':sum(v)/len(v)} for k,v in groups.items()}
        tmp=a.out/'results.tmp';tmp.write_text(json.dumps(metadata,indent=2));tmp.replace(a.out/'results.json')
    with (a.out/'simulator.log').open('w') as log:
        proc=subprocess.Popen([str(simulator),'-m','smolvla_pipeline.sim_worker','--repo',str(repo),
            '--config',str(config),'--suite',a.suite],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
            env={**os.environ,'MUJOCO_GL':'egl','OMP_NUM_THREADS':'2'})
        def receive():
            msg=read_message(proc.stdout)
            if msg is None:raise RuntimeError('Simulator exited; inspect simulator.log')
            if not msg['ok']:raise RuntimeError(msg['error'])
            return msg
        def rpc(message):write_message(proc.stdin,message);return receive()
        try:
            tasks=receive()['tasks'];stop=min(len(tasks),a.task_start+a.task_limit) if a.task_limit else len(tasks)
            if a.task_start>=stop:raise ValueError('No selected tasks')
            metadata['selected_tasks']=stop-a.task_start;metadata['suite_total_tasks']=len(tasks)
            metadata['complete']=False;persist()
            for ti in range(a.task_start,stop):
                for ep in range(episodes):
                    begin=time.monotonic();obs=rpc({'cmd':'reset','task_index':ti,'episode':ep,'seed':a.seed+ep})
                    model.reset();success=obs['success'];steps=0
                    with torch.inference_mode():
                        while not success and steps<HORIZONS[a.suite]:
                            images=np.stack([apply_flip(image,manifest['image_flips'][key]).transpose(2,0,1)
                                for image,key in zip(obs['images'],['agentview_rgb','eye_in_hand_rgb'])])
                            raw={'images':torch.from_numpy(images)[None],
                                 'state':torch.from_numpy(np.asarray(obs['state'],dtype=np.float32))[None],
                                 'task':[tasks[ti]['language']]}
                            batch=prepare(model,raw,manifest['stats'],'cuda')
                            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):action=model.select_action(batch)
                            action=unnormalize_action(action[0],manifest['stats'])
                            action=np.clip(action,-1,1);action[6]=1.0 if action[6]>0 else -1.0
                            obs=rpc({'cmd':'step','action':action.tolist()});steps+=1;success=obs['success']
                            if obs['done']:break
                    row={'task_index':ti,'task':tasks[ti]['name'],'episode':ep,'success':bool(success),
                         'steps':steps,'seconds':time.monotonic()-begin}
                    if a.benchmark=='plus':
                        classification=classes.get(tasks[ti]['name'],{})
                        row.update(category=classification.get('category','unclassified'),difficulty=classification.get('difficulty_level'))
                    metadata['results'].append(row);persist();print(json.dumps(row),flush=True)
            metadata['complete']=True;persist()
        finally:
            try:write_message(proc.stdin,{'cmd':'close'});proc.wait(timeout=20)
            except Exception:proc.terminate();proc.wait(timeout=10)
    print(json.dumps({'success_rate':metadata['success_rate'],'episodes':metadata['completed_episodes']}),flush=True)


if __name__=='__main__':main()
