"""Paired PI05 adapter evaluation, restricted to robot initial states on Plus."""
import argparse,hashlib,json,os,subprocess,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from .model import load_base,prepare,decode
from .sim_worker import read_message,write_message
from training.common import apply_flip

ROOT=Path(__file__).resolve().parents[1]
HORIZONS={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--benchmark',choices=['standard','plus'],default='plus')
    p.add_argument('--suite',choices=list(HORIZONS),required=True)
    p.add_argument('--episodes',type=int);p.add_argument('--task-limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=10000);p.add_argument('--action-steps',type=int,default=5)
    a=p.parse_args();episodes=a.episodes or (1 if a.benchmark=='plus' else 50)
    if a.out.exists():p.error('Use a fresh evaluation output')
    if episodes<1 or not 1<=a.action_steps<=50 or a.task_limit<0:p.error('Invalid evaluation counts')
    if a.benchmark=='plus' and episodes!=1:p.error('Plus requires one episode per official variant')
    manifest=json.loads((a.checkpoint.parent/'manifest.json').read_text())
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
    model=load_base(manifest['args']['base'],rank=manifest['args']['rank'])
    adapter=load_file(str(a.checkpoint/'adapter.safetensors'))
    expected={n for n,v in model.named_parameters() if v.requires_grad}
    if set(adapter)!=expected:raise ValueError('Adapter tensor names do not match the model')
    model.load_state_dict(adapter,strict=False);model.eval();model.config.n_action_steps=a.action_steps
    # Queue length is fixed at construction; rebuild it after changing n_action_steps.
    model.reset()
    tokenizer=AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224',padding_side='right')
    simulator=ROOT/'robo_env/bin/python' if a.benchmark=='standard' else ROOT.parent/'robo_plus/bin/python'
    repo=ROOT.parent/'external/LIBERO' if a.benchmark=='standard' else ROOT.parent/'LIBERO-plus'
    classes={}
    if a.benchmark=='plus':
        classification=json.loads((repo/'libero/libero/benchmark/task_classification.json').read_text())
        classes={x['name']:x for x in classification[a.suite]}
    a.out.mkdir(parents=True)
    metadata={'checkpoint':str(a.checkpoint.resolve()),'benchmark':a.benchmark,'suite':a.suite,
        'category_filter':'Robot Initial States' if a.benchmark=='plus' else 'standard',
        'seed':a.seed,'episodes_per_task':episodes,'action_steps':a.action_steps,'horizon':HORIZONS[a.suite],
        'adapter_sha256':hashlib.sha256((a.checkpoint/'adapter.safetensors').read_bytes()).hexdigest(),
        'benchmark_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'code_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['evaluate.py','model.py','sim_worker.py']},
        'instruction_source':'BDDL language_instruction','complete':False,'results':[]}
    def persist():
        rows=metadata['results'];n=len(rows);s=sum(r['success'] for r in rows)
        metadata.update(completed_episodes=n,successes=s,success_rate=s/n if n else None)
        groups=defaultdict(list)
        for r in rows:groups[str(r.get('difficulty','standard'))].append(int(r['success']))
        metadata['per_difficulty']={k:{'successes':sum(v),'episodes':len(v),'rate':sum(v)/len(v)} for k,v in groups.items()}
        temp=a.out/'results.tmp';temp.write_text(json.dumps(metadata,indent=2));temp.replace(a.out/'results.json')
    with (a.out/'simulator.log').open('w') as log:
        proc=subprocess.Popen([str(simulator),'-m','pi05_pipeline.sim_worker','--repo',str(repo),
            '--config',str(ROOT/'configs'/('libero_'+a.benchmark)),'--suite',a.suite],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env={**os.environ,'MUJOCO_GL':'egl','OMP_NUM_THREADS':'2'})
        def receive():
            msg=read_message(proc.stdout)
            if msg is None:raise RuntimeError('Simulator exited; inspect simulator.log')
            if not msg['ok']:raise RuntimeError(msg['error'])
            return msg
        def rpc(msg):write_message(proc.stdin,msg);return receive()
        try:
            tasks=receive()['tasks']
            selected=[i for i,t in enumerate(tasks) if a.benchmark=='standard' or classes.get(t['name'],{}).get('category')=='Robot Initial States']
            if not selected:raise ValueError('No tasks match the official category filter')
            if a.benchmark=='plus':
                expected={name for name,c in classes.items() if c['category']=='Robot Initial States'}
                if {tasks[i]['name'] for i in selected}!=expected:raise ValueError('Official category/task inventory mismatch')
            if a.task_limit:selected=selected[:a.task_limit]
            metadata['selected_tasks']=[tasks[i]['name'] for i in selected];persist()
            for ti in selected:
                for ep in range(episodes):
                    tick=time.monotonic();episode_seed=a.seed+ti*1000+ep
                    torch.manual_seed(episode_seed);obs=rpc({'cmd':'reset','task_index':ti,'episode':ep,'seed':episode_seed})
                    model.reset();success=obs['success'];steps=0
                    with torch.inference_mode():
                        while not success and steps<HORIZONS[a.suite]:
                            images=np.stack([apply_flip(image,manifest['image_flips'][key]).transpose(2,0,1)
                                for image,key in zip(obs['images'],['agentview_rgb','eye_in_hand_rgb'])])
                            raw={'images':torch.from_numpy(images)[None],
                                 'state':torch.from_numpy(np.asarray(obs['state'],dtype=np.float32))[None],
                                 'task':[tasks[ti]['language']]}
                            with torch.autocast('cuda',dtype=torch.bfloat16):action=model.select_action(prepare(raw,manifest['stats'],tokenizer,'cuda'))
                            action=decode(action[0],manifest['stats'])[:7]
                            if not np.isfinite(action).all():raise FloatingPointError('Nonfinite policy action')
                            action=np.clip(action,-1,1);action[6]=1. if action[6]>0 else -1.
                            obs=rpc({'cmd':'step','action':action.tolist()});steps+=1;success=obs['success']
                            if obs['done']:break
                    row={'task_index':ti,'task':tasks[ti]['name'],'language':tasks[ti]['language'],'episode':ep,
                         'seed':episode_seed,'success':bool(success),'steps':steps,'seconds':time.monotonic()-tick}
                    if a.benchmark=='plus':row['difficulty']=classes[tasks[ti]['name']]['difficulty_level']
                    metadata['results'].append(row);persist();print(json.dumps(row),flush=True)
            metadata['complete']=True;persist()
        finally:
            try:write_message(proc.stdin,{'cmd':'close'});proc.wait(timeout=20)
            except Exception:proc.terminate();proc.wait(timeout=10)
    print(json.dumps({'success_rate':metadata['success_rate'],'episodes':metadata['completed_episodes']}),flush=True)

if __name__=='__main__':main()
