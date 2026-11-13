"""Durable, single-GPU generation -> training -> evaluation experiment runner.

Run from the remote project with .venv-smolvla/bin/python. This does not submit
jobs elsewhere. A failed stage stops the sequence and records the error.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
MODES=['clean','fixed','curriculum']


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def interval(successes,n):
    if not n:return None
    p=successes/n;z=1.96;d=1+z*z/n
    mid=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,mid-half),min(1,mid+half)]


def aggregate(folder):
    groups=defaultdict(list);suites=defaultdict(dict);categories=defaultdict(list)
    for p in folder.glob('eval/*/*/*/block_*/results.json'):
        r=json.loads(p.read_text())
        if not r.get('complete'):continue
        mode,benchmark,suite=p.parts[-5:-2]
        groups[(mode,benchmark)].extend(r['results'])
        suites[(mode,benchmark)].setdefault(suite,[]).extend(r['results'])
        for row in r['results']:categories[(mode,benchmark,row.get('category','standard'))].append(row)
    def stats(rows):
        n=len(rows);s=sum(int(r['success']) for r in rows)
        return {'episodes':n,'successes':s,'success_rate':s/n if n else None,'wilson_95_percent':interval(s,n)}
    report={}
    for (mode,benchmark),rows in groups.items():
        report.setdefault(mode,{})[benchmark]={**stats(rows),
            'suites':{k:stats(v) for k,v in suites[(mode,benchmark)].items()},
            'categories':{k[2]:stats(v) for k,v in categories.items() if k[:2]==(mode,benchmark)}}
    atomic(folder/'comparison.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if not a.name.replace('_','').replace('-','').isalnum():p.error('Use a simple run name')
    folder=ROOT/'results'/a.name;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env={**os.environ,'MUJOCO_GL':'egl','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
         'LIBERO_CONFIG_PATH':str(ROOT/'configs/libero_standard')}
    model=str(ROOT/'.venv-smolvla/bin/python');sim=str(ROOT/'robo_env/bin/python')
    recovery=ROOT/'data'/('contact_'+a.name)
    env['RECOVERY_ROOT']=str(recovery)
    checkpoint={mode:ROOT/'checkpoints'/f'smolvla_{mode}_{a.name}' for mode in MODES}
    plan={'run':a.name,'steps_per_mode':a.steps,'seed':0,'effective_batch':16,
          'modes':MODES,'suites':SUITES,'generation':{'levels':5,'repeats':1,'demos_per_task':50,
          'max_position_m':.03,'max_rotation_rad':.15,'workers':a.workers},
          'checkpoint_selection':'minimum fixed held-out clean flow loss; never LIBERO+ scores',
          'standard_trials_per_task':50,'plus_trials_per_variant':1,
          'recovery_root':str(recovery),'checkpoints':{k:str(v) for k,v in checkpoint.items()},
          'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(__file__).parent.glob('*.py'))}}
    plan_path=folder/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Experiment plan changed; use a new name')
    atomic(plan_path,plan)
    if a.plan_only:print(json.dumps(plan,indent=2));return
    state_path=folder/'status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'completed_stages':[]}
    state.update(pid=os.getpid(),status='running',started_at=time.time());atomic(state_path,state)
    def stage(name,command):
        if name in state['completed_stages']:return
        if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Less than 8 GiB free; stopped before next stage')
        state.update(stage=name,command=command,updated_at=time.time());atomic(state_path,state)
        print(json.dumps({'stage':name,'event':'start'}),flush=True)
        with (folder/(name+'.log')).open('a') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed with exit {result.returncode}; inspect its log')
        state['completed_stages'].append(name);state['updated_at']=time.time();atomic(state_path,state)
    try:
        stage('generate',[sim,'-u','-m','smolvla_pipeline.generate_all','--output',str(recovery),
            '--targets',str(ROOT/'configs/contact_targets.json'),'--levels','5','--repeats','1',
            '--workers',str(a.workers)])
        for mode in MODES:
            # Launcher refuses accidental overwrites. Explicit repair is needed
            # for an interrupted training stage, using the saved latest checkpoint.
            stage('train_'+mode,['bash','experiments/train_smolvla.sh',mode,a.name,
                '--steps',str(a.steps),'--keep-last','1','--validation-batches','32'])
        chosen={}
        for mode,path in checkpoint.items():
            step=json.loads((path/'best_validation.json').read_text())['step']
            chosen[mode]=path/f'step_{step:06d}'
        atomic(folder/'selected_checkpoints.json',{k:str(v) for k,v in chosen.items()})
        classification=json.loads((ROOT.parent/'LIBERO-plus/libero/libero/benchmark/task_classification.json').read_text())
        for benchmark in ['standard','plus']:
            for mode in MODES:
                for suite in SUITES:
                    count=10 if benchmark=='standard' else len(classification[suite])
                    block=1 if benchmark=='standard' else 50
                    for start in range(0,count,block):
                        name=f'eval_{benchmark}_{mode}_{suite}_{start:05d}'
                        out=folder/'eval'/mode/benchmark/suite/f'block_{start:05d}'
                        existing=out/'results.json'
                        if existing.exists() and json.loads(existing.read_text()).get('complete'):
                            if name not in state['completed_stages']:state['completed_stages'].append(name)
                            continue
                        stage(name,[model,'-u','-m','smolvla_pipeline.evaluate','--checkpoint',str(chosen[mode]),
                            '--benchmark',benchmark,'--suite',suite,'--episodes','50' if benchmark=='standard' else '1',
                            '--task-start',str(start),'--task-limit',str(min(block,count-start)),
                            '--out',str(out),'--seed','10000'])
                        aggregate(folder)
        aggregate(folder);state.update(status='complete',finished_at=time.time());atomic(state_path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error),updated_at=time.time());atomic(state_path,state)
        raise


if __name__=='__main__':main()
