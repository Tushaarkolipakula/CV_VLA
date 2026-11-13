"""Run contact-recovery generation across suites with per-task progress records."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/home/tushaar/data/LIBERO-datasets'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suites',nargs='+',default=['libero_spatial','libero_object','libero_goal','libero_10'])
    p.add_argument('--targets',type=Path,help='Optional JSON task-name to target-object overrides')
    p.add_argument('--levels',type=int,default=5)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--demos',type=int,default=0)
    p.add_argument('--pos-max',type=float,default=.03)
    p.add_argument('--rot-max',type=float,default=.15)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--min-free-gb',type=float,default=20)
    a=p.parse_args()
    if not 1 <= a.workers <= 4:p.error('--workers must be between 1 and 4')
    from augmentation.generate_curriculum_dataset import find_bddl
    overrides=json.loads(a.targets.read_text()) if a.targets else {}
    plan=[]
    for suite in a.suites:
        files=sorted((a.source/suite).glob('*.hdf5'))
        if len(files)!=10:raise ValueError(f'{suite}: expected 10 task files, found {len(files)}')
        for path in files:
            task=path.stem.removesuffix('_demo')
            match=re.search(r'\(:obj_of_interest\s+([^)]*)\)',Path(find_bddl(task,suite)).read_text(),re.S)
            if not match:raise ValueError(f'No object-of-interest declaration for {task}')
            target=overrides.get(task,match.group(1).split()[0])
            plan.append({'source':str(path),'suite':suite,'task':task,'target_object':target,
                         'target_rule':'explicit_override' if task in overrides else 'first_declared_object_of_interest'})
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(plan,indent=2),flush=True)
    if a.dry_run:return
    def run(job):
        folder=a.output/job['suite'];folder.mkdir(exist_ok=True)
        dest=folder/Path(job['source']).name
        if dest.exists():
            with h5py.File(dest,'r') as f:
                if not f['data'].attrs.get('generation_complete',False):raise ValueError(f'Incomplete output {dest}')
                manifest=json.loads(f['data'].attrs['generation_manifest'])
                for key in ['levels','repeats','demos','pos_max','rot_max','seed']:
                    if manifest['args'][key]!=getattr(a,key):raise ValueError(f'Existing dataset settings differ: {dest}/{key}')
                if manifest['args']['target_object']!=job['target_object']:raise ValueError(f'Existing target differs: {dest}')
            return {**job,'status':'existing_complete'}
        else:
            if shutil.disk_usage(a.output).free < a.min_free_gb*1024**3:
                return {**job,'status':'failed','error':'Insufficient free space; generation not started'}
            cmd=[sys.executable,'-u','-m','smolvla_pipeline.generate','--task-file',job['source'],
                '--target-object',job['target_object'],'--output-dir',str(folder),
                '--levels',str(a.levels),'--repeats',str(a.repeats),'--demos',str(a.demos),
                '--pos-max',str(a.pos_max),'--rot-max',str(a.rot_max),'--seed',str(a.seed)]
            log=folder/(job['task']+'.log')
            with log.open('a') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
            return {**job,'status':'complete' if result.returncode==0 else 'failed',
                    'returncode':result.returncode,'log':str(log)}
    records=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run,job):job for job in plan}
        for future in as_completed(futures):
            try:records.append(future.result())
            except Exception as error:records.append({**futures[future],'status':'failed','error':str(error)})
            temporary=a.output/'status.tmp';temporary.write_text(json.dumps(records,indent=2));temporary.replace(a.output/'status.json')
            print(json.dumps(records[-1]),flush=True)
    if any(r['status']=='failed' for r in records):raise SystemExit(1)


if __name__=='__main__':main()
