"""Private pipe-connected simulator worker; keeps benchmark dependencies isolated."""
import argparse
import os
from pathlib import Path
import pickle
import struct
import sys
import traceback

import numpy as np


def read_message(stream):
    size=stream.read(8)
    if not size:return None
    if len(size)!=8:raise EOFError('Truncated simulator message header')
    remaining=struct.unpack('<Q',size)[0];parts=[]
    while remaining:
        data=stream.read(remaining)
        if not data:raise EOFError('Truncated simulator message')
        parts.append(data);remaining-=len(data)
    # Messages are exchanged only with our own directly spawned local process.
    return pickle.loads(b''.join(parts))


def write_message(stream,value):
    data=pickle.dumps(value,protocol=4)
    stream.write(struct.pack('<Q',len(data)));stream.write(data);stream.flush()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--suite',required=True)
    a=p.parse_args()
    output=sys.stdout.buffer;sys.stdout=sys.stderr
    os.environ['LIBERO_CONFIG_PATH']=str(Path(a.config).resolve())
    os.environ.setdefault('MUJOCO_GL','egl')
    sys.path.insert(0,str(Path(a.repo).resolve()))
    env=None
    try:
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.bddl_utils import get_problem_info
        from training.common import proprio_from_env_obs
        suite=benchmark.get_benchmark_dict()[a.suite]()
        tasks=[suite.get_task(i) for i in range(suite.n_tasks)]
        def bddl_path(task):
            path=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
            if not path.exists() and '_view_' in path.name:
                path=path.with_name(path.name.split('_view_')[0]+'.bddl')
            if not path.is_file():raise FileNotFoundError(path)
            return path
        # Task filenames contain perturbation suffixes. Read the actual command.
        write_message(output,{'ok':True,'tasks':[{'name':t.name,'language':get_problem_info(str(bddl_path(t)))['language_instruction']} for t in tasks]})
        def pack(obs):
            return {'images':[obs['agentview_image'],obs['robot0_eye_in_hand_image']],
                    'state':proprio_from_env_obs(obs),'success':bool(env.check_success())}
        while True:
            request=read_message(sys.stdin.buffer)
            if request is None or request['cmd']=='close':break
            try:
                if request['cmd']=='reset':
                    if env is not None:env.close()
                    ti=request['task_index'];task=tasks[ti]
                    bddl=str(bddl_path(task))
                    env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=128,camera_widths=128,
                                           camera_names=['agentview','robot0_eye_in_hand'])
                    env.seed(request['seed']);env.reset()
                    states=suite.get_task_init_states(ti)
                    state_index=request['episode']
                    if state_index>=len(states):raise ValueError('Requested initial state index exceeds available states')
                    obs=env.set_init_state(states[state_index])
                    for _ in range(10):obs,_,_,_=env.step(np.array([0,0,0,0,0,0,-1.0]))
                    write_message(output,{'ok':True,**pack(obs)})
                elif request['cmd']=='step':
                    obs,_,done,_=env.step(np.asarray(request['action'],dtype=np.float32))
                    write_message(output,{'ok':True,'done':bool(done),**pack(obs)})
                else:raise ValueError('Unknown worker command')
            except Exception:
                write_message(output,{'ok':False,'error':traceback.format_exc()})
    except Exception:
        write_message(output,{'ok':False,'error':traceback.format_exc()})
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
