"""Generate contact-triggered physical perturbation recovery + successful continuation.

Run using robo_env (the existing simulator environment), not the model environment.
The original expert model XML is restored. Injected disturbance actions are never
training targets. A sample is accepted only if recovery AND task completion succeed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from augmentation.generate_curriculum_dataset import make_env,find_bddl,servo_action,rot_error,random_unit
from training.common import CAMERAS,apply_flip,detect_flip


def restore_demo(env,group,state_index):
    from libero.libero import get_libero_path
    from libero.libero.utils.utils import postprocess_model_xml
    env.reset()
    xml=postprocess_model_xml(group.attrs['model_file'],{})
    root=ET.fromstring(xml)
    for element in root.iter():
        path=element.get('file')
        if path and '/assets/' in path and 'robosuite' not in path:
            element.set('file',str(Path(get_libero_path('assets'))/path.split('/assets/',1)[1]))
    env.reset_from_xml_string(ET.tostring(root,encoding='unicode'))
    env.sim.reset()
    return env.set_init_state(group['states'][state_index])


def contact_groups(env,target):
    objects={**env.env.objects_dict,**env.env.fixtures_dict}
    if target not in objects:raise ValueError(f'Unknown target {target}; available: {list(objects)}')
    targets={env.sim.model.geom_name2id(n) for n in objects[target].contact_geoms}
    names=env.robots[0].gripper.important_geoms
    fingers={env.sim.model.geom_name2id(n) for key in ['left_finger','right_finger'] for n in names[key]}
    return fingers,targets


def contact_pairs(env,fingers,targets):
    pairs=[]
    for c in env.sim.data.contact[:env.sim.data.ncon]:
        if c.dist>0:continue
        if (c.geom1 in fingers and c.geom2 in targets) or (c.geom2 in fingers and c.geom1 in targets):
            pairs.append([env.sim.model.geom_id2name(c.geom1),env.sim.model.geom_id2name(c.geom2)])
    return pairs


def first_contact(env,group,target):
    restore_demo(env,group,0)
    fingers,targets=contact_groups(env,target)
    for t in range(1,len(group['states'])):
        env.set_init_state(group['states'][t])
        pairs=contact_pairs(env,fingers,targets)
        if pairs:return t,pairs
    return None,[]


def record(obs,action,frames,flips):
    import robosuite.utils.transform_utils as T
    for ds,live in CAMERAS:frames[ds].append(apply_flip(obs[live],flips[ds]))
    frames['gripper_states'].append(obs['robot0_gripper_qpos'].copy())
    frames['ee_pos'].append(obs['robot0_eef_pos'].copy())
    frames['ee_ori'].append(T.quat2axisangle(obs['robot0_eef_quat']))
    frames['actions'].append(np.asarray(action,dtype=np.float32).copy())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-file',type=Path,required=True)
    p.add_argument('--target-object',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--demos',type=int,default=0,help='0 means all source demos')
    p.add_argument('--levels',type=int,default=5)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--pos-max',type=float,default=.03,help='Maximum commanded displacement in metres')
    p.add_argument('--rot-max',type=float,default=.15,help='Maximum commanded rotation in radians')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--trigger-only',action='store_true')
    a=p.parse_args()
    if a.levels<1 or a.repeats<1 or a.pos_max<=0 or a.rot_max<0:p.error('Invalid perturbation settings')
    src=a.task_file.expanduser().resolve();name=src.stem.removesuffix('_demo');out=a.output_dir.expanduser().resolve()
    if out==src.parent:p.error('Output must differ from source')
    out.mkdir(parents=True,exist_ok=True)
    dest=out/src.name
    if dest.exists():p.error('Output exists; use a new version directory')
    report_path=out/(name+'_generation.json')
    if report_path.exists():p.error('Generation report exists; use a new version directory')
    rng=np.random.default_rng(a.seed)
    report={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'events':[],'accepted':0}
    temp=None
    import robosuite.utils.transform_utils as T
    with h5py.File(src,'r') as source:
        keys=sorted(source['data'],key=lambda k:int(k.split('_')[-1]))
        if a.demos:keys=keys[:a.demos]
        h,w=source['data'][keys[0]]['obs/agentview_rgb'].shape[1:3]
        env=make_env(find_bddl(name,src.parent.name),h,w)
        try:
            output=None;data=None
            if not a.trigger_only:
                fd,temp=tempfile.mkstemp(prefix='.'+name+'-',suffix='.partial',dir=out);os.close(fd)
                output=h5py.File(temp,'w');data=output.create_group('data')
                for key,value in source['data'].attrs.items():data.attrs[key]=value
                data.attrs['task_name']=name;data.attrs['num_levels']=a.levels
            try:
                for key in keys:
                    group=source['data'][key];t,pairs=first_contact(env,group,a.target_object)
                    event={'source_demo':key,'contact_step':t,'contact_geoms':pairs}
                    report['events'].append(event);print(json.dumps(event),flush=True)
                    if t is None or a.trigger_only:continue
                    # Raw source obs[t-1] is the post-action observation for state[t].
                    obs=restore_demo(env,group,t)
                    flips={ds:detect_flip(obs[live],group['obs'][ds][t-1])[0] for ds,live in CAMERAS}
                    if 'image_flips' in data.attrs and json.loads(data.attrs['image_flips'])!=flips:
                        raise ValueError('Inconsistent image orientation across source demos')
                    data.attrs['image_flips']=json.dumps(flips)
                    for level in range(1,a.levels+1):
                        for repeat in range(a.repeats):
                            obs=restore_demo(env,group,t)
                            home_pos=obs['robot0_eef_pos'].copy();home_quat=obs['robot0_eef_quat'].copy()
                            grip=float(group['actions'][t,6]);magnitude=a.pos_max*level/a.levels
                            rotation=a.rot_max*level/a.levels
                            direction=random_unit(rng,3)
                            target_pos=home_pos+direction*magnitude
                            target_quat=T.mat2quat(T.quat2mat(T.axisangle2quat(random_unit(rng,3)*rotation))@T.quat2mat(home_quat))
                            for _ in range(5):
                                action,_,_=servo_action(obs,target_pos,target_quat,grip)
                                obs,_,_,_=env.step(action)
                            achieved_pos=float(np.linalg.norm(obs['robot0_eef_pos']-home_pos))
                            achieved_rot=float(np.linalg.norm(rot_error(home_quat,obs['robot0_eef_quat'])))
                            frames={k:[] for k in ['agentview_rgb','eye_in_hand_rgb','gripper_states','ee_pos','ee_ori','actions']}
                            recovered=False;success=False
                            for _ in range(80):
                                action,dp,dr=servo_action(obs,home_pos,home_quat,grip)
                                if dp<.005 and dr<.04:recovered=True;break
                                record(obs,action,frames,flips);obs,_,_,_=env.step(action)
                            recovery_steps=len(frames['actions'])
                            # Original actions follow the original pre-action state[t].
                            # Verify their continuation; never accept pose-only success.
                            if recovered and recovery_steps and achieved_pos>=.001:
                                for action in group['actions'][t:]:
                                    record(obs,action,frames,flips);obs,_,_,_=env.step(action)
                                    if env.check_success():success=True;break
                            result={'source_demo':key,'level':level,'repeat':repeat,'recovered':recovered,
                                    'task_success':success,'recovery_steps':recovery_steps,
                                    'achieved_pos_m':achieved_pos,'achieved_rot_rad':achieved_rot}
                            report['events'].append(result);print(json.dumps(result),flush=True)
                            if not success:continue
                            g=data.create_group(f"recovery_{report['accepted']}");o=g.create_group('obs')
                            for field,values in frames.items():
                                target=g if field=='actions' else o
                                target.create_dataset(field,data=np.asarray(values),compression='gzip',compression_opts=1)
                            g.attrs.update(kind='recovery',noise_level=level,source_demo=key,
                                trigger_type='contact',target_object=a.target_object,contact_step=t,
                                contact_geoms=json.dumps(pairs),task_success=True,recovery_steps=recovery_steps,
                                pos_mag_m=achieved_pos,rot_mag_rad=achieved_rot,num_samples=len(frames['actions']))
                            report['accepted']+=1
                if data is not None:
                    data.attrs['generation_complete']=True;data.attrs['generation_manifest']=json.dumps(report)
            finally:
                if output is not None:output.close()
            if temp and report['accepted']:
                os.link(temp,dest);os.unlink(temp)
            elif temp:
                # Keep failed diagnostic output, without an .hdf5 extension so it
                # cannot silently enter training as an empty/invalid dataset.
                print(f'No accepted samples; diagnostic staging retained: {temp}',flush=True)
        finally:env.close()
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps({'accepted':report['accepted'],'report':str(report_path)}),flush=True)
    if not a.trigger_only and report['accepted']==0:raise SystemExit(2)


if __name__=='__main__':main()
