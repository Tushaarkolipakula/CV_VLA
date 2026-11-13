"""Independent initial-joint-state recovery data; never reads LIBERO+ test states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import h5py
import numpy as np
from smolvla_pipeline.generate import restore_demo, record
from augmentation.generate_curriculum_dataset import make_env, find_bddl, servo_action
from training.common import CAMERAS, detect_flip


def save_json(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def penetrations(sim):
    result={}
    for c in sim.data.contact[:sim.data.ncon]:
        pair=tuple(sorted((int(c.geom1),int(c.geom2))))
        result[pair]=min(result.get(pair,0.),float(c.dist))
    return result


def generate(source, output, demos=3, levels=5, repeats=1, seed=1729):
    source=Path(source).resolve();output=Path(output).resolve()
    if output==source.parent:raise ValueError('Output must differ from source')
    output.mkdir(parents=True,exist_ok=True)
    name=source.stem.removesuffix('_demo');dest=output/source.name
    report_path=output/(name+'_generation.json')
    if dest.exists() or report_path.exists():raise FileExistsError('Use a new dataset version')
    report={'source':str(source),'trigger':'initial_joint_state','seed':seed,'levels':levels,
            'repeats':repeats,'requested_demos':demos,'accepted':0,'events':[],
            'joint_distribution':'independent uniform [-0.1*level, +0.1*level] radians',
            'source_bytes':source.stat().st_size,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    seed_task=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
    rng=np.random.default_rng(np.random.SeedSequence([seed,seed_task]))
    env=None;partial=None
    try:
        with h5py.File(source,'r') as f:
            keys=sorted(f['data'],key=lambda k:int(k.split('_')[-1]))
            if demos:keys=keys[:demos]
            h,w=f['data'][keys[0]]['obs/agentview_rgb'].shape[1:3]
            env=make_env(find_bddl(name,source.parent.name),h,w)
            fd,partial=tempfile.mkstemp(prefix='.'+name,suffix='.partial',dir=output);os.close(fd)
            with h5py.File(partial,'w') as out:
                data=out.create_group('data');data.attrs.update(f['data'].attrs)
                data.attrs.update(task_name=name,num_levels=levels,dataset_version='initial_joint_v2_compact')
                for key in keys:
                    g=f['data'][key]
                    # Paired clean replay checks restoration and expert labels first.
                    obs=restore_demo(env,g,1)
                    flips={ds:detect_flip(obs[live],g['obs'][ds][0])[0] for ds,live in CAMERAS}
                    if 'image_flips' in data.attrs and json.loads(data.attrs['image_flips'])!=flips:raise ValueError('Mixed camera orientations')
                    data.attrs['image_flips']=json.dumps(flips)
                    obs=restore_demo(env,g,0);clean_ok=False
                    for action in g['actions']:
                        obs,_,done,_=env.step(action)
                        if env.check_success():clean_ok=True;break
                        if done:break
                    report['events'].append({'source_demo':key,'clean_replay_success':clean_ok})
                    if not clean_ok:
                        save_json(report_path,report);continue
                    for level in range(1,levels+1):
                        for repeat in range(repeats):
                            if shutil.disk_usage(output).free<10*1024**3:raise RuntimeError('Less than 10 GiB free')
                            obs=restore_demo(env,g,0)
                            home_pos=obs['robot0_eef_pos'].copy();home_quat=obs['robot0_eef_quat'].copy()
                            robot=env.robots[0];idx=np.asarray(robot._ref_joint_pos_indexes,dtype=int)
                            velidx=np.asarray(robot._ref_joint_vel_indexes,dtype=int)
                            q0=env.sim.data.qpos.copy();baseline_contacts=penetrations(env.sim)
                            delta=rng.uniform(-.1*level,.1*level,len(idx))
                            proposed=q0[idx]+delta
                            joint_ids=[env.sim.model.joint_name2id(j) for j in robot.robot_model.joints]
                            limits=np.asarray(env.sim.model.jnt_range[joint_ids]).T
                            if limits.shape!=(2,len(idx)):raise ValueError(f'Unexpected joint limits {limits.shape}')
                            event={'source_demo':key,'level':level,'repeat':repeat,'joint_delta_rad':delta.tolist(),'task_success':False}
                            if np.any(proposed<=limits[0]+.01) or np.any(proposed>=limits[1]-.01):
                                event['rejection']='joint_limits';report['events'].append(event);continue
                            env.sim.data.qpos[idx]=proposed;env.sim.data.qvel[velidx]=0;env.sim.forward()
                            obs=env.set_init_state(env.sim.get_state().flatten())
                            if any(depth<baseline_contacts.get(pair,0.)-.002 for pair,depth in penetrations(env.sim).items()):
                                event['rejection']='new_penetration_over_2mm';report['events'].append(event)
                                save_json(report_path,report);continue
                            # Non-robot generalized coordinates must not change at injection.
                            other=np.ones(len(q0),dtype=bool);other[idx]=False
                            if not np.allclose(env.sim.data.qpos[other],q0[other],atol=1e-7):raise RuntimeError('Injection changed object/gripper state')
                            event['eef_displacement_m']=float(np.linalg.norm(obs['robot0_eef_pos']-home_pos))
                            frames={k:[] for k in ['agentview_rgb','eye_in_hand_rgb','gripper_states','ee_pos','ee_ori','actions']}
                            grip=float(g['actions'][0,6]);recovered=False;success=False
                            for _ in range(100):
                                action,dp,dr=servo_action(obs,home_pos,home_quat,grip)
                                if dp<.005 and dr<.04:recovered=True;break
                                record(obs,action,frames,flips);obs,_,done,_=env.step(action)
                                if done:break
                            _,dp,dr=servo_action(obs,home_pos,home_quat,grip)
                            recovered=recovered or (dp<.005 and dr<.04)
                            recovery_steps=len(frames['actions'])
                            if recovered and recovery_steps:
                                for action in g['actions']:
                                    record(obs,action,frames,flips);obs,_,done,_=env.step(action)
                                    if env.check_success():success=True;break
                                    if done:break
                            event.update(recovered=recovered,recovery_steps=recovery_steps,task_success=success)
                            report['events'].append(event)
                            if success:
                                rec=data.create_group('recovery_'+str(report['accepted']));o=rec.create_group('obs')
                                for field,values in frames.items():
                                    # Train only at corrective frames. Keep 49 continuation
                                    # frames so every 50-action target has its real future.
                                    array=np.asarray(values[:recovery_steps+49],dtype=np.uint8 if field in ['agentview_rgb','eye_in_hand_rgb'] else np.float32)
                                    (rec if field=='actions' else o).create_dataset(field,data=array,compression='gzip',compression_opts=1)
                                rec.attrs.update(kind='recovery',source_demo=key,noise_level=level,task_success=True,
                                    trigger_type='initial_joint_state',recovery_steps=recovery_steps,joint_delta_rad=delta,
                                    num_samples=min(len(frames['actions']),recovery_steps+49),clean_replay_success=True,
                                    verified_rollout_steps=len(frames['actions']),stored_window='recovery_plus_49')
                                report['accepted']+=1
                            out.flush();save_json(report_path,report);print(json.dumps(event),flush=True)
                data.attrs['generation_complete']=True;data.attrs['generation_manifest']=json.dumps(report)
            if report['accepted']:os.link(partial,dest)
            report['complete']=True;save_json(report_path,report)
            print(json.dumps({'file':str(dest),'accepted':report['accepted']}),flush=True)
    except BaseException as e:
        report.update(complete=False,error=str(e));save_json(report_path,report);raise
    finally:
        if env is not None:env.close()
        if partial is not None:Path(partial).unlink(missing_ok=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--demos',type=int,default=3);p.add_argument('--levels',type=int,default=5)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--seed',type=int,default=1729)
    a=p.parse_args()
    if not 1<=a.levels<=5 or a.repeats<1 or a.demos<0:p.error('Invalid generation counts')
    generate(a.source,a.output,a.demos,a.levels,a.repeats,a.seed)
