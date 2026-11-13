"""Fine-tune pinned pretrained SmolVLA on HDF5 with a recovery curriculum."""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .data import ChunkData
from .model import prepare,masked_flow_loss

MODEL='lerobot/smolvla_base'
REVISION='d9f33c94a60fb382c90dea2164c96845bd955e28'


def curriculum(step,steps,mode,max_level):
    if mode=='clean':return 0,0.0
    if mode=='fixed':return max_level,0.3
    progress=step/max(steps-1,1)
    if progress<0.1:return 0,0.0
    return min(max_level,1+int((progress-0.1)/0.6*max_level)),min(0.3,0.1+(progress-0.1)*0.4)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['clean','fixed','curriculum'],default='clean')
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--clean-action-offset',type=int,choices=[0,1],default=1)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--validate-every',type=int,default=500)
    p.add_argument('--validation-batches',type=int,default=32)
    p.add_argument('--keep-last',type=int,default=2,help='Keep N recent full checkpoints plus best validation; 0 keeps all')
    p.add_argument('--resume',type=Path)
    p.add_argument('--model',default=MODEL)
    p.add_argument('--revision',default=REVISION)
    p.add_argument('--vlm-path',default=None,help='Pinned local VLM config/tokenizer snapshot')
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if Path(a.model).is_dir():a.revision=None
    if min(a.steps,a.batch_size,a.accumulation,a.save_every,a.validate_every,a.validation_batches)<=0:p.error('Counts must be positive')
    if a.keep_last<0:p.error('--keep-last must be nonnegative')
    if a.out.exists() and not a.resume:p.error('Output exists; choose a fresh run directory or explicit --resume')
    if a.resume and a.resume.resolve().parent!=a.out.resolve():
        p.error('Resume into the same run directory; use --model for a separate fine-tuning experiment')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    ds=ChunkData(a.data,seed=a.seed,clean_offset=a.clean_action_offset)
    if a.mode!='clean' and ds.max_level==0:p.error('No accepted contact-triggered recoveries: generate and validate them first')
    stats=ds.stats()
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import FeatureType,PolicyFeature
    if a.resume:
        cfg=PreTrainedConfig.from_pretrained(a.resume/'policy')
    else:
        cfg=PreTrainedConfig.from_pretrained(a.model,revision=a.revision)
        cfg.input_features={'observation.state':PolicyFeature(FeatureType.STATE,(8,)),
            'observation.images.camera1':PolicyFeature(FeatureType.VISUAL,(3,128,128)),
            'observation.images.camera2':PolicyFeature(FeatureType.VISUAL,(3,128,128))}
        cfg.output_features={'action':PolicyFeature(FeatureType.ACTION,(7,))}
        cfg.n_action_steps=5
        cfg.freeze_vision_encoder=True;cfg.train_expert_only=True;cfg.train_state_proj=True
    cfg.device=a.device;cfg.push_to_hub=False
    # All VLM + expert weights are loaded from the complete pretrained policy.
    # Avoid redundantly downloading the original VLM's weights during construction.
    cfg.load_vlm_weights=False
    if a.vlm_path:cfg.vlm_model_name=a.vlm_path
    if cfg.chunk_size!=ds.chunk_size:raise ValueError('Checkpoint chunk length differs from dataset adapter')
    policy=SmolVLAPolicy.from_pretrained(a.resume/'policy' if a.resume else a.model,
        config=cfg,revision=None if a.resume else a.revision,strict=True)
    optimizer=torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad],lr=a.lr,
                               betas=(0.9,0.95),weight_decay=1e-10)
    warmup=min(1000,max(1,a.steps//10))
    def multiplier(step):
        if step<warmup:return (step+1)/warmup
        return .025+.975*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(a.steps-warmup,1))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    train_id_set=set(ds.train)
    manifest={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'dataset_files':ds.files,'val_sources':ds.val_sources,'stats':stats,'image_flips':ds.flips,
        'state_order':['gripper_qpos(2)','eef_pos(3)','eef_axisangle(3)'],
        'model':a.model,'revision':a.revision,'trainable_parameters':sum(x.numel() for x in policy.parameters() if x.requires_grad),
        'recovery_train_demos_by_task':{task:sum(d['task']==task and d['kind']=='recovery' for i,d in enumerate(ds.demos) if i in train_id_set)
                                       for task in sorted({d['task'] for d in ds.demos})},
        'total_parameters':sum(x.numel() for x in policy.parameters()),
        'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in ['train.py','data.py','model.py']}}
    start=0;best_val=float('inf');best_step=None
    if a.resume:
        saved=json.loads((a.resume/'manifest.json').read_text())
        for key in ['dataset_files','val_sources','stats','image_flips','source_sha256']:
            if saved[key]!=manifest[key]:raise ValueError(f'Resume mismatch: {key}')
        for key in ['steps','mode','seed','batch_size','accumulation','lr','clean_action_offset','model','revision','validation_batches']:
            if saved['args'][key]!=manifest['args'][key]:raise ValueError(f'Resume argument changed: {key}')
        state=torch.load(a.resume/'training.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start=state['step']
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['random_rng']);np.random.set_state(state['numpy_rng'])
        best_val=state['best_val'];best_step=state['best_step']
        metrics=a.out/'metrics.jsonl'
        if metrics.exists():
            rows=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
            if rows and rows[-1]['step']!=start:raise ValueError('Resume the latest step or choose a new output directory')
    a.out.mkdir(parents=True,exist_ok=bool(a.resume))
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'event':'loaded_pretrained','parameters':manifest['total_parameters'],
                      'trainable':manifest['trainable_parameters'],'start_step':start}),flush=True)
    def save(step):
        dest=a.out/f'step_{step:06d}';tmp=a.out/f'.step_{step:06d}.tmp'
        if dest.exists() or tmp.exists():raise FileExistsError(dest)
        tmp.mkdir();policy.save_pretrained(tmp/'policy')
        (tmp/'manifest.json').write_text(json.dumps(manifest,indent=2))
        torch.save({'step':step,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'best_val':best_val,'best_step':best_step,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'random_rng':random.getstate(),'numpy_rng':np.random.get_state()},tmp/'training.pt')
        tmp.rename(dest)
        marker=a.out/'latest.tmp';marker.write_text(dest.name+'\n');marker.replace(a.out/'latest.txt')
        if best_step is not None:
            (a.out/'best_validation.json').write_text(json.dumps({'step':best_step,'clean_val_loss':best_val}))
        if a.keep_last:
            checkpoints=sorted(p for p in a.out.glob('step_*') if p.is_dir() and not p.is_symlink()
                               and p.name.removeprefix('step_').isdigit() and (p/'training.pt').exists())
            keep={p.name for p in checkpoints[-a.keep_last:]}
            if best_step is not None:keep.add(f'step_{best_step:06d}')
            for old in checkpoints:
                if old.name not in keep:shutil.rmtree(old)
    amp=a.device.startswith('cuda')
    for step in range(start,a.steps):
        begin=time.monotonic();policy.train();optimizer.zero_grad(set_to_none=True)
        level,ratio=curriculum(step,a.steps,a.mode,ds.max_level)
        rng=np.random.default_rng(np.random.SeedSequence([a.seed,step]));total=0.0;recovery_count=0
        for _ in range(a.accumulation):
            raw=ds.batch(rng,a.batch_size,level,ratio)
            recovery_count+=int(raw['is_recovery'].sum())
            batch=prepare(policy,raw,stats,a.device)
            with torch.autocast(device_type='cuda' if amp else 'cpu',dtype=torch.bfloat16,enabled=amp):
                loss=masked_flow_loss(policy,batch)
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at step {step}')
            (loss/a.accumulation).backward();total+=loss.item()/a.accumulation
        grad=torch.nn.utils.clip_grad_norm_([x for x in policy.parameters() if x.requires_grad],10)
        if not torch.isfinite(grad):raise FloatingPointError('Non-finite gradient')
        optimizer.step();scheduler.step()
        row={'step':step+1,'loss':total,'level':level,'recovery_ratio':ratio,
             'actual_recovery_fraction':recovery_count/(a.batch_size*a.accumulation),
             'lr':optimizer.param_groups[0]['lr'],'seconds':time.monotonic()-begin}
        if (step+1)%a.validate_every==0 or step+1==a.steps:
            # Use fixed clean validation batches and a separate RNG context.
            policy.eval();vals=[]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if amp else []),torch.no_grad():
                torch.manual_seed(a.seed+991)
                vrng=np.random.default_rng(a.seed+991)
                for _ in range(a.validation_batches):
                    b=prepare(policy,ds.batch(vrng,a.batch_size,0,0,validation=True),stats,a.device)
                    with torch.autocast(device_type='cuda' if amp else 'cpu',dtype=torch.bfloat16,enabled=amp):
                        vals.append(masked_flow_loss(policy,b).item())
            row['clean_val_loss']=float(np.mean(vals))
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step<5 or (step+1)%10==0 or 'clean_val_loss' in row:print(json.dumps(row),flush=True)
        improved=row.get('clean_val_loss',float('inf'))<best_val
        if improved:best_val=row['clean_val_loss'];best_step=step+1
        if improved or (step+1)%a.save_every==0 or step+1==a.steps:save(step+1)
    ds.close()


if __name__=='__main__':main()
