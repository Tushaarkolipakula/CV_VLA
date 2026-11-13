"""PI05 LoRA experiment: fixed and curriculum have matched recovery probability."""
import argparse,hashlib,json,math,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer
from .data import ChunkData
from .model import load_base,prepare,loss

def schedule(step,steps,mode):
    if mode=='clean':return 0,0.
    # Constant recovery probability in both augmented runs isolates the level schedule.
    return (5 if mode=='fixed' else min(5,1+int(4*step/max(1,steps*.7)))),.1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base',default=Path('results/pi05_setup/base_path.txt').read_text().strip())
    p.add_argument('--mode',choices=['clean','fixed','curriculum'],default='clean')
    p.add_argument('--steps',type=int,default=30000);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--lr',type=float,default=2.5e-5)
    p.add_argument('--save-every',type=int,default=500);p.add_argument('--validation-batches',type=int,default=32)
    p.add_argument('--resume',type=Path);a=p.parse_args()
    if min(a.steps,a.batch_size,a.accumulation,a.save_every,a.validation_batches)<1:p.error('Counts must be positive')
    if a.out.exists() and not a.resume:p.error('Output already exists')
    if a.resume and a.resume.resolve().parent!=a.out.resolve():p.error('Resume into original run only')
    tokenizer=AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224',padding_side='right')
    if torch.cuda.mem_get_info()[0]<18*1024**3:raise RuntimeError('Need at least 18 GiB free; other GPU jobs left untouched')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    ds=ChunkData(a.data,seed=0) # same source split for every training seed and mode
    if a.mode!='clean' and ds.max_level!=5:raise ValueError('All five recovery levels required')
    stats=ds.stats()
    manifest={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'stats':stats,
        'dataset_files':ds.files,'val_sources':ds.val_sources,'image_flips':ds.flips,
        'source_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['train.py','data.py','model.py']}}
    policy=load_base(a.base,rank=a.rank)
    trainable={n:v for n,v in policy.named_parameters() if v.requires_grad}
    optimizer=torch.optim.AdamW(trainable.values(),lr=a.lr,betas=(.9,.95),weight_decay=.01)
    warmup=min(1000,max(1,a.steps//10))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda t:(t+1)/warmup if t<warmup else .1+.9*.5*(1+math.cos(math.pi*min(1,(t-warmup)/max(1,a.steps-warmup)))))
    start=0;best=float('inf');best_step=None
    if a.resume:
        old=json.loads((a.out/'manifest.json').read_text())
        for k in ['stats','dataset_files','val_sources','source_sha256']:
            if old[k]!=manifest[k]:raise ValueError(f'Resume mismatch: {k}')
        for k in ['steps','seed','mode','batch_size','accumulation','rank','lr','base','validation_batches']:
            if old['args'][k]!=manifest['args'][k]:raise ValueError(f'Resume argument mismatch: {k}')
        w=load_file(str(a.resume/'adapter.safetensors'))
        if set(w)!=set(trainable):raise ValueError('Adapter parameter mismatch')
        policy.load_state_dict(w,strict=False)
        state=torch.load(a.resume/'training.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['step'];best=state['best'];best_step=state['best_step']
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        lines=(a.out/'metrics.jsonl').read_text().splitlines()
        trailing=[line for line in lines if json.loads(line)['step']>start]
        if trailing:
            with (a.out/f'interrupted_metrics_{time.time_ns()}.jsonl').open('x') as f:f.write('\n'.join(trailing)+'\n')
            kept=[line for line in lines if json.loads(line)['step']<=start]
            (a.out/'metrics.jsonl').write_text('\n'.join(kept)+'\n')
            lines=kept
        if lines and json.loads(lines[-1])['step']!=start:raise ValueError('Checkpoint/metrics disagreement')
    a.out.mkdir(parents=True,exist_ok=bool(a.resume));(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'event':'strict_base_loaded','trainable_parameters':sum(v.numel() for v in trainable.values())}),flush=True)
    def save(step):
        target=a.out/f'step_{step:06d}';temp=a.out/f'.step_{step:06d}.partial';temp.mkdir()
        save_file({k:v.detach().cpu().contiguous() for k,v in trainable.items()},str(temp/'adapter.safetensors'))
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':step,'best':best,
            'best_step':best_step,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp/'training.pt')
        temp.rename(target);(a.out/'latest.txt').write_text(target.name)
        (a.out/'best_validation.json').write_text(json.dumps({'step':best_step,'loss':best}))
        for old in a.out.glob('step_*'):
            if old.name not in {target.name,f'step_{best_step:06d}'} and (old/'adapter.safetensors').exists():shutil.rmtree(old)
    try:
        for step in range(start,a.steps):
            tick=time.monotonic();policy.train();optimizer.zero_grad(set_to_none=True)
            level,ratio=schedule(step,a.steps,a.mode);rng=np.random.default_rng(np.random.SeedSequence([a.seed,step]));total=0;recoveries=0
            for _ in range(a.accumulation):
                raw=ds.batch(rng,a.batch_size,level,ratio);recoveries+=int(raw['is_recovery'].sum())
                with torch.autocast('cuda',dtype=torch.bfloat16):value=loss(policy,prepare(raw,stats,tokenizer,'cuda'))
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
                (value/a.accumulation).backward();total+=value.item()/a.accumulation
            if step==start:
                for family in ['language_model','gemma_expert']:
                    magnitude=sum(float(v.grad.float().square().sum()) for n,v in trainable.items() if family in n and v.grad is not None)
                    if not math.isfinite(magnitude) or magnitude<=0:raise RuntimeError(f'No finite nonzero gradient in {family}')
                    print(json.dumps({'event':'gradient_verified','family':family,'squared_norm':magnitude}),flush=True)
            norm=torch.nn.utils.clip_grad_norm_(trainable.values(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();scheduler.step()
            row={'step':step+1,'loss':total,'max_level':level,'recovery_fraction':recoveries/(a.batch_size*a.accumulation),
                'seconds':time.monotonic()-tick,'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3}
            if (step+1)%a.save_every==0 or step+1==a.steps:
                policy.eval();values=[]
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
                    torch.manual_seed(991);vrng=np.random.default_rng(991)
                    for _ in range(a.validation_batches):
                        raw=ds.batch(vrng,a.batch_size,0,0,validation=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):values.append(loss(policy,prepare(raw,stats,tokenizer,'cuda')).item())
                row['clean_val_loss']=float(np.mean(values))
                if row['clean_val_loss']<best:best=row['clean_val_loss'];best_step=step+1
            with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step<5 or (step+1)%10==0 or 'clean_val_loss' in row:print(json.dumps(row),flush=True)
            if 'clean_val_loss' in row:save(step+1)
    finally:ds.close()

if __name__=='__main__':main()
