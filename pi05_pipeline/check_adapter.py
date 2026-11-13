"""Verify a saved adapter and recovery preprocessing without updating weights."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from .model import load_base,prepare,loss,decode
from .data import ChunkData

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--benchmark-batch',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(2);manifest=json.loads((a.checkpoint.parent/'manifest.json').read_text())
    weights=load_file(str(a.checkpoint/'adapter.safetensors'))
    for family in ['language_model','gemma_expert']:
        values=[v for k,v in weights.items() if family in k and 'lora_B' in k]
        if not values or not any(torch.count_nonzero(v).item()>0 for v in values):raise ValueError(f'No learned adapter changes in {family}')
    model=load_base(manifest['args']['base'],rank=manifest['args']['rank'])
    expected={n for n,v in model.named_parameters() if v.requires_grad}
    if set(weights)!=expected:raise ValueError('Saved tensor set mismatch')
    model.load_state_dict(weights,strict=False)
    for n,v in model.named_parameters():
        if n in weights and not torch.equal(v.detach().cpu(),weights[n]):raise ValueError('Reload mismatch: '+n)
    roots=list(manifest['args']['data'])+['data/initial_pilot_v3/libero_goal']
    ds=ChunkData(roots)
    try:
        ids=[i for i,d in enumerate(ds.demos) if d['kind']=='recovery']
        if not ids:raise ValueError('No pilot recoveries')
        items=[ds.sample(i,0) for i in ids[:4]]
        raw={k:[x[k] for x in items] if k=='task' else torch.from_numpy(np.stack([x[k] for x in items])) for k in items[0]}
        tokenizer=AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224',padding_side='right')
        model.eval();torch.manual_seed(193)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            batch=prepare(raw,manifest['stats'],tokenizer,'cuda');value=loss(model,batch)
            actions=model.predict_action_chunk(batch)
        decoded=decode(actions,manifest['stats'])
        if not np.isfinite(decoded).all() or not torch.isfinite(value):raise ValueError('Nonfinite recovery loss/action')
        print(json.dumps({'passed':True,'adapter_tensors':len(weights),'recovery_loss':value.item(),
            'action_shape':list(decoded.shape),'checks':['learned adapters in both transformers','exact saved tensor reload','finite recovery loss','finite inferred actions']}),flush=True)
        if a.benchmark_batch:
            model.train();torch.cuda.reset_peak_memory_stats()
            for i in range(3):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.monotonic()
                raw=ds.batch(np.random.default_rng(i),a.benchmark_batch,0,0)
                with torch.autocast('cuda',dtype=torch.bfloat16):value=loss(model,prepare(raw,manifest['stats'],tokenizer,'cuda'))
                value.backward();torch.cuda.synchronize()
                if not torch.isfinite(value):raise FloatingPointError('Benchmark loss is nonfinite')
                print(json.dumps({'benchmark_batch':a.benchmark_batch,'iteration':i,'seconds':time.monotonic()-tick,
                    'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'weights_updated':False}),flush=True)
    finally:ds.close()

if __name__=='__main__':main()
