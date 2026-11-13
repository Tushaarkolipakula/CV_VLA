"""Shared preprocessing and a valid-action-only SmolVLA flow-matching loss."""
import numpy as np
import torch


def prepare(policy, batch, stats, device):
    out={}
    for i,key in enumerate(policy.config.image_features):
        out[key]=batch['images'][:,i].to(device=device,dtype=torch.float32)/255.0
    for key,name in [('observation.state','state'),('action','action')]:
        if name not in batch:continue
        x=batch[name].to(device=device,dtype=torch.float32)
        mean=torch.tensor(stats[name]['mean'],device=device)
        std=torch.tensor(stats[name]['std'],device=device)
        out[key]=(x-mean)/std
    tokenizer=policy.model.vlm_with_expert.processor.tokenizer
    tokens=tokenizer([t.rstrip()+'\n' for t in batch['task']],padding=policy.config.pad_language_to,
                     truncation=True,max_length=policy.config.tokenizer_max_length,return_tensors='pt')
    out['observation.language.tokens']=tokens['input_ids'].to(device)
    out['observation.language.attention_mask']=tokens['attention_mask'].to(device).bool()
    if 'valid' in batch:out['valid']=batch['valid'].to(device)
    return out


def masked_flow_loss(policy,batch):
    # LeRobot v0.4.4's wrapper uses an inconsistent padding key and averages
    # padded action dimensions. Restrict loss explicitly to valid steps and 7
    # robot action dimensions so short recovery segments are weighted correctly.
    images,masks=policy.prepare_images(batch)
    losses=policy.model.forward(images,masks,batch['observation.language.tokens'],
        batch['observation.language.attention_mask'],policy.prepare_state(batch),policy.prepare_action(batch))
    return reduce_valid_loss(losses,batch['valid'],7)


def reduce_valid_loss(losses,valid,action_dim):
    selected=losses[:,:,:action_dim]
    weights=valid.unsqueeze(-1).to(selected.dtype)
    # Each sampled observation has equal weight, independently of remaining
    # episode length. Padding never contributes a target.
    per_sample=(selected*weights).sum((1,2))/(weights.sum((1,2))*action_dim).clamp_min(1)
    return per_sample.mean()


def unnormalize_action(action,stats):
    return action.float().cpu().numpy()*np.asarray(stats['action']['std'])+np.asarray(stats['action']['mean'])
