"""Strict base loading and PI05-specific preprocessing. No random-weight fallback."""
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from peft import LoraConfig,inject_adapter_in_model
from smolvla_pipeline.model import reduce_valid_loss

CAMERA_KEYS=['observation.images.base_0_rgb','observation.images.left_wrist_0_rgb','observation.images.right_wrist_0_rgb']

def load_base(path,device='cuda',rank=8):
    cfg=PreTrainedConfig.from_pretrained(path);cfg.device='meta';cfg.dtype='bfloat16'
    cfg.freeze_vision_encoder=True;cfg.train_expert_only=False;cfg.gradient_checkpointing=True
    cfg.n_action_steps=5
    cfg.input_features={k:PolicyFeature(FeatureType.VISUAL,(3,224,224)) for k in CAMERA_KEYS}
    cfg.input_features['observation.state']=PolicyFeature(FeatureType.STATE,(8,))
    cfg.output_features={'action':PolicyFeature(FeatureType.ACTION,(7,))}
    # Allocate no random full-size model on the 32-GB host. Assign real
    # pretrained tensors directly, then reconstruct only nonpersistent buffers.
    with torch.device('meta'):policy=PI05Policy(cfg)
    weights=policy._fix_pytorch_state_dict_keys(load_file(str(Path(path)/'model.safetensors')),cfg)
    weights={k if k.startswith('model.') else 'model.'+k:v for k,v in weights.items()}
    params=dict(policy.named_parameters(remove_duplicate=False))
    for missing in set(policy.state_dict())-set(weights):
        aliases=[k for k in weights if k in params and missing in params and params[k] is params[missing]]
        if not aliases:raise RuntimeError(f'Missing pretrained tensor: {missing}')
        weights[missing]=weights[aliases[0]]
    policy.load_state_dict(weights,strict=True,assign=True);del weights,params
    for name,module in policy.named_modules():
        for key,buffer in list(module.named_buffers(recurse=False)):
            if buffer.device.type!='meta':continue
            if key=='position_ids':
                value=torch.arange(buffer.numel(),dtype=buffer.dtype,device='cpu').reshape(buffer.shape)
            elif key=='inv_freq' and hasattr(module,'rope_init_fn'):
                value,scale=module.rope_init_fn(module.config,device=torch.device('cpu'))
                module.attention_scaling=scale;module.original_inv_freq=value
            else:raise RuntimeError(f'Unknown nonpersistent meta buffer: {name}.{key}')
            module.register_buffer(key,value,persistent=key not in module._non_persistent_buffers_set)
    # Freeze the base before injecting into the transformer submodule. PEFT
    # expects a Transformers config; the outer LeRobot config is a dataclass.
    policy.model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
    policy.requires_grad_(False)
    lora=LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=0.,
        target_modules=r'(?:.*\.)?(language_model|gemma_expert)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$')
    policy.model.paligemma_with_expert=inject_adapter_in_model(lora,policy.model.paligemma_with_expert)
    for name,p in policy.named_parameters():
        if p.requires_grad:p.data=p.data.float()
    trainable=[n for n,p in policy.named_parameters() if p.requires_grad]
    if not trainable or not any('gemma_expert' in n for n in trainable) or not any('language_model' in n for n in trainable):
        raise RuntimeError('LoRA did not attach to both language and action attention')
    policy.to(device);policy.config.device=device;return policy

def normalize(value,stats):
    low=torch.as_tensor(stats['q01'],device=value.device,dtype=value.dtype)
    high=torch.as_tensor(stats['q99'],device=value.device,dtype=value.dtype)
    return 2*(value-low)/(high-low).clamp_min(1e-6)-1

def prepare(raw,stats,tokenizer,device):
    state=normalize(raw['state'].float(),stats['state'])
    state=torch.nn.functional.pad(state,(0,32-state.shape[-1]))
    bins=np.digitize(state.numpy(),np.linspace(-1,1,257)[:-1])-1
    prompts=[]
    for task,values in zip(raw['task'],bins):
        task=task.strip().replace('_',' ').replace('\n',' ')
        prompts.append(f"Task: {task}, State: {' '.join(map(str,values))};\nAction: ")
    tokens=tokenizer(prompts,padding='max_length',truncation=True,max_length=200,return_tensors='pt')
    batch={CAMERA_KEYS[i]:raw['images'][:,i].float().to(device)/255 for i in range(2)}
    batch['observation.language.tokens']=tokens['input_ids'].to(device)
    batch['observation.language.attention_mask']=tokens['attention_mask'].to(device).bool()
    if 'action' in raw:
        batch['action']=normalize(raw['action'].float(),stats['action']).to(device)
        batch['valid']=raw['valid'].to(device)
    return batch

def loss(policy,batch):
    images,masks=policy._preprocess_images(batch)
    losses=policy.model.forward(images,masks,batch['observation.language.tokens'],
        batch['observation.language.attention_mask'],policy.prepare_action(batch))
    return reduce_valid_loss(losses,batch['valid'],7)

def decode(action,stats):
    low=np.asarray(stats['action']['q01']);high=np.asarray(stats['action']['q99'])
    return (action.float().cpu().numpy()+1)*np.maximum(high-low,1e-6)/2+low
