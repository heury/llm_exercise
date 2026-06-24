"""
训练工具函数集合
"""
import os
import random
import math
import numpy as np
import datasets
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_omni import MiniMindOmni
    


# 기능: is_main_process 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# 기능: Logger 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def Logger(content):
    if is_main_process():
        print(content)


# 기능: get_lr에서 현재 상태의 필요 값을 조회해 반환합니다.
def get_lr(current_step, total_steps, lr):
    # mmv와 동일하게 초기 lr=1.0*base_lr, 최종 lr=0.1*base_lr 사용
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# 기능: init_distributed_mode 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # ?-DDP ??
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# 기능: Python, NumPy, PyTorch, CUDA 난수 시드를 고정해 학습 재현성을 맞춥니다.
def setup_seed(seed:
    int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 기능: log_model_params 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def log_model_params(model, ignore_patterns=['audio_encoder', 'vision_encoder']):
    # 기능: should_count 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
    def should_count(n):
        return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    cfg = model.config
    n_routed = getattr(cfg, 'n_routed_experts', getattr(cfg, 'num_experts', 0))
    n_active = getattr(cfg, 'num_experts_per_tok', 0)
    n_shared = getattr(cfg, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


# 기능: init_omni_model 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def init_omni_model(omni_config, from_weight='full_sft', tokenizer_path='../../minimind_model', audio_encoder_path='../../minimind_model/SenseVoiceSmall', vision_model_path='../../minimind_model/siglip2-base-p32-256-ve', save_dir='../../minimind_out', device='cuda', freeze_backbone='none', from_resume=0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindOmni(omni_config, audio_encoder_path=audio_encoder_path, vision_model_path=vision_model_path)
    
    if from_weight != 'none':
        moe_suffix = '_moe' if omni_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            param_shapes = {k: v.shape for k, v in model.named_parameters()}
            incompatible = {k for k, v in weights.items() if k in param_shapes and v.shape != param_shapes[k]}
            if incompatible:
                Logger(f'跳过shape不匹配的权重: {incompatible}')
                weights = {k: v for k, v in weights.items() if k not in incompatible}
            model.load_state_dict(weights, strict=False)
            Logger(f'已加载权重: {weight_path}')
            if from_resume == 0 and omni_config.talker_hidden_size == omni_config.hidden_size:
                n_talker = omni_config.num_talker_hidden_layers
                n_thinker = len(model.thinker.layers)
                has_talker = any(k.startswith('talker.layers.') for k in weights)
                if not has_talker and n_talker > 0:
                    for i in range(n_talker):
                        src = n_thinker - n_talker + i
                        model.talker.layers[i].load_state_dict(model.thinker.layers[src].state_dict())
                    Logger(f'Talker层初始化: 复制thinker layers[{n_thinker-n_talker}:{n_thinker}] → talker layers[0:{n_talker}]')
    
    # 해당 단계의 처리 흐름을 설명
    if freeze_backbone == 'all':
        # 해당 단계의 처리 흐름을 설명
        for param in model.model.parameters():
            param.requires_grad = False
    elif freeze_backbone == 'last1':
        # 해당 단계의 처리 흐름을 설명
        for param in model.model.parameters():
            param.requires_grad = False
        # 해당 단계의 처리 흐름을 설명
        if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            for param in model.model.layers[-1].parameters():
                param.requires_grad = True
    return model.to(device), tokenizer


# 기능: omni_checkpoint 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def omni_checkpoint(omni_config, weight='pretrain_omni', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../../minimind_out/checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if omni_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        # 동결된 audio_encoder / vision_encoder 파라미터는 저장하지 않고 사전학습 경로에서 다시 로드
        clean_state_dict = {k: v for k, v in raw_model.state_dict().items() if not k.startswith('audio_encoder.') and not k.startswith('vision_encoder.')}
        state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # ?? ??
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


# 기능: vlm_collate_fn 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def vlm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values


# 역할: `SkipBatchSampler` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class SkipBatchSampler(Sampler):
    # 기능: SkipBatchSampler 객체가 사용할 계층과 상태를 초기화합니다.
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    # 기능: __iter__ 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    # 기능: __len__ 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)

