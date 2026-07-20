# DLLM (이산 확산 언어 모델) 전체 SFT 훈련 스크립트
# AR(자기회귀) 사전훈련 가중치를 로드한 후 확산 기반 디노이징 목표로 파인튜닝

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind_dllm import MiniMindDLLMConfig, MiniMindForMaskedDiffusion
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

# q_proj와 k_proj 레이어만 훈련 가능하도록 나머지 파라미터를 동결
def freeze_all_except_qk(model):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'q_proj' in name or 'k_proj' in name:
            param.requires_grad = True
            trainable_params += param.numel()
    Logger(f'[QK] 총 파라미터: {total_params:,} | 훈련 가능 파라미터: {trainable_params:,}')
    Logger(f'[QK] 훈련 가능 파라미터 비율: {trainable_params/total_params*100:.2f}%')
    return trainable_params

# DLLM 모델 초기화 및 AR 사전훈련 가중치 로드
def init_model_dllm(config, from_weight, device):
    tokenizer = AutoTokenizer.from_pretrained('../../models', local_files_only=True)
    model = MiniMindForMaskedDiffusion(config)
    if from_weight and from_weight != 'none':
        moe_suffix = '_moe' if config.use_moe else ''
        ckp = f'../../checkouts/{from_weight}_{config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(ckp):
            state_dict = torch.load(ckp, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            Logger(f'[DLLM] {ckp} 에서 가중치를 로드했습니다')
        else:
            raise FileNotFoundError(f'[DLLM] 가중치를 찾을 수 없습니다: {ckp}, from_weight 인자를 확인하세요')
    return model.to(device), tokenizer

# 한 에폭의 훈련 루프 실행
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (X, Y) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        # Y에서 -100이 아닌 위치가 실제 손실을 계산할 유효 토큰
        loss_mask = (Y != -100)
        last_step = step
        # 코사인 학습률 스케줄링
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 전방 확산: 균등 분포에서 노이즈 수준 t를 샘플링
            t = torch.rand(X.shape[0], device=args.device)
            base_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 입력 토큰에 노이즈를 추가하여 마스크된 시퀀스 생성
            noisy_X, corruption_mask, p_mask = base_model.add_noise_to_tokens(X, t, pad_token_id=tokenizer.pad_token_id)

            # 손실 마스크와 노이즈 마스크의 교집합에서만 손실 계산
            corruption_mask = corruption_mask & loss_mask
            # 패딩 위치는 원본 토큰 유지
            noisy_X = torch.where(loss_mask, noisy_X, X)

            attention_mask = (X != tokenizer.pad_token_id).long()
            # 모델이 마스크된 토큰에서 원본 토큰을 예측하도록 훈련
            res = model(input_ids=noisy_X, attention_mask=attention_mask, labels=X, corruption_mask=corruption_mask, p_mask=p_mask, n_valid=loss_mask.sum())
            loss = res.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        # 그래디언트 누적 완료 시 옵티마이저 스텝 실행
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 훈련 로그 출력
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            mask_ratio = corruption_mask.float().mean().item()
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, mask_ratio: {mask_ratio:.2f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "mask_ratio": mask_ratio, "learning_rate": current_lr, "epoch_time": eta_min})

        # 체크포인트 저장 (메인 프로세스에서만)
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # FP16으로 변환하여 저장 (디스크 공간 절약)
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../../checkouts', scaler=scaler)
            model.train()
            del state_dict

        # GPU 메모리 절약을 위해 사용 완료된 텐서 삭제
        del X, Y, noisy_X, corruption_mask, p_mask, res, loss

    # 에폭 종료 시 남은 누적 그래디언트 처리
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DLLM Full SFT")
    parser.add_argument("--save_dir", type=str, default="../../checkouts", help="모델 저장 디렉토리")
    parser.add_argument('--save_weight', default='dllm', type=str, help="저장 가중치 접두사")
    parser.add_argument("--epochs", type=int, default=5, help="훈련 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="초기 학습률")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="훈련 장치")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="혼합 정밀도 타입")
    parser.add_argument("--num_workers", type=int, default=8, help="데이터 로딩 스레드 수")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="그래디언트 누적 스텝 수")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="그래디언트 클리핑 임계값")
    parser.add_argument("--log_interval", type=int, default=100, help="로그 출력 간격")
    parser.add_argument("--save_interval", type=int, default=1000, help="모델 저장 간격")
    parser.add_argument('--hidden_size', default=768, type=int, help="은닉층 차원")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="은닉층 수")
    parser.add_argument('--max_seq_len', default=768, type=int, help="최대 시퀀스 길이")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 아키텍처 사용 여부 (0=아니오, 1=예)")
    parser.add_argument("--data_path", type=str, default="../../datasets/sft_t2t_mini.jsonl", help="훈련 데이터 경로")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="AR 가중치 초기화 기반 (none이면 가중치 없이 훈련)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="자동 감지 및 이어서 훈련 (0=아니오, 1=예)")
    parser.add_argument("--use_wandb", action="store_true", help="wandb 사용 여부")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DLLM-SFT", help="wandb 프로젝트명")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="torch.compile 가속 사용 (0=아니오, 1=예)")
    parser.add_argument("--train_qk_only", default=0, type=int, choices=[0, 1], help="q_proj와 k_proj만 훈련 (0=아니오, 1=예)")
    args = parser.parse_args()

    # ========== 1. 환경 및 랜덤 시드 초기화 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 디렉토리, 모델 파라미터 설정, 체크포인트 확인 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindDLLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../../checkouts') if args.from_resume == 1 else None

    # ========== 3. 혼합 정밀도 설정 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. wandb 설정 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DLLM-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 모델, 데이터, 옵티마이저 정의 ==========
    model, tokenizer = init_model_dllm(lm_config, args.from_weight, device=args.device)
    if args.train_qk_only:
        freeze_all_except_qk(model)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 체크포인트에서 상태 복원 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 컴파일 및 분산 래핑 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 훈련 시작 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 앞의 {start_step}개 step 건너뛰기, step {start_step + 1}부터 시작')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 분산 프로세스 정리 ==========
    if dist.is_initialized(): dist.destroy_process_group()
