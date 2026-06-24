# LoRA(Low-Rank Adaptation) 모듈: 저랭크 행렬을 이용한 효율적 미세조정 구현
import torch
from torch import optim, nn


# LoRA 네트워크 구조 정의
# LoRA 레이어: 저랭크 행렬 A, B를 통해 파라미터 효율적 미세조정 수행
class LoRA(nn.Module):
    # 기능: LoRA 저랭크 A/B projection과 rank를 초기화합니다.
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA의 랭크(rank), 저랭크 행렬의 크기를 제어
        self.A = nn.Linear(in_features, rank, bias=False)  # 저랭크 행렬 A
        self.B = nn.Linear(rank, out_features, bias=False)  # 저랭크 행렬 B
        # 행렬 A를 가우시안 분포로 초기화
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 행렬 B를 0으로 초기화
        self.B.weight.data.zero_()

    # 기능: LoRA A/B projection으로 원본 Linear에 더할 보정값을 계산합니다.
    def forward(self, x):
        return self.B(self.A(x))


# 모델의 정방 Linear 레이어에 LoRA를 적용
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 명시적 바인딩
            # 기능: 원본 Linear 출력에 LoRA 보정 출력을 더하는 monkey patch forward입니다.
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


# 저장된 LoRA 가중치를 모델에 로드
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


# 모델의 LoRA 가중치만 추출하여 저장
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


# LoRA 가중치를 원본 모델에 병합하여 저장
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
