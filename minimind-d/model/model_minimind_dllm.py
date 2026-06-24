# 이산 확산 언어 모델(DLLM) 아키텍처 정의
# MiniMind 기반의 마스크 확산 모델: 반복적 디노이징으로 텍스트를 생성하는 비자기회귀(non-autoregressive) 모델

import math, torch, torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MaskedLMOutput
from model.model_minimind import MiniMindConfig, MiniMindModel

# Gumbel 노이즈를 로짓에 추가하여 확률적 샘플링을 수행하는 함수
def add_gumbel_noise(logits, temperature):
    if temperature == 0: return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    # Gumbel-max 트릭: 로짓의 지수에 Gumbel 노이즈를 나눠서 확률적 argmax 생성
    return logits.exp() / ((-torch.log(noise)) ** temperature)

# DLLM 전용 설정 클래스: 마스크 토큰 ID와 마스크 엡실론 추가
# 역할: `MiniMindDLLMConfig` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class MiniMindDLLMConfig(MiniMindConfig):
    model_type = "minimind_dllm"
    # 기능: MiniMindDLLMConfig 객체가 사용할 계층과 상태를 초기화합니다.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = kwargs.get("mask_token_id", 27)      # [MASK] 토큰 ID
        self.mask_epsilon = kwargs.get("mask_epsilon", 0.001)      # 노이즈 스케줄링의 최소 마스킹 확률

# DLLM 백본 모델: 양방향 어텐션 사용 (is_causal=False)
# 역할: `MiniMindDLLMModel` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class MiniMindDLLMModel(MiniMindModel):
    # 기능: MiniMindDLLMModel 객체가 사용할 계층과 상태를 초기화합니다.
    def __init__(self, config:
        MiniMindDLLMConfig):
        super().__init__(config)
        # 확산 모델은 전체 시퀀스를 동시에 처리하므로 인과적 마스킹을 비활성화
        for layer in self.layers: layer.self_attn.is_causal = False

    # 입력 토큰에 노이즈를 추가하여 마스크된 시퀀스를 생성 (전방 확산 과정)
    def add_noise_to_tokens(self, input_ids, t, eps=None, pad_token_id=0):
        batch_size, seq_len = input_ids.shape
        eps = eps if eps is not None else self.config.mask_epsilon
        # 선형 노이즈 스케줄: t=0이면 거의 마스킹 없음, t=1이면 거의 전부 마스킹
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask.unsqueeze(-1).expand(batch_size, seq_len)
        # 베르누이 샘플링으로 마스킹할 위치 결정
        corruption_mask = torch.rand(batch_size, seq_len, device=input_ids.device) < p_mask
        # 패딩 토큰은 마스킹하지 않음
        corruption_mask = corruption_mask & (input_ids != pad_token_id)
        # 마스킹 위치를 [MASK] 토큰으로 교체
        noisy_input_ids = torch.where(corruption_mask, self.config.mask_token_id, input_ids)
        return noisy_input_ids, corruption_mask, p_mask

# 마스크 확산 생성을 위한 최상위 모델 클래스
# 역할: `MiniMindForMaskedDiffusion` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class MiniMindForMaskedDiffusion(PreTrainedModel, GenerationMixin):
    config_class = MiniMindDLLMConfig
    # 기능: MiniMindForMaskedDiffusion 객체가 사용할 계층과 상태를 초기화합니다.
    def __init__(self, config:
        MiniMindDLLMConfig = None):
        self.config = config or MiniMindDLLMConfig()
        super().__init__(self.config)
        self.model = MiniMindDLLMModel(self.config)
        # 언어 모델 헤드: 히든 상태를 어휘 로짓으로 변환
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 임베딩 가중치를 출력 헤드와 공유 (weight tying)
        self.model.embed_tokens.weight = self.lm_head.weight

    # 순전파: 마스크된 입력에서 원본 토큰을 예측
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, labels=None, corruption_mask=None, p_mask=None, n_valid=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None and corruption_mask is not None and p_mask is not None:
            # 크로스 엔트로피 손실 계산 (마스크된 위치만)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), reduction='none').view(labels.shape)
            denom = n_valid if n_valid is not None else corruption_mask.sum().clamp_min(1)
            # 중요도 가중 손실: p_mask로 나누어 낮은 노이즈 수준에서의 손실에 더 큰 가중치 부여
            loss = (loss[corruption_mask] / p_mask[corruption_mask]).sum() / denom
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)

    # 노이즈 추가를 백본 모델에 위임
    def add_noise_to_tokens(self, input_ids, t, eps=None, pad_token_id=0):
        return self.model.add_noise_to_tokens(input_ids, t, eps, pad_token_id)

    # 반복적 디노이징을 통한 텍스트 생성 (역방향 확산 과정)
    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens=128, temperature=0.5, top_k=50, steps=128, eos_token_id=None, tokenizer=None, cfg_scale=0.0, **kwargs):
        input_ids = kwargs.get("input_ids", inputs)
        bsz, prompt_len, device = input_ids.shape[0], input_ids.shape[1], input_ids.device
        mask_id = self.config.mask_token_id
        eos_id = eos_token_id or self.config.eos_token_id
        block_size = kwargs.get("block_size", max_new_tokens)

        # 프롬프트 뒤에 [MASK] 토큰으로 채운 전체 시퀀스 초기화
        T = prompt_len + max_new_tokens
        x = torch.full((bsz, T), eos_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids
        x[:, prompt_len:] = mask_id
        unmasked_index = (x != mask_id)

        # 블록 단위 디노이징: 긴 시퀀스를 블록으로 나누어 순차적으로 생성
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = steps

        for b in range(num_blocks):
            block_end = min(prompt_len + (b + 1) * block_size, T)

            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                mask_count = mask_index[:, :block_end].sum(-1).min().item()
                if mask_count == 0: break
                # 남은 스텝에 비례하여 이번 스텝에서 언마스킹할 토큰 수 결정
                n_unmask = max(1, round(mask_count / (steps_per_block - i)))

                # Classifier-Free Guidance (CFG): 조건부/무조건부 로짓 혼합
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(input_ids=x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(input_ids=x).logits

                # Top-k 필터링: 상위 k개 토큰만 남기고 나머지 확률 제거
                if top_k > 0: logits[logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]] = -float('inf')

                # Gumbel 노이즈로 샘플링하여 후보 토큰 생성
                x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
                p = F.softmax(logits, dim=-1)
                # 각 후보 토큰의 신뢰도(확률) 계산
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                # 현재 블록 범위 밖의 토큰은 선택되지 않도록 신뢰도를 -inf로 설정
                x0_p[:, block_end:] = -float('inf')

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))

                # 신뢰도가 가장 높은 n_unmask개 토큰을 선택하여 언마스킹
                for j in range(bsz):
                    _, idx = torch.topk(confidence[j], k=min(n_unmask, int(mask_count)))
                    x[j, idx] = x0[j, idx]

            # 스트리밍 모드에서 블록별 중간 결과 출력
            if tokenizer and kwargs.get("stream", False):
                print(f"[Block {b+1}/{num_blocks}] {tokenizer.decode(x[0, prompt_len:], skip_special_tokens=False)}")

        return x
