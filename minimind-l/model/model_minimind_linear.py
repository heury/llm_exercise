# MiniMind 선형 어텐션 모델 정의 파일
# 소프트맥스 어텐션을 GatedDeltaNet 선형 어텐션으로 대체한 경량 언어 모델

import math, torch, torch.nn.functional as F, logging
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

logger = logging.getLogger(__name__)

# CUDA 인과 합성곱 가속 라이브러리 로드 시도
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    logger.info('causal-conv1d 감지됨, CUDA 합성곱 가속 활성화')
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
    logger.warning('causal-conv1d 사용 불가, PyTorch 기본 합성곱으로 대체')

# Triton 기반 선형 어텐션 가속 라이브러리 (FLA) 로드 시도
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    logger.info('flash-linear-attention (FLA) 감지됨, Triton 선형 어텐션 가속 활성화')
except ImportError:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    logger.warning('flash-linear-attention (FLA) 사용 불가, PyTorch 기본 선형 어텐션으로 대체')


# 모델 설정 클래스: 히든 크기, 레이어 수, MoE, 선형 어텐션 등 모든 하이퍼파라미터 관리
# 역할: `MiniMindConfig` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    # 기능: MiniMindConfig에 모델 차원, 레이어, attention, RoPE, MoE 설정을 저장합니다.
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # 기본 모델 구조 설정
        self.hidden_size = hidden_size                  # 은닉 차원 크기
        self.num_hidden_layers = num_hidden_layers      # 트랜스포머 레이어 수
        self.use_moe = use_moe                          # MoE (혼합 전문가) 사용 여부
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)        # 어휘 사전 크기
        self.bos_token_id = kwargs.get("bos_token_id", 1)       # 문장 시작 토큰 ID
        self.eos_token_id = kwargs.get("eos_token_id", 2)       # 문장 종료 토큰 ID
        self.flash_attn = kwargs.get("flash_attn", True)        # Flash Attention 사용 여부
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)      # 어텐션 헤드 수
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)     # KV 헤드 수 (GQA)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 헤드당 차원
        self.hidden_act = kwargs.get("hidden_act", 'silu')      # 활성화 함수
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)  # FFN 중간 크기
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 최대 위치 임베딩
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)    # RMSNorm 엡실론 값
        self.rope_theta = kwargs.get("rope_theta", 1e6)         # RoPE 기저 주파수
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        # YaRN 방식의 RoPE 스케일링 설정 (추론 시 긴 시퀀스 처리용)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE 전용 설정 (use_moe=False이면 무시됨)
        self.num_experts = kwargs.get("num_experts", 4)                     # 전문가 수
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)     # 토큰당 활성화 전문가 수
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)            # top-k 확률 정규화
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 라우터 보조 손실 계수
        ### GatedDeltaNet (선형 어텐션) 전용 설정
        self.full_attention_interval = kwargs.get("full_attention_interval", 4)  # 풀 어텐션 레이어 간격
        self.linear_conv_kernel_dim = kwargs.get("linear_conv_kernel_dim", 4)   # 선형 어텐션 합성곱 커널 크기
        self.linear_key_head_dim = kwargs.get("linear_key_head_dim", self.head_dim)
        self.linear_value_head_dim = kwargs.get("linear_value_head_dim", self.head_dim)
        self.linear_num_key_heads = kwargs.get("linear_num_key_heads", self.num_attention_heads)
        self.linear_num_value_heads = kwargs.get("linear_num_value_heads", self.num_attention_heads)
        # 레이어 유형 결정: full_attention_interval 간격마다 풀 어텐션, 나머지는 선형 어텐션
        self.layer_types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                self.layer_types.append("full_attention")
            else:
                self.layer_types.append("linear_attention")


# RMS 정규화: LayerNorm의 경량 대안으로 평균을 제거하고 제곱평균제곱근만 사용
# 역할: `RMSNorm` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class RMSNorm(torch.nn.Module):
    # 기능: RMSNorm scale parameter와 epsilon을 초기화합니다.
    def __init__(self, dim:
        int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # 기능: RMSNorm의 평균제곱근 역수로 tensor를 정규화합니다.
    def norm(self, x):
        # x / sqrt(mean(x^2) + eps) 계산
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # 기능: RMSNorm 정규화 결과에 학습 가능한 scale을 곱합니다.
    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


# 게이트 RMS 정규화: RMSNorm에 SiLU 게이트 메커니즘을 추가하여 출력 조절
# 역할: `RMSNormGated` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class RMSNormGated(nn.Module):
    # 기능: RMSNormGated scale parameter와 gate 적용용 epsilon을 초기화합니다.
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    # 기능: RMSNormGated에서 gate를 곱한 뒤 RMS 정규화를 적용합니다.
    def forward(self, x, gate=None):
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = (self.weight * x_normed.type_as(x))
        # 정규화된 값에 SiLU 게이트를 곱하여 출력 흐름 제어
        return x_normed * F.silu(gate.float()).type_as(x)


# L2 정규화: 벡터를 단위 크기로 정규화 (선형 어텐션의 쿼리/키에 사용)
# 기능: 선형 attention의 query/key vector를 L2 norm 기준으로 정규화합니다.
def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# PyTorch 기반 청크 게이트 델타 규칙 구현 (FLA 라이브러리 없을 때 대체용)
# 시퀀스를 청크 단위로 나누어 선형 어텐션을 효율적으로 계산하는 알고리즘
# q: 쿼리, k: 키, v: 값, g: 게이트(감쇠율), beta: 업데이트 강도
# 기능: PyTorch 연산만으로 chunked gated delta rule attention을 계산합니다.
def torch_chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=128, initial_state=None, output_final_state=False):
    # 입력 텐서를 (B, H, T, D) 형태로 변환하고 float으로 캐스팅
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    # 시퀀스 길이를 청크 크기의 배수로 패딩
    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = [F.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    T_pad = T + pad
    scale = Dk ** -0.5
    q = q * scale
    # beta로 가중된 키/값 계산
    v_beta, k_beta = v * beta.unsqueeze(-1), k * beta.unsqueeze(-1)
    # 청크 단위로 텐서 재구성
    q, k, v, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)]
    g = g.reshape(B, H, -1, chunk_size)
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    # 게이트 값의 누적합으로 감쇠 마스크 생성
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    # 청크 내부 어텐션 행렬 계산 (인과적 마스킹 적용)
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask_upper, 0)
    # 행 단위 누적 보정 (델타 규칙의 핵심 연산)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, device=attn.device, dtype=attn.dtype)
    v = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    # 순환 상태 S 초기화 (Dk x Dv 크기의 메모리 행렬)
    S = torch.zeros(B, H, Dk, Dv, device=v.device, dtype=v.dtype) if initial_state is None else initial_state.float()
    out = torch.zeros_like(v)
    mask_causal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    num_chunks = T_pad // chunk_size
    # 각 청크를 순차적으로 처리하며 상태 S를 업데이트
    for i in range(num_chunks):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        # 청크 내부 어텐션 (intra-chunk)
        attn_intra = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_causal, 0)
        # 이전 청크 상태에서 값 복원
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        # 청크 간 어텐션 (inter-chunk): 이전 상태를 쿼리로 참조
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ S
        out[:, :, i] = attn_inter + attn_intra @ v_new
        # 상태 업데이트: 감쇠된 이전 상태 + 새로운 키-값 정보
        S = S * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    if not output_final_state:
        S = None
    # 패딩 제거 후 원래 시퀀스 길이로 복원
    out = out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2).contiguous().to(q.dtype)
    return out, S


# GatedDeltaNet: 소프트맥스 어텐션을 대체하는 선형 어텐션 모듈
# 델타 규칙 기반의 순환 상태 업데이트로 O(n) 복잡도의 시퀀스 처리 실현
# 1D 인과 합성곱 + 게이트 메커니즘 + L2 정규화된 쿼리/키 사용
# 역할: `GatedDeltaNet` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class GatedDeltaNet(nn.Module):
    # 기능: GatedDeltaNet QKV projection, causal conv, gate, beta/decay projection을 초기화합니다.
    def __init__(self, config:
        MiniMindConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads    # 값 헤드 수
        self.num_k_heads = config.linear_num_key_heads      # 키 헤드 수
        self.head_k_dim = config.linear_key_head_dim        # 키 헤드 차원
        self.head_v_dim = config.linear_value_head_dim      # 값 헤드 차원
        self.key_dim = self.head_k_dim * self.num_k_heads   # 전체 키 차원
        self.value_dim = self.head_v_dim * self.num_v_heads # 전체 값 차원
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        # 1D 인과 합성곱: QKV를 시간 축으로 혼합하여 지역 패턴 포착
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1)
        # 시간 감쇠 파라미터 (Mamba 스타일)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log_())
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        # 프로젝션 레이어들
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)     # 출력 프로젝션
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)  # QKV 프로젝션
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)    # 게이트 z 프로젝션
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # beta (업데이트 강도) 프로젝션
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # 감쇠율 프로젝션

    # 순전파: AMP 비활성화 상태에서 float32로 연산 (수치 안정성 확보)
    # 기능: GatedDeltaNet에서 Gated Delta Rule 선형 attention을 계산합니다.
    def forward(self, x, conv_state=None, recurrent_state=None, use_cache=False):
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self._forward(x.float(), conv_state, recurrent_state, use_cache, input_dtype)

    # 기능: _forward 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
    def _forward(self, x, conv_state, recurrent_state, use_cache, input_dtype):
        B, T, _ = x.shape
        w, b = self.conv1d.weight.squeeze(1), self.conv1d.bias
        # 순환 모드: 캐시가 있고 시퀀스 길이가 1이면 토큰 단위 추론
        use_recurrent = conv_state is not None and T == 1
        # 입력을 Q, K, V로 프로젝션
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(B, T, -1, self.head_v_dim)       # 게이트 값
        beta, a = self.in_proj_b(x).sigmoid(), self.in_proj_a(x)       # 업데이트 강도와 감쇠 파라미터
        if use_recurrent:
            # 순환 모드: 합성곱 상태를 단일 스텝으로 업데이트
            if causal_conv1d_update is not None and x.is_cuda:
                mixed_qkv = causal_conv1d_update(mixed_qkv, conv_state, w, b, "silu")
            else:
                # CUDA 가속 없을 때 수동 합성곱 업데이트
                xc = torch.cat([conv_state, mixed_qkv], dim=-1)
                conv_state.copy_(xc[:, :, 1:])
                mixed_qkv = (xc * w).sum(-1, keepdim=True)
                if b is not None: mixed_qkv = mixed_qkv + b.unsqueeze(-1)
                mixed_qkv = F.silu(mixed_qkv)
        else:
            # 병렬 모드: 전체 시퀀스에 인과 합성곱 적용
            if use_cache:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - 1 - mixed_qkv.shape[-1], 0))[:, :, -(self.conv_kernel_size - 1):]
            if causal_conv1d_fn is not None and mixed_qkv.is_cuda:
                mixed_qkv = causal_conv1d_fn(x=mixed_qkv, weight=w, bias=b, activation="silu")
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :T])
        # QKV 분리 후 L2 정규화 적용 (선형 어텐션의 핵심)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = l2norm(q.reshape(B, T, -1, self.head_k_dim))
        k = l2norm(k.reshape(B, T, -1, self.head_k_dim))
        v = v.reshape(B, T, -1, self.head_v_dim)
        # 게이트 값 계산: A_log와 dt_bias를 이용한 지수 감쇠율
        g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias))
        # GQA 스타일: 키 헤드 수가 적으면 반복하여 값 헤드 수에 맞춤
        if self.num_v_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if use_recurrent:
            # 순환 모드: 단일 토큰 처리 (추론 시 사용)
            if fused_recurrent_gated_delta_rule is not None and q.is_cuda:
                try:
                    out, recurrent_state = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=recurrent_state, output_final_state=use_cache)
                except Exception as e:
                    logger.warning_once(f"FLA fused_recurrent 커널 실패: {e}, PyTorch로 대체합니다.")
                    out = None
            else:
                out = None
            if out is None:
                # PyTorch 기본 순환 연산: 상태를 감쇠하고 델타 규칙으로 업데이트
                scale = q.shape[-1] ** -0.5
                q_t, k_t, v_t = q.squeeze(1) * scale, k.squeeze(1), v.squeeze(1)
                recurrent_state = recurrent_state * g.squeeze(1).exp().unsqueeze(-1).unsqueeze(-1)
                kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
                recurrent_state = recurrent_state + k_t.unsqueeze(-1) * ((v_t - kv_mem) * beta.squeeze(1).unsqueeze(-1)).unsqueeze(-2)
                out = (recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2).unsqueeze(1)
        else:
            # 병렬 모드: 청크 단위로 선형 어텐션 계산 (학습 시 사용)
            if chunk_gated_delta_rule is not None and q.is_cuda:
                try:
                    out, recurrent_state = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=use_cache)
                except Exception as e:
                    logger.warning_once(f"FLA 청크 커널 실패: {e}, PyTorch로 대체합니다.")
                    out, recurrent_state = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=use_cache)
            else:
                out, recurrent_state = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=use_cache)
        # 게이트 RMSNorm 적용 후 출력 프로젝션
        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        out = self.out_proj(out.reshape(B, T, -1))
        if out.dtype != input_dtype: out = out.to(input_dtype)
        return out, (conv_state, recurrent_state) if use_cache else None


# RoPE(회전 위치 임베딩) 주파수 사전 계산
# YaRN 스케일링 지원으로 학습 시 최대 길이를 초과하는 시퀀스 처리 가능
# 기능: precompute_freqs_cis 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def precompute_freqs_cis(dim:
    int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 기본 주파수 계산: 1 / (base^(2i/d))
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        # YaRN 스케일링: 저주파/고주파 성분에 서로 다른 스케일링 적용
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 램프 함수로 저주파(외삽)/고주파(보간) 사이를 부드럽게 전환
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    # 위치별 cos/sin 값 계산
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


# RoPE 적용: 쿼리와 키에 회전 위치 임베딩을 곱하여 위치 정보 주입
# 기능: apply_rotary_pos_emb 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # 벡터의 앞/뒤 절반을 교환하고 부호 반전하여 회전 행렬 효과 구현
    # 기능: RoPE 회전을 위해 hidden 차원의 앞뒤 절반을 교차하고 부호를 반전합니다.
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


# GQA용 KV 반복: KV 헤드 수가 쿼리 헤드 수보다 적을 때 반복하여 맞춤
# 기능: repeat_kv 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def repeat_kv(x:
    torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


# 표준 소프트맥스 어텐션: full_attention_interval 간격의 레이어에서 사용
# GQA(그룹 쿼리 어텐션) 지원, Flash Attention 자동 감지
# 역할: `Attention` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class Attention(nn.Module):
    # 기능: Attention Q/K/V/O projection, head 구성, dropout, cache 설정을 초기화합니다.
    def __init__(self, config:
        MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads         # 쿼리 헤드 수
        self.n_local_kv_heads = self.num_key_value_heads        # KV 헤드 수
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # GQA 반복 횟수
        self.head_dim = config.head_dim
        # Q, K, V, O 프로젝션 레이어
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # PyTorch SDPA(Flash Attention) 사용 가능 여부 확인
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    # 순전파: RoPE 적용, KV 캐시 처리, 어텐션 점수 계산
    # 기능: Attention에서 RoPE가 적용된 causal self-attention을 계산합니다.
    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        # RoPE 위치 임베딩 적용
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # KV 캐시 연결 (자기회귀 생성 시)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # GQA: KV 헤드를 반복하여 쿼리 헤드 수에 맞춤
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention 경로 (효율적인 인과적 어텐션)
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 표준 어텐션 경로: 점수 계산 -> 인과적 마스킹 -> 소프트맥스
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


# SwiGLU 피드포워드 네트워크: gate_proj * up_proj 구조로 LLaMA 스타일 FFN
# 역할: `FeedForward` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class FeedForward(nn.Module):
    # 기능: FeedForward gate/up/down projection으로 구성된 FFN을 초기화합니다.
    def __init__(self, config:
        MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)   # 게이트 프로젝션
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)   # 다운 프로젝션
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)     # 업 프로젝션
        self.act_fn = ACT2FN[config.hidden_act]

    # 기능: FeedForward에서 SiLU gate와 up projection을 곱해 FFN 출력을 만듭니다.
    def forward(self, x):
        # SwiGLU: activation(gate(x)) * up(x) -> down 프로젝션
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# MoE (혼합 전문가) 피드포워드: 라우터가 토큰별로 최적의 전문가를 선택
# 각 토큰은 top-k 전문가만 활성화하여 계산 효율성 확보
# 역할: `MOEFeedForward` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class MOEFeedForward(nn.Module):
    # 기능: MOEFeedForward 전문가 FFN 목록과 token routing gate를 초기화합니다.
    def __init__(self, config:
        MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # 라우터 게이트
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.act_fn = ACT2FN[config.hidden_act]

    # 기능: MOEFeedForward에서 token별 top-k 전문가 출력을 가중합합니다.
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        # 라우터: 각 토큰에 대한 전문가별 점수 계산
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        # 각 전문가에 해당 토큰을 라우팅하여 가중 합산
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 학습 시 미사용 전문가도 그래디언트 흐름 유지
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        # 보조 손실: 전문가 간 부하 균형을 유도
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)


# 트랜스포머 블록: 어텐션(선형 또는 풀) + FFN으로 구성
# 레이어 유형에 따라 GatedDeltaNet 또는 표준 Attention 사용
# 역할: `MiniMindBlock` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class MiniMindBlock(nn.Module):
    # 기능: MiniMindBlock attention, FFN/MoE, norm 계층을 한 블록으로 초기화합니다.
    def __init__(self, layer_id:
        int, config: MiniMindConfig):
        super().__init__()
        self.layer_type = config.layer_types[layer_id]
        # 레이어 유형에 따라 선형 어텐션 또는 풀 어텐션 선택
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_id)
        else:
            self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MoE 사용 여부에 따라 FFN 선택
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    # 기능: MiniMindBlock에서 attention residual과 FFN/MoE residual을 차례로 적용합니다.
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 잔차 연결을 위해 입력 저장
        residual = hidden_states
        if self.layer_type == "linear_attention":
            # 선형 어텐션: 합성곱 상태와 순환 상태를 캐시로 사용
            conv_state = past_key_value[0] if past_key_value is not None else None
            recurrent_state = past_key_value[1] if past_key_value is not None else None
            hidden_states, present_key_value = self.linear_attn(
                self.input_layernorm(hidden_states), conv_state, recurrent_state, use_cache
            )
        else:
            # 풀 어텐션: 표준 KV 캐시 사용
            hidden_states, present_key_value = self.self_attn(
                self.input_layernorm(hidden_states), position_embeddings,
                past_key_value, use_cache, attention_mask
            )
        # 잔차 연결 + FFN
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# MiniMind 본체 모델: 임베딩 + 트랜스포머 블록 스택 + 최종 정규화
# 선형 어텐션과 풀 어텐션 레이어를 교대로 배치
# 역할: `MiniMindModel` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class MiniMindModel(nn.Module):
    # 기능: MiniMindModel token embedding, Transformer blocks, final norm, RoPE cache를 초기화합니다.
    def __init__(self, config:
        MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 토큰 임베딩
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)         # 최종 정규화
        # RoPE 주파수 사전 계산 후 버퍼로 등록
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # 기능: MiniMindModel에서 token embedding을 전체 Transformer block에 통과시킵니다.
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 풀 어텐션 레이어의 KV 캐시에서 시작 위치 결정
        start_pos = 0
        for i, lt in enumerate(self.config.layer_types):
            if lt == "full_attention" and past_key_values[i] is not None:
                start_pos = past_key_values[i][0].shape[1]
                break
        # 토큰 임베딩 + 드롭아웃
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 현재 위치에 해당하는 RoPE 임베딩 슬라이스
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        # 모든 트랜스포머 블록을 순차적으로 통과
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        # MoE 레이어들의 보조 손실 합산
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


# 역할: `MiniMindForCausalLM` 관련 설정, 하위 모듈, 실행 상태를 하나의 객체로 묶어 관리합니다.
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    # 기능: MiniMindForCausalLM 본체 모델과 LM head를 연결합니다.
    def __init__(self, config:
        MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    # 기능: MiniMindForCausalLM에서 hidden state를 vocab logits로 변환하고 labels가 있으면 LM loss를 계산합니다.
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    # 기능: generate 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = 0
            if past_key_values:
                for i, lt in enumerate(self.model.config.layer_types):
                    if lt == "full_attention" and past_key_values[i] is not None:
                        past_len = past_key_values[i][0].shape[1]
                        break
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
