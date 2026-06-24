# MiniMind-D (Diffusion Language Model)

MiniMind 기반의 **Discrete Diffusion Language Model (dLLM)** 구현체입니다.
기존 autoregressive(AR) 방식 대신, 마스킹된 토큰을 반복적으로 디노이징하여 텍스트를 생성합니다.

> 출처: [minimind discussions #618](https://github.com/jingyaogong/minimind/discussions/618)

## 디렉토리 구조

```
minimind-d/
├── model/
│   └── model_minimind_dllm.py   # dLLM 모델 정의
├── trainer/
│   └── train_dllm.py            # 훈련 스크립트
├── scripts/
│   ├── chat_dllm.py             # TUI 채팅 (Rich 기반)
│   └── web_dllm.py              # 웹 데모 (Flask + SSE)
├── eval_dllm.py                 # 평가/추론 스크립트
└── README.md
```

## 의존성

```
torch
transformers
rich          # chat_dllm.py용
flask         # web_dllm.py용
```

---

## 파일별 설명

### 1. `model/model_minimind_dllm.py` — 모델 정의

dLLM의 핵심 모델 파일입니다. MiniMind의 AR 모델을 상속받아 양방향(bidirectional) attention으로 변경하고,
마스크 기반 디퓨전 훈련/생성 로직을 추가합니다.

### 2. `trainer/train_dllm.py` — 훈련 스크립트

SFT 데이터셋으로 dLLM을 훈련합니다. 기존 AR 가중치(`full_sft`)를 초기값으로 로드한 뒤,
마스크 디퓨전 목적함수로 파인튜닝합니다.

**주요 기능:**
- 분산 훈련 (DDP) 지원
- QK-only 파인튜닝 (`--train_qk_only 1`): q_proj, k_proj만 학습하여 causal → bidirectional attention 전환
- 체크포인트 이어서 훈련 (`--from_resume 1`)
- 혼합 정밀도 (bfloat16/float16)
- swanlab(wandb) 로깅

### 3. `eval_dllm.py` — 평가/추론

훈련된 dLLM 가중치를 로드하여 대화형 추론을 수행합니다.
자동 테스트 모드(미리 정의된 프롬프트)와 수동 입력 모드를 지원합니다.

### 4. `scripts/chat_dllm.py` — TUI 채팅 인터페이스

Rich 라이브러리 기반 터미널 UI로, 디노이징 과정을 실시간으로 시각화합니다.
마스크 토큰(`[M]`)이 점진적으로 실제 토큰으로 대체되는 과정을 색상으로 표시합니다.

- 빨간색: 아직 마스킹된 토큰
- 밝은 초록색: 방금 드러난 새 토큰
- 흰색: 이미 확정된 토큰

### 5. `scripts/web_dllm.py` — 웹 데모

Flask 서버 + SSE(Server-Sent Events) 스트리밍으로 브라우저에서 디노이징 과정을 실시간 관찰할 수 있습니다.
다크 테마 UI에서 토큰 뷰/텍스트 뷰 전환이 가능합니다.

---

## 실행 방법

### 훈련

```bash
# 기본 훈련 (AR 가중치 기반으로 dLLM 파인튜닝)
cd trainer
python train_dllm.py --from_weight full_sft --epochs 5 --batch_size 32

# QK-only 파인튜닝 (빠른 전환 학습)
python train_dllm.py --from_weight full_sft --train_qk_only 1

# 분산 훈련 (멀티 GPU)
torchrun --nproc_per_node=2 train_dllm.py --from_weight full_sft

# 처음부터 훈련 (AR 가중치 없이)
python train_dllm.py --from_weight none

# 이어서 훈련
python train_dllm.py --from_resume 1
```

**주요 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--from_weight` | `full_sft` | 초기화에 사용할 AR 가중치 이름 (`none`이면 처음부터) |
| `--save_weight` | `dllm` | 저장할 가중치 접두사 |
| `--hidden_size` | `768` | 은닉층 차원 |
| `--num_hidden_layers` | `8` | Transformer 레이어 수 |
| `--max_seq_len` | `768` | 최대 시퀀스 길이 |
| `--train_qk_only` | `0` | QK-only 파인튜닝 활성화 |
| `--use_moe` | `0` | MoE 아키텍처 사용 여부 |
| `--data_path` | `../dataset/sft_t2t_mini.jsonl` | 훈련 데이터 경로 |

### 평가/추론

```bash
# 기본 추론
python eval_dllm.py

# 파라미터 조정
python eval_dllm.py --steps 64 --block_size 64 --temperature 0.3 --max_new_tokens 512

# 대화 히스토리 포함 (최근 4턴)
python eval_dllm.py --historys 4
```

### TUI 채팅

```bash
cd scripts
python chat_dllm.py

# 파라미터 조정
python chat_dllm.py --steps 64 --block_size 64 --temperature 0.3

# pretrain 가중치 사용
python chat_dllm.py --weight dllm_pt
```

### 웹 데모

```bash
cd scripts
python web_dllm.py
# 브라우저에서 http://localhost:5001 접속
```

---

## 모델 클래스 및 함수 상세 (`model/model_minimind_dllm.py`)

### `add_gumbel_noise(logits, temperature)`

Gumbel 노이즈를 logits에 적용하는 함수입니다. 디노이징 생성 시 샘플링 다양성을 조절합니다.

- `temperature = 0`: 노이즈 없이 원본 logits 반환 (greedy)
- `temperature > 0`: Gumbel 분포 기반 노이즈 적용. 값이 클수록 랜덤성 증가
- 수치 안정성을 위해 float64로 변환하여 계산

**수식:** `logits.exp() / (-log(U))^temperature` (U ~ Uniform(0,1))

---

### `MiniMindDLLMConfig(MiniMindConfig)`

dLLM 전용 설정 클래스입니다. 기존 `MiniMindConfig`를 상속하며 다음 필드를 추가합니다.

| 필드 | 기본값 | 설명 |
|---|---|---|
| `mask_token_id` | `27` | 마스크 토큰의 vocabulary ID |
| `mask_epsilon` | `0.001` | 노이즈 스케줄의 최소 마스킹 확률. 0이면 노이즈가 아예 없는 상태가 가능 |

---

### `MiniMindDLLMModel(MiniMindModel)`

dLLM용 백본 모델입니다. `MiniMindModel`을 상속하며 핵심적인 차이점은
**모든 self-attention 레이어를 non-causal(양방향)로 전환**한다는 것입니다.

```python
for layer in self.layers: layer.self_attn.is_causal = False
```

#### `add_noise_to_tokens(input_ids, t, eps=None, pad_token_id=0)`

훈련 시 입력 토큰에 마스크 노이즈를 추가하는 forward diffusion 함수입니다.

**파라미터:**
- `input_ids` — 원본 토큰 시퀀스 `(batch_size, seq_len)`
- `t` — 타임스텝 텐서 `(batch_size,)`. 0~1 사이 값으로, 1에 가까울수록 더 많이 마스킹
- `eps` — 최소 마스킹 확률 (기본: `config.mask_epsilon`)
- `pad_token_id` — 패딩 토큰 ID (마스킹 대상에서 제외)

**반환값:**
- `noisy_input_ids` — 마스크가 적용된 토큰 시퀀스
- `corruption_mask` — 마스킹된 위치를 나타내는 boolean 마스크
- `p_mask` — 각 위치의 마스킹 확률

**노이즈 스케줄:** `p_mask = (1 - eps) * t + eps`
- `t=0` → `p_mask = eps` (거의 마스킹 없음)
- `t=1` → `p_mask = 1` (거의 전부 마스킹)

---

### `MiniMindForMaskedDiffusion(PreTrainedModel, GenerationMixin)`

dLLM의 최상위 모델 클래스입니다. HuggingFace `PreTrainedModel`과 `GenerationMixin`을 상속합니다.

**구조:**
- `self.model` — `MiniMindDLLMModel` (양방향 Transformer 백본)
- `self.lm_head` — 은닉 상태 → 어휘 확률 투영 (`Linear(hidden_size, vocab_size)`)
- embedding weight tying: `embed_tokens.weight = lm_head.weight`

#### `forward(input_ids, attention_mask, labels, corruption_mask, p_mask, n_valid, ...)`

모델의 순전파 함수입니다. 훈련 시 디퓨전 손실을 계산합니다.

**파라미터:**
- `input_ids` — 노이즈가 적용된 입력 토큰
- `attention_mask` — 어텐션 마스크
- `labels` — 원본 정답 토큰 (손실 계산용)
- `corruption_mask` — 마스킹된 위치 (boolean). 이 위치에서만 손실 계산
- `p_mask` — 각 위치의 마스킹 확률. 손실 가중치로 사용 (확률이 낮은 위치일수록 가중치 높음)
- `n_valid` — 유효 토큰 수 (손실 정규화 분모)

**손실 계산:**
```
loss = sum( CE(logits[masked], labels[masked]) / p_mask[masked] ) / n_valid
```
마스킹 확률(`p_mask`)로 나누어 **importance weighting**을 적용합니다.
낮은 확률로 마스킹된(= 거의 정답에 가까운 상태의) 토큰에 더 높은 가중치를 부여합니다.

**반환값:** `MaskedLMOutput(loss, logits, hidden_states)`

#### `generate(inputs, max_new_tokens, temperature, top_k, steps, block_size, cfg_scale, ...)`

반복적 디노이징으로 텍스트를 생성하는 추론 함수입니다.

**생성 과정:**

1. 프롬프트 뒤에 `max_new_tokens`개의 `[MASK]` 토큰을 배치
2. 생성 영역을 `block_size` 단위 블록으로 분할
3. 각 블록에서 `steps`번 디노이징 반복:
   - 전체 시퀀스를 모델에 입력하여 logits 계산
   - Gumbel 노이즈 + top-k 필터링으로 후보 토큰 샘플링
   - **confidence 기반 스케줄링**: 모델이 가장 확신하는 마스크 위치부터 우선 언마스킹
   - 한 스텝에 언마스킹할 개수: `남은 마스크 수 / 남은 스텝 수`
4. 모든 블록이 완료되면 최종 시퀀스 반환

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `max_new_tokens` | `128` | 생성할 최대 토큰 수 |
| `temperature` | `0.5` | Gumbel 노이즈 온도. 0이면 greedy, 높을수록 다양 |
| `top_k` | `50` | 상위 k개 토큰만 샘플링 후보로 유지 |
| `steps` | `128` | 블록당 디노이징 반복 횟수 |
| `block_size` | `max_new_tokens` | 블록 크기. 작을수록 semi-autoregressive에 가까움 |
| `cfg_scale` | `0.0` | Classifier-Free Guidance 강도. 0이면 비활성 |

**CFG(Classifier-Free Guidance):**
`cfg_scale > 0`이면 조건부/비조건부 logits를 함께 계산하여 조건부 생성을 강화합니다.
```
logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
```
