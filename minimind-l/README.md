# MiniMind-L (Linear Attention)

MiniMind의 Linear Attention 변형 모델로, 기존 Softmax Attention 레이어 일부를 **GatedDeltaNet** 기반 Linear Attention으로 교체한 하이브리드 아키텍처입니다.

> 출처: https://github.com/jingyaogong/minimind/discussions/704

---

## 설치

```bash
# 기본 의존성 (필수)
pip install torch transformers

# 선택적 가속 라이브러리 (강력 권장 - 학습 속도 차이가 큼)
pip install flash-linear-attention   # Triton 기반 linear attention 가속
pip install causal-conv1d --no-build-isolation  # CUDA conv1d 가속
```

가속 라이브러리가 없어도 PyTorch 네이티브 fallback으로 정상 동작합니다.

---

## 실행 방법

`run_linear.py`를 통해 기존 minimind 학습/평가 스크립트를 그대로 사용할 수 있습니다.

### 학습 (SFT)

```bash
# 단일 GPU
python run_linear.py trainer/train_full_sft.py --save_weight sft_linear --from_weight full_sft

# 멀티 GPU (N = GPU 수)
torchrun --nproc_per_node N run_linear.py trainer/train_full_sft.py --save_weight sft_linear --from_weight full_sft
```

### 평가

```bash
python run_linear.py eval_llm.py --weight sft_linear
```

---

## 파일 구조 및 설명

```
minimind-l/
├── README.md                              # 본 문서
├── run_linear.py                          # 실행 엔트리포인트
└── model/
    └── model_minimind_linear.py           # Linear Attention 모델 정의
```

### `run_linear.py`

기존 minimind 코드베이스를 수정하지 않고 Linear Attention 모델을 사용하기 위한 엔트리포인트입니다.
`model.model_minimind` 모듈 import를 `model.model_minimind_linear`로 동적 리다이렉트하여,
기존 학습/평가 스크립트(`train_full_sft.py`, `eval_llm.py` 등)를 그대로 재사용할 수 있게 합니다.

### `model/model_minimind_linear.py`

Linear Attention(GatedDeltaNet)이 적용된 MiniMind 모델의 전체 구현 파일입니다.
기본 설정에서 8개 레이어 중 매 4번째 레이어만 Full Attention을 사용하고, 나머지 레이어는 Linear Attention을 사용합니다.

---

## 클래스 및 함수 설명

### Config

| 이름 | 설명 |
|------|------|
| `MiniMindConfig` | 모델의 모든 하이퍼파라미터를 관리하는 설정 클래스. `PretrainedConfig`를 상속하며, hidden_size, num_hidden_layers 등 기본 설정 외에 GatedDeltaNet 전용 설정(`full_attention_interval`, `linear_conv_kernel_dim`, `linear_key_head_dim` 등)과 MoE 설정을 포함합니다. `full_attention_interval`에 따라 각 레이어의 타입(`full_attention` / `linear_attention`)을 자동 결정합니다. |

### 정규화 (Normalization)

| 이름 | 설명 |
|------|------|
| `RMSNorm` | Root Mean Square Layer Normalization. LayerNorm 대비 평균 계산을 생략하여 효율적입니다. Full Attention 레이어와 모델 전반에서 사용됩니다. |
| `RMSNormGated` | Gate 메커니즘이 추가된 RMSNorm. 입력 `x`를 정규화한 후 `gate` 신호에 SiLU를 적용하여 곱합니다. GatedDeltaNet의 출력 정규화에 사용됩니다. |

### Linear Attention 핵심 컴포넌트

| 이름 | 설명 |
|------|------|
| `l2norm` | L2 정규화 함수. GatedDeltaNet에서 Query와 Key 벡터를 단위 구 위에 매핑하는 데 사용됩니다. |
| `torch_chunk_gated_delta_rule` | Gated Delta Rule의 PyTorch 네이티브 구현. 시퀀스를 chunk 단위로 분할하여 linear attention을 수행합니다. `flash-linear-attention` 라이브러리가 없을 때 fallback으로 사용됩니다. chunk 내부에서는 인트라 어텐션을, chunk 간에는 재귀적 상태(S)를 통해 인터 어텐션을 계산합니다. |
| `GatedDeltaNet` | Linear Attention의 핵심 모듈. 1D Causal Convolution으로 로컬 패턴을 포착한 뒤, Gated Delta Rule로 장기 의존성을 학습합니다. 학습 시에는 chunk 기반 병렬 처리를, 추론 시에는 재귀(recurrent) 모드로 효율적 디코딩을 수행합니다. `causal-conv1d`와 `flash-linear-attention` 라이브러리가 있으면 자동으로 가속 커널을 사용합니다. |

### Full Attention 컴포넌트

| 이름 | 설명 |
|------|------|
| `precompute_freqs_cis` | RoPE(Rotary Position Embedding)의 cos/sin 주파수를 사전 계산합니다. YaRN 스케일링을 지원하여 학습 시보다 긴 시퀀스에서도 외삽이 가능합니다. |
| `apply_rotary_pos_emb` | 사전 계산된 cos/sin 값을 Query와 Key에 적용하여 위치 정보를 부여합니다. |
| `repeat_kv` | GQA(Grouped Query Attention)를 위해 Key/Value 헤드를 Query 헤드 수만큼 반복 확장합니다. |
| `Attention` | 표준 Multi-Head Attention (GQA 지원). QKV Norm, RoPE, Flash Attention(`scaled_dot_product_attention`)을 적용하며, KV Cache를 통한 효율적 추론을 지원합니다. 하이브리드 구조에서 매 N번째 레이어(`full_attention_interval`)에 배치됩니다. |

### Feed-Forward 네트워크

| 이름 | 설명 |
|------|------|
| `FeedForward` | SwiGLU 구조의 FFN. `gate_proj`와 `up_proj`를 SiLU 활성화로 결합한 후 `down_proj`로 차원을 복원합니다. |
| `MOEFeedForward` | Mixture-of-Experts FFN. 라우터 게이트가 각 토큰을 Top-K 전문가에 분배하며, 학습 시 로드 밸런싱을 위한 auxiliary loss를 계산합니다. `use_moe=True`일 때 FeedForward 대신 사용됩니다. |

### 모델 구조

| 이름 | 설명 |
|------|------|
| `MiniMindBlock` | 단일 Transformer 블록. `layer_types` 설정에 따라 `GatedDeltaNet`(linear attention) 또는 `Attention`(full attention) 중 하나를 선택하고, Pre-Norm 잔차 연결 후 FFN(또는 MoE)을 적용합니다. |
| `MiniMindModel` | Transformer 백본. 토큰 임베딩, N개의 `MiniMindBlock` 레이어 스택, 최종 RMSNorm으로 구성됩니다. RoPE 주파수를 버퍼로 미리 계산하여 등록하고, Full Attention 레이어의 KV Cache로부터 위치 오프셋을 자동 계산합니다. |
| `MiniMindForCausalLM` | 최종 Causal Language Model. `MiniMindModel` 위에 `lm_head`(언어 모델 헤드)를 추가하며, 임베딩 가중치를 공유(weight tying)합니다. Cross-entropy loss 계산과 Top-K/Top-P 샘플링 기반 텍스트 생성(`generate`)을 지원합니다. |

---

## 아키텍처 요약

```
Layer 0: Linear Attention (GatedDeltaNet)
Layer 1: Linear Attention (GatedDeltaNet)
Layer 2: Linear Attention (GatedDeltaNet)
Layer 3: Full Attention   (Softmax + RoPE + GQA)  ← 매 4번째
Layer 4: Linear Attention (GatedDeltaNet)
Layer 5: Linear Attention (GatedDeltaNet)
Layer 6: Linear Attention (GatedDeltaNet)
Layer 7: Full Attention   (Softmax + RoPE + GQA)  ← 매 4번째
```

- **Linear Attention 레이어**: O(T) 복잡도로 긴 시퀀스를 효율적으로 처리
- **Full Attention 레이어**: 정확한 전역 관계 포착을 위해 주기적으로 배치
- 두 방식의 장점을 결합한 하이브리드 구조
