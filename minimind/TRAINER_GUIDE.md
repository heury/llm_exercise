# MiniMind Trainer 정리

이 문서는 `minimind/trainer` 아래의 학습 스크립트를 읽기 쉽게 정리한 안내서입니다.
기준이 되는 스크립트는 `train_pretrain.py`이며, 나머지 trainer는 pretrain 대비 어떤 목적과 로직이 달라지는지 중심으로 설명합니다.

## 전체 학습 흐름 요약

MiniMind의 일반적인 학습 파이프라인은 다음 순서로 이어집니다.

1. `train_tokenizer.py`: tokenizer를 새로 만들 때만 사용하는 참고용 스크립트
2. `train_pretrain.py`: 일반 텍스트로 언어 모델의 기본 next-token 예측 능력 학습
3. `train_full_sft.py`: 대화 형식 데이터로 supervised fine-tuning 수행
4. `train_lora.py`: 전체 모델 대신 LoRA 어댑터만 학습하는 경량 SFT
5. `train_dpo.py`: chosen/rejected 선호 데이터로 직접 선호 최적화
6. `train_ppo.py`, `train_grpo.py`: 보상 모델 기반 RLHF/RLAIF
7. `train_agent.py`: 도구 호출이 포함된 agentic RL
8. `train_distillation.py`: teacher 모델의 출력을 student 모델로 증류

공통적으로 대부분의 trainer는 다음 유틸리티를 공유합니다.

- `init_distributed_mode`: DDP 환경 초기화
- `setup_seed`: 재현 가능한 학습을 위한 시드 고정
- `init_model`: tokenizer와 `MiniMindForCausalLM` 로드
- `lm_checkpoint`: 학습 가중치와 resume 상태 저장/복원
- `SkipBatchSampler`: 중단 지점 이후 step부터 이어서 학습
- `Logger`, `is_main_process`: rank 0 중심 로깅

## 사전 준비
- uv sync
- uv tool install huggingface_hub
- hf download caspar/minimind_dataset --repo-type dataset --local-dir ./datasets

## train_pretrain.py

`train_pretrain.py`는 MiniMind 언어 모델을 처음부터 또는 기존 가중치에서 이어서 사전학습하는 기본 trainer입니다.
가장 단순한 형태의 학습 루프이며, 이후 SFT, DPO, PPO, GRPO 등은 이 구조를 바탕으로 데이터셋, 손실 함수, 모델 구성, rollout 또는 reward 계산을 추가합니다.

### 목적

일반 텍스트 데이터에서 다음 토큰을 예측하도록 모델을 학습합니다.
입력 토큰 `input_ids`를 모델에 넣고, 한 칸 shift된 `labels`에 대해 causal language modeling loss를 계산합니다.

### 사용 데이터

- 데이터셋 클래스: `PretrainDataset`
- 기본 경로: `./datasets/pretrain_english_wikitext2_mini.jsonl`
- 주요 처리:
  - 긴 텍스트를 tokenizer로 토큰화
  - `max_seq_len` 길이에 맞춰 자르거나 패딩
  - `input_ids`, `labels` 쌍을 반환

### 주요 인자

- `--save_weight pretrain`: 저장될 가중치 접두사
- `--epochs 2`: 전체 반복 횟수
- `--batch_size 32`: 배치 크기
- `--learning_rate 5e-4`: 초기 학습률
- `--accumulation_steps 8`: 그래디언트 누적 step 수
- `--max_seq_len 340`: 학습 시퀀스 길이
- `--use_moe 0`: MoE 모델 사용 여부
- `--from_weight none`: 시작 가중치. `none`이면 처음부터 학습
- `--from_resume 0`: resume checkpoint 자동 복원 여부
- `--use_compile 0`: `torch.compile` 사용 여부

### 전체 실행 흐름

1. 인자 파싱
   - 저장 경로, 모델 크기, 학습률, 데이터 경로, DDP/compile/resume 여부를 CLI 인자로 받습니다.

2. 분산 학습과 시드 초기화
   - `init_distributed_mode()`로 DDP 환경을 확인합니다.
   - DDP가 켜져 있으면 각 rank에 맞춰 `cuda:{local_rank}`를 사용합니다.
   - rank와 epoch를 섞어 seed를 고정해 데이터 순서를 제어합니다.

3. 모델 설정과 checkpoint 확인
   - `MiniMindConfig(hidden_size, num_hidden_layers, use_moe)`를 만듭니다.
   - `from_resume=1`이면 `lm_checkpoint(..., model=None)`을 호출해 resume 파일을 찾습니다.

4. mixed precision 설정
   - CPU면 `nullcontext`를 사용합니다.
   - CUDA면 `torch.cuda.amp.autocast(dtype=...)`를 사용합니다.
   - `float16`일 때는 `GradScaler`가 활성화됩니다.

5. wandb 또는 swanlab 로깅 준비
   - `--use_wandb`가 켜져 있고 main process이면 run을 생성합니다.
   - resume checkpoint에 저장된 `wandb_id`가 있으면 같은 run으로 이어서 기록합니다.

6. 모델, tokenizer, 데이터셋, optimizer 생성
   - `init_model(lm_config, args.from_weight, device=args.device)`로 모델과 tokenizer를 로드합니다.
   - `PretrainDataset`으로 학습 데이터를 구성합니다.
   - DDP 환경이면 `DistributedSampler`를 사용합니다.
   - optimizer는 `AdamW(model.parameters(), lr=args.learning_rate)`입니다.

7. resume 상태 복원
   - checkpoint가 있으면 model, optimizer, scaler 상태를 복원합니다.
   - 저장된 `epoch`, `step`부터 이어서 학습합니다.

8. compile과 DDP 래핑
   - `--use_compile=1`이면 `torch.compile(model)`을 적용합니다.
   - DDP 환경이면 `DistributedDataParallel(model, device_ids=[local_rank])`로 감쌉니다.

9. epoch 학습
   - epoch마다 sampler seed를 갱신합니다.
   - resume 시 `SkipBatchSampler`로 이미 처리한 batch를 건너뜁니다.
   - `train_epoch`를 호출합니다.

10. 분산 프로세스 종료
   - DDP가 켜져 있으면 barrier 후 process group을 정리합니다.

### train_epoch 상세 흐름

`train_epoch(epoch, loader, iters, start_step=0, wandb=None)`는 실제 학습 step을 수행합니다.

1. 배치 로드
   - `input_ids`, `labels`를 GPU/CPU device로 이동합니다.

2. learning rate 스케줄링
   - `get_lr(current_step, total_steps, base_lr)`로 cosine 형태의 learning rate를 계산합니다.
   - optimizer param group에 현재 lr을 반영합니다.

3. forward와 loss 계산
   - `model(input_ids, labels=labels)`를 호출합니다.
   - 기본 language modeling loss에 MoE 보조 손실 `res.aux_loss`를 더합니다.
   - gradient accumulation을 위해 loss를 `accumulation_steps`로 나눕니다.

4. backward
   - `scaler.scale(loss).backward()`로 mixed precision backward를 수행합니다.

5. optimizer step
   - 현재 step이 `accumulation_steps`의 배수이면 optimizer step을 수행합니다.
   - scaler unscale 후 gradient clipping을 적용합니다.
   - `scaler.step`, `scaler.update`, `optimizer.zero_grad` 순서로 갱신합니다.

6. logging
   - `log_interval`마다 현재 loss, logits loss, aux loss, learning rate, epoch 남은 시간을 출력합니다.
   - wandb가 있으면 같은 값을 기록합니다.

7. checkpoint 저장
   - `save_interval`마다 main process에서만 저장합니다.
   - 모델 state dict는 half precision CPU tensor로 저장합니다.
   - `lm_checkpoint`로 resume용 optimizer/scaler/epoch/step도 함께 저장합니다.

8. 마지막 누적 gradient 처리
   - epoch 끝에 accumulation boundary에 닿지 않은 gradient가 남아 있으면 한 번 더 optimizer step을 수행합니다.

### 저장 산출물

- 일반 가중치: `./checkouts/pretrain_{hidden_size}.pth`
- MoE 가중치: `./checkouts/pretrain_{hidden_size}_moe.pth`
- resume checkpoint: `./checkouts/pretrain_{hidden_size}[_moe]_resume.pth`

### 실습
- 학습: uv run .\trainer\train_pretrain.py
- 테스트: uv run eval_llm.py --weight pretrain

## train_full_sft.py

`train_full_sft.py`는 pretrain 모델을 대화형 지시 데이터에 맞게 전체 파라미터로 미세조정합니다.

pretrain 대비 달라지는 점:

- 데이터셋이 `PretrainDataset`에서 `SFTDataset`으로 바뀝니다.
- 기본 데이터 경로가 `./datasets/sft_t2t_mini.jsonl`입니다.
- 기본 시작 가중치가 `from_weight=pretrain`입니다.
- 저장 접두사가 `full_sft`입니다.
- 학습률이 `5e-4`에서 `1e-5`로 낮아집니다.
- `accumulation_steps` 기본값이 8에서 1로 줄어듭니다.
- `max_seq_len` 기본값이 340에서 768로 늘어납니다.
- 손실 구조는 pretrain과 거의 같지만, `SFTDataset`이 assistant 응답 구간만 label로 남기고 prompt/system/user 구간은 `-100`으로 마스킹합니다.

즉, 학습 루프는 pretrain과 거의 동일하고, 핵심 차이는 “일반 텍스트 next-token 학습”에서 “대화 응답 구간 supervised learning”으로 데이터와 label masking이 바뀐다는 점입니다.

### 실습
- 모델:hf download jingyaogong/minimind-3-pytorch pretrain_768.pth --local-dir ./checkouts
- 학습: uv run .\trainer\train_pretrain.py
- 테스트: uv run eval_llm.py --weight full_sft
- 
## train_lora.py

`train_lora.py`는 전체 모델을 업데이트하지 않고 LoRA 어댑터만 학습합니다.

pretrain 대비 달라지는 점:

- 데이터셋은 `SFTDataset`을 사용합니다.
- 기본 시작 가중치는 `from_weight=full_sft`입니다.
- `apply_lora(model)`로 모델의 선형 계층에 LoRA 모듈을 붙입니다.
- LoRA 파라미터만 `requires_grad=True`로 두고 나머지 파라미터는 동결합니다.
- optimizer는 전체 모델이 아니라 LoRA 파라미터만 받습니다.
- 저장은 일반 `lm_checkpoint` 중심이 아니라 `save_lora`로 LoRA 가중치만 별도 저장합니다.
- `torch.compile`과 monkey-patched LoRA forward가 충돌할 수 있어 compile을 자동으로 끄는 처리가 있습니다.

결과적으로 pretrain/full SFT보다 학습 가능한 파라미터 수가 매우 작고, 특정 도메인 데이터에 빠르게 적응시키는 용도입니다.

### 실습
- 모델:hf download jingyaogong/minimind-3-pytorch pretrain_768.pth --local-dir ./checkouts
- 학습: uv run .\trainer\train_pretrain.py
- 테스트: uv run eval_llm.py --weight lora_medical

## train_dpo.py

`train_dpo.py`는 선호 데이터의 chosen/rejected 응답 쌍을 이용해 DPO(Direct Preference Optimization)를 수행합니다.

pretrain 대비 달라지는 점:

- 데이터셋이 `DPODataset`으로 바뀝니다.
- 기본 시작 가중치는 `from_weight=full_sft`입니다.
- 학습 모델 외에 같은 초기 가중치의 `ref_model`을 하나 더 만들고 동결합니다.
- 입력은 `chosen_input_ids`, `rejected_input_ids`와 각각의 loss mask로 구성됩니다.
- 일반 CE loss 대신 `dpo_loss`를 사용합니다.
- `logits_to_log_probs`로 chosen/rejected 응답 구간의 token log probability를 합산합니다.
- policy와 reference의 chosen/rejected log-ratio 차이를 비교해 chosen 응답의 상대 선호도를 높입니다.
- 주요 추가 인자는 `--beta`이며, policy가 reference에서 얼마나 강하게 벗어날지 조절합니다.

pretrain이 “정답 토큰을 맞히기”라면, DPO는 “좋은 응답과 나쁜 응답의 상대적 선호 차이를 키우기”입니다.

### 실습
- 학습: uv run .\trainer\train_dpo.py
- 테스트: uv run eval_llm.py --weight dpo

## train_ppo.py

`train_ppo.py`는 reward model을 이용해 PPO 방식으로 actor 모델을 강화학습합니다.

pretrain 대비 달라지는 점:

- 데이터셋은 `RLAIFDataset`입니다.
- 기본 시작 가중치는 `from_weight=full_sft`입니다.
- 학습 대상 actor 모델 외에 다음 모델들이 추가됩니다.
  - `ref_model`: KL 기준이 되는 고정 reference model
  - `critic_model`: value를 예측하는 모델
  - `reward_model`: 외부 응답 품질 점수를 주는 모델
- `rollout_engine`이 추가되어 현재 actor로 응답을 생성합니다.
- 생성된 응답에 대해 휴리스틱 보상과 reward model 점수를 합산합니다.
- reward는 응답 마지막 토큰 위치에 붙이고, GAE로 advantage와 return을 계산합니다.
- actor는 PPO clipped objective로 업데이트합니다.
- critic은 value loss로 별도 업데이트합니다.
- KL penalty와 early stop으로 reference model에서 너무 멀어지는 것을 제한합니다.
- actor learning rate와 critic learning rate가 분리됩니다.

pretrain이 정적 데이터셋을 그대로 읽는 반면, PPO는 “현재 모델이 직접 답을 생성하고, 그 답을 평가해 다시 학습”하는 online RL 구조입니다.

## train_grpo.py

`train_grpo.py`는 GRPO(Group Relative Policy Optimization) 또는 CISPO 방식으로 RLAIF 학습을 수행합니다.

pretrain 대비 달라지는 점:

- 데이터셋은 `RLAIFDataset`입니다.
- 기본 시작 가중치는 `from_weight=full_sft`입니다.
- `rollout_engine`으로 프롬프트당 여러 응답을 생성합니다.
- `num_generations` 기본값은 6입니다.
- 각 응답에 대해 reward model 점수와 휴리스틱 보상을 계산합니다.
- 같은 프롬프트에서 생성된 여러 응답의 reward 평균/표준편차로 group-relative advantage를 만듭니다.
- PPO처럼 critic/value model을 두지 않습니다.
- reference model과의 KL penalty는 유지합니다.
- `loss_type`에 따라 두 손실을 선택합니다.
  - `grpo`: PPO식 ratio clipping
  - `cispo`: ratio 상한만 두는 CISPO 방식

PPO와 비교하면 critic이 없고, 한 프롬프트에서 여러 샘플을 뽑아 그룹 내부 상대 보상으로 advantage를 만드는 점이 핵심입니다.

### 실습
- 보상 모델: hf download internlm/internlm2-1_8b-reward --local-dir ./models/internlm2-1_8b-reward
- 학습: uv run .\trainer\train_grpo.py
- 테스트: uv run eval_llm.py --weight grpo

## train_agent.py

`train_agent.py`는 tool call이 포함된 agent 행동을 GRPO/CISPO 방식으로 학습합니다.

pretrain 대비 달라지는 점:

- 데이터셋은 `AgentRLDataset`입니다.
- 입력 샘플이 단순 prompt가 아니라 `messages`, `tools`, `gt`를 포함합니다.
- rollout이 단일 응답 생성에서 multi-turn agent 실행으로 바뀝니다.
- 모델 응답에서 `<tool_call>...</tool_call>`을 파싱합니다.
- tool 이름과 arguments를 검증한 뒤 실제 도구 함수를 실행하고, 결과를 `role=tool` 메시지로 다시 대화에 넣습니다.
- 최대 `max_turns` 동안 모델 응답과 도구 결과를 반복합니다.
- 보상은 다음 요소를 합칩니다.
  - tool call 태그 형식
  - 유효한 도구 이름 사용
  - arguments 파싱 성공 여부
  - GT 정답과 도구 결과 일치도
  - 최종 응답 품질
  - 반복 패널티
  - reward model 점수
- 업데이트 손실은 GRPO/CISPO 계열로 `train_grpo.py`와 유사합니다.

즉, `train_grpo.py`가 “답변 품질” 중심 RL이라면, `train_agent.py`는 “도구를 호출하고 관찰 결과를 반영하는 행동”까지 학습합니다.

## train_distillation.py

`train_distillation.py`는 teacher 모델의 분포를 student 모델에 전달하는 지식 증류 trainer입니다.

pretrain 대비 달라지는 점:

- 데이터셋은 `SFTDataset`입니다.
- student와 teacher 두 모델을 만듭니다.
- teacher는 보통 더 크거나 MoE 모델이며, 학습 중 동결됩니다.
- student는 학습 대상입니다.
- 손실은 두 가지를 섞습니다.
  - hard label CE loss: 실제 label에 대한 일반 supervised loss
  - soft distillation loss: teacher logits와 student logits의 KL divergence
- `alpha`로 hard/soft 손실 비율을 조절합니다.
- `temperature`로 teacher/student softmax 분포를 부드럽게 만듭니다.
- 기본 설정은 student dense 모델, teacher MoE 모델을 가정합니다.

pretrain이 정답 토큰만 보고 학습한다면, distillation은 teacher가 각 토큰 후보에 부여한 확률 분포까지 따라 배우게 합니다.

### 실습
- reference 모델: hf download jingyaogong/minimind-3-pytorch full_sft_768_moe.pth --local-dir ./checkouts
- 학습: uv run .\trainer\train_distillation.py
- 테스트: uv run eval_llm.py --weight full_dist

## train_tokenizer.py

`train_tokenizer.py`는 모델 가중치 학습 trainer라기보다 tokenizer 생성/검증용 스크립트입니다.

pretrain 대비 달라지는 점:

- PyTorch 모델 학습 루프가 없습니다.
- `tokenizers` 라이브러리로 BPE tokenizer를 학습합니다.
- special token과 buffer token을 정의합니다.
- 학습 후 chat template, encode/decode 일관성, 압축률, 스트리밍 decode 등을 확인합니다.
- MiniMind에는 기본 tokenizer가 이미 포함되어 있으므로 일반 학습 흐름에서는 다시 학습하지 않는 것이 좋습니다.

tokenizer를 바꾸면 기존 모델 가중치와 token id 체계가 맞지 않기 때문에, 실험 목적이 아니라면 `train_pretrain.py`부터 시작하는 편이 안전합니다.

## trainer_utils.py와 rollout_engine.py의 역할

### trainer_utils.py

각 trainer가 반복해서 쓰는 공통 기능을 제공합니다.

- 모델 파라미터 수 로깅
- DDP main process 판별
- cosine learning rate 계산
- 분산 학습 초기화
- seed 고정
- checkpoint 저장/복원
- 모델/tokenizer 초기화
- resume 시 batch skip sampler
- reward model wrapper

### rollout_engine.py

PPO, GRPO, Agent RL에서 사용하는 생성 백엔드를 추상화합니다.

- `TorchRolloutEngine`: 현재 PyTorch 모델의 `generate`를 직접 호출
- `SGLangRolloutEngine`: SGLang 서버에 HTTP 요청을 보내 생성
- `compute_per_token_logps`: 생성된 토큰의 log probability 계산
- `create_rollout_engine`: CLI 인자에 따라 torch/sglang 엔진 선택

pretrain/SFT 계열은 정적 데이터셋을 읽어 바로 loss를 계산하지만, RL 계열은 먼저 모델이 응답을 생성해야 하므로 rollout engine이 필요합니다.

## trainer별 핵심 차이 표

| 파일 | 목적 | 데이터셋 | 시작 가중치 | 핵심 손실/알고리즘 | pretrain 대비 핵심 차이 |
| --- | --- | --- | --- | --- | --- |
| `train_pretrain.py` | 기본 언어 모델 사전학습 | `PretrainDataset` | `none` | CE + MoE aux loss | 기준 학습 루프 |
| `train_full_sft.py` | 대화 응답 SFT | `SFTDataset` | `pretrain` | CE + MoE aux loss | assistant 응답 구간만 학습 |
| `train_lora.py` | LoRA 경량 SFT | `SFTDataset` | `full_sft` | CE | LoRA 파라미터만 학습 |
| `train_dpo.py` | 선호 최적화 | `DPODataset` | `full_sft` | DPO loss | chosen/rejected 비교, ref model 사용 |
| `train_ppo.py` | reward 기반 RL | `RLAIFDataset` | `full_sft` | PPO actor loss + value loss | rollout, reward model, critic 추가 |
| `train_grpo.py` | group-relative RL | `RLAIFDataset` | `full_sft` | GRPO/CISPO | 여러 샘플의 그룹 상대 보상 사용, critic 없음 |
| `train_agent.py` | tool-use agent RL | `AgentRLDataset` | `full_sft` | GRPO/CISPO | tool call 실행/검증과 multi-turn rollout |
| `train_distillation.py` | teacher-student 증류 | `SFTDataset` | `full_sft` 계열 | CE + KL distillation | teacher logits 분포를 student가 모방 |
| `train_tokenizer.py` | tokenizer 학습/검증 | 텍스트 iterator | 없음 | BPE tokenizer 학습 | 모델 학습이 아니라 tokenizer 생성 |

