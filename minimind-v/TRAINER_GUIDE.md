# MiniMind-V Trainer 정리

이 문서는 `minimind-v/trainer` 아래의 학습 스크립트를 읽기 쉽게 정리한 안내서입니다.
기준이 되는 스크립트는 `train_pretrain_vlm.py`이며, `train_sft_vlm.py`는 pretrain 대비 어떤 점이 달라지는지 중심으로 설명합니다.

`minimind-v`는 [minimind](../minimind)의 순수 언어 모델(LLM)에 SigLIP2 비전 인코더와 vision projection을 붙여 이미지-텍스트 대화가 가능하도록 확장한 자매 프로젝트입니다. LLM 자체의 구조와 학습 방식은 `minimind/TRAINER_GUIDE.md`를 참고하세요.

## 전체 학습 흐름 요약

MiniMind-V의 학습 파이프라인은 다음 순서로 이어집니다.

1. `train_pretrain_vlm.py` (선택): 이미지-caption 쌍으로 vision projection을 LLM의 언어 공간에 정렬
2. `train_sft_vlm.py`: 대화형 이미지 지시 데이터로 SFT 수행

SFT 기본 데이터셋(`sft_vlm_english_flickr8k_mini.parquet`)은 pretrain 단계의 기본 데이터셋과 동일한 파일이므로, pretrain 단계는 완전히 건너뛰고 `--from_weight full_sft`로 SFT를 바로 시작해도 됩니다. 다만 pretrain을 먼저 한 번 돌리면 projection이 미리 정렬되어 SFT 수렴이 더 안정적입니다.

공통적으로 두 trainer는 다음 유틸리티를 공유합니다(`trainer_utils.py`).

- `init_distributed_mode`: DDP 환경 초기화
- `setup_seed`: 재현 가능한 학습을 위한 시드 고정
- `init_vlm_model`: tokenizer와 `MiniMindVLM` 로드, `freeze_llm` 정책에 따라 파라미터 동결
- `vlm_checkpoint`: 모델/옵티마이저/스케일러 상태 저장과 resume 복원(원자적 저장)
- `SkipBatchSampler`: 중단 지점 이후 step부터 이어서 학습
- `vlm_collate_fn`: `input_ids`/`labels`/`pixel_values`를 배치로 묶는 collate 함수
- `Logger`, `is_main_process`: rank 0 중심 로깅

## 사용 데이터 형식

두 trainer 모두 `VLMDataset`(`dataset/lm_dataset.py`)으로 parquet 파일을 읽습니다. parquet은 컬럼 두 개로 구성됩니다.

- `conversations`: 대화 JSON 문자열. `<image>` 자리표시자가 포함되며, 로드 시 `image_token_len`(기본 64)개의 `<|image_pad|>` 토큰으로 치환됩니다.
- `image_bytes`: JPEG로 인코딩된 원본 바이트가 그대로 저장된 binary 컬럼(별도 파일 경로나 base64 없이 raw bytes를 직접 담습니다).

## 사전 준비

- uv sync
- uv tool install huggingface_hub
- 데이터셋: `hf download caspar/minimind_dataset --repo-type dataset --local-dir ./datasets` (minimind와 동일한 저장소를 공유하며, 여기에 `sft_i2t.parquet`와 `sft_vlm_english_flickr8k_mini.parquet`가 포함되어 있습니다. 원본 `pretrain_i2t.parquet`는 이 저장소에 없으므로 `train_pretrain_vlm.py`의 기본 `--data_path`는 대신 `sft_vlm_english_flickr8k_mini.parquet`(동일한 `conversations`/`image_bytes` 스키마)를 가리키도록 되어 있습니다. `train_sft_vlm.py`의 기본 `--data_path`도 같은 `sft_vlm_english_flickr8k_mini.parquet`를 사용하며, 더 큰 `sft_i2t.parquet`로 학습하려면 `--data_path`로 직접 지정하세요.)
- 비전 인코더: `hf download jingyaogong/siglip2-base-p32-256-ve --local-dir ./models/siglip2-base-p32-256-ve`
- 베이스 언어 모델 가중치(VLM 초기화용): `minimind`의 `train_full_sft.py`로 학습해 만든 `minimind/checkouts/full_sft_768.pth`를 `minimind-v/checkouts`로 복사하거나, `hf download jingyaogong/minimind-3-pytorch full_sft_768.pth --local-dir ./checkouts`로 받아옵니다.

준비가 끝나면 `minimind-v` 디렉터리 기준 구조는 다음과 같습니다(`minimind`과 별개로 `minimind-v` 자체에 `checkouts/`, `datasets/`, `models/`를 둡니다).

```text
minimind-v/
├── checkouts/
│   └── full_sft_768.pth
├── datasets/
│   └── sft_vlm_english_flickr8k_mini.parquet
├── models/
│   └── siglip2-base-p32-256-ve/
└── trainer/
    ├── train_pretrain_vlm.py
    ├── train_sft_vlm.py
    └── trainer_utils.py
```

trainer 스크립트의 `--save_dir`, `--data_path`, `init_vlm_model`의 `tokenizer_path`/`vision_model_path` 기본값은 모두 `minimind-v/` 를 작업 디렉터리로 실행한다는 가정 하에 `./checkouts`, `./datasets`, `./models`로 설정되어 있습니다.

## train_pretrain_vlm.py

`train_pretrain_vlm.py`는 이미지-caption 쌍만으로 vision projection을 언어 임베딩 공간에 정렬하는 선택적 사전 학습 단계입니다.

### 목적

SigLIP2 비전 인코더(항상 동결)가 뽑아낸 patch feature를, LLM이 이해할 수 있는 semantic space로 투영하는 `vision_proj` 계층을 학습합니다. 기본값(`--freeze_llm 2`)에서는 LLM과 vision encoder를 모두 완전히 동결하고 projection만 학습하므로, 원본 LLM 가중치를 건드리지 않은 채 시각 토큰을 언어 공간에 깔끔히 맞추는 것이 목표입니다.

### 사용 데이터

- 데이터셋 클래스: `VLMDataset`
- 기본 경로: `./datasets/sft_vlm_english_flickr8k_mini.parquet` (원본 `pretrain_i2t.parquet`는 이 exercise 데이터셋에 없어 대신 사용. 원본 pretrain 데이터를 쓰려면 `--data_path`로 직접 지정)

### 주요 인자

- `--save_weight pretrain_vlm`: 저장될 가중치 접두사
- `--epochs 2`, `--batch_size 16`, `--learning_rate 4e-4`
- `--max_seq_len 450`: 학습 시퀀스 길이
- `--use_moe 0`: MoE 아키텍처 사용 여부
- `--freeze_llm 2`: 동결 정책(0=전체 학습, 1=proj+첫/마지막 LLM 레이어, 2=proj만 학습)
- `--from_weight full_sft`: minimind에서 SFT까지 학습된 언어 모델 가중치(`full_sft_768.pth`)에서 시작
- `--from_resume 0`: resume checkpoint 자동 복원 여부

### 전체 실행 흐름

1. 인자 파싱
   - 저장 경로, 모델 크기, 학습률, 데이터 경로, freeze 정책, DDP/compile/resume 여부를 CLI 인자로 받습니다.

2. 분산 학습과 시드 초기화
   - `init_distributed_mode()`로 DDP 환경을 확인하고, rank별로 seed를 고정합니다.

3. 모델 설정과 checkpoint 확인
   - `VLMConfig(hidden_size, num_hidden_layers, max_seq_len, use_moe)`를 만듭니다.
   - `from_resume=1`이면 `vlm_checkpoint(..., model=None)`으로 resume 파일을 찾습니다.

4. mixed precision 설정
   - CPU면 `nullcontext`, CUDA면 `torch.cuda.amp.autocast(dtype=...)`를 사용합니다(기본 `bfloat16`).

5. wandb 또는 swanlab 로깅 준비
   - `--use_wandb`가 켜져 있고 main process이면 run을 생성하고, resume 시 저장된 `wandb_id`로 이어서 기록합니다.

6. 모델, tokenizer, 데이터셋, optimizer 생성
   - `init_vlm_model(vlm_config, args.from_weight, device=args.device, freeze_llm=args.freeze_llm)`로 모델/tokenizer/전처리기를 로드하고 동결 정책을 적용합니다.
   - `VLMDataset`으로 학습 데이터를 구성합니다.
   - optimizer는 `AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)`로, 동결되지 않은 파라미터만 대상으로 합니다.

7. resume 상태 복원
   - checkpoint가 있으면 model, optimizer, scaler 상태와 `epoch`/`step`을 복원합니다.

8. compile과 DDP 래핑
   - `--use_compile=1`이면 `torch.compile(model)`을 적용합니다.
   - DDP 환경이면 `freqs_cos`/`freqs_sin` 버퍼를 무시하도록 설정 후 `DistributedDataParallel`로 감쌉니다.

9. epoch 학습
   - `vlm_collate_fn`으로 `pixel_values`를 포함한 배치를 구성하고, resume 시 `SkipBatchSampler`로 이미 처리한 batch를 건너뜁니다.

10. 분산 프로세스 종료

### train_epoch 상세 흐름

1. 배치 로드
   - `input_ids`, `labels`, `pixel_values`(dict 또는 tensor)를 device로 이동합니다.

2. learning rate 스케줄링과 forward
   - `model(input_ids, labels=labels, pixel_values=pixel_values)`를 호출해 `loss + aux_loss`(MoE 보조 손실)를 계산합니다.

3. backward와 optimizer step
   - `scaler.scale(loss).backward()` 후 `accumulation_steps` 배수마다 clipping, `scaler.step`, `zero_grad`를 수행합니다.

4. logging
   - `log_interval`마다 loss/logits_loss/aux_loss/lr/남은 시간을 출력하고 wandb에 기록합니다.

5. checkpoint 저장
   - `save_interval`마다 main process에서만 저장합니다.
   - `vision_encoder.`로 시작하는 파라미터는 저장에서 제외합니다(고정된 사전학습 가중치라 다시 저장할 필요가 없음).
   - half precision + CPU로 이동 후 저장하며, `vlm_checkpoint`가 resume용 optimizer/scaler/epoch/step을 tmp 파일 저장 후 `os.replace`로 원자적으로 교체합니다(중단 시 가중치 손상 방지).

6. 마지막 누적 gradient 처리
   - epoch 끝에 accumulation boundary에 닿지 않은 gradient가 남아 있으면 한 번 더 optimizer step을 수행합니다.

### 저장 산출물

- 일반 가중치: `./checkouts/pretrain_vlm_{hidden_size}.pth`
- MoE 가중치: `./checkouts/pretrain_vlm_{hidden_size}_moe.pth`
- resume checkpoint: `./checkouts/pretrain_vlm_{hidden_size}[_moe]_resume.pth`

### 실습

- 모델: hf download jingyaogong/minimind-3-pytorch full_sft_768.pth --local-dir ./checkouts (또는 minimind의 train_full_sft.py로 직접 학습)
- 학습: uv run .\trainer\train_pretrain_vlm.py
- 테스트: uv run eval_vlm.py --weight pretrain_vlm

## train_sft_vlm.py

`train_sft_vlm.py`는 pretrain_vlm(또는 순수 LLM) 가중치를 실제 이미지 기반 지시 대화 데이터로 미세조정합니다.

pretrain_vlm 대비 달라지는 점:

- 기본 데이터 경로는 pretrain과 동일하게 `./datasets/sft_vlm_english_flickr8k_mini.parquet`를 사용합니다(더 큰 `sft_i2t.parquet`로 학습하려면 `--data_path`로 직접 지정).
- 기본 시작 가중치가 `from_weight=pretrain_vlm`입니다.
- 저장 접두사가 `sft_vlm`입니다.
- 학습률이 `4e-4`에서 `5e-6`로 크게 낮아집니다.
- `batch_size` 기본값이 16에서 4로 줄어듭니다(시퀀스 길이 증가로 메모리 사용량이 늘어나기 때문).
- `max_seq_len` 기본값이 450에서 768로 늘어납니다.
- `freeze_llm` 기본값이 2에서 1로 바뀌어, projection 외에 LLM의 **첫 번째와 마지막 레이어**까지 학습 가능해집니다. 첫 레이어는 시각 토큰이 LLM에 들어온 직후 교차 모달 융합을 담당하고, 마지막 레이어는 응답의 형식/스타일을 결정하는 반면, 중간 레이어는 pretrain 지식을 그대로 유지합니다.
- 나머지 학습 루프, checkpoint 저장 방식(vision_encoder 파라미터 제외, half precision, atomic save)은 pretrain_vlm과 동일합니다.

즉, pretrain_vlm이 "vision projection을 언어 공간에 정렬"이라면, SFT는 "정렬된 표현을 바탕으로 실제 이미지 질의응답을 수행하도록 LLM 일부까지 미세조정"하는 단계입니다.

### 실습

- 모델: 1단계(`train_pretrain_vlm.py`)를 먼저 실행해 `./checkouts/pretrain_vlm_768.pth`를 만들거나, `hf download jingyaogong/minimind-3v-pytorch pretrain_vlm_768.pth --local-dir ./checkouts`로 받아옵니다. pretrain을 건너뛰려면 `--from_weight full_sft`를 지정해 `./checkouts/full_sft_768.pth`에서 바로 시작합니다.
- 학습: uv run .\trainer\train_sft_vlm.py
- 테스트: uv run eval_vlm.py --weight sft_vlm --image_dir ./dataset/eval_images/

> 참고: `eval_vlm.py`의 `--image_dir` 기본값은 `../datasets/eval_images/`인데, 실제 샘플 이미지는 `minimind-v/dataset/eval_images/`에 들어 있어 기본값 그대로는 디렉터리를 찾지 못합니다. 위처럼 `--image_dir ./dataset/eval_images/`를 직접 지정해야 합니다.

## trainer_utils.py의 역할

각 trainer가 반복해서 쓰는 공통 기능을 제공합니다.

- `get_model_params`: 학습 가능 파라미터 수 로깅. `vision_encoder`는 통계에서 제외합니다(항상 동결되는 고정 파라미터라 학습 파라미터 논의에 의미가 없음).
- `is_main_process`, `Logger`: DDP main process 판별과 rank 0 전용 로깅
- `get_lr`: cosine 형태의 learning rate 계산
- `init_distributed_mode`: DDP 환경 초기화
- `setup_seed`: 시드 고정
- `init_vlm_model`: tokenizer와 `MiniMindVLM` 로드. `from_weight`가 `none`이 아니면 `{save_dir}/{from_weight}_{hidden_size}[_moe].pth`를 불러오고, `freeze_llm` 값(0/1/2)에 따라 `requires_grad`를 설정합니다. `vision_proj`를 제외한 전체를 우선 동결한 뒤, 정책에 맞게 일부를 다시 해제하는 순서로 동작합니다.
- `vlm_checkpoint`: 모델 저장/복원. `vision_encoder` 파라미터는 저장에서 제외되며, tmp 파일 저장 후 `os.replace`로 원자적으로 교체해 중단 시 가중치 손상을 방지합니다. GPU 수가 바뀌어 resume하는 경우 저장된 `step`을 자동으로 환산합니다.
- `SkipBatchSampler`: resume 시 이미 처리한 batch를 건너뛰는 sampler
- `vlm_collate_fn`: `input_ids`/`labels`는 그대로 stack하고, `pixel_values`가 dict(SigLIP2 NaFlex 출력 형식)이면 key별로, 아니면 그대로 stack합니다.

## trainer별 핵심 차이 표

| 파일 | 목적 | 데이터셋 | 시작 가중치 | freeze_llm | pretrain_vlm 대비 핵심 차이 |
| --- | --- | --- | --- | --- | --- |
| `train_pretrain_vlm.py` | vision projection을 언어 공간에 정렬 | `sft_vlm_english_flickr8k_mini.parquet`(원본 `pretrain_i2t.parquet` 대체) | `full_sft` | `2`(proj만 학습) | 기준 학습 루프, LLM/vision encoder 완전 동결, 높은 lr(4e-4) |
| `train_sft_vlm.py` | 이미지 기반 지시 대화 SFT | `sft_vlm_english_flickr8k_mini.parquet` | `pretrain_vlm` | `1`(proj + 첫/마지막 LLM 레이어) | 낮은 lr(5e-6), 긴 시퀀스(768), 실제 대화형 데이터로 학습 |
