# DLLM 모델 평가 및 대화형 추론 스크립트
# 훈련된 마스크 확산 모델을 로드하여 반복적 디노이징으로 텍스트를 생성

import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer
from model.model_minimind_dllm import MiniMindDLLMConfig, MiniMindForMaskedDiffusion
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

# 모델 및 토크나이저 초기화, 체크포인트에서 가중치 로드
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    model = MiniMindForMaskedDiffusion(MiniMindDLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    ))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    # 모델 파라미터 수 출력
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

# 기능: main 함수에서 필요한 데이터 변환과 모델 호출 로직을 수행합니다.
def main():
    parser = argparse.ArgumentParser(description="MiniMind DLLM Inference")
    parser.add_argument('--load_from', default='../models', type=str, help="모델 로드 경로")
    parser.add_argument('--save_dir', default='../checkouts', type=str, help="모델 가중치 디렉토리")
    parser.add_argument('--weight', default='dllm', type=str, help="가중치 이름 접두사")
    parser.add_argument('--hidden_size', default=768, type=int, help="은닉층 차원")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="은닉층 수")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 아키텍처 사용 여부")
    parser.add_argument('--max_new_tokens', default=256, type=int, help="최대 생성 길이")
    parser.add_argument('--steps', default=32, type=int, help="디노이징 총 스텝 수")
    parser.add_argument('--block_size', default=32, type=int, help="블록당 토큰 수")
    parser.add_argument('--temperature', default=0.4, type=float, help="생성 온도 (0-1)")
    parser.add_argument('--top_k', default=50, type=int, help="top-k 샘플링")
    parser.add_argument('--historys', default=0, type=int, help="대화 히스토리 턴 수 (짝수)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="실행 장치")
    args = parser.parse_args()

    # 테스트 프롬프트 (중국어 LLM용 테스트 입력이므로 원문 유지)
    prompts = [
        '请介绍一下你自己',
        '推荐一部好看的科幻电影',
        '如何快速学习一门新的编程语言'
    ]

    conversation = []
    model, tokenizer = init_model(args)
    # 사용자에게 자동/수동 모드 선택 요청
    input_mode = int(input('[0] 자동 테스트\n[1] 수동 입력\n'))

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        # 히스토리 길이 제한 (최근 N턴만 유지)
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 채팅 템플릿을 적용하여 입력 토큰 생성
        inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        # 반복적 디노이징을 통한 텍스트 생성
        generated = model.generate(
            inputs=inputs["input_ids"],
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            block_size=args.block_size,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
        # 프롬프트 이후의 생성된 토큰만 추출
        gen_ids = generated[0, inputs['input_ids'].shape[1]:].tolist()
        # EOS 토큰 이후의 출력 제거
        if tokenizer.eos_token_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(tokenizer.eos_token_id)]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        print(f'🧠: {response}\n')

if __name__ == "__main__":
    main()
