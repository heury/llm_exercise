"""
MiniMind 모델 추론 및 대화 평가 스크립트

MiniMind 모델을 로드하여 자동 테스트 또는 수동 입력 모드로
대화형 추론을 수행하는 스크립트입니다.
"""

import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

# 모델과 토크나이저를 초기화하고 로드하는 함수
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    # torch 네이티브 가중치 로드 경로
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        # LoRA 가중치 적용 (설정된 경우)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        # transformers 형식 모델 로드
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    # half precision으로 변환 후 평가 모드로 설정
    return model.half().eval().to(args.device), tokenizer

# 메인 함수: 인자 파싱, 모델 로드, 대화 루프 실행
def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="MiniMind 모델 추론 및 대화")
    parser.add_argument('--load_from', default='../minimind_model', type=str, help="모델 로드 경로 (model=네이티브 torch 가중치, 기타 경로=transformers 형식)")
    parser.add_argument('--save_dir', default='../minimind_out', type=str, help="모델 가중치 디렉토리")
    parser.add_argument('--weight', default='full_sft', type=str, help="가중치 이름 접두사 (pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA 가중치 이름 (None=사용 안 함, 선택: lora_identity, lora_medical)")
    parser.add_argument('--hidden_size', default=768, type=int, help="은닉층 차원")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="은닉층 개수")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 아키텍처 사용 여부 (0=아니오, 1=예)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="RoPE 위치 인코딩 외삽 활성화 (4배, 위치 인코딩 문제만 해결)")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="최대 생성 길이 (주의: 모델의 실제 긴 텍스트 능력과 다름)")
    parser.add_argument('--temperature', default=0.85, type=float, help="생성 온도, 무작위성 제어 (0-1, 클수록 무작위)")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus 샘플링 임계값 (0-1)")
    parser.add_argument('--open_thinking', default=0, type=int, help="적응형 사고 활성화 여부 (0=아니오, 1=예)")
    parser.add_argument('--historys', default=0, type=int, help="이전 대화 유지 횟수 (짝수여야 함, 0=이전 대화 미포함)")
    parser.add_argument('--show_speed', default=1, type=int, help="디코드 속도 표시 (tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="실행 디바이스")
    args = parser.parse_args()
    
    # 자동 테스트용 프롬프트 목록 (중국어 LLM 테스트 입력)
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    # 모델 초기화 및 입력 모드 선택
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 자동 테스트\n[1] 수동 입력\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 대화 루프: 자동 테스트 또는 수동 입력 모드
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        # 대화 이력 관리
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        # 입력 포맷 구성: pretrain은 원시 텍스트, 그 외는 채팅 템플릿 적용
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        # 모델 추론 및 스트리밍 출력
        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        # 응답 디코딩 및 대화 이력에 추가
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        # 생성 속도 출력
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()