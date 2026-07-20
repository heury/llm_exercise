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

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind 모델 추론 및 대화")
    parser.add_argument('--load_from', default='./model', type=str, help="모델 로드 경로(model=네이티브 torch 가중치, 다른 경로=transformers 형식)")
    parser.add_argument('--save_dir', default='../checkouts', type=str, help="모델 가중치 디렉터리")
    parser.add_argument('--weight', default='full_sft', type=str, help="가중치 이름 접두사(pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA 가중치 이름(None이면 사용하지 않음, 선택: lora_identity, lora_medical)")
    parser.add_argument('--hidden_size', default=768, type=int, help="은닉층 차원")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="은닉층 수")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 아키텍처 사용 여부(0=아니오, 1=예)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="RoPE 위치 인코딩 외삽 활성화(4배, 위치 인코딩 문제만 해결)")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="최대 생성 길이(주의: 모델의 실제 장문 처리 능력을 의미하지 않음)")
    parser.add_argument('--temperature', default=0.85, type=float, help="생성 온도. 무작위성을 제어합니다(0~1, 클수록 더 무작위)")
    parser.add_argument('--top_p', default=0.95, type=float, help="뉴클리어스 샘플링 임계값(0~1)")
    parser.add_argument('--open_thinking', default=0, type=int, help="적응형 thinking 활성화 여부(0=아니오, 1=예)")
    parser.add_argument('--historys', default=0, type=int, help="포함할 과거 대화 턴 수(짝수여야 하며 0이면 포함하지 않음)")
    parser.add_argument('--show_speed', default=1, type=int, help="디코딩 속도 표시(tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="실행 장치")
    args = parser.parse_args()
    
    prompts = [
        'What are your strengths?',
        'Why is the sky blue?',
        'Write a Python function that computes the Fibonacci sequence',
        'Explain the basic process of "photosynthesis"',
        'How should I go out if it rains tomorrow?',
        'Compare the pros and cons of keeping a cat versus a dog as a pet',
        'Explain what machine learning is',
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 자동 테스트\n[1] 수동 입력\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[속도]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()