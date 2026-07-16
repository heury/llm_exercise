import time
import argparse
import os
import warnings
import torch
import random
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindVLM(
            VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
            vision_model_path="../models/siglip2-base-p32-256-ve"
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniMindVLM.get_vision_model("../models/siglip2-base-p32-256-ve")
    get_model_params(model, model.config)
    model = model.eval()
    if "cuda" in args.device: model = model.half()
    return model.to(args.device), tokenizer, model.processor


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V 대화")
    parser.add_argument('--load_from', default='../models', type=str, help="모델 로드 경로(model=네이티브 torch 가중치, 다른 경로=transformers 형식)")
    parser.add_argument('--save_dir', default='../checkouts', type=str, help="모델 가중치 디렉터리")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="가중치 이름 접두사(pretrain_vlm, sft_vlm)")
    parser.add_argument('--hidden_size', default=768, type=int, help="은닉층 차원")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="은닉층 수")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 아키텍처 사용 여부(0=아니오, 1=예)")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="최대 생성 길이")
    parser.add_argument('--temperature', default=0.7, type=float, help="생성 온도. 무작위성을 제어합니다(0~1, 클수록 더 무작위)")
    parser.add_argument('--top_p', default=0.85, type=float, help="뉴클리어스 샘플링 임계값(0~1)")
    parser.add_argument('--image_dir', default='../datasets/eval_images/', type=str, help="테스트 이미지 디렉터리")
    parser.add_argument('--show_speed', default=1, type=int, help="디코딩 속도 표시(tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="실행 장치")
    parser.add_argument('--open_thinking', default=0, type=int, help="적응형 thinking 활성화 여부(0=아니오, 1=예)")
    args = parser.parse_args()
    
    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # image_dir의 모든 이미지를 자동 테스트
    prompt = "<image>\n이 이미지의 주요 물체와 장면을 설명해 주세요."
    # prompt = "<image>\n이미지를 말로 설명해 주세요."
    for image_file in sorted(os.listdir(args.image_dir)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            setup_seed(random.randint(1, 31415926))
            image_path = os.path.join(args.image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in MiniMindVLM.image2tensor(image, preprocess).items()}
            
            messages = [{"role": "user", "content": prompt.replace('<image>', model.config.image_special_token * model.config.image_token_len)}]
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
            inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)
            
            print(f'[이미지]: {image_file}')
            print(f"💬: {repr(prompt)}")
            print('🤖: ', end='')
            st = time.time()
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, pixel_values=pixel_values
            )
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            print(f'\n[속도]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
