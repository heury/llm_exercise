import argparse
import os
import random
import time
import warnings
import torch
import soundfile as sf
from PIL import Image
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForCausalLM, MimiModel
from model.model_omni import MiniMindOmni, OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import setup_seed, log_model_params
warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindOmni(
            OmniConfig(
                hidden_size=args.hidden_size, 
                num_hidden_layers=args.num_hidden_layers, 
                use_moe=bool(args.use_moe)
            ),
            audio_encoder_path="../models/SenseVoiceSmall",
            vision_model_path="../models/siglip2-base-p32-256-ve"
        )
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.audio_encoder, model.audio_processor = MiniMindOmni.load_sensevoice("../models/SenseVoiceSmall")
        model.vision_encoder, model.vision_processor = MiniMindOmni.load_vision("../models/siglip2-base-p32-256-ve")
    log_model_params(model)
    if model.audio_encoder is not None: model.audio_encoder.to(args.device)
    if model.vision_encoder is not None: model.vision_encoder.to(args.device)
    model.mimi_model = MimiModel.from_pretrained("../models/mimi").eval()
    return model.half().eval().to(args.device), tokenizer


def eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, output_name, pixel_values=None, history=None, audio_lens=None, ref_codes=None, spk_emb=None):
    messages = (history or []) + [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
    x = torch.tensor(tokenizer(inputs_text).data['input_ids'], dtype=torch.long, device=args.device)[None, ...]

    audio_frames = []
    with torch.no_grad():
        res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_p=args.top_p, stream=True,
                               return_audio_codes=True, open_thinking=bool(args.open_thinking),
                               audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values,
                               ref_codes=ref_codes, spk_emb=spk_emb)
        print('📒 [Thinker]: ', end='', flush=True)
        history_idx = 0
        for y, audio_frame in res_y:
            if y is not None:
                answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                if answer and answer[-1] != '�':
                    print(answer[history_idx:], end='', flush=True)
                    history_idx = len(answer)
            if audio_frame:
                audio_frames.append(audio_frame)
        print()

        if audio_frames:
            print(f'🎹 [Talker]: {len(audio_frames)} 프레임', end=" ")
            if args.decode_audio:
                try:
                    codes = [f for f in audio_frames if f and len(f) == 8]
                    if not codes:
                        print('⚠️  생성된 Mimi codes가 비어 있어 저장을 건너뜁니다.')
                        return
                    mimi_codes = torch.tensor(codes, dtype=torch.long).T.unsqueeze(0).to(args.device)
                    filtered = torch.where(mimi_codes >= 2049, torch.zeros_like(mimi_codes), mimi_codes)
                    audio = model.mimi_model.decode(filtered).audio_values
                    output_path = os.path.join(args.output_dir, output_name)
                    wav_path = output_path.rsplit('.', 1)[0] + '.wav'
                    sf.write(wav_path, audio.squeeze().float().cpu().numpy(), 24000)
                    AudioSegment.from_wav(wav_path).export(output_path, format='mp3', bitrate='64k')
                    os.remove(wav_path)
                    print(f'| 오디오 디코딩 저장 위치: {output_path}')
                except Exception as e:
                    print(f'⚠️  오디오 저장 실패: {str(e)}')
            else:
                print("(decode_audio=off)\n")


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O 대화")
    parser.add_argument('--load_from', default='../models', type=str, help="모델 로드 경로(model=네이티브 torch 가중치)")
    parser.add_argument('--save_dir', default='../checkouts', type=str, help="모델 가중치 디렉터리")
    parser.add_argument('--weight', default='sft_omni', type=str, help="가중치 이름 접두사")
    parser.add_argument('--hidden_size', default=768, type=int, help="은닉층 차원")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="은닉층 수")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 아키텍처 사용 여부")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="최대 생성 길이")
    parser.add_argument('--temperature', default=0.7, type=float, help="Thinker 생성 온도")
    parser.add_argument('--top_p', default=0.85, type=float, help="뉴클리어스 샘플링 임계값")
    parser.add_argument('--output_dir', default='../checkouts/output_audio/', type=str, help="출력 오디오 저장 디렉터리")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="실행 장치")
    parser.add_argument('--audio_dir', default='../datasets/eval_omni/', type=str, help="테스트 오디오 디렉터리")
    parser.add_argument('--image_dir', default='../datasets/eval_omni/', type=str, help="테스트 이미지 디렉터리")
    parser.add_argument('--open_thinking', default=0, type=int, help="thinking 모드 활성화 여부(0=아니오, 1=예, thinking 모드에서는 audio 출력 비활성화)")
    parser.add_argument('--decode_audio', default=1, type=int, help="오디오 출력 디코딩 여부(0=아니오, 1=예)")
    parser.add_argument('--mode', default='0', type=str, help="평가 모드: -1=전체 0=텍스트 1=멀티턴 2=오디오 3=음색 복제 4=이미지 5=혼합(쉼표 조합, 예: 2,5)")
    parser.add_argument('--prompt_lang', default=0, type=int, choices=[0, 1, 2], help="질문 언어: 0=영어 1=중국어 2=영어+중국어")
    args = parser.parse_args()
    modes = set(args.mode.replace(',', '').replace('-1', '012345'))
    
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = init_model(args)
    setup_seed(int(time.time()) % 31415926)

    if '0' in modes:
        print('\n\n==================== 텍스트 -> {텍스트, 오디오} ====================')
        test_prompts_en = [
            "우주에 관한 흥미로운 사실을 알려 주세요.", "커피 한 잔을 어떻게 만들면 되나요?", "오늘 날씨는 어떤가요?",
            "내일 비가 올까요?", "농담 하나 해 주세요.", "저를 위해 노래를 불러 줄 수 있나요?", "자기소개를 해 주세요."
        ]
        test_prompts_zh = [
            "우주에 관한 흥미로운 사실을 알려 주세요.", "커피 한 잔을 어떻게 만들면 되나요?", "오늘 날씨는 어떤가요?",
            "내일 비가 올까요?", "농담 하나 해 주세요.", "저를 위해 노래를 불러 줄 수 있나요?", "자기소개를 해 주세요."
        ]
        test_prompts = [test_prompts_en, test_prompts_zh, test_prompts_en + test_prompts_zh][args.prompt_lang]
        for idx, prompt in enumerate(test_prompts):
            print(f'\n📝 [text-{idx+1}]: {prompt}')
            eval_sample(model, tokenizer, args, idx, prompt, None, f"text-{idx:02d}.mp3")

    if '1' in modes:
        print('\n\n==================== 멀티턴 -> {텍스트, 오디오} ====================')
        multi_turn_tests_zh = [
            {
                "history": [
                    {"role": "user", "content": "안녕하세요"},
                    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
                ],
                "prompt": "할 일을 찾고 있어요. 추천할 만한 것이 있나요?"
            },
            {
                "history": [
                    {"role": "user", "content": "안녕하세요"},
                    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
                    {"role": "user", "content": "할 일을 찾고 있어요. 추천할 만한 것이 있나요?"},
                    {"role": "assistant", "content": "음악을 듣거나 책을 읽으면서 마음을 조금 쉬게 해 보세요."}
                ],
                "prompt": "좋아요, 그렇게 해 볼게요. 고마워요."
            }
        ]
        multi_turn_tests_en = [
            {
                "history": [
                    {"role": "user", "content": "안녕하세요"},
                    {"role": "assistant", "content": "안녕하세요! How can I help you?"}
                ],
                "prompt": "할 일을 찾고 있어요. 추천할 만한 것이 있나요?"
            },
            {
                "history": [
                    {"role": "user", "content": "안녕하세요"},
                    {"role": "assistant", "content": "안녕하세요! How can I help you?"},
                    {"role": "user", "content": "할 일을 찾고 있어요. 추천할 만한 것이 있나요?"},
                    {"role": "assistant", "content": "음악을 듣거나 책을 읽으면서 조금 쉬어 보세요."}
                ],
                "prompt": "좋아요, 그렇게 해 볼게요. 고마워요."
            }
        ]
        multi_turn_tests = [multi_turn_tests_en, multi_turn_tests_zh, multi_turn_tests_en + multi_turn_tests_zh][args.prompt_lang]
        for idx, test in enumerate(multi_turn_tests):
            print(f'\n💬 [multi-{idx+1}]')
            for msg in test["history"]: print(f'   {msg["role"]}: {msg["content"]}')
            print(f'   user: {test["prompt"]}')
            eval_sample(model, tokenizer, args, idx, test["prompt"], None, f"multi-{idx:02d}.mp3", history=test["history"])

    if '2' in modes:
        print('\n\n==================== 오디오 -> {텍스트, 오디오} ====================')
        audio_files_en = sorted([f for f in os.listdir(args.audio_dir) if f.startswith('audio-en-') and f.lower().endswith(('.mp3', '.wav'))])
        audio_files_zh = sorted([f for f in os.listdir(args.audio_dir) if f.startswith('audio-zh-') and f.lower().endswith(('.mp3', '.wav'))])
        audio_files = [audio_files_en, audio_files_zh, audio_files_en + audio_files_zh][args.prompt_lang]
        for idx, audio_file in enumerate(audio_files):
            print(f'\n🎤 [audio-{idx+1}]: {audio_file}')
            mel, valid_len = OmniDataset.process_audio(os.path.join(args.audio_dir, audio_file), model.audio_processor)
            audio_inputs = mel.unsqueeze(0).to(args.device)
            audio_lens = torch.tensor([valid_len], device=args.device)
            audio_token_len = valid_len or 1
            prompt = model.config.audio_special_token * audio_token_len
            eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, f"audio-{idx:02d}-{os.path.splitext(audio_file)[0]}.mp3", audio_lens=audio_lens)

    if '3' in modes:
        print('\n\n==================== 음색 복제 -> {텍스트, 오디오} ====================')
        clone_prompts_en = ["안녕하세요, 자기소개를 해 주세요.", "오늘 날씨는 어떤가요?", "농담 하나 해 주세요."]
        clone_prompts_zh = ["안녕하세요, 자기소개를 해 주세요.", "오늘 날씨는 어떤가요?", "농담 하나 해 주세요."]
        clone_prompts = [clone_prompts_en, clone_prompts_zh, clone_prompts_en + clone_prompts_zh][args.prompt_lang]
        voices_pt = '../models/speaker/voices_unseen.pt'
        voices = [('default', None, None)]
        if os.path.exists(voices_pt):
            voice_data = torch.load(voices_pt, map_location=args.device)
            for speaker, v in sorted(voice_data.items()):
                rc = v['ref_codes'].unsqueeze(0).to(args.device)
                se = v['spk_emb'].half().unsqueeze(0).to(args.device) if 'spk_emb' in v else None
                voices.append((speaker, rc, se))
        for speaker, rc, se in voices:
            info = f'ref_codes: {rc.shape[2]} 프레임, spk_emb: {"+" if se is not None else "-"}' if rc is not None else ('spk_emb only' if se is not None else 'default')
            print(f'\n🎵 [clone: {speaker}] {info}')
            for idx, prompt in enumerate(clone_prompts):
                print(f'  📝 [text-{idx+1}]: {prompt}')
                history = [{"role": "system", "content": "당신은 전문 음성 어시스턴트입니다. 주어진 음색 스타일로 사용자의 질문에 답하세요. 가능한 자세하고 가치 있는 정보를 제공하세요."}]
                eval_sample(model, tokenizer, args, idx, prompt, None, f"clone-{speaker}-{idx:02d}.mp3", ref_codes=rc, history=history, spk_emb=se)

    if '4' in modes:
        print('\n\n==================== 이미지 -> {텍스트, 오디오} ====================')
        image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for idx, image_file in enumerate(image_files):
            print(f'\n🖼️ [image-{idx+1}]: {image_file}')
            image = Image.open(os.path.join(args.image_dir, image_file)).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in model.vision_processor(images=image, return_tensors="pt").items()}
            prompts = [["이 이미지를 설명해 주세요."], ["이 이미지를 설명해 주세요."], ["이 이미지를 설명해 주세요.", "이 이미지를 설명해 주세요."]][args.prompt_lang]
            for lang_idx, prompt_text in enumerate(prompts):
                prompt = prompt_text + "\n\n" + model.config.image_special_token * model.config.image_token_len
                eval_sample(model, tokenizer, args, idx, prompt, None, f"image-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3", pixel_values=pixel_values)

    if '5' in modes:
        print('\n\n==================== text+audio+이미지 -> {텍스트, 오디오} ====================')
        img_audio_files = sorted([f for f in os.listdir(args.audio_dir) if f.startswith('img-') and f.lower().endswith(('.mp3', '.wav'))])
        image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        text_hints = [["답변해 주세요: "], ["답변해 주세요: "], ["답변해 주세요: ", "답변해 주세요: "]][args.prompt_lang]
        for idx, image_file in enumerate(image_files):
            audio_file = random.choice(img_audio_files)
            image = Image.open(os.path.join(args.image_dir, image_file)).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in model.vision_processor(images=image, return_tensors="pt").items()}
            for lang_idx, text_hint in enumerate(text_hints):
                print(f'\n🌀 [mix-{idx+1}-{lang_idx}]: {text_hint} | {audio_file} | {image_file}')
                mel, valid_len = OmniDataset.process_audio(os.path.join(args.audio_dir, audio_file), model.audio_processor)
                audio_inputs = mel.unsqueeze(0).to(args.device)
                audio_lens = torch.tensor([valid_len], device=args.device)
                audio_token_len = valid_len or 1
                prompt = text_hint + model.config.audio_special_token * audio_token_len + "\n\n" + model.config.image_special_token * model.config.image_token_len
                eval_sample(model, tokenizer, args, idx, prompt, audio_inputs, f"mix-{idx:02d}-{lang_idx}-{os.path.splitext(image_file)[0]}.mp3", pixel_values=pixel_values, audio_lens=audio_lens)


if __name__ == "__main__":
    main()

