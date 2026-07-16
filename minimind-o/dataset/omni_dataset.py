import torch
import os
import math
import random
import numpy as np
import soundfile as sf
import librosa
import json
import io
from PIL import Image
from scipy.signal import resample
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "당신은 지식이 풍부한 AI입니다. 정확한 정보를 제공하기 위해 최선을 다하세요.",
        "당신은 minimind입니다. 작지만 유용한 언어 모델입니다.",
        "당신은 전문 AI 어시스턴트입니다. 가치 있는 답변을 제공하세요.",
        "당신은 minimind입니다. 사용자의 문제 해결을 최선을 다해 도와주세요.",
        "당신은 신뢰할 수 있는 AI입니다. 정확한 답변을 제공하세요.",
        "당신은 도움이 되는 AI 어시스턴트입니다.",
        "당신은 minimind입니다. 가벼운 지능형 어시스턴트입니다.",
        "당신은 친절한 챗봇입니다. 사용자의 질문에 신중하게 답하세요.",
        "당신은 지식이 풍부한 AI입니다. 정확한 정보를 제공하기 위해 최선을 다하세요.",
        "당신은 minimind입니다. 작지만 유용한 언어 모델입니다."
    ]
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class OmniDataset(Dataset):
    def __init__(self, data_path, tokenizer, audio_processor=None, vision_processor=None,
                 max_length=1200, audio_special_token='<|audio_pad|>', image_special_token='<|image_pad|>',
                 audio_stop_token=2050,  # <|audio_stop|>
                 audio_pad_token=2049,  # <|audio_pad|>
                 audio_spk_token=2051,  # <|audio_spk|>
                 audio_vocab_size=2112,  # 2048 mimi codes + 64 special tokens
                 scheduled_sampling=0.05,
                 image_token_len=64):
        super().__init__()
        tables = [pa.Table.from_batches(pq.ParquetFile(p.strip()).iter_batches()) for p in data_path.split(',')]
        tables = [t.cast(pa.schema([f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f for f in t.schema])) for t in tables]
        self.table = pa.concat_tables(tables, promote_options='default')
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.max_length = max_length
        self.audio_token = audio_special_token
        self.image_token_len = image_token_len
        self.image_token = image_special_token * image_token_len
        self.audio_stop_token = audio_stop_token
        self.audio_pad_token = audio_pad_token
        self.audio_spk_token = audio_spk_token
        self.audio_vocab_size = audio_vocab_size
        self.scheduled_sampling_prob = scheduled_sampling
        self.text_vocab_size = len(tokenizer)
        self.image_token_id = tokenizer.encode(image_special_token, add_special_tokens=False)[0]
        self.audio_token_id = tokenizer.encode(audio_special_token, add_special_tokens=False)[0]
        self.think_end_ids = tokenizer.encode('</think>\n\n', add_special_tokens=False)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    @staticmethod
    def process_audio(audio_path, audio_processor):
        """오디오를 로드해 fbank로 전처리하고 (fbank (T,560), valid_len=encoder 출력 프레임 수)를 반환"""
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        inputs = audio_processor(wav.astype(np.float32), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return inputs.input_features.squeeze(0), valid_len

    def augment_wav(self, wav, sr=16000):
        # 무작위 속도 변경(0.7~1.6x): 오디오 길이와 음높이를 바꿔 빠른/느린 말속도를 포괄
        if random.random() < 0.5:
            speed = random.uniform(0.7, 1.6)
            wav = resample(wav, int(len(wav) / speed)).astype(np.float32)
        # 무작위 잡음 추가: 약한 가우시안 백색잡음을 더해 녹음 환경 차이를 모사
        if random.random() < 0.3:
            noise = np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
            wav = wav + noise
        # 무작위 볼륨: 진폭을 0.8~1.2배 조정해 말소리 크기 변화를 모사
        if random.random() < 0.3:
            wav = wav * random.uniform(0.8, 1.2)
        # 무작위 시간 마스킹: 0.25초 구간을 0으로 만들어 짧은 무음/패킷 손실을 모사
        if random.random() < 0.2 and len(wav) > sr:
            start = random.randint(0, len(wav) - sr // 4)
            wav[start:start + sr // 4] = 0
        # 무작위 저역통과 필터: 이동 평균으로 고주파를 흐리게 해 전화/저품질 마이크를 모사
        if random.random() < 0.2:
            k = random.choice([3, 5, 7])
            wav = np.convolve(wav, np.ones(k) / k, mode='same').astype(np.float32)
        # 무작위 리버브: 지수 감쇠 임펄스 응답을 합성해 방 반사/메아리를 모사
        if random.random() < 0.3:
            ir_len = int(sr * random.uniform(0.05, 0.2))
            ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0, 10, ir_len))
            ir[0] = 1.0
            ir /= np.sqrt(np.sum(ir ** 2) + 1e-6)
            wav = np.convolve(wav, ir, mode='same').astype(np.float32)
        # 무작위 핑크 노이즈: 1/f 잡음으로 실내 배경 소음(에어컨/먼 사람 목소리)을 모사
        if random.random() < 0.2:
            pink = np.cumsum(np.random.randn(len(wav))).astype(np.float32)
            pink /= np.max(np.abs(pink)) + 1e-6
            wav = wav + pink * random.uniform(0.003, 0.015)
        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    def augment_mel(self, fbank):
        # fbank: (T, 560) — SenseVoice LFR 이후 특징(시간 차원이 앞, 주파수 차원이 뒤)
        T, D = fbank.shape
        # SpecAugment 주파수 마스킹: 1~64차원을 무작위로 지워 특정 주파수 대역 과적합을 방지
        if random.random() < 0.5:
            f = random.randint(1, 64)
            f0 = random.randint(0, D - f)
            fbank[:, f0:f0 + f] = 0
        # SpecAugment 시간 마스킹: 1~min(10,T) 프레임을 무작위로 지워 불완전 입력에 대한 내성을 높임
        if random.random() < 0.5 and T > 1:
            t = random.randint(1, min(10, T))
            t0 = random.randint(0, T - t)
            fbank[t0:t0 + t, :] = 0
        return fbank

    def load_audio_inputs(self, audio_bytes):
        if not audio_bytes: return None, 0
        wav, sr = sf.read(io.BytesIO(audio_bytes))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        wav = self.augment_wav(wav.astype(np.float32))
        inputs = self.audio_processor(wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return self.augment_mel(inputs.input_features.squeeze(0)), valid_len

    def load_image_inputs(self, image_bytes):
        if not image_bytes or self.vision_processor is None: return None
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inputs = self.vision_processor(images=image, return_tensors="pt")
        if hasattr(inputs, 'keys'): return {k: v for k, v in inputs.items()}
        return inputs.pixel_values

    def create_chat_prompt(self, conversations, audio_features_length=0):
        conversations = pre_processing_chat(conversations)
        messages = []
        is_last_user = lambda i: i == max(j for j, t in enumerate(conversations) if t['role'] == 'user')
        for idx, turn in enumerate(conversations):
            role, content = turn['role'], turn['content']
            if role == 'user' and is_last_user(idx) and audio_features_length > 0:
                ap = self.audio_token * audio_features_length
                r = random.random()
                if r < 0.4: content = ap
                elif r < 0.6: content = content
                elif r < 0.8: content = ap + '\n\n' + content
                else: content = content + '\n\n' + ap
            if '<image>' in content:
                r = random.random()
                if r < 0.2: content = '<image>\n' + content.replace('<image>', '').strip()
                elif r < 0.4: content = '<image>\n\n' + content.replace('<image>', '').strip()
                elif r < 0.6: content = content.replace('<image>', '').strip() + '\n' + '<image>'
                else: content = content.replace('<image>', '').strip() + '\n\n' + '<image>'
            messages.append({"role": role, "content": content})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return post_processing_chat(prompt)

    
    def generate_text_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        ranges = []
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                ranges.append((start, end))
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels, ranges
    
    def apply_scheduled_sampling(self, input_ids, audio_labels, text_labels):
        """Scheduled Sampling: 일부 GT를 무작위 값으로 대체해 모델이 잘못된 이력에서 회복하도록 학습"""
        if self.scheduled_sampling_prob <= 0:
            return input_ids
        audio_mask = (audio_labels != -100).any(dim=0) & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        for i in range(8):
            input_ids[i] = torch.where(audio_mask, torch.randint(0, self.audio_vocab_size, input_ids[i].shape), input_ids[i])
        # image token의 연속성 보존
        text_mask = (text_labels != -100) & (input_ids[8] != self.image_token_id) & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        input_ids[8] = torch.where(text_mask, torch.randint(0, self.text_vocab_size, input_ids[8].shape), input_ids[8])
        return input_ids

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        question_audios = self.table['question_audios'][index].as_py() if 'question_audios' in self.table.column_names else []
        answer_audios = self.table['answer_audios'][index].as_py() if 'answer_audios' in self.table.column_names else []
        image_bytes = self.table['image_bytes'][index].as_py() if 'image_bytes' in self.table.column_names else []
        if image_bytes and not isinstance(image_bytes, list): image_bytes = [image_bytes]
        ref_audios = self.table['ref_audios'][index].as_py() if 'ref_audios' in self.table.column_names else []
        spk_emb_raw = self.table['spk_emb'][index].as_py() if 'spk_emb' in self.table.column_names else []
        
        # 임의의 턴까지 잘라냄(각 턴=user+assistant)
        asst_indices = [i for i, t in enumerate(conversations) if t['role'] == 'assistant']
        if len(asst_indices) > 1:
            rand_idx = random.randint(0, len(asst_indices) - 1)
            # 무작위 턴에서 시작해 길이가 안전해질 때까지 앞으로 되돌림
            for i in range(rand_idx, -1, -1):
                conversations = conversations[:asst_indices[i] + 1]
                test_prompt = self.create_chat_prompt(conversations, 0)
                if len(self.tokenizer(test_prompt).input_ids) + 100 < self.max_length:
                    break
        
        # 마지막 user의 이미지를 로드(user 턴 인덱스로 접근, audio와 동일)
        pixel_values = None
        user_count = sum(1 for t in conversations if t['role'] == 'user')
        if image_bytes and len(image_bytes) > 0 and self.vision_processor:
            pixel_values = self.load_image_inputs(image_bytes[0])
        
        # 마지막 user의 audio만 로드(user 턴 인덱스로 접근)
        audio_inputs, audio_len, audio_features_length = None, 0, 0
        user_count = sum(1 for t in conversations if t['role'] == 'user')
        if question_audios and user_count > 0 and user_count <= len(question_audios) and self.audio_processor:
            audio_bytes = question_audios[user_count - 1]
            if audio_bytes:
                mel, valid_len = self.load_audio_inputs(audio_bytes)
                if mel is not None:
                    audio_inputs = mel.unsqueeze(0)
                    audio_len = valid_len
                    audio_features_length = valid_len or 1
        
        # 혼합 학습에서 오디오가 없는 샘플은 dummy tensor를 반환해 batch 인덱스를 최대한 맞춤 (SenseVoice: T x 560)
        if audio_inputs is None and self.audio_processor:
            audio_inputs = torch.zeros(1, 1, 560)
            audio_len = 0
        if pixel_values is None and self.vision_processor:
            pixel_values = {'pixel_values': torch.zeros(1, 3, 256, 256)}
        
        # answer_audios에서 마지막 assistant의 audio codes를 가져옴
        last_audio_codes = None
        asst_count = sum(1 for t in conversations if t['role'] == 'assistant')
        if answer_audios and asst_count > 0 and asst_count <= len(answer_audios):
            tokens = answer_audios[asst_count - 1]
            if tokens:
                audio_codes_8layers = [[] for _ in range(8)]
                for i in range(0, len(tokens) - 7, 8):
                    for j in range(8): audio_codes_8layers[j].append(tokens[i + j])
                for layer in audio_codes_8layers: layer.append(self.audio_stop_token)
                last_audio_codes = audio_codes_8layers
        
        # prompt 생성(text input_ids)
        prompt = self.create_chat_prompt(conversations, audio_features_length)
        if pixel_values is not None: prompt = prompt.replace('<image>', self.image_token)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        # input_ids를 max_length까지 PAD
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # labels 생성(마지막 assistant만 학습)
        text_labels, assistant_ranges = self.generate_text_labels(input_ids)
        for start, end in assistant_ranges[:-1]:
            mask_end = min(end + len(self.eos_id), self.max_length)
            text_labels[start:mask_end] = [-100] * (mask_end - start)
        
        # 7층 audio targets 생성(마지막 assistant만 채움)
        Y_audio_layers = [[self.audio_pad_token] * self.max_length for _ in range(8)]
        audio_labels = [[-100] * self.max_length for _ in range(8)]
        if assistant_ranges and last_audio_codes:
            assistant_start, assistant_end = assistant_ranges[-1]
            for pos in range(assistant_start, min(assistant_end, assistant_start + 50)):
                if input_ids[pos:pos + len(self.think_end_ids)] == self.think_end_ids:
                    assistant_start = pos + len(self.think_end_ids)
                    break
            # spk_emb 자리 확보 + ref_codes 오른쪽 정렬(50% 확률로 ref_codes를 drop하고 spk만 유지)
            has_spk = bool(spk_emb_raw)
            has_ref = bool(ref_audios) and random.random() > 0.5
            spk_reserve = 1 if has_spk else 0
            if has_ref:
                ref_codes = [[] for _ in range(8)]
                for i in range(0, len(ref_audios) - 7, 8):
                    for j in range(8): ref_codes[j].append(ref_audios[i + j])
                ref_len = len(ref_codes[0])
                ref_start = max(spk_reserve, assistant_start - ref_len)
                for layer_idx in range(8):
                    codes = ref_codes[layer_idx][-(assistant_start - ref_start):] if ref_len > (assistant_start - ref_start) else ref_codes[layer_idx]
                    for i, code in enumerate(codes):
                        Y_audio_layers[layer_idx][ref_start + i] = code
            else:
                ref_start = assistant_start
            if has_spk and ref_start > 0:
                spk_pos = ref_start - 1
                for layer_idx in range(8):
                    Y_audio_layers[layer_idx][spk_pos] = self.audio_spk_token
            # target codes를 assistant_start 뒤에 채움(loss에 참여)
            for layer_idx in range(8):
                codes = last_audio_codes[layer_idx]
                start_pos = assistant_start + layer_idx + 1
                for i, code in enumerate(codes):
                    if start_pos + i < self.max_length:
                        Y_audio_layers[layer_idx][start_pos + i] = code
                        audio_labels[layer_idx][start_pos + i] = code
        
        # 9채널 입력 구성: input_ids = (9, T) = 8채널 audio + 1채널 text
        X_audio = torch.tensor([layer[:-1] for layer in Y_audio_layers], dtype=torch.long)  # (8, T-1)
        X_text = torch.tensor(input_ids[:-1], dtype=torch.long)  # (T-1,)
        input_ids = torch.cat((X_audio, X_text.unsqueeze(0)), dim=0)  # (9, T-1)
        text_labels = torch.tensor(text_labels[1:], dtype=torch.long)  # (T-1,)
        audio_labels = torch.tensor([layer[1:] for layer in audio_labels], dtype=torch.long)  # (8, T-1)
        
        input_ids = self.apply_scheduled_sampling(input_ids, audio_labels, text_labels)
        spk_emb = torch.tensor(spk_emb_raw, dtype=torch.float32) if spk_emb_raw else torch.zeros(192)
        return input_ids, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb


# parquet 데이터 읽기 테스트
if __name__ == '__main__':
    for path in ['sft_a2a.parquet']:
        if not os.path.exists(path): continue
        t = pa.Table.from_batches(pq.ParquetFile(path).iter_batches())
        conversations = json.loads(t['conversations'][0].as_py())
        answer_audios = t['answer_audios'][0].as_py() if 'answer_audios' in t.column_names else []
        user_msg = conversations[0]
        asst_msg = conversations[1] if len(conversations) > 1 else {}
        print(f'{path}: {len(t)}개, 열{t.column_names}')
        print(f'  User: {user_msg["content"][:50]}...')
        print(f'  Asst: {asst_msg.get("content", "")[:50]}...')
        if answer_audios:
            print(f'  answer_audios: {len(answer_audios)}턴, 첫 턴 {len(answer_audios[0])}tokens')