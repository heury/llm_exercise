import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import torch
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model.model_vlm import MiniMindVLM
from datasets import Dataset as HFDataset
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 데이터는 온전히 보존하고 처리하지 않음
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
    # 확률적으로 system 추가
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 80% 확률로 빈 thinking 태그 제거
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class VLMDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512, image_special_token='<|image_pad|>', image_token_len=64):
        super().__init__()
        self.dataset = HFDataset.from_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.dataset)

    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            content = turn['content'].replace('<image>', self.image_special_token) if turn.get('role') != 'system' else turn['content']
            messages.append({"role": turn['role'], "content": content})
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int):
        row = self.dataset[index]
        conversations = json.loads(row['conversations']) if isinstance(row['conversations'], str) else row['conversations']
        image_bytes = row['image_bytes']
        if not isinstance(image_bytes, list): image_bytes = [image_bytes]
        
        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        image_inputs_list = [MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes]
        if hasattr(image_inputs_list[0], 'keys'):
            image_data = {k: torch.cat([inp[k] for inp in image_inputs_list], dim=0) for k in image_inputs_list[0].keys()}
        else:
            image_data = torch.stack(image_inputs_list)
        # # === 디버그 출력 ===
        # print(f"\n--- 샘플 {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_data

# parquet 데이터 읽기 및 시각화 테스트
if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        pf = pq.ParquetFile(path); n = pf.num_row_groups; t = pa.concat_tables([pf.read_row_group(i * n // 5).slice(0, 1) for i in range(5)]); fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            img_data = t['image_bytes'][i].as_py(); img_data = img_data[0] if isinstance(img_data, list) else img_data
            ax[i].imshow(Image.open(io.BytesIO(img_data))); ax[i].axis('off')
            ax[i].set_title(json.loads(t['conversations'][i].as_py())[1]['content'][:30], fontsize=8)
        out = path.replace('.parquet', '_preview.png'); plt.savefig(out); print(f'저장됨: {out}, 총 {pf.metadata.num_rows}개')
