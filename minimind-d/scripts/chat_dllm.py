# DLLM TUI(터미널 UI) 채팅 스크립트
# Rich 라이브러리를 사용하여 디노이징 과정을 실시간 시각화하는 터미널 채팅 인터페이스

import sys, os, argparse, math, shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model_minimind_dllm import MiniMindDLLMConfig, MiniMindForMaskedDiffusion, add_gumbel_noise

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.text import Text

model, tokenizer, device = None, None, 'cpu'

# DLLM 모델 및 토크나이저 로드
def load_model(args):
    global model, tokenizer, device
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    config = MiniMindDLLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    model = MiniMindForMaskedDiffusion(config)
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    if os.path.exists(ckp):
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        print(f"[가중치 로드] {ckp}")
    model = model.half().eval().to(device)

# 이전 스텝의 토큰 ID (새로 드러난 토큰 강조 표시에 사용)
prev_ids = None

# 토큰 시퀀스를 Rich Text로 렌더링 (마스크/EOS/새 토큰을 색상으로 구분)
def render_tokens(x, prompt_len, mask_id, eos_id):
    global prev_ids
    ids = x[prompt_len:].tolist()
    t = Text()
    for i, tid in enumerate(ids):
        if tid == mask_id:
            # 마스크 토큰: 빨간색으로 표시
            t.append("[M]", style="dim red")
        elif tid == eos_id:
            # EOS 토큰: 녹색으로 표시
            t.append("[E]", style="dim green")
        elif prev_ids is not None and i < len(prev_ids) and prev_ids[i] == mask_id:
            # 이번 스텝에서 새로 드러난 토큰: 밝은 녹색으로 강조
            t.append(tokenizer.decode([tid]), style="bold bright_green")
        else:
            t.append(tokenizer.decode([tid]), style="white")
    prev_ids = ids[:]
    return t

# TUI 레이아웃 업데이트: 진행률 표시줄과 토큰 패널 갱신
def update_tui(live, layout, progress, task_id, x, prompt_len, mask_id, eos_id, step, total_steps):
    masks_left = int((x[prompt_len:] == mask_id).sum().item())
    total_gen = len(x[prompt_len:])
    revealed = total_gen - masks_left
    pct = f"{int(100 * step / max(total_steps, 1))}%"
    progress.update(task_id, completed=step, total=total_steps, masks=masks_left, pct=pct)
    layout["text"].update(Panel(
        render_tokens(x, prompt_len, mask_id, eos_id),
        title=f"[bold cyan]Denoising[/bold cyan]  [dim]{revealed}/{total_gen} revealed[/dim]",
        subtitle=f"[dim]Step {step}/{total_steps}[/dim]",
        border_style="bright_blue", padding=(1, 1),
    ))
    layout["progress"].update(Panel(progress, border_style="dim"))
    live.refresh()

# TUI에서 디노이징 과정을 실시간으로 시각화하며 텍스트 생성
def stream_generate_tui(input_ids, prompt_len, args):
    global prev_ids
    prev_ids = None
    mask_id = model.config.mask_token_id
    eos_id = tokenizer.eos_token_id
    block_size, steps, max_tokens = args.block_size, args.steps, args.max_new_tokens

    # 블록 수와 총 스텝 수 계산
    num_blocks = math.ceil(max_tokens / block_size)
    total_steps = num_blocks * steps
    # 전체 시퀀스를 [MASK]로 초기화
    T = prompt_len + max_tokens
    x = torch.full((1, T), eos_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    x[:, prompt_len:] = mask_id
    unmasked_index = (x != mask_id)

    try:
        tw = shutil.get_terminal_size().columns
    except Exception:
        tw = 120
    console = Console(width=tw)

    # Rich 진행률 표시줄 설정
    progress = Progress(
        SpinnerColumn(), TextColumn("[bold blue]Diffusion"), BarColumn(), MofNCompleteColumn(),
        TextColumn("•"), TextColumn("[cyan]Masks: {task.fields[masks]}"),
        TextColumn("•"), TextColumn("[magenta]{task.fields[pct]:>4s}"), TimeRemainingColumn(), expand=True,
    )
    layout = Layout()
    layout.split_column(Layout(name="text", ratio=1), Layout(name="progress", size=3))
    task_id = progress.add_task("gen", total=total_steps, masks=max_tokens, pct="0%")

    # 실시간 TUI 디노이징 루프
    with Live(layout, console=console, auto_refresh=False, screen=True) as live:
        global_step = 0
        for b in range(num_blocks):
            block_end = min(prompt_len + (b + 1) * block_size, T)

            for step in range(steps):
                mask_index = (x == mask_id)
                mask_count = mask_index[:, :block_end].sum(-1).min().item()
                if mask_count == 0: break
                # 남은 스텝에 비례하여 언마스킹할 토큰 수 결정
                n_unmask = max(1, round(mask_count / (steps - step)))

                # 모델 추론: 마스크된 위치의 토큰 예측
                with torch.no_grad():
                    logits = model(input_ids=x).logits
                # Top-k 필터링
                if args.top_k > 0:
                    logits[logits < torch.topk(logits, args.top_k, dim=-1)[0][..., -1:]] = -float('inf')
                # Gumbel 노이즈로 샘플링
                x0 = torch.argmax(add_gumbel_noise(logits, args.temperature), dim=-1)
                p = F.softmax(logits.float(), dim=-1)
                # 신뢰도 기반으로 가장 확실한 토큰부터 언마스킹
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, block_end:] = -float('inf')
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))
                _, idx = torch.topk(confidence[0], k=min(n_unmask, int(mask_count)))
                x[0, idx] = x0[0, idx]

                global_step += 1
                update_tui(live, layout, progress, task_id, x[0], prompt_len, mask_id, eos_id, global_step, total_steps)

            # EOS 토큰이 생성되면 조기 종료
            if eos_id and (x[:, prompt_len:] == eos_id).any(): break

    # 생성된 텍스트 디코딩 및 출력
    gen_ids = x[0, prompt_len:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    final_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"\n🤖️: {final_text}\n")
    return final_text

# 대화 루프: 자동 테스트 또는 수동 입력 모드로 채팅 실행
def chat_loop(args):
    is_pt = 'pt' in args.weight
    # 테스트 프롬프트 (중국어 LLM용 테스트 입력이므로 원문 유지)
    prompts = ['你有什么特长？', '为什么天空是蓝色的', '请用Python写一个计算斐波那契数列的函数', '解释一下"光合作用"的基本过程']

    print(f"\n{'='*50}")
    print(f"  MiniMind DLLM TUI Chat ({'pretrain' if is_pt else 'sft'} mode)")
    print(f"  steps={args.steps} block={args.block_size} max_tokens={args.max_new_tokens}")
    print(f"  temperature={args.temperature} top_k={args.top_k}")
    print(f"{'='*50}\n")

    mode = input('[0] 자동 테스트\n[1] 수동 입력\n> ').strip()
    prompt_iter = prompts if mode == '0' else iter(lambda: input('\n💬 > '), '')

    for prompt in prompt_iter:
        if mode == '0':
            print(f'\n💬: {prompt}')

        # 사전훈련 모드와 SFT 모드에서 입력 형식이 다름
        if is_pt:
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        else:
            tpl = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(tpl, return_tensors="pt", truncation=True).input_ids.to(device)

        # TUI 디노이징 시각화와 함께 텍스트 생성
        stream_generate_tui(input_ids, input_ids.shape[1], args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DLLM TUI Chat")
    parser.add_argument('--load_from', default='../../models', type=str, help="토크나이저 경로")
    parser.add_argument('--save_dir', default='../../checkouts', type=str, help="가중치 디렉토리")
    parser.add_argument('--weight', default='dllm', type=str, help="가중치 접두사")
    parser.add_argument('--hidden_size', default=768, type=int, help="은닉층 차원")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="은닉층 수")
    parser.add_argument('--use_moe', default=0, type=int, help="MoE 아키텍처 사용 여부")
    parser.add_argument('--max_new_tokens', default=128, type=int, help="최대 생성 길이")
    parser.add_argument('--steps', default=32, type=int, help="디노이징 총 스텝 수")
    parser.add_argument('--block_size', default=32, type=int, help="블록당 토큰 수")
    parser.add_argument('--temperature', default=0.5, type=float, help="샘플링 온도")
    parser.add_argument('--top_k', default=50, type=int, help="top-k 샘플링")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="추론 장치")
    args = parser.parse_args()

    try:
        load_model(args)
        chat_loop(args)
    except KeyboardInterrupt:
        print("\n종료")
