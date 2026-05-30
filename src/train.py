# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    # 데이터를 모델과 같은 device(CPU/GPU)로 옮겨야 연산이 가능.
    # DataLoader가 만든 텐서는 기본적으로 CPU에 있으므로 매 배치마다 .to(device) 필요.
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # GPTModel.forward는 targets를 주면 (loss, logits)를 반환 (model.py 참고).
    # 여기서는 loss만 쓰고 logits는 _로 버림.
    loss, _ = model(input_batch, target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    total_loss = 0
    # 빈 loader면 0으로 나누는 사고를 막기 위해 nan을 반환.
    if len(data_loader) == 0:
        return float("nan")

    # num_batches를 안 주면 전체 배치를 다 본다.
    elif num_batches is None:
        num_batches = len(data_loader)

    # 줬더라도 실제 배치 수를 넘지 않게 clip (eval에서 일부만 빠르게 보고 싶을 때 사용).
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # .item(): GPU 텐서를 파이썬 float으로 꺼냄.
            # 텐서째로 누적하면 계산 그래프가 계속 쌓여 메모리가 새므로 스칼라로 변환해서 더함.
            total_loss += loss.item()

        else:
            break
    # 본 배치 수로 나눠 평균 loss 반환.
    return total_loss / num_batches


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    # 학습을 중단했다 이어서(resume) 하려면 가중치뿐 아니라
    # optimizer 상태(Adam의 모멘텀 등)와 진행 위치(epoch, global_step)까지 함께 저장해야 함.
    checkpoint = {
        # .state_dict(): 모듈의 학습 가능한 파라미터를 {이름: 텐서} 딕셔너리로 반환.
        "model_state_dict": model.state_dict(),
        # optimizer도 step별 통계(예: Adam의 1차/2차 모멘트)를 갖고 있어 같이 저장.
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    # torch.save는 내부적으로 pickle로 직렬화해서 파일에 기록.
    torch.save(checkpoint, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    # map_location=device: 저장 당시 device(예: GPU)와 지금 device가 달라도
    # 지정한 device로 텐서를 매핑해서 로드 (GPU에서 저장→CPU에서 로드 같은 경우 대비).
    checkpoint = torch.load(path, map_location=device)
    # 저장해둔 파라미터를 현재 모델 구조에 덮어씀 (구조가 같아야 함).
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer가 None일 수 있음(추론만 할 때) → 있을 때만 복원.
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # 이어서 학습할 수 있도록 멈췄던 위치(epoch, global_step)를 돌려줌.
    return checkpoint["epoch"], checkpoint["global_step"]


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    # generate_text_simple(model.py)의 확장판: greedy뿐 아니라
    # top-k로 후보를 추리고 temperature로 무작위성을 조절하는 샘플링까지 지원.
    for _ in range(max_new_tokens):
        # context_size보다 길어지면 모델이 못 받으므로 최근 context_size개 토큰만 입력으로 사용.
        idx_cond = idx[:, -context_size:]
        # 생성은 학습이 아니므로 gradient 계산/저장이 불필요 → no_grad로 메모리·속도 절약.
        with torch.no_grad():
            logits = model(idx_cond)
        # 다음 토큰은 "마지막 위치"의 예측만 필요 → 시간축에서 -1만 슬라이스.
        logits = logits[:, -1, :]  # 마지막 위치 (batch, vocab)

        # ===== 여기가 새로운 부분 =====
        # 1. top_k 적용 (있으면)
        # 2. temperature 적용 후 샘플링, 또는 greedy
        # =============================
        if top_k is not None:
            # 확률 상위 k개만 후보로 남기고 나머지는 -inf로 만들어 softmax에서 0이 되게 함.
            # → 말도 안 되는 저확률 토큰이 뽑히는 걸 방지.
            top_vals, _ = torch.topk(logits, top_k)
            min_val = top_vals[:, -1]  # 상위 k개 중 최솟값(=합격 커트라인)
            # 커트라인 미만인 logit을 모두 -inf로. torch.where(조건, 참일때, 거짓일때).
            logits = torch.where(logits < min_val, float("-inf"), logits)

        if temperature > 0:
            # temperature로 분포를 조절: <1이면 뾰족(보수적), >1이면 평평(다양).
            # logit을 T로 나눈 뒤 softmax → 확률 분포.
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # multinomial: 확률에 비례해 무작위로 1개 추첨 (= 샘플링).
            next_id = torch.multinomial(probs, num_samples=1)  # 확률대로 추첨
        else:
            # temperature가 0이면 무작위성 제거 → 가장 확률 높은 토큰 선택(greedy).
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy

        # 3. eos_id 나오면 멈추기 (있으면)
        # 배치 내 모든 시퀀스가 종료 토큰을 내면 더 생성할 필요가 없으므로 중단.
        if eos_id is not None and (next_id == eos_id).all():
            break

        # 뽑은 토큰을 시퀀스 끝에 이어 붙이고 다음 루프에서 다시 입력으로 사용 (자기회귀).
        idx = torch.cat([idx, next_id], dim=1)

    return idx


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    # eval 모드: dropout 끄고 등 추론용 동작으로 전환 (학습 중 샘플을 찍어볼 때 호출).
    model.eval()
    # 문자열 → 토큰 ID 리스트.
    ids = tokenizer.encode(start_context)
    # 모델은 (batch, seq) 모양을 기대 → unsqueeze(0)로 batch 차원 1을 앞에 붙이고 device로 이동.
    encoded = torch.tensor(ids).unsqueeze(0).to(device) # unsqueeze(0) : 0번 위치에 크기 1짜리 차원을 끼워넣어라
    token_ids = generate(model, encoded, max_new_tokens, context_size, temperature, top_k)
    # squeeze(0): batch 차원 제거 → 1D 토큰열. .tolist()로 파이썬 리스트로 만들어 decode.
    text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(text)
    # 다시 train 모드로 돌려놔야 이후 학습에서 dropout 등이 정상 동작.
    model.train()


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 학습 도중 현재 성능을 점검하는 함수. train/val loss를 함께 측정해 과적합 여부를 가늠.
    # eval 모드 + no_grad로 평가 (가중치 갱신 없이 순전파만).
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )   # train_loader로 loss, num_batches=eval_iter (전체가 아닌 일부 배치만 빠르게)
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )     # val_loader로 loss, num_batches=eval_iter
    # 평가가 끝나면 학습을 계속할 수 있게 train 모드 복귀.
    model.train()
    return train_loss, val_loss

def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    # 전체 학습을 지휘하는 메인 루프. start_epoch/global_step 인자가 있어
    # checkpoint에서 이어서(resume) 학습하는 것도 지원.
    train_losses = []

    for epoch in range(start_epoch, num_epochs):
        model.train()  # 매 epoch 시작 시 train 모드 보장 (평가 후 eval로 바뀌어 있을 수 있음).
        for input_batch, target_batch in train_loader:
            # === 학습 4박자 === (모든 딥러닝 학습의 기본 사이클)
            optimizer.zero_grad()  # 1) 이전 step의 gradient 초기화 (안 하면 누적됨).
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 2) 순전파+loss.
            loss.backward()        # 3) 역전파: 각 파라미터의 gradient 계산.
            optimizer.step()       # 4) gradient 방향으로 파라미터 갱신.

            global_step += 1  # 전체 학습 동안 누적되는 step 카운터 (eval/저장 주기 판단용).

            # === eval_freq마다 평가 + 샘플 출력 ===
            # 일정 step마다 train/val loss를 찍고, 실제 생성 샘플도 출력해 진척을 눈으로 확인.
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                print(f"Step {global_step}: train {train_loss:.3f}, val {val_loss:.3f}")
                generate_and_print_sample(model, tokenizer, device, start_context)

        # === epoch 끝: train loss 기록 ===
        # epoch마다 대표 train loss를 한 번 측정해 리스트에 쌓음 (나중에 plot_losses로 그릴 용도).
        epoch_train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        train_losses.append(epoch_train_loss)

        # === ckpt_freq마다 checkpoint 저장 ===
        # ckpt_freq epoch마다 중간 저장 → 중단되어도 마지막 checkpoint부터 재개 가능.
        # (epoch + 1)을 쓰는 이유: epoch는 0부터 시작하므로 사람이 읽는 "1번째"와 맞추기 위함.
        if ckpt_freq is not None and (epoch + 1) % ckpt_freq == 0:
            save_checkpoint(model, optimizer, epoch, global_step, f"checkpoint_epoch{epoch + 1}.pt")

    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
