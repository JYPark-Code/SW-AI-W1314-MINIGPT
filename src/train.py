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
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
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
    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)

    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()

        else:
            break
    return total_loss / num_batches


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # 마지막 위치 (batch, vocab)

        # ===== 여기가 새로운 부분 =====
        # 1. top_k 적용 (있으면)
        # 2. temperature 적용 후 샘플링, 또는 greedy
        # =============================
        if top_k is not None:
            top_vals, _ = torch.topk(logits, top_k)
            min_val = top_vals[:, -1]  # 상위 k개 중 최솟값
            logits = torch.where(logits < min_val, float("-inf"), logits)

        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # 확률대로 추첨
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy

        # 3. eos_id 나오면 멈추기 (있으면)
        if eos_id is not None and (next_id == eos_id).all():
            break

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
    model.eval()
    ids = tokenizer.encode(start_context)
    encoded = torch.tensor(ids).unsqueeze(0).to(device) # unsqueeze(0) : 0번 위치에 크기 1짜리 차원을 끼워넣어라
    token_ids = generate(model, encoded, max_new_tokens, context_size, temperature, top_k)
    text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(text)
    model.train()


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )   # train_loader로 loss, num_batches=eval_iter
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )     # val_loader로 loss, num_batches=eval_iter
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
    train_losses = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # === 학습 4박자 ===
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            global_step += 1

            # === eval_freq마다 평가 + 샘플 출력 ===
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                print(f"Step {global_step}: train {train_loss:.3f}, val {val_loss:.3f}")
                generate_and_print_sample(model, tokenizer, device, start_context)

        # === epoch 끝: train loss 기록 ===
        epoch_train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        train_losses.append(epoch_train_loss)

        # === ckpt_freq마다 checkpoint 저장 ===
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
