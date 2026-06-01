# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """

    # 1. TSV 읽는 헬퍼 (download_data.py의 _read_nsmc_tsv와 동일)
    def read_tsv(path):
        rows = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("document") or "").strip()
                label = row.get("label")
                if not text or label not in {"0", "1"}:
                    continue
                rows.append({"text": text, "label": int(label)})
        return rows

    # 2. train 읽고 쓰기
    train_rows = read_tsv(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    # 3. train/val 나누기
    val_size = max(1, int(len(train_rows) * val_ratio))
    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]

    # 4. test (있으면)
    test_data = read_tsv(test_tsv_path) if test_tsv_path is not None else []

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        ids = self.tokenizer.encode(text)

        # 1. max_length보다 겉면 자르기
        ids = ids[:self.max_length]

        # 2. 짧으면 pad_id로 채우기
        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))

        # 3. Tensor (label 점수 그대로)
        return torch.tensor(ids, dtype=torch.long), label



class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        pad_id = 0

        # 1. backbone 에서 hidden state 뽑기 ( 방법 A - lm_head 빼고)
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x, causal_mask=True)
        x = self.gpt.final_norm(x) # (B, T, emb_dim)

        # 2. pad 아닌 마지막 토큰 위치 찾기
        lengths = (input_ids != pad_id).sum(dim=1) # (B,)
        last_idx = lengths - 1                     # (B,)

        # 3. 각 리뷰에서 그 위치의 hidden state 뽑기
        # (B, T, emb_dim)에서 각 batch마다 last_idx 위치 -> (B, emb_dim)
        pooled = x[torch.arange(x.size(0)), last_idx] # 인덱싱

        # 4. dropout -> classifier
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)             #(8, num_labels)

        # 5. labels 가 없으면 loss
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        return logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # === 학습 4박자 ===
        optimizer.zero_grad()
        loss, logits = model(input_ids, labels)  # forward가 (loss, logits) 반환
        loss.backward()
        optimizer.step()

        # === 측정 ===
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)  # 예측 label (B,)
        correct += (preds == labels).sum().item()  # 맞힌 개수
        total += labels.size(0)  # 전체 개수

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss, logits = model(input_ids, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy
