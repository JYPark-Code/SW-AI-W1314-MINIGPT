# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout

    nn.Module 상속이 하는 일:
    - 클래스 속성으로 nn.Parameter / nn.Module을 할당하면 자동으로 등록되어
      .parameters()로 순회 가능 (optimizer가 이걸로 학습 대상 파라미터를 모음).
    - .to(device), .train()/.eval(), state_dict() 등이 자식 모듈까지 재귀 전파됨.
    - super().__init__()을 빼먹으면 위 등록 메커니즘이 안 깔려서 파라미터가 사라진다.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        # nn.Embedding(N, D)는 사실상 (N, D) 모양의 learnable weight matrix 하나.
        # forward(idx)는 weight[idx]와 동등 (gather/lookup), 단 autograd 추적됨.
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        # position도 똑같은 lookup table. 학습 가능한 절대 위치 임베딩 (GPT-2 방식).
        # sinusoidal과 달리 학습되므로 context_length보다 긴 시퀀스는 처리 불가.
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        # nn.Dropout(p)는 학습 모드에서만 작동. eval 모드(.eval())에선 identity가 된다.
        # 살아남은 원소는 1/(1-p)로 스케일 업되어 기대값이 보존됨.
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        seq_len = x.shape[1]
        # torch.arange(N) = [0, 1, ..., N-1]
        # device=x.device가 중요: 입력이 GPU에 있으면 positions도 GPU에 만들어야
        # 같이 lookup할 수 있음. CPU/GPU 텐서 섞이면 런타임 에러.
        positions = torch.arange(seq_len, device=x.device)

        # tok shape: (B, T, D) — 각 토큰 ID를 D차원 벡터로 변환
        tok = self.token_embedding(x)
        # pos shape: (T, D) — 위치별 D차원 벡터
        pos = self.position_embedding(positions)

        # broadcasting: (B, T, D) + (T, D) → 뒤 차원부터 맞춤
        # (T, D)가 (1, T, D)로 확장되고 B 차원으로 복제되어 모든 배치에 같은 pos가 더해짐.
        # 위치 임베딩은 배치와 무관하므로 이게 의도된 동작.
        return self.dropout(tok + pos)
