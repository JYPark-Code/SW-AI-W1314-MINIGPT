# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        # Q : Query, K : Key, V : Value
        self.qkv = nn.Linear(d_model, 3* d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        B, T, C  = x.shape

        # 1. qkv 뽑고 셋으로 쪼개기
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. head 분리: (B, T, C) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. score = Q·Kᵀ / √head_dim
        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # 4. causal mask  ← 여기는 비워둠, 다음에 같이
        # (causal_mask가 True이면 미래 위치를 -inf로)
        if causal_mask:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # 5. softmax → dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. weight · V
        context = attn_weights @ v  # (B, n_heads, T, head_dim)

        # 7. head 합치고 out_proj
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(context)

        if return_attention_weights:
            return out, attn_weights
        return out
