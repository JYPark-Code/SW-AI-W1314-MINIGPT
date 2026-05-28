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

    표기 약속:
    - B = batch_size, T = seq_len (시간축), C = d_model
    - head_dim = C / n_heads
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
        # nn.Linear(in, out)은 y = x @ W^T + b. W는 (out, in) 모양으로 저장됨.
        # Q/K/V를 각각 따로 nn.Linear 3개 만드는 대신 (d_model → 3*d_model) 하나로
        # 합치면 단일 matmul 호출이 되어 GPU에서 훨씬 빠름 (커널 launch 1번).
        # 같은 차원의 행렬곱이라 결과는 분리해서 만든 것과 수학적으로 동일.
        self.qkv = nn.Linear(d_model, 3* d_model, bias=qkv_bias)
        # head별 결과를 합친 뒤 한 번 더 섞어주는 선형변환 (정보 mixing 역할).
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        B, T, C  = x.shape

        # 1. qkv 뽑고 셋으로 쪼개기
        # qkv shape: (B, T, 3*C). 마지막 축을 3등분 → 각 (B, T, C).
        # .chunk(n, dim)은 마지막 축을 n조각으로 나눠 tuple로 반환 (view라 copy 없음).
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. head 분리: (B, T, C) → (B, n_heads, T, head_dim)
        # .view(B, T, n_heads, head_dim): 같은 메모리를 다른 모양으로 해석. C = n_heads * head_dim.
        # .transpose(1, 2): T축과 n_heads축을 swap. 이후 head 차원이 batch처럼 행동해서
        #   각 head가 독립적인 attention을 병렬로 계산할 수 있게 됨.
        # 주의: transpose는 stride만 바꿔서 메모리상 비연속(non-contiguous)이 됨.
        #   matmul(@)은 비연속 텐서를 받지만, 뒤에서 view를 다시 쓰려면 contiguous() 필요.
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. score = Q·Kᵀ / √head_dim
        # k.transpose(2, 3): (B, h, T, head_dim) → (B, h, head_dim, T)
        # @는 torch.matmul. 4D 텐서끼리 곱하면 앞쪽 (B, h)는 batch로 보고
        #   마지막 두 축에 대해서만 행렬곱 → (T, head_dim) @ (head_dim, T) = (T, T).
        # 결과 attn_scores shape: (B, h, T, T). [i, j]는 i가 j를 얼마나 볼지의 점수.
        attn_scores = q @ k.transpose(2, 3)
        # √head_dim으로 스케일: head_dim이 크면 dot product 분산이 커져
        # softmax 기울기가 saturate되는 걸 막음 ("Attention is All You Need" 스케일링).
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # 4. causal mask: 미래 토큰을 못 보게 차단
        if causal_mask:
            # torch.ones(T, T): T×T 1로 채운 행렬
            # torch.triu(..., diagonal=1): 대각선 위쪽(=미래 위치)만 남기고 나머지 0
            #   예) T=4면
            #   [[0,1,1,1],
            #    [0,0,1,1],
            #    [0,0,0,1],
            #    [0,0,0,0]]
            # .bool()로 mask용 boolean 텐서로 변환.
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            # .masked_fill(mask, val): mask가 True인 자리에 val을 채움.
            # (T, T) mask가 (B, h, T, T) scores에 broadcast됨.
            # -inf를 넣는 이유: 다음 줄 softmax에서 exp(-inf)=0이 되어 그 위치의 attention 가중치가 정확히 0.
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # 5. softmax → dropout
        # dim=-1: 마지막 축(key 위치 T)을 따라 정규화. 각 query 행의 합이 1이 됨.
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # attention dropout: 일부 연결을 끊어 regularization (원 논문 권장 방식).
        attn_weights = self.dropout(attn_weights)

        # 6. weight · V
        # (B, h, T, T) @ (B, h, T, head_dim) = (B, h, T, head_dim)
        # 각 query 위치가 자기 가중치로 V를 weighted sum한 결과.
        context = attn_weights @ v

        # 7. head 합치고 out_proj
        # transpose로 다시 (B, T, h, head_dim) 만들고
        # .contiguous(): transpose 후 메모리가 흩어져 있으므로, view를 쓰기 위해 연속 메모리로 복사.
        #   (.reshape를 쓰면 contiguous를 알아서 해주지만 명시적으로 분리하는 게 디버깅에 좋음)
        # .view(B, T, C): h * head_dim = C로 펼침 → 모든 head를 concat한 효과.
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(context)

        if return_attention_weights:
            return out, attn_weights
        return out
