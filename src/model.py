# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        # keepdim=True: 축을 줄이지 않고 크기 1로 남겨 (B, T, 1) 모양 유지
        #   → 뒤에서 (B, T, C)인 x와 broadcasting으로 자연스럽게 빼고 나눌 수 있음.
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False: 분산을 N-1이 아닌 N으로 나눔. LayerNorm 정의가 모집단 분산을 쓰기 때문.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # +eps: 분산이 0에 가까울 때 0으로 나누는 걸 막는 수치 안정화 장치.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # gamma(scale)/beta(shift): 정규화로 잃은 표현력을 모델이 다시 학습하도록 하는 파라미터.
        #   gamma=1, beta=0으로 시작해 필요하면 분포를 늘이거나 옮길 수 있음.
        return self.gamma * norm_x + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        # GELU의 tanh 근사식 (GPT-2가 사용한 버전).
        # ReLU처럼 음수를 딱 자르지 않고 부드럽게 통과시켜, 작은 음수도 약간 살려줌 → 학습 안정.
        # 정확한 GELU는 정규분포 CDF를 쓰지만, 이 tanh 근사가 계산이 싸고 거의 동일.
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                                         (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        # 차원을 mult배(보통 4배)로 늘렸다 다시 줄이는 구조.
        # 넓은 중간층에서 비선형(GELU) 변환을 충분히 한 뒤 원래 차원으로 압축 → 표현력 확보.
        # attention이 토큰 간 정보를 섞는다면, FFN은 각 토큰 내부에서 위치별로 변환을 가함.
        self.layers = nn.Sequential(
            nn.Linear(d_model, mult*d_model),  # 확장: d_model → 4*d_model
            GELU(),                            # 비선형
            nn.Linear(mult*d_model, d_model),  # 압축: 4*d_model → d_model (원래 차원으로 복귀)
            nn.Dropout(dropout),               # 과적합 방지
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        # nn.Sequential은 등록된 레이어를 정의한 순서대로 차례로 통과시킴.
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        self.attention = MultiHeadAttention(
            d_model = d_model,
            n_heads = n_heads,
            drop_rate = drop_rate,
            qkv_bias = qkv_bias
        )
        self.ffn = FeedForward(d_model, drop_rate)
        # norm을 sub-layer마다 따로 두 개: attention 앞(norm1)과 ffn 앞(norm2)에 각각 적용.
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        # ---- sub-layer 1: causal self-attention ----
        # Pre-LN 구조(GPT-2 방식): residual로 더할 원본을 먼저 따로 보관(shortcut)하고,
        # 정규화된 입력을 sub-layer에 통과시킨 뒤 원본을 더함.
        # residual connection은 gradient가 깊은 층까지 그대로 흐르게 해 깊은 망 학습을 가능케 함.
        shortcut = x
        x = self.norm1(x)                              # 먼저 정규화 (Pre-LN)
        x = self.attention(x, causal_mask=causal_mask)  # 토큰 간 정보 교환
        x = self.dropout(x)
        x = x + shortcut                               # 원본 더하기 (residual)

        # ---- sub-layer 2: feed-forward ----
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)                                # 토큰별 위치 단위 변환
        x = self.dropout(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        # token + position 임베딩으로 ID를 벡터로 변환 (embeddings.py 참고).
        self.embedding = InputEmbedding(
            vocab_size = config["vocab_size"],
            emb_dim = config["emb_dim"],
            context_length = config["context_length"],
            drop_rate = config["drop_rate"],
        )

        # TransformerBlock을 n_layers개 쌓음. nn.ModuleList로 담아야
        # 자식 모듈로 등록되어 파라미터가 .parameters()/state_dict()에 잡힘 (일반 list면 누락됨).
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config["emb_dim"],
                n_heads=config["n_heads"],
                drop_rate=config["drop_rate"],
                qkv_bias=config["qkv_bias"],
            )
            for _ in range(config["n_layers"])
        ])

        # 마지막 블록 출력을 한 번 더 정규화 (GPT-2 구조).
        self.final_norm = LayerNorm(config["emb_dim"])
        # LM head: hidden 벡터(emb_dim)를 vocab 크기 logit으로 사상 → 다음 토큰 점수.
        # bias=False: GPT-2 관례 (출력층 bias 생략).
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)

        -----------------------------------

        idx (token IDs)
          ↓ embedding
          ↓ blocks 차례로 통과 (for문)
          ↓ final_norm
          ↓ lm_head
        logits
          ↓ targets 있으면 → loss도 계산
        """
        x = self.embedding(idx)                  # (B, T) → (B, T, emb_dim)
        for block in self.blocks:                # N개 블록을 차례로 통과
            x = block(x, causal_mask=True)       # causal: 미래 토큰 못 보게
        x = self.final_norm(x)
        logits = self.lm_head(x)                 # (B, T, vocab_size)

        if targets is not None:
            # cross_entropy는 (N, C)와 (N,) 모양을 기대 → 배치·시간축을 하나로 평탄화.
            # logits.reshape(-1, vocab): (B*T, vocab), targets.reshape(-1): (B*T,)
            # 각 위치마다 "정답 토큰을 얼마나 잘 맞췄나"를 평균낸 값이 loss.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            return loss, logits

        return logits



def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""

    # 가장 단순한 생성: 매 step 확률이 가장 높은 토큰만 고르는 greedy 방식
    # (train.py의 generate는 여기에 temperature/top-k 샘플링을 추가한 확장판).
    for _ in range(max_new_tokens):
        # context_size를 넘으면 모델이 처리 못 하므로 최근 context_size개만 입력으로 사용.
        idx_cond = idx[:, -context_size:]
        # 생성은 추론이므로 gradient 불필요 → no_grad로 메모리·속도 절약.
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[: , -1 , :]                         # 마지막 위치 예측만 사용 (B, vocab)
        probas = torch.softmax(logits, dim=-1)              # logit → 확률
        next_id = torch.argmax(probas, dim=-1, keepdim=True)  # 최고 확률 토큰 선택 (greedy)
        idx = torch.cat((idx, next_id), dim=1)              # 시퀀스 끝에 이어 붙이고 반복 (자기회귀)

    return idx