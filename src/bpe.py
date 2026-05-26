# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256

def get_pair_counts(ids):
    """이웃한 token pair의 빈도를 센다.

        Args:
            ids: token ID 리스트
        Returns:
            {(a, b): 빈도} 형태의 dict
    """
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_pair(ids, pair, new_id):
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            # 같은 경우
            result.append(new_id)  # 새 ID 하나 넣고
            i += 2  # 두 칸 점프
        else:
            # 아닌 경우
            result.append(ids[i])  # 지금 원소 그대로 넣고
            i += 1  # 한 칸
    return result

class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        special_token = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for i in range(4):
            token = special_token[i]
            idx = i

            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

        for v in range(256):
            token = bytes([v])
            idx = v + 4
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        self._init_special_tokens()
        ids = [ b + BYTE_OFFSET for b in corpus.encode("utf-8")] # 1단계
        next_id = BYTE_OFFSET + NUM_BYTES                        # 260

        while len(self.id_to_token) < self.vocab_size:  # 4단계: 목표까지 반복
            counts = get_pair_counts(ids)  # 2단계: 빈도 세기
            if not counts:  # pair가 없으면 멈춤
                break
            best_pair = max(counts, key=counts.get)  # 3-(a): 최다 pair
            # 3-(b): ids에서 best_pair를 next_id로 치환
            ids = merge_pair(ids, best_pair, next_id)
            # self.merges, id_to_token, token_to_id 갱신
            self.merges.append(best_pair)  # 학습 순서 기록: [(11,12), ...]
            self.id_to_token[next_id] = best_pair  # 260 → (11,12)
            self.token_to_id[best_pair] = next_id  # (11,12) → 260
            next_id += 1


    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        ids = [b + BYTE_OFFSET for b in text.encode("utf-8")]   #1
        for pair in self.merges:                                #2
            new_id = self.token_to_id[pair]
            ids = merge_pair(ids, pair, new_id)

        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        def expand(x):
            if x < BYTE_OFFSET:
                return []                     # A: 특수토큰 skip
            elif x < BYTE_OFFSET + NUM_BYTES:
                return [x]                    # B: byte 바닥
            else:
                a, b = self.id_to_token[x]
                return expand(a) + expand(b)  # C: 재귀

        all_bytes = []
        for x in ids:
            all_bytes += expand(x)
        raw_bytes = [v - BYTE_OFFSET for v in all_bytes]
        return bytes(raw_bytes).decode("utf-8")
