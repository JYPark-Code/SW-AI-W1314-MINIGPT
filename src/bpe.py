# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json


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

        예: [10, 11, 12, 11, 12] → {(10,11):1, (11,12):2, (12,11):1}
    """
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        # dict.get(key, default): 키가 없으면 default 반환. defaultdict 대안.
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_pair(ids, pair, new_id):
    """ids에 등장하는 pair를 모두 new_id 하나로 치환.

    겹침 방지를 위해 left-to-right로 훑으며 매칭 시 2칸 점프하는 방식.
    예: ids=[1,2,1,2,2], pair=(1,2), new_id=99 → [99, 99, 2]
    """
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
        기본 사전 초기화:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록.

        bytes([v])는 길이 1짜리 bytes 객체. str과 달리 dict key로 쓸 수 있고
        byte-level BPE에선 임의의 유니코드(특히 한글)도 깨짐 없이 다룰 수 있어 안전함.
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
        코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        알고리즘:
        1) UTF-8 byte 시퀀스로 변환 → 각 byte를 +4 offset 줘서 special token ID와 겹치지 않게.
        2) 가장 자주 등장하는 (a, b) 쌍을 찾음.
        3) 새 ID로 치환하고, merges/사전 업데이트.
        4) vocab_size에 도달하거나 더 합칠 pair가 없을 때까지 반복.

        merges는 학습된 순서대로 저장됨 — encode 시 이 순서를 그대로 적용해야
        같은 token화 결과가 나옴 (순서 바뀌면 다른 segmentation이 됨).
        """
        self._init_special_tokens()
        # corpus.encode("utf-8") → bytes 객체 (정수 시퀀스처럼 iterate 가능, 각 원소 0~255)
        # +BYTE_OFFSET(=4): special token과 충돌 안 나도록 byte 0 → ID 4부터 시작
        ids = [ b + BYTE_OFFSET for b in corpus.encode("utf-8")] # 1단계
        next_id = BYTE_OFFSET + NUM_BYTES                        # 260

        while len(self.id_to_token) < self.vocab_size:  # 4단계: 목표까지 반복
            counts = get_pair_counts(ids)  # 2단계: 빈도 세기
            if not counts:  # pair가 없으면 멈춤
                break
            # max(dict, key=dict.get): value가 최대인 key 반환. 동률이면 먼저 나온 것.
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
        vocabulary와 merge rule을 JSON 파일로 저장합니다.

        merges만 저장해도 충분한 이유:
        - special 토큰(0~3)과 byte 토큰(4~259)은 _init_special_tokens()으로 재구성 가능.
        - merges 순서만 그대로면 id_to_token/token_to_id를 완전히 복원할 수 있음.
        JSON 제약: tuple은 list로, bytes는 표현 불가 → 여기선 merge가 (int,int) 튜플이라
        list 변환만으로 충분.
        """
        data = {
            "vocab_size" : self.vocab_size,
            "merges" : [list(pair) for pair in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            # ensure_ascii=False: 기본값(True)은 비ASCII를 \uXXXX로 이스케이프함.
            # 여기선 merges가 int뿐이라 사실상 영향 없지만, 한글 보존 관례로 False.
            json.dump(data, f, ensure_ascii=False)


    def load(self, path: str | Path):
        """save()로 저장한 JSON 파일을 읽어 사전을 완전히 복원합니다."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self._init_special_tokens()  # 2번: 기본 사전 깔기
        self.merges = []  # 깨끗이 비우고 다시 채움

        next_id = BYTE_OFFSET + NUM_BYTES  # 260부터
        for pair_list in data["merges"]:
            # JSON에선 tuple→list로 직렬화됐으므로 다시 tuple로 (dict key로 쓰려면 hashable 필요)
            pair = tuple(pair_list)  # [36,240] → (36,240)
            self.merges.append(pair)
            self.id_to_token[next_id] = pair
            self.token_to_id[pair] = next_id
            next_id += 1


    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        문자열 → token ID 리스트.

        핵심: train에서 학습한 merges를 학습된 순서대로 그대로 적용해야 함.
        순서가 달라지면 greedy하게 합쳐지는 모양이 달라져 다른 segmentation이 나옴.
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
        token ID 리스트 → 문자열 복원.

        merge로 만든 토큰은 (a, b) 쌍을 다시 a와 b로 재귀 분해해서 최종적으론
        byte 시퀀스까지 풀어낸 뒤, 한 번에 UTF-8 디코딩.
        byte별로 따로 decode하면 멀티바이트 글자(한글 등)가 깨짐.
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
        # ID → 원본 byte값(0~255)로 되돌리기
        raw_bytes = [v - BYTE_OFFSET for v in all_bytes]
        # bytes(int_list).decode("utf-8"): 정수 리스트를 bytes로 만들고 한 번에 UTF-8 해석
        return bytes(raw_bytes).decode("utf-8")
