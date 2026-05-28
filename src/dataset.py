# -*- coding: utf-8 -*-
"""GPT 사전 학습용 Dataset/DataLoader 과제 템플릿."""

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """
    token ID 리스트를 다음 토큰 예측용 input/target 쌍으로 자릅니다.

    예: token_ids=[10, 11, 12, 13], context_length=3
    - input:  [10, 11, 12]
    - target: [11, 12, 13]

    torch.utils.data.Dataset은 추상 클래스로, __len__과 __getitem__ 두 메서드만
    구현하면 DataLoader가 알아서 indexing/batching/shuffling을 처리한다.
    (내부적으로 DataLoader는 Sampler가 뽑은 인덱스로 dataset[idx]를 호출함)
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
        stride: int | None = None,
    ):
        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        # 길이 N짜리 시퀀스에서 (input=context_length, target=다음 1칸) 쌍을 stride 간격으로 뽑을 때
        # 마지막 샘플은 start + context_length + 1 <= N 을 만족해야 함
        # => start_max = N - context_length - 1, stride로 나누고 +1
        self._length = (len(self.token_ids) - self.context_length -1) // self.stride + 1


    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        idx번째 (input_ids, target_ids) 쌍을 LongTensor로 반환.

        target은 input을 1칸 shift한 것. 즉 input[t]를 보고 target[t]를 예측하도록 학습됨.

        Returns:
            input_ids: (context_length,)
            target_ids: (context_length,)
        """
        start = idx * self.stride
        # torch.tensor(...)는 파이썬 리스트를 복사해서 새 텐서를 만든다.
        # dtype=torch.long(int64)으로 강제하는 이유: nn.Embedding의 index 인자로 쓰려면
        # 정수형이어야 하고, PyTorch는 관례적으로 인덱스에 long을 요구한다.
        input_ids = torch.tensor(self.token_ids[start: start + self.context_length], dtype=torch.long)
        target_ids = torch.tensor(self.token_ids[start + 1: start + 1 + self.context_length], dtype=torch.long)
        return input_ids, target_ids


def create_dataloader(
    token_ids: list[int],
    context_length: int,
    batch_size: int = 8,
    stride: int | None = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """GPTDataset을 만들고 torch.utils.data.DataLoader로 감싸 반환합니다.

    DataLoader가 해주는 일:
    - Sampler로 인덱스 시퀀스 생성 (shuffle=True면 random, False면 sequential)
    - 각 인덱스에 대해 dataset[idx] 호출
    - batch_size개를 모은 뒤 default_collate로 stack
      (튜플 (input, target)이면 ([input,...], [target,...])로 묶고 각각 torch.stack)
    - num_workers>0이면 subprocess로 병렬 prefetch (Windows는 spawn 비용이 커서 0이 보통 빠름)
    - drop_last=True면 batch_size로 안 나뉘는 마지막 자투리 배치를 버림
    """

    dataset = GPTDataset(token_ids, context_length, stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return loader
