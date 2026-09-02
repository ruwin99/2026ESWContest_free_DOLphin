from __future__ import annotations

import math
import random
from collections.abc import Iterator

from torch.utils.data import Sampler


class HalfNegativeBatchSampler(Sampler[list[int]]):
    """Every stage-2 batch contains at least 50% clean/artifact samples."""

    def __init__(self, scenarios: list[str], batch_size: int, seed: int) -> None:
        if batch_size < 2:
            raise ValueError("Balanced Stage-2 sampler requires batch_size >= 2")
        self.negative = [i for i, value in enumerate(scenarios) if value in {"clean", "artifact_hard_negative"}]
        self.other = [i for i, value in enumerate(scenarios) if i not in set(self.negative)]
        if not self.negative or not self.other:
            raise ValueError("Stage-2 sampler requires both clean/artifact negatives and defect samples")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil((len(self.negative) + len(self.other)) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        negative = self.negative.copy()
        other = self.other.copy()
        rng.shuffle(negative)
        rng.shuffle(other)
        neg_count = math.ceil(self.batch_size / 2)
        other_count = self.batch_size - neg_count
        for batch_index in range(len(self)):
            batch = [negative[(batch_index * neg_count + i) % len(negative)] for i in range(neg_count)]
            batch.extend(other[(batch_index * other_count + i) % len(other)] for i in range(other_count))
            rng.shuffle(batch)
            yield batch
