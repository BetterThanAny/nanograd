from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from nanograd.data.dataset import Dataset


def default_collate(samples: list) -> tuple:
    """Stack list of (x, y, ...) tuples into batched arrays."""
    if not samples:
        return ()
    # single-item samples (not tuple) — treat as 1-tuple
    if not isinstance(samples[0], tuple):
        return (np.stack(samples),)
    cols = list(zip(*samples))
    return tuple(np.stack(c) for c in cols)


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(indices)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue
            samples = [self.dataset[int(i)] for i in batch_idx]
            yield self.collate_fn(samples)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
