from __future__ import annotations

from typing import Sized, Tuple

import numpy as np


class Dataset:
    """Abstract dataset. Override __len__ and __getitem__."""

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def __getitem__(self, idx: int):  # pragma: no cover
        raise NotImplementedError


class TensorDataset(Dataset):
    """Wrap numpy arrays; indexing returns a tuple of slices."""

    def __init__(self, *arrays: np.ndarray):
        if not arrays:
            raise ValueError("at least one array required")
        n = len(arrays[0])
        for a in arrays:
            if len(a) != n:
                raise ValueError("all arrays must have same length along dim 0")
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, idx):
        return tuple(a[idx] for a in self.arrays)


class TransformDataset(Dataset):
    """Apply a function on each sample."""

    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        return self.transform(self.base[idx])
