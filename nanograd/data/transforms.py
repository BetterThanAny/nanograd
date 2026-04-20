"""Image data augmentation transforms.

All transforms take and return numpy arrays. Shape conventions:
  - (C, H, W) for a single image
  - (N, C, H, W) for a batch

Compose via :class:`Compose` or :class:`SampleTransform` (applies per-sample inside Dataset).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


class Transform:
    def __call__(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


class Normalize(Transform):
    """Subtract mean and divide by std, per-channel.

    Accepts (C, H, W) or (N, C, H, W).
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.p:
            return x[..., ::-1].copy()
        return x


class RandomCrop(Transform):
    """Pad then crop (H, W) with random offset."""

    def __init__(self, size: int | Tuple[int, int], padding: int = 0, seed: Optional[int] = None):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.padding = padding
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            pw = ((0, 0),) * (x.ndim - 2) + ((self.padding, self.padding), (self.padding, self.padding))
            x = np.pad(x, pw)
        H, W = x.shape[-2], x.shape[-1]
        th, tw = self.size
        top = int(self.rng.integers(0, H - th + 1))
        left = int(self.rng.integers(0, W - tw + 1))
        return x[..., top : top + th, left : left + tw]


class ToFloat(Transform):
    """Cast to float32 and optionally divide by 255."""

    def __init__(self, scale: float = 1.0 / 255.0):
        self.scale = scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32) * self.scale


class SampleTransform:
    """Wrap a Dataset so that a per-sample transform is applied at __getitem__.

    Assumes each sample is either an array or a (data, target) tuple; the
    transform is applied to ``data`` only.
    """

    def __init__(self, dataset, transform: Transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, tuple):
            data, *rest = sample
            return (self.transform(data), *rest)
        return self.transform(sample)
