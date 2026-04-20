"""CIFAR-10 loader. Downloads the official pickle archive and caches locally."""
from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

from nanograd.data.dataset import Dataset


URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _default_cache() -> Path:
    return Path.home() / ".nanograd" / "cifar10"


def _download_and_extract(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_dir / "cifar-10-batches-py"
    if root.exists() and (root / "data_batch_1").exists():
        return root
    tgz = cache_dir / "cifar-10-python.tar.gz"
    if not tgz.exists():
        print(f"[cifar] downloading {URL}")
        with urllib.request.urlopen(URL, timeout=300) as r, open(tgz, "wb") as f:
            f.write(r.read())
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(cache_dir)
    return root


def _load_batch(path: Path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32)
    y = np.array(d[b"labels"], dtype=np.int64)
    return X, y


def load_cifar10(
    cache_dir: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = Path(cache_dir) if cache_dir else _default_cache()
    root = _download_and_extract(cache)
    X_parts, y_parts = [], []
    for i in range(1, 6):
        X, y = _load_batch(root / f"data_batch_{i}")
        X_parts.append(X)
        y_parts.append(y)
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    X_test, y_test = _load_batch(root / "test_batch")
    return X_train, y_train, X_test, y_test


class CIFAR10(Dataset):
    CLASSES = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    def __init__(
        self,
        train: bool = True,
        cache_dir: str | Path | None = None,
        normalize: bool = True,
    ):
        X_tr, y_tr, X_te, y_te = load_cifar10(cache_dir)
        self.X = (X_tr if train else X_te).astype(np.float32)
        self.y = y_tr if train else y_te
        if normalize:
            # standard per-channel mean/std
            mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32) * 255
            std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32) * 255
            self.X = (self.X - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
        else:
            self.X /= 255.0

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
