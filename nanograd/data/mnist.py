"""MNIST loader.

Source priority (uses http proxy if set):
  1. Local cache at ``cache_dir`` (default: ~/.nanograd/mnist)
  2. Download from cvdf-datasets S3 mirror
  3. Fallback to yann.lecun.com

Returns uint8 images (N, 28, 28) and int64 labels (N,).
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

from nanograd.data.dataset import Dataset


MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _default_cache() -> Path:
    return Path.home() / ".nanograd" / "mnist"


def _download(fname: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / fname
    if dst.exists():
        return dst
    last_err = None
    for mirror in MIRRORS:
        url = mirror + fname
        try:
            print(f"[mnist] downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
                f.write(r.read())
            return dst
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"all MNIST mirrors failed, last error: {last_err}")


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad magic {magic}")
        buf = f.read(n * rows * cols)
        return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad magic {magic}")
        return np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)


def load_mnist(
    cache_dir: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test). Images are uint8 (N, 28, 28)."""
    cache = Path(cache_dir) if cache_dir else _default_cache()
    paths = {k: _download(v, cache) for k, v in FILES.items()}
    X_train = _read_images(paths["train_images"])
    y_train = _read_labels(paths["train_labels"])
    X_test = _read_images(paths["test_images"])
    y_test = _read_labels(paths["test_labels"])
    return X_train, y_train, X_test, y_test


class MNIST(Dataset):
    """MNIST dataset; flattens images and converts to float32 by default."""

    def __init__(
        self,
        train: bool = True,
        cache_dir: str | Path | None = None,
        flatten: bool = True,
        normalize: bool = True,
    ):
        X_tr, y_tr, X_te, y_te = load_mnist(cache_dir)
        self.X = X_tr if train else X_te
        self.y = y_tr if train else y_te
        if normalize:
            self.X = self.X.astype(np.float32) / 255.0
        else:
            self.X = self.X.astype(np.float32)
        if flatten:
            self.X = self.X.reshape(len(self.X), -1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
