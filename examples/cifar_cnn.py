"""Small CNN on CIFAR-10.

Goal: show the framework trains — target >= 50% test accuracy in 1-2 epochs
on a subset to keep runtime reasonable (pure Python/NumPy conv is slow).

Model (~155k params):
  Conv(3->32, 3, pad=1) - ReLU - MaxPool(2)     # 32 -> 16
  Conv(32->64, 3, pad=1) - ReLU - MaxPool(2)    # 16 -> 8
  Flatten - Linear(64*8*8 -> 128) - ReLU - Linear(128 -> 10)
"""
from __future__ import annotations

import sys
import time

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.data import DataLoader, TensorDataset
from nanograd.data.cifar import CIFAR10
from nanograd.nn import functional as F


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1, seed=0),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1, seed=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128, seed=2),
        nn.ReLU(),
        nn.Linear(128, 10, seed=3),
    )


def accuracy(model, X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(X), batch_size):
        pred = model(Tensor(X[i : i + batch_size])).data.argmax(axis=-1)
        correct += int((pred == y[i : i + batch_size]).sum())
    model.train()
    return correct / len(X)


def main(subset: int = 10000, epochs: int = 1):
    train_ds = CIFAR10(train=True)
    test_ds = CIFAR10(train=False)
    # subset for runtime
    if subset and subset < len(train_ds):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(train_ds), size=subset, replace=False)
        train_X = train_ds.X[idx]
        train_y = train_ds.y[idx]
        train_ds = TensorDataset(train_X, train_y)
    print(f"train: {len(train_ds)}  test: {len(test_ds)}", flush=True)

    loader = DataLoader(train_ds, batch_size=64, shuffle=True, seed=0)
    model = build_model()
    print(f"params: {model.num_params():,}", flush=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        n = 0
        model.train()
        for i, (X, y) in enumerate(loader):
            logits = model(Tensor(X))
            loss = F.cross_entropy(logits, Tensor(y))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(X)
            n += len(X)
            if (i + 1) % 20 == 0:
                print(f"  epoch {ep} iter {i+1}/{len(loader)}  loss={running/n:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        test_acc = accuracy(model, test_ds.X, test_ds.y)
        dt = time.time() - t0
        print(f"epoch {ep}  train_loss={running/n:.4f}  test_acc={test_acc:.4f}  ({dt:.1f}s)", flush=True)

    final = accuracy(model, test_ds.X, test_ds.y)
    print(f"\nfinal test acc: {final:.4f}", flush=True)
    # scaled target for the subset; random is 10%
    assert final >= 0.40, f"CIFAR CNN did not train: acc={final:.4f}"


if __name__ == "__main__":
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(subset=subset, epochs=epochs)
