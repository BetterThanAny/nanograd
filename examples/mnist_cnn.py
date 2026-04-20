"""Train a small CNN on MNIST. Target: >=98% test accuracy.

Model (~60k params):
  Conv(1->16, 3x3, pad=1) - ReLU - MaxPool(2)   # 28 -> 14
  Conv(16->32, 3x3, pad=1) - ReLU - MaxPool(2)  # 14 -> 7
  Flatten - Linear(32*7*7 -> 64) - ReLU - Linear(64 -> 10)
"""
from __future__ import annotations

import time

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.data import DataLoader
from nanograd.data.mnist import MNIST
from nanograd.nn import functional as F


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1, seed=0),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1, seed=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64, seed=2),
        nn.ReLU(),
        nn.Linear(64, 10, seed=3),
    )


def accuracy(model, X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(X), batch_size):
        pred = model(Tensor(X[i : i + batch_size])).data.argmax(axis=-1)
        correct += int((pred == y[i : i + batch_size]).sum())
    model.train()
    return correct / len(X)


def main():
    train_ds = MNIST(train=True, flatten=False)
    test_ds = MNIST(train=False, flatten=False)
    # add channel dim: (N, 28, 28) -> (N, 1, 28, 28)
    train_ds.X = train_ds.X[:, None, :, :]
    test_ds.X = test_ds.X[:, None, :, :]
    print(f"train: {len(train_ds)}  test: {len(test_ds)}")

    loader = DataLoader(train_ds, batch_size=64, shuffle=True, seed=0)
    model = build_model()
    print(f"params: {model.num_params():,}")
    opt = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2
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
            if (i + 1) % 100 == 0:
                print(f"  epoch {ep} iter {i+1}/{len(loader)}  loss={running/n:.4f}  ({time.time()-t0:.1f}s)")
        train_loss = running / n
        test_acc = accuracy(model, test_ds.X, test_ds.y)
        dt = time.time() - t0
        print(f"epoch {ep}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}  ({dt:.1f}s)")

    final_acc = accuracy(model, test_ds.X, test_ds.y)
    print(f"\nfinal test acc: {final_acc:.4f}")
    # accept 98% target
    assert final_acc >= 0.98, f"CNN did not reach 98%: {final_acc:.4f}"


if __name__ == "__main__":
    main()
