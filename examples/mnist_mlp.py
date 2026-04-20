"""Train an MLP on MNIST. Target: >=95% test accuracy in <= 5 epochs."""
from __future__ import annotations

import time

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.data import DataLoader
from nanograd.data.mnist import MNIST
from nanograd.nn import functional as F


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(784, 128, seed=0),
        nn.ReLU(),
        nn.Linear(128, 64, seed=1),
        nn.ReLU(),
        nn.Linear(64, 10, seed=2),
    )


def accuracy(model, X: np.ndarray, y: np.ndarray, batch_size: int = 512) -> float:
    correct = 0
    for i in range(0, len(X), batch_size):
        logits = model(Tensor(X[i : i + batch_size])).data
        pred = logits.argmax(axis=-1)
        correct += int((pred == y[i : i + batch_size]).sum())
    return correct / len(X)


def main():
    train_ds = MNIST(train=True)
    test_ds = MNIST(train=False)
    print(f"train: {len(train_ds)}  test: {len(test_ds)}")

    loader = DataLoader(train_ds, batch_size=128, shuffle=True, seed=0)
    model = build_model()
    print(f"params: {model.num_params():,}")
    opt = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        n = 0
        for X, y in loader:
            logits = model(Tensor(X))
            loss = F.cross_entropy(logits, Tensor(y))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(X)
            n += len(X)
        dt = time.time() - t0
        train_loss = running / n
        test_acc = accuracy(model, test_ds.X, test_ds.y)
        print(f"epoch {ep}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}  ({dt:.1f}s)")

    final_acc = accuracy(model, test_ds.X, test_ds.y)
    print(f"\nfinal test acc: {final_acc:.4f}")
    assert final_acc >= 0.95, f"MNIST MLP did not reach 95%: {final_acc:.4f}"


if __name__ == "__main__":
    main()
