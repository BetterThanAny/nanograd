"""Train a small Vision Transformer on MNIST. Target: test acc > 90% on 10k-sample subset."""
from __future__ import annotations

import time

import numpy as np

from nanograd import Tensor, optim
from nanograd.data import DataLoader
from nanograd.data.dataset import Dataset
from nanograd.data.mnist import MNIST
from nanograd.models import ViT
from nanograd.nn import functional as F


class MNISTImages(Dataset):
    """Wrap MNIST but keep (1, 28, 28) shape for ViT."""

    def __init__(self, train: bool, n: int | None = None, seed: int = 0):
        base = MNIST(train=train, flatten=False, normalize=True)
        X = base.X[:, None, :, :]  # (N, 1, 28, 28)
        y = base.y
        if n is not None and n < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=n, replace=False)
            X, y = X[idx], y[idx]
        self.X, self.y = X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def accuracy(model, X, y, batch_size: int = 256) -> float:
    correct = 0
    for i in range(0, len(X), batch_size):
        logits = model(Tensor(X[i : i + batch_size])).data
        correct += int((logits.argmax(-1) == y[i : i + batch_size]).sum())
    return correct / len(X)


def main():
    train = MNISTImages(train=True, n=10000, seed=0)
    test = MNISTImages(train=False, n=2000, seed=1)
    print(f"train {len(train)} test {len(test)}")

    loader = DataLoader(train, batch_size=64, shuffle=True, seed=0)
    model = ViT(
        image_size=28,
        patch_size=7,
        embed_dim=64,
        depth=4,
        num_heads=4,
        ff_dim=128,
        seed=0,
    )
    print(f"params: {model.num_params():,}")
    opt = optim.Adam(model.parameters(), lr=2e-3)
    sched = optim.CosineAnnealingLR(opt, T_max=6)

    epochs = 6
    for ep in range(1, epochs + 1):
        t0 = time.time()
        run, n = 0.0, 0
        for X, y in loader:
            logits = model(Tensor(X))
            loss = F.cross_entropy(logits, Tensor(y))
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(X)
            n += len(X)
        dt = time.time() - t0
        acc = accuracy(model, test.X, test.y)
        print(f"epoch {ep}  loss={run/n:.4f}  test_acc={acc:.4f}  lr={opt.lr:.5f}  ({dt:.1f}s)")
        sched.step()

    final = accuracy(model, test.X, test.y)
    print(f"\nfinal test acc: {final:.4f}")
    assert final > 0.90, f"ViT did not reach 90%: {final:.4f}"


if __name__ == "__main__":
    main()
