"""Train a small U-Net on MNIST as an autoencoder. Target: MSE < 0.05 on held-out subset."""
from __future__ import annotations

import time

import numpy as np

from nanograd import Tensor, optim
from nanograd.data import DataLoader
from nanograd.data.dataset import Dataset
from nanograd.data.mnist import MNIST
from nanograd.models.unet import UNet
from nanograd.nn import functional as F


class MNISTImages(Dataset):
    def __init__(self, train: bool, n: int | None = None, seed: int = 0):
        base = MNIST(train=train, flatten=False, normalize=True)
        X = base.X[:, None, :, :]
        if n is not None and n < len(X):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=n, replace=False)
            X = X[idx]
        self.X = X

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.X[i]  # self-supervised


def eval_mse(model, X, batch_size: int = 256) -> float:
    total, n = 0.0, 0
    for i in range(0, len(X), batch_size):
        xb = Tensor(X[i : i + batch_size])
        y = model(xb).data
        total += float(((y - X[i : i + batch_size]) ** 2).sum())
        n += X[i : i + batch_size].size
    return total / n


def main():
    train = MNISTImages(train=True, n=1500, seed=0)
    test = MNISTImages(train=False, n=500, seed=1)
    print(f"train {len(train)} test {len(test)}", flush=True)

    loader = DataLoader(train, batch_size=32, shuffle=True, seed=0)
    model = UNet(in_channels=1, out_channels=1, base=8, seed=0)
    print(f"params: {model.num_params():,}", flush=True)
    opt = optim.Adam(model.parameters(), lr=3e-3)

    epochs = 2
    for ep in range(1, epochs + 1):
        t0 = time.time()
        run, n = 0.0, 0
        for X, _y in loader:
            xb = Tensor(X)
            y = model(xb)
            loss = F.mse_loss(y, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(X)
            n += len(X)
        dt = time.time() - t0
        test_mse = eval_mse(model, test.X)
        print(f"epoch {ep}  train_mse={run/n:.4f}  test_mse={test_mse:.4f}  ({dt:.1f}s)", flush=True)

    final = eval_mse(model, test.X)
    print(f"\nfinal test mse: {final:.4f}")
    assert final < 0.05, f"U-Net AE did not reach MSE<0.05: {final:.4f}"


if __name__ == "__main__":
    main()
