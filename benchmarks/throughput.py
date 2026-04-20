"""Throughput microbenchmarks across workloads.

Measures forward + backward steps/sec for:
  - MLP on MNIST-sized batch
  - CNN (LeNet-ish) on MNIST-sized batch
  - LSTM on small sequence
  - TransformerBlock on small seq/embed

No PyTorch comparison — this is absolute throughput for nanograd itself.
"""
from __future__ import annotations

import time

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F


def bench(name: str, setup_fn, iters: int = 30, warmup: int = 3):
    model, step = setup_fn()
    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    dt = time.perf_counter() - t0
    params = model.num_params()
    print(f"{name:25s}  params={params:>8,}  {iters/dt:>6.1f} steps/s  ({dt*1000/iters:>6.1f} ms/step)")


def mlp_setup(batch: int = 64):
    X = np.random.default_rng(0).standard_normal((batch, 784)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 10, size=(batch,))
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    opt = optim.Adam(model.parameters(), lr=1e-3)

    def step():
        logits = model(Tensor(X))
        loss = F.cross_entropy(logits, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model, step


def cnn_setup(batch: int = 32):
    X = np.random.default_rng(0).standard_normal((batch, 1, 28, 28)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 10, size=(batch,))
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    opt = optim.Adam(model.parameters(), lr=1e-3)

    def step():
        logits = model(Tensor(X))
        loss = F.cross_entropy(logits, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model, step


def lstm_setup(batch: int = 32, T: int = 16, D: int = 32, H: int = 64):
    X = np.random.default_rng(0).standard_normal((batch, T, D)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 10, size=(batch,))
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(D, H)
            self.head = nn.Linear(H, 10)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h)

    model = Net()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    def step():
        logits = model(Tensor(X))
        loss = F.cross_entropy(logits, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model, step


def transformer_setup(batch: int = 8, T: int = 16, D: int = 32, H: int = 2):
    X = np.random.default_rng(0).standard_normal((batch, T, D)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 10, size=(batch * T,))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.TransformerBlock(D, H)
            self.head = nn.Linear(D, 10)

        def forward(self, x):
            y = self.block(x)  # (B, T, D)
            B, T, D = y.shape
            return self.head(y).reshape(B * T, 10)

    model = Net()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    def step():
        logits = model(Tensor(X))
        loss = F.cross_entropy(logits, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model, step


if __name__ == "__main__":
    print("nanograd throughput benchmarks\n")
    bench("MLP (b=64, 784)",   mlp_setup)
    bench("CNN (b=32, 1x28)",  cnn_setup, iters=15)
    bench("LSTM (b=32, T=16)", lstm_setup)
    bench("Transformer (b=8,T=16)", transformer_setup, iters=15)
