"""Integration tests — verify end-to-end training flows.

Non-slow tests run by default. `-m slow` for MNIST/CIFAR.
"""
import numpy as np
import pytest

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F


def test_xor_converges():
    """Tiny MLP must solve XOR."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    model = nn.Sequential(
        nn.Linear(2, 8, seed=0),
        nn.Tanh(),
        nn.Linear(8, 1, seed=1),
    )
    opt = optim.Adam(model.parameters(), lr=0.1)

    for _ in range(500):
        pred = model(Tensor(X))
        loss = F.bce_with_logits_loss(pred, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()

    probs = F.sigmoid(model(Tensor(X))).data.reshape(-1)
    assert loss.item() < 0.05
    assert np.all((probs > 0.5) == y.reshape(-1).astype(bool))


def test_small_classification():
    """Train a small MLP on a synthetic 3-class gaussian blobs. Must reach 95%+ acc."""
    rng = np.random.default_rng(0)
    centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    N = 200
    X, y = [], []
    for cls, c in enumerate(centers):
        X.append(rng.standard_normal((N, 2)).astype(np.float32) * 0.3 + c)
        y.append(np.full(N, cls, dtype=np.int64))
    X = np.concatenate(X)
    y = np.concatenate(y)
    # shuffle
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    model = nn.Sequential(
        nn.Linear(2, 16, seed=0),
        nn.ReLU(),
        nn.Linear(16, 3, seed=1),
    )
    opt = optim.Adam(model.parameters(), lr=0.05)

    for _ in range(300):
        logits = model(Tensor(X))
        loss = F.cross_entropy(logits, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()

    logits = model(Tensor(X))
    pred = logits.data.argmax(axis=-1)
    acc = (pred == y).mean()
    assert acc >= 0.95, f"acc={acc}"


@pytest.mark.slow
def test_mnist_mlp_reaches_95():
    from nanograd.data import DataLoader
    from nanograd.data.mnist import MNIST

    train_ds = MNIST(train=True)
    test_ds = MNIST(train=False)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, seed=0)
    model = nn.Sequential(
        nn.Linear(784, 128, seed=0),
        nn.ReLU(),
        nn.Linear(128, 64, seed=1),
        nn.ReLU(),
        nn.Linear(64, 10, seed=2),
    )
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        for X, y in loader:
            logits = model(Tensor(X))
            loss = F.cross_entropy(logits, Tensor(y))
            opt.zero_grad()
            loss.backward()
            opt.step()
    # eval
    correct = 0
    B = 512
    for i in range(0, len(test_ds.X), B):
        pred = model(Tensor(test_ds.X[i : i + B])).data.argmax(axis=-1)
        correct += int((pred == test_ds.y[i : i + B]).sum())
    acc = correct / len(test_ds.X)
    assert acc >= 0.95, f"MNIST MLP acc={acc}"
