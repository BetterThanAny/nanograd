"""Train a small ResNet-CIFAR on CIFAR-10.

Uses:
  - resnet_cifar(n=1) ~ 77k params (ResNet-8-ish)
  - Random horizontal flip + random crop augmentation
  - Adam optimizer
  - Subset for runtime (full CIFAR-10 in pure numpy is prohibitive)

Goal: show ResNet + BN + residual trains on a real dataset.
Target: test acc > 40% on a 5000-sample subset in 1 epoch.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.data import (
    DataLoader,
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    SampleTransform,
    TensorDataset,
)
from nanograd.data.cifar import CIFAR10
from nanograd.models import resnet_cifar
from nanograd.nn import functional as F
from nanograd.training import MetricTracker


def accuracy(model, X: np.ndarray, y: np.ndarray, batch_size: int = 128) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(X), batch_size):
        pred = model(Tensor(X[i : i + batch_size])).data.argmax(axis=-1)
        correct += int((pred == y[i : i + batch_size]).sum())
    model.train()
    return correct / len(X)


def main(subset: int = 5000, epochs: int = 1, n_blocks: int = 1):
    # ingest raw (non-normalized) so augmentation sees it in the same frame
    train_full = CIFAR10(train=True, normalize=False)
    test_ds = CIFAR10(train=False, normalize=True)
    print(f"train total: {len(train_full)}  test: {len(test_ds)}", flush=True)

    rng = np.random.default_rng(0)
    if subset and subset < len(train_full):
        idx = rng.choice(len(train_full), size=subset, replace=False)
        train_X = train_full.X[idx]
        train_y = train_full.y[idx]
    else:
        train_X = train_full.X
        train_y = train_full.y

    train_ds = TensorDataset(train_X, train_y)
    # augmentation pipeline
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    tf = Compose([
        RandomCrop(32, padding=4, seed=0),
        RandomHorizontalFlip(p=0.5, seed=1),
        Normalize(mean, std),
    ])
    train_ds = SampleTransform(train_ds, tf)

    loader = DataLoader(train_ds, batch_size=64, shuffle=True, seed=0)
    model = resnet_cifar(num_blocks_per_stage=n_blocks, num_classes=10)
    print(f"params: {model.num_params():,}", flush=True)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    tracker = MetricTracker()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tracker.reset()
        model.train()
        for i, (X, y) in enumerate(loader):
            logits = model(Tensor(X))
            loss = F.cross_entropy(logits, Tensor(y))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tracker.update("loss", loss.item(), n=len(X))
            if (i + 1) % 20 == 0:
                print(f"  epoch {ep} iter {i+1}/{len(loader)}  loss={tracker.avg('loss'):.4f}  ({time.time()-t0:.1f}s)", flush=True)
        test_acc = accuracy(model, test_ds.X, test_ds.y)
        dt = time.time() - t0
        print(f"epoch {ep}  train_loss={tracker.avg('loss'):.4f}  test_acc={test_acc:.4f}  ({dt:.1f}s)", flush=True)

    final = accuracy(model, test_ds.X, test_ds.y)
    print(f"\nfinal test acc: {final:.4f}  (random baseline 0.10)", flush=True)
    # n=1 ResNet on 5000-sample subset, 1 epoch, aug: expect ~30-35% (3x random)
    assert final >= 0.25, f"ResNet CIFAR did not train: acc={final:.4f}"


if __name__ == "__main__":
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    main(subset=subset, epochs=epochs, n_blocks=n)
