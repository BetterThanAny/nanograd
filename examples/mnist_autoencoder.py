"""Conv autoencoder on MNIST — demonstrates ConvTranspose2d.

Encoder: Conv(1->16, s2) -> Conv(16->32, s2)   # 28->14->7
Decoder: ConvT(32->16, k2, s2) -> ConvT(16->1, k2, s2)  # 7->14->28

Target: MSE loss drops well below random (~0.1 initial) to ~0.02 in 1 epoch on subset.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.data import DataLoader, TensorDataset
from nanograd.data.mnist import MNIST
from nanograd.nn import functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, seed=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, seed=1),
            nn.ReLU(),
        )
        # encoded: (N, 32, 7, 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0, seed=2),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=0, seed=3),   # 14->28
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def main(subset: int = 5000, epochs: int = 1):
    train_full = MNIST(train=True, flatten=False)
    # add channel dim and slice subset
    X = train_full.X[:subset, None, :, :]
    print(f"train subset: {X.shape[0]} images", flush=True)

    # dataset yields (image, image) — x == target
    ds = TensorDataset(X, X)
    loader = DataLoader(ds, batch_size=64, shuffle=True, seed=0)

    model = Autoencoder()
    print(f"params: {model.num_params():,}", flush=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        n = 0
        model.train()
        for i, (X_batch, _) in enumerate(loader):
            recon = model(Tensor(X_batch))
            loss = F.mse_loss(recon, Tensor(X_batch))
            opt.zero_grad()
            loss.backward()
            # clip for stability
            optim.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running += loss.item() * len(X_batch)
            n += len(X_batch)
            if (i + 1) % 20 == 0:
                print(f"  epoch {ep} iter {i+1}/{len(loader)}  loss={running/n:.6f}  ({time.time()-t0:.1f}s)", flush=True)
        print(f"epoch {ep}  train_mse={running/n:.6f}  ({time.time()-t0:.1f}s)", flush=True)

    # eval: reconstruction MSE on a held-out chunk
    model.eval()
    test_X = train_full.X[subset:subset + 256, None, :, :]
    recon = model(Tensor(test_X)).data
    final_mse = float(np.mean((recon - test_X) ** 2))
    print(f"\nheldout reconstruction MSE: {final_mse:.6f}", flush=True)
    # should be well below initial (~0.1) — typical good autoencoder < 0.03
    assert final_mse < 0.05, f"autoencoder did not reconstruct: mse={final_mse:.4f}"


if __name__ == "__main__":
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(subset=subset, epochs=epochs)
