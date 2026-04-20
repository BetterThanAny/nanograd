"""Small DCGAN-style GAN on MNIST.

Demonstrates ConvTranspose2d + BatchNorm2d + BCE-with-logits loss + two-optimizer training loop.

Generator: z (noise) -> ConvTranspose chain -> 28x28 image
Discriminator: image -> Conv chain -> scalar logit

Goal: show that D/G losses stabilize around equilibrium and samples look better than pure noise.
This is not a full GAN training run — just a smoke test of the components together.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.data.mnist import MNIST
from nanograd.nn import functional as F


LATENT = 64


class Generator(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.fc = nn.Linear(latent, 128 * 7 * 7, seed=0)
        self.bn0 = nn.BatchNorm2d(128)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, seed=1)  # 7->14
        self.bn1 = nn.BatchNorm2d(64)
        self.up2 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, seed=2)   # 14->28

    def forward(self, z):
        x = self.fc(z).reshape(z.shape[0], 128, 7, 7)
        x = F.relu(self.bn0(x))
        x = F.relu(self.bn1(self.up1(x)))
        x = F.tanh(self.up2(x))  # output in [-1, 1]
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 4, stride=2, padding=1, seed=10)   # 28->14
        self.c2 = nn.Conv2d(32, 64, 4, stride=2, padding=1, seed=11)  # 14->7
        self.bn = nn.BatchNorm2d(64)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 1, seed=12)

    def forward(self, x):
        x = F.leaky_relu(self.c1(x), 0.2)
        x = F.leaky_relu(self.bn(self.c2(x)), 0.2)
        return self.fc(self.flat(x))


def main(subset: int = 2000, steps: int = 200, batch_size: int = 32):
    train_full = MNIST(train=True, flatten=False)
    X_real = train_full.X[:subset, None, :, :]  # (N, 1, 28, 28)
    X_real = X_real * 2 - 1  # scale to [-1, 1] to match tanh
    print(f"real: {X_real.shape}", flush=True)

    G = Generator()
    D = Discriminator()
    g_opt = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    print(f"G params={G.num_params():,}  D params={D.num_params():,}", flush=True)

    rng = np.random.default_rng(0)
    initial_d_loss = None
    initial_g_loss = None

    t0 = time.time()
    for step in range(1, steps + 1):
        # sample real
        idx = rng.integers(0, len(X_real), size=batch_size)
        real = X_real[idx]
        # sample fake
        z = rng.standard_normal((batch_size, LATENT)).astype(np.float32)
        fake = G(Tensor(z)).detach()

        # discriminator step
        d_real_logits = D(Tensor(real))
        d_fake_logits = D(fake)
        d_loss = F.bce_with_logits_loss(d_real_logits, Tensor(np.ones_like(d_real_logits.data))) \
            + F.bce_with_logits_loss(d_fake_logits, Tensor(np.zeros_like(d_fake_logits.data)))
        d_opt.zero_grad()
        d_loss.backward()
        optim.clip_grad_norm_(D.parameters(), max_norm=5.0)
        d_opt.step()

        # generator step
        z = rng.standard_normal((batch_size, LATENT)).astype(np.float32)
        fake = G(Tensor(z))
        g_logits = D(fake)
        # generator wants D to output 1 (real)
        g_loss = F.bce_with_logits_loss(g_logits, Tensor(np.ones_like(g_logits.data)))
        g_opt.zero_grad()
        g_loss.backward()
        optim.clip_grad_norm_(G.parameters(), max_norm=5.0)
        g_opt.step()

        if initial_d_loss is None:
            initial_d_loss = d_loss.item()
            initial_g_loss = g_loss.item()

        if step % 20 == 0:
            print(f"step {step:3d}  d_loss={d_loss.item():.4f}  g_loss={g_loss.item():.4f}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\ninitial: d={initial_d_loss:.4f} g={initial_g_loss:.4f}", flush=True)
    print(f"final  : d={d_loss.item():.4f} g={g_loss.item():.4f}", flush=True)

    # Stability check: both losses must be finite + bounded
    assert np.isfinite(d_loss.item())
    assert np.isfinite(g_loss.item())
    assert d_loss.item() < 10.0, f"d_loss exploded: {d_loss.item()}"
    assert g_loss.item() < 10.0, f"g_loss exploded: {g_loss.item()}"


if __name__ == "__main__":
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    main(subset=subset, steps=steps)
