"""Variational Autoencoder on MNIST.

Demonstrates:
  - Encoder that outputs mu and log_var
  - Reparameterization trick: z = mu + exp(0.5*log_var) * epsilon
  - Composite loss = reconstruction (BCE-with-logits) + KL divergence

Goal: ELBO improves clearly from random baseline within 3 epochs on a subset.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.data import DataLoader, TensorDataset
from nanograd.data.mnist import MNIST
from nanograd.nn import functional as F
from nanograd.training import EMA


LATENT = 16


class Encoder(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, seed=0)
        self.fc2 = nn.Linear(256, 128, seed=1)
        self.mu = nn.Linear(128, latent, seed=2)
        self.logvar = nn.Linear(128, latent, seed=3)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.fc1 = nn.Linear(latent, 128, seed=4)
        self.fc2 = nn.Linear(128, 256, seed=5)
        self.out = nn.Linear(256, 784, seed=6)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.out(h)  # logits


class VAE(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.latent = latent
        self.enc = Encoder(latent)
        self.dec = Decoder(latent)

    def forward(self, x, rng):
        mu, logvar = self.enc(x)
        # reparameterization: z = mu + sigma * eps
        eps = Tensor(rng.standard_normal(mu.shape).astype(np.float32))
        # sigma = exp(0.5 * logvar)
        sigma = (logvar * 0.5).exp()
        z = mu + sigma * eps
        recon_logits = self.dec(z)
        return recon_logits, mu, logvar


def elbo_loss(recon_logits: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    # reconstruction via BCE-with-logits, summed over pixels, averaged over batch
    bce_per = F.bce_with_logits_loss(recon_logits, x)  # already mean over everything
    # want sum over pixels then mean over batch. Scale back up.
    N_px = recon_logits.shape[1]
    recon = bce_per * N_px  # convert mean-per-element back to mean-per-image summed-over-pixels
    # KL: 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)   [per-sample], mean over batch
    kl_per_sample = ((mu * mu + logvar.exp() - Tensor(np.ones_like(logvar.data)) - logvar) * 0.5).sum(axis=-1)
    kl = kl_per_sample.mean()
    return recon + kl, recon, kl


def main(subset: int = 5000, epochs: int = 3):
    train = MNIST(train=True, flatten=True)
    X = train.X[:subset]
    print(f"train subset: {X.shape[0]}", flush=True)

    # (image, image) dataset
    ds = TensorDataset(X, X)
    loader = DataLoader(ds, batch_size=64, shuffle=True, seed=0)

    model = VAE()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.95)
    print(f"params: {model.num_params():,}", flush=True)

    rng = np.random.default_rng(0)
    first_elbo = None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        elbo_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        n = 0
        for X_batch, _ in loader:
            recon_logits, mu, logvar = model(Tensor(X_batch), rng)
            loss, recon, kl = elbo_loss(recon_logits, Tensor(X_batch), mu, logvar)
            opt.zero_grad()
            loss.backward()
            optim.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            ema.update()
            elbo_sum += loss.item() * len(X_batch)
            recon_sum += recon.item() * len(X_batch)
            kl_sum += kl.item() * len(X_batch)
            n += len(X_batch)
        avg = elbo_sum / n
        if first_elbo is None:
            first_elbo = avg
        dt = time.time() - t0
        print(f"epoch {ep}  ELBO={avg:.2f}  recon={recon_sum/n:.2f}  KL={kl_sum/n:.2f}  ({dt:.1f}s)", flush=True)

    # Eval with EMA weights swapped in — should be at least as good (noisy SGD smoothed)
    eval_rng = np.random.default_rng(42)
    eval_X = Tensor(X[:256])

    def eval_elbo() -> float:
        logits, mu, lv = model(eval_X, eval_rng)
        return elbo_loss(logits, eval_X, mu, lv)[0].item()

    raw_eval = eval_elbo()
    with ema.swap_into(model):
        ema_eval = eval_elbo()
    print(f"\nELBO: {first_elbo:.2f} → {avg:.2f}", flush=True)
    print(f"eval ELBO (raw)={raw_eval:.2f}   (EMA)={ema_eval:.2f}", flush=True)
    assert avg < first_elbo - 10, f"VAE did not learn: {first_elbo:.2f} → {avg:.2f}"


if __name__ == "__main__":
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(subset=subset, epochs=epochs)
