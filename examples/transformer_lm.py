"""Tiny decoder-only Transformer LM on a repeating corpus.

Architecture:
  Embedding(V, D) + SinusoidalPositionalEncoding
  -> N x TransformerBlock(D, H)
  -> LayerNorm
  -> Linear(D, V)

Causal attention mask ensures each position only attends to earlier positions.

Demonstrates:
  - Embedding + positional encoding
  - MultiHeadAttention with causal mask
  - LayerNorm
  - Greedy autoregressive sampling
  - Gradient clipping
"""
from __future__ import annotations

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F


CORPUS = "the quick brown fox jumps over the lazy dog. " * 30


def build_model(vocab_size: int, d_model: int = 32, n_layers: int = 2, n_heads: int = 2, max_len: int = 128) -> nn.Module:
    class TransformerLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model, seed=0)
            self.pos = nn.SinusoidalPositionalEncoding(max_len, d_model)
            self.blocks = nn.Sequential(*[
                nn.TransformerBlock(d_model, n_heads, seed=i * 10)
                for i in range(n_layers)
            ])
            self.ln_final = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, seed=99)

        def forward(self, idx):
            B, T = idx.shape
            x = self.emb(idx)         # (B, T, D)
            x = self.pos(x)           # add positional
            # causal mask: position t attends to <= t
            mask = np.tril(np.ones((T, T), dtype=bool))[None, None, :, :]
            for blk in self.blocks:
                # TransformerBlock.forward takes optional mask
                x = blk(x, mask=mask)
            x = self.ln_final(x)
            return self.head(x)  # (B, T, V)

    return TransformerLM()


def main():
    # vocabulary
    chars = sorted(set(CORPUS))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    V = len(chars)
    ids = np.array([c2i[c] for c in CORPUS], dtype=np.int64)
    T = 32

    model = build_model(V, d_model=32, n_layers=2, n_heads=2, max_len=T + 4)
    opt = optim.Adam(model.parameters(), lr=3e-3)
    print(f"vocab={V}  params={model.num_params():,}", flush=True)

    rng = np.random.default_rng(0)
    for step in range(1, 301):
        start = rng.integers(0, len(ids) - T - 1)
        x = ids[start : start + T][None, :]
        y = ids[start + 1 : start + T + 1]
        logits = model(x)
        # reshape for cross_entropy: (T, V) vs (T,)
        B, Tc, V_ = logits.shape
        logits_flat = logits.reshape(B * Tc, V_)
        loss = F.cross_entropy(logits_flat, Tensor(y))
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss={loss.item():.4f}", flush=True)

    # greedy generation
    seed = "the quick"
    cur = np.array([c2i[c] for c in seed], dtype=np.int64)
    generated = list(seed)
    for _ in range(40):
        # take last T tokens (if exceeded)
        inp = cur[-T:] if len(cur) > T else cur
        logits = model(inp[None, :])        # (1, T, V)
        nxt = int(np.argmax(logits.data[0, -1]))
        generated.append(i2c[nxt])
        cur = np.append(cur, nxt)
    out = "".join(generated)
    print(f"\nseed='{seed}' → {out!r}", flush=True)
    assert "fox" in out or "lazy" in out or "dog" in out, f"Transformer LM did not learn: {out}"


if __name__ == "__main__":
    main()
