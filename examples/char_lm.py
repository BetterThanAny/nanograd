"""Char-level LSTM language model on a tiny corpus.

Trains to predict the next character of a repeating pattern. Demonstrates:
  - Embedding layer
  - LSTM on sequences
  - CrossEntropyLoss with token targets
"""
from __future__ import annotations

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F


CORPUS = "hello world, goodbye world, " * 20


def main():
    chars = sorted(set(CORPUS))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    V = len(chars)
    ids = np.array([c2i[c] for c in CORPUS], dtype=np.int64)
    T = 40  # context length

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, 32, seed=0)
            self.lstm = nn.LSTM(32, 64, seed=1)
            self.head = nn.Linear(64, V, seed=2)

        def forward(self, idx):
            x = self.emb(idx)           # (B, T, 32)
            out, _ = self.lstm(x)       # (B, T, 64)
            B, Tc, H = out.shape
            return self.head(out.reshape(B * Tc, H))  # (B*T, V)

    model = Net()
    opt = optim.Adam(model.parameters(), lr=3e-3)
    print(f"vocab={V}  params={model.num_params():,}")

    rng = np.random.default_rng(0)
    for step in range(1, 401):
        # sample a random window
        start = rng.integers(0, len(ids) - T - 1)
        x = ids[start : start + T][None, :]     # (1, T)
        y = ids[start + 1 : start + T + 1]      # (T,)
        logits = model(x)
        loss = F.cross_entropy(logits, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss={loss.item():.4f}")

    # sample greedily from "hello"
    seed = "hello"
    cur = np.array([c2i[c] for c in seed], dtype=np.int64)
    generated = list(seed)
    for _ in range(40):
        logits = model(cur[None, :])
        last = logits.data[-1]  # logits for last step
        nxt = int(np.argmax(last))
        generated.append(i2c[nxt])
        cur = np.append(cur, nxt)
    out = "".join(generated)
    print(f"\nseed='{seed}'  generated: {out!r}")
    # should contain recognizable corpus patterns
    assert "world" in out or "hello" in out[5:], f"LM did not learn: {out}"


if __name__ == "__main__":
    main()
