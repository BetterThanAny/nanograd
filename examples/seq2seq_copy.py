"""Seq2seq LSTM encoder-decoder on a copy task.

Task: given a sequence of symbols, reproduce it exactly.
    [5, 2, 3, 7] -> [5, 2, 3, 7]

Demonstrates:
  - Encoder LSTM compresses input to a hidden state
  - Decoder LSTM with teacher forcing reconstructs the sequence
  - Cross-entropy per-step loss
  - Greedy autoregressive inference at eval time
"""
from __future__ import annotations

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F


VOCAB = 10
SEQ_LEN = 5
HIDDEN = 64
# Special tokens
BOS = VOCAB       # vocab index = VOCAB (i.e., the extra "start" token)
EOS = VOCAB + 1   # end-of-sequence
FULL_VOCAB = VOCAB + 2


def make_batch(batch_size: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Produce a batch. src: (B, T) content tokens 0..9; tgt: (B, T+1) same with BOS prefix."""
    src = rng.integers(0, VOCAB, size=(batch_size, SEQ_LEN)).astype(np.int64)
    # decoder input: BOS + src; decoder target: src + EOS (shifted)
    bos = np.full((batch_size, 1), BOS, dtype=np.int64)
    eos = np.full((batch_size, 1), EOS, dtype=np.int64)
    dec_in = np.concatenate([bos, src], axis=1)                    # (B, T+1)
    dec_tgt = np.concatenate([src, eos], axis=1)                   # (B, T+1)
    return src, dec_in, dec_tgt


class Seq2seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(FULL_VOCAB, 32, seed=0)
        self.tgt_emb = nn.Embedding(FULL_VOCAB, 32, seed=1)
        self.encoder = nn.LSTM(32, HIDDEN, num_layers=1, seed=2)
        self.decoder = nn.LSTM(32, HIDDEN, num_layers=1, seed=3)
        self.head = nn.Linear(HIDDEN, FULL_VOCAB, seed=4)

    def forward(self, src_ids, dec_in_ids):
        x_src = self.src_emb(src_ids)             # (B, T, 32)
        _, enc_state = self.encoder(x_src)        # state = (h, c), each (1, B, H)

        x_dec = self.tgt_emb(dec_in_ids)          # (B, T+1, 32)
        dec_out, _ = self.decoder(x_dec, enc_state)  # (B, T+1, H)
        B, Tc, H = dec_out.shape
        return self.head(dec_out.reshape(B * Tc, H))

    def greedy_decode(self, src_ids, max_len: int = SEQ_LEN + 2) -> np.ndarray:
        """Autoregressive greedy generation."""
        self.eval()
        x_src = self.src_emb(src_ids)
        _, state = self.encoder(x_src)
        B = src_ids.shape[0]
        cur = np.full((B, 1), BOS, dtype=np.int64)
        produced = []
        for _ in range(max_len):
            x = self.tgt_emb(cur[:, -1:])              # (B, 1, 32)
            out, state = self.decoder(x, state)        # (B, 1, H)
            logits = self.head(out.reshape(B, -1))     # (B, V)
            nxt = logits.data.argmax(axis=-1)          # (B,)
            produced.append(nxt)
            cur = np.concatenate([cur, nxt[:, None]], axis=1)
        self.train()
        return np.stack(produced, axis=1)  # (B, max_len)


def main():
    rng = np.random.default_rng(0)
    model = Seq2seq()
    opt = optim.Adam(model.parameters(), lr=3e-3)
    print(f"params: {model.num_params():,}", flush=True)

    for step in range(1, 401):
        src, dec_in, dec_tgt = make_batch(32, rng)
        logits = model(src, dec_in)              # (B*(T+1), V)
        loss = F.cross_entropy(logits, Tensor(dec_tgt.reshape(-1)))
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss={loss.item():.4f}", flush=True)

    # evaluate: accuracy of exact sequence match on held-out batch
    src, _, _ = make_batch(64, rng)
    pred = model.greedy_decode(src)
    # trim predictions at EOS (if present) and compare to src
    correct = 0
    for i in range(64):
        p = pred[i]
        # stop at EOS if any
        eos_pos = np.where(p == EOS)[0]
        trimmed = p[: eos_pos[0] if len(eos_pos) else SEQ_LEN]
        if len(trimmed) == SEQ_LEN and np.array_equal(trimmed, src[i]):
            correct += 1
    acc = correct / 64
    print(f"\ngreedy exact-match accuracy: {acc:.4f}", flush=True)
    assert acc > 0.8, f"seq2seq copy accuracy too low: {acc:.2f}"


if __name__ == "__main__":
    main()
