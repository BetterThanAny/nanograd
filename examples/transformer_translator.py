"""Encoder-decoder Transformer on a sequence-reversal task.

Input:  random digit sequence [3, 7, 1, 9, 4]
Target: reversed                [4, 9, 1, 7, 3]

Reversal is hard for purely autoregressive models without attention because
position t of the output depends on position (T-1-t) of the input — i.e., long-range.
The Transformer's cross-attention is the natural mechanism for this.

Architecture:
  Encoder: emb + pos -> N × TransformerBlock (self-attn, no mask)
  Decoder: emb + pos -> N × TransformerDecoderBlock (causal self-attn + cross-attn)
  Head: Linear(D, V)
"""
from __future__ import annotations

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F


VOCAB = 10
SEQ_LEN = 6
D_MODEL = 48
N_HEADS = 4
N_LAYERS = 2
BOS = VOCAB
EOS = VOCAB + 1
FULL_VOCAB = VOCAB + 2


def make_batch(batch_size: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = rng.integers(0, VOCAB, size=(batch_size, SEQ_LEN)).astype(np.int64)
    tgt = src[:, ::-1].copy()  # reversed
    bos = np.full((batch_size, 1), BOS, dtype=np.int64)
    eos = np.full((batch_size, 1), EOS, dtype=np.int64)
    dec_in = np.concatenate([bos, tgt], axis=1)
    dec_out = np.concatenate([tgt, eos], axis=1)
    return src, dec_in, dec_out


class Translator(nn.Module):
    def __init__(self):
        super().__init__()
        # src and tgt share token space
        self.src_emb = nn.Embedding(FULL_VOCAB, D_MODEL, seed=0)
        self.tgt_emb = nn.Embedding(FULL_VOCAB, D_MODEL, seed=1)
        self.pos_src = nn.SinusoidalPositionalEncoding(64, D_MODEL)
        self.pos_tgt = nn.SinusoidalPositionalEncoding(64, D_MODEL)

        # encoder
        self.enc_blocks = nn.Sequential(*[
            nn.TransformerBlock(D_MODEL, N_HEADS, seed=10 + i) for i in range(N_LAYERS)
        ])
        self.enc_ln = nn.LayerNorm(D_MODEL)

        # decoder
        self._dec_blocks = []
        for i in range(N_LAYERS):
            b = nn.TransformerDecoderBlock(D_MODEL, N_HEADS, seed=100 + i * 5)
            setattr(self, f"dec_{i}", b)
            self._dec_blocks.append(b)
        self.dec_ln = nn.LayerNorm(D_MODEL)

        self.head = nn.Linear(D_MODEL, FULL_VOCAB, seed=200)

    def encode(self, src_ids):
        x = self.src_emb(src_ids)
        x = self.pos_src(x)
        x = self.enc_blocks(x)
        return self.enc_ln(x)

    def decode(self, dec_in_ids, context):
        x = self.tgt_emb(dec_in_ids)
        x = self.pos_tgt(x)
        T = x.shape[1]
        causal = np.tril(np.ones((T, T), dtype=bool))[None, None, :, :]
        for blk in self._dec_blocks:
            x = blk(x, context, causal_mask=causal)
        return self.head(self.dec_ln(x))

    def forward(self, src_ids, dec_in_ids):
        ctx = self.encode(src_ids)
        return self.decode(dec_in_ids, ctx)

    def greedy_decode(self, src_ids, max_len=SEQ_LEN + 2):
        self.eval()
        ctx = self.encode(src_ids)
        B = src_ids.shape[0]
        cur = np.full((B, 1), BOS, dtype=np.int64)
        produced = []
        for _ in range(max_len):
            logits = self.decode(cur, ctx)          # (B, T, V)
            nxt = logits.data[:, -1].argmax(axis=-1)  # (B,)
            produced.append(nxt)
            cur = np.concatenate([cur, nxt[:, None]], axis=1)
        self.train()
        return np.stack(produced, axis=1)


def main():
    rng = np.random.default_rng(0)
    model = Translator()
    opt = optim.Adam(model.parameters(), lr=3e-3)
    print(f"params: {model.num_params():,}", flush=True)

    for step in range(1, 601):
        src, dec_in, dec_tgt = make_batch(64, rng)
        logits = model(src, dec_in)                # (B, T+1, V)
        B, Tc, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * Tc, V), Tensor(dec_tgt.reshape(-1)))
        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step:4d}  loss={loss.item():.4f}", flush=True)

    # eval
    src, _, _ = make_batch(64, rng)
    pred = model.greedy_decode(src)
    correct = 0
    for i in range(64):
        p = pred[i]
        eos_pos = np.where(p == EOS)[0]
        trimmed = p[: eos_pos[0] if len(eos_pos) else SEQ_LEN]
        expected = src[i][::-1]
        if len(trimmed) == SEQ_LEN and np.array_equal(trimmed, expected):
            correct += 1
    acc = correct / 64
    print(f"\ngreedy exact-match (reversal) accuracy: {acc:.4f}", flush=True)
    # show sample
    print(f"example: src={src[0].tolist()}  pred={pred[0].tolist()}  expected={src[0][::-1].tolist()}", flush=True)
    assert acc > 0.7, f"translator accuracy too low: {acc:.2f}"


if __name__ == "__main__":
    main()
