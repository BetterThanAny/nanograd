"""Multi-head attention and a minimal Transformer block."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from nanograd.function import Function
from nanograd.nn import functional as F
from nanograd.nn.layers import LayerNorm, Linear
from nanograd.nn.module import Module, Parameter
from nanograd.tensor import Tensor


def sinusoidal_positional_encoding(max_len: int, dim: int) -> np.ndarray:
    """Classic Transformer positional encoding (Vaswani et al. 2017).

    Returns (max_len, dim) ndarray of sin/cos frequencies.
    Works for both even and odd ``dim``.
    """
    pe = np.zeros((max_len, dim), dtype=np.float32)
    pos = np.arange(max_len, dtype=np.float32)[:, None]
    # div has len ceil(dim/2); even-indexed cols use full div, odd-indexed cols use first floor(dim/2)
    div = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: dim // 2])
    return pe


class SinusoidalPositionalEncoding(Module):
    """Adds fixed sinusoidal positional encoding to input."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        pe = sinusoidal_positional_encoding(max_len, dim)
        self.register_buffer("pe", pe)
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        T = x.shape[-2]
        assert T <= self.max_len, f"seq length {T} exceeds max_len {self.max_len}"
        return x + Tensor(self.pe[:T])


class LearnedPositionalEncoding(Module):
    """Learnable position embedding."""

    def __init__(self, max_len: int, dim: int, seed: Optional[int] = None):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.weight = Parameter(rng.standard_normal((max_len, dim)).astype(np.float32) * 0.02)
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        T = x.shape[-2]
        assert T <= self.max_len
        return x + self.weight[:T]


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[np.ndarray] = None,
) -> Tensor:
    """q, k, v: (..., T, D). Return (..., T, D)."""
    d = q.shape[-1]
    scores = q @ k.transpose(*range(k.ndim - 2), k.ndim - 1, k.ndim - 2)
    scores = scores * (1.0 / math.sqrt(d))
    if mask is not None:
        # broadcast-add a large negative where mask == 0
        scores = scores + Tensor(np.where(mask, 0.0, -1e9).astype(np.float32))
    attn = F.softmax(scores, axis=-1)
    return attn @ v


class MultiHeadAttention(Module):
    """Self-attention (Q=K=V from same input)."""

    def __init__(self, embed_dim: int, num_heads: int, seed: Optional[int] = None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = Linear(embed_dim, 3 * embed_dim, seed=seed)
        self.out = Linear(embed_dim, embed_dim, seed=(seed + 1) if seed is not None else None)

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        H, Dh = self.num_heads, self.head_dim
        q = qkv[:, :, :D].reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        k = qkv[:, :, D : 2 * D].reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2 * D :].reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out(attn)


class MultiHeadCrossAttention(Module):
    """Cross-attention: Q from ``query_in``, K/V from ``context``.

    Used in encoder-decoder Transformers to let the decoder attend to the encoder output.
    """

    def __init__(self, embed_dim: int, num_heads: int, seed: Optional[int] = None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, seed=seed)
        s = (seed + 1) if seed is not None else None
        self.kv_proj = Linear(embed_dim, 2 * embed_dim, seed=s)
        s = (seed + 2) if seed is not None else None
        self.out = Linear(embed_dim, embed_dim, seed=s)

    def forward(self, query_in: Tensor, context: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        B, T_q, D = query_in.shape
        _, T_kv, _ = context.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(query_in).reshape(B, T_q, H, Dh).transpose(0, 2, 1, 3)
        kv = self.kv_proj(context)  # (B, T_kv, 2D)
        k = kv[:, :, :D].reshape(B, T_kv, H, Dh).transpose(0, 2, 1, 3)
        v = kv[:, :, D:].reshape(B, T_kv, H, Dh).transpose(0, 2, 1, 3)
        attn = scaled_dot_product_attention(q, k, v, mask)  # (B, H, T_q, Dh)
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T_q, D)
        return self.out(attn)


class TransformerDecoderBlock(Module):
    """Decoder block: self-attn (causal) + cross-attn + MLP, all with residuals + pre-LN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        s = seed or 0
        self.ln1 = LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, seed=s)
        self.ln2 = LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, seed=s + 100)
        self.ln3 = LayerNorm(embed_dim)
        self.fc1 = Linear(embed_dim, ff_dim, seed=s + 200)
        self.fc2 = Linear(ff_dim, embed_dim, seed=s + 201)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        causal_mask: Optional[np.ndarray] = None,
        cross_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        x = x + self.self_attn(self.ln1(x), mask=causal_mask)
        x = x + self.cross_attn(self.ln2(x), context, mask=cross_mask)
        y = self.ln3(x)
        y = F.gelu(self.fc1(y))
        y = self.fc2(y)
        return x + y


class TransformerBlock(Module):
    """LN -> MHA -> residual -> LN -> MLP -> residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, seed=seed)
        self.ln2 = LayerNorm(embed_dim)
        self.fc1 = Linear(embed_dim, ff_dim, seed=(seed + 2) if seed is not None else None)
        self.fc2 = Linear(ff_dim, embed_dim, seed=(seed + 3) if seed is not None else None)

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        # MLP(x) = fc2(gelu(fc1(x)))
        y = self.ln2(x)
        y = F.gelu(self.fc1(y))
        y = self.fc2(y)
        return x + y
