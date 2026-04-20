"""Tests for cross-attention and the decoder block."""
import numpy as np
import pytest

from nanograd import Tensor, nn
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(111)


def _rt(shape, rng):
    return Tensor(rng.uniform(-0.5, 0.5, size=shape).astype(np.float32), requires_grad=True)


def test_cross_attention_shape(rng):
    m = nn.MultiHeadCrossAttention(8, 2, seed=0)
    q = Tensor(rng.standard_normal((2, 5, 8)).astype(np.float32))
    ctx = Tensor(rng.standard_normal((2, 7, 8)).astype(np.float32))  # different seq length
    out = m(q, ctx)
    assert out.shape == (2, 5, 8)


def test_cross_attention_gradcheck(rng):
    m = nn.MultiHeadCrossAttention(4, 2, seed=0)
    q = _rt((1, 2, 4), rng)
    ctx = _rt((1, 3, 4), rng)
    gradcheck(lambda q, c: m(q, c).sum(), [q, ctx], atol=1e-1, rtol=1e-1)


def test_transformer_decoder_block_shape(rng):
    m = nn.TransformerDecoderBlock(8, 2, seed=0)
    x = Tensor(rng.standard_normal((1, 4, 8)).astype(np.float32))
    ctx = Tensor(rng.standard_normal((1, 6, 8)).astype(np.float32))
    out = m(x, ctx)
    assert out.shape == (1, 4, 8)


def test_transformer_decoder_causal_mask(rng):
    m = nn.TransformerDecoderBlock(8, 2, seed=0)
    x = Tensor(rng.standard_normal((1, 4, 8)).astype(np.float32))
    ctx = Tensor(rng.standard_normal((1, 6, 8)).astype(np.float32))
    T = 4
    causal = np.tril(np.ones((T, T), dtype=bool))[None, None, :, :]
    out = m(x, ctx, causal_mask=causal)
    assert out.shape == (1, 4, 8)
    assert not np.any(np.isnan(out.data))
