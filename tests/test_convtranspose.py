"""Tests for ConvTranspose2d."""
import numpy as np
import pytest

from nanograd import Tensor, nn
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(83)


def _rt(shape, rng):
    return Tensor(rng.uniform(-1, 1, size=shape).astype(np.float32), requires_grad=True)


def test_convtranspose2d_shape_stride1_pad0(rng):
    m = nn.ConvTranspose2d(2, 3, 3, stride=1, padding=0, seed=0)
    x = Tensor(rng.standard_normal((1, 2, 4, 4)).astype(np.float32))
    y = m(x)
    # H_out = (4-1)*1 - 0 + 3 = 6
    assert y.shape == (1, 3, 6, 6)


def test_convtranspose2d_shape_stride2_pad1(rng):
    m = nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1, seed=0)
    x = Tensor(rng.standard_normal((1, 3, 4, 4)).astype(np.float32))
    y = m(x)
    # H_out = (4-1)*2 - 2 + 3 = 7
    assert y.shape == (1, 2, 7, 7)


def test_convtranspose2d_gradcheck_input(rng):
    m = nn.ConvTranspose2d(2, 3, 3, stride=1, padding=0, seed=0)
    x = _rt((1, 2, 3, 3), rng)
    gradcheck(lambda x: m(x).sum(), [x], atol=1e-2, rtol=1e-2)


def test_convtranspose2d_gradcheck_weight(rng):
    m = nn.ConvTranspose2d(2, 3, 3, stride=1, padding=0, seed=0)
    x = Tensor(rng.standard_normal((1, 2, 3, 3)).astype(np.float32))

    def f(w):
        from nanograd.nn.conv import _ConvTranspose2dFn
        out = _ConvTranspose2dFn.apply(x, w, stride=1, padding=0)
        return out + m.bias.reshape(1, 3, 1, 1)

    gradcheck(lambda w: f(w).sum(), [m.weight], atol=1e-2, rtol=1e-2)


def test_convtranspose2d_matches_manual(rng):
    """ConvTranspose with stride=2, k=2, pad=0 should upsample spatially."""
    m = nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0, bias=False, seed=0)
    x = Tensor(np.ones((1, 1, 2, 2), dtype=np.float32))
    y = m(x).data
    # output size = (2-1)*2 + 2 = 4
    assert y.shape == (1, 1, 4, 4)


def test_convtranspose2d_stride2_gradcheck(rng):
    """Regression: verify gradcheck passes for stride=2, padding=1."""
    m = nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1, seed=0)
    x = _rt((1, 2, 3, 3), rng)
    gradcheck(lambda x: m(x).sum(), [x], atol=1e-2, rtol=1e-2)


def test_convtranspose2d_backward_runs():
    m = nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1, seed=0)
    x = Tensor(np.random.default_rng(0).standard_normal((2, 2, 4, 4)).astype(np.float32), requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert x.grad is not None
    assert m.weight.grad is not None
    assert m.bias.grad is not None
    assert not np.any(np.isnan(x.grad))
