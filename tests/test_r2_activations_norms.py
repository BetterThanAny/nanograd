"""Tests for new activations (ELU/SiLU/Mish) and norms (GroupNorm/InstanceNorm2d)."""
import numpy as np
import pytest

from nanograd import Tensor, nn
from nanograd.nn import functional as F
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(71)


def _rt(shape, rng, lo=-1.0, hi=1.0):
    return Tensor(rng.uniform(lo, hi, size=shape).astype(np.float32), requires_grad=True)


# ---------- activations ----------


def test_elu_positive_is_identity():
    x = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    y = F.elu(x).data
    assert np.allclose(y, [1.0, 2.0])


def test_elu_negative_saturates():
    x = Tensor(np.array([-1.0, -5.0], dtype=np.float32))
    y = F.elu(x, alpha=1.0).data
    # expected: alpha*(exp(-1) - 1), alpha*(exp(-5) - 1)
    assert np.allclose(y, [np.exp(-1) - 1, np.exp(-5) - 1], atol=1e-5)


def test_elu_gradcheck(rng):
    x = _rt((3, 4), rng)
    gradcheck(lambda x: F.elu(x, 1.0), [x], atol=1e-2, rtol=1e-2)


def test_silu_gradcheck(rng):
    x = _rt((3, 4), rng)
    gradcheck(lambda x: F.silu(x), [x], atol=1e-2, rtol=1e-2)


def test_silu_alias_swish():
    from nanograd.nn.functional import swish
    x = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    assert np.allclose(F.silu(x).data, swish(x).data)


def test_mish_gradcheck(rng):
    x = _rt((3, 4), rng)
    gradcheck(lambda x: F.mish(x), [x], atol=1e-2, rtol=1e-2)


def test_activation_modules(rng):
    for M in (nn.ELU, nn.SiLU, nn.Mish):
        m = M()
        x = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
        y = m(x)
        assert y.shape == (3, 4)


# ---------- GroupNorm ----------


def test_groupnorm_stats(rng):
    m = nn.GroupNorm(2, 4)
    x = Tensor(rng.standard_normal((3, 4, 5, 5)).astype(np.float32) * 2 + 1)
    y = m(x)
    # per (N, group) stats should be normalized
    yg = y.data.reshape(3, 2, 2, 5, 5)
    assert np.allclose(yg.mean(axis=(2, 3, 4)), 0.0, atol=1e-5)
    assert np.allclose(yg.var(axis=(2, 3, 4)), 1.0, atol=1e-3)


def test_groupnorm_gradcheck(rng):
    m = nn.GroupNorm(2, 4)
    x = _rt((2, 4, 3, 3), rng)
    gradcheck(lambda x: m(x).sum(), [x], atol=1e-1, rtol=1e-1)


def test_groupnorm_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        nn.GroupNorm(3, 4)  # 4 channels not divisible by 3


# ---------- InstanceNorm2d ----------


def test_instancenorm2d_stats(rng):
    m = nn.InstanceNorm2d(4)
    x = Tensor(rng.standard_normal((3, 4, 5, 5)).astype(np.float32) * 2 + 1)
    y = m(x)
    # per (N, C) spatial stats normalized
    assert np.allclose(y.data.mean(axis=(-2, -1)), 0.0, atol=1e-5)
    assert np.allclose(y.data.var(axis=(-2, -1)), 1.0, atol=1e-3)


def test_instancenorm2d_no_affine(rng):
    m = nn.InstanceNorm2d(4, affine=False)
    assert m.weight is None and m.bias is None
    assert list(m.parameters()) == []
    x = Tensor(rng.standard_normal((2, 4, 3, 3)).astype(np.float32))
    y = m(x)
    assert y.shape == x.shape
