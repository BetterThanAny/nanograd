"""Tests for cat/stack/pad ops and AdaptiveAvgPool2d."""
import numpy as np
import pytest

import nanograd as ng
from nanograd import Tensor, nn
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(61)


def _rt(shape, rng):
    return Tensor(rng.uniform(-1, 1, size=shape).astype(np.float32), requires_grad=True)


def test_cat_shape(rng):
    a = _rt((2, 3), rng)
    b = _rt((2, 4), rng)
    c = ng.cat([a, b], axis=1)
    assert c.shape == (2, 7)


def test_cat_values():
    a = Tensor(np.ones((2, 3), dtype=np.float32))
    b = Tensor(np.zeros((2, 2), dtype=np.float32))
    c = ng.cat([a, b], axis=1)
    assert np.allclose(c.data[:, :3], 1.0)
    assert np.allclose(c.data[:, 3:], 0.0)


def test_cat_gradcheck(rng):
    a = _rt((2, 3), rng)
    b = _rt((2, 4), rng)
    gradcheck(lambda a, b: ng.cat([a, b], axis=1).sum(), [a, b])


def test_stack_shape(rng):
    a = _rt((3,), rng)
    b = _rt((3,), rng)
    c = _rt((3,), rng)
    s = ng.stack([a, b, c], axis=0)
    assert s.shape == (3, 3)


def test_stack_gradcheck(rng):
    a = _rt((2, 3), rng)
    b = _rt((2, 3), rng)
    gradcheck(lambda a, b: ng.stack([a, b], axis=0).sum(), [a, b])


def test_pad_shape(rng):
    x = _rt((2, 3), rng)
    y = ng.pad(x, [(1, 1), (0, 2)])
    assert y.shape == (4, 5)


def test_pad_gradcheck(rng):
    x = _rt((2, 3), rng)
    gradcheck(lambda x: ng.pad(x, [(1, 1), (0, 2)]).sum(), [x])


def test_adaptive_avg_pool_global(rng):
    m = nn.AdaptiveAvgPool2d(1)
    x = Tensor(rng.standard_normal((2, 3, 8, 8)).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 3, 1, 1)
    # should equal mean over H, W
    expected = x.data.mean(axis=(-2, -1), keepdims=True)
    assert np.allclose(y.data, expected)


def test_adaptive_avg_pool_grid(rng):
    m = nn.AdaptiveAvgPool2d(2)
    x = Tensor(rng.standard_normal((1, 2, 8, 8)).astype(np.float32))
    y = m(x)
    assert y.shape == (1, 2, 2, 2)


def test_adaptive_avg_pool_gradcheck(rng):
    m = nn.AdaptiveAvgPool2d(1)
    x = _rt((1, 2, 4, 4), rng)
    gradcheck(lambda x: m(x).sum(), [x], atol=1e-2, rtol=1e-2)
