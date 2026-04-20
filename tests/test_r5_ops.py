"""Tests for R5 core ops: where/clamp/masked_fill/cumsum/argmax/argmin/topk."""
import numpy as np
import pytest

import nanograd as ng
from nanograd import Tensor
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(91)


def _rt(shape, rng):
    return Tensor(rng.uniform(-1, 1, size=shape).astype(np.float32), requires_grad=True)


# ---------- where ----------


def test_where_values():
    a = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = Tensor(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    cond = np.array([True, False, True])
    y = ng.where(cond, a, b)
    assert np.allclose(y.data, [1.0, 20.0, 3.0])


def test_where_gradcheck(rng):
    a = _rt((3, 4), rng)
    b = _rt((3, 4), rng)
    cond = rng.random((3, 4)) > 0.5
    gradcheck(lambda a, b: ng.where(cond, a, b).sum(), [a, b])


def test_where_with_tensor_cond():
    a = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    b = Tensor(np.array([10.0, 20.0], dtype=np.float32))
    cond = Tensor(np.array([1, 0], dtype=np.int64))
    y = ng.where(cond, a, b)
    assert np.allclose(y.data, [1.0, 20.0])


# ---------- clamp ----------


def test_clamp_values():
    a = Tensor(np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32))
    y = a.clamp(-1.0, 1.0)
    assert np.allclose(y.data, [-1.0, -0.5, 0.5, 1.0])


def test_clamp_one_sided():
    a = Tensor(np.array([-2.0, 0.5, 2.0], dtype=np.float32))
    y = a.clamp(minv=0.0)
    assert np.allclose(y.data, [0.0, 0.5, 2.0])


def test_clamp_gradcheck(rng):
    # avoid boundary where grad is undefined
    a = Tensor(rng.uniform(-0.5, 0.5, (3, 4)).astype(np.float32), requires_grad=True)
    gradcheck(lambda x: x.clamp(-1.0, 1.0).sum(), [a])
    # with active clamping
    a = Tensor(rng.uniform(-3, 3, (3, 4)).astype(np.float32), requires_grad=True)
    # avoid exactly hitting bounds
    a.data = np.where(np.abs(np.abs(a.data) - 1.0) < 0.1, a.data + 0.3, a.data)
    gradcheck(lambda x: x.clamp(-1.0, 1.0).sum(), [a], atol=1e-2, rtol=1e-2)


# ---------- masked_fill ----------


def test_masked_fill_values():
    a = Tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    mask = np.array([False, True, False, True])
    y = a.masked_fill(mask, -1.0)
    assert np.allclose(y.data, [1.0, -1.0, 3.0, -1.0])


def test_masked_fill_gradcheck(rng):
    a = _rt((3, 4), rng)
    mask = rng.random((3, 4)) > 0.5
    gradcheck(lambda x: x.masked_fill(mask, 0.0).sum(), [a])


# ---------- cumsum ----------


def test_cumsum_values():
    a = Tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    y = ng.cumsum(a, axis=0)
    assert np.allclose(y.data, [1.0, 3.0, 6.0, 10.0])


def test_cumsum_gradcheck_1d(rng):
    a = _rt((5,), rng)
    gradcheck(lambda x: ng.cumsum(x, axis=0).sum(), [a])


def test_cumsum_gradcheck_2d_axis1(rng):
    a = _rt((3, 4), rng)
    gradcheck(lambda x: ng.cumsum(x, axis=1).sum(), [a])


def test_cumsum_backward_correct():
    """The grad of sum(cumsum(x)) wrt x_i is (n - i), since x_i appears in cumsum[i..n-1]."""
    a = Tensor(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), requires_grad=True)
    y = ng.cumsum(a, axis=0).sum()
    y.backward()
    # grad should be [4, 3, 2, 1]
    assert np.allclose(a.grad, [4.0, 3.0, 2.0, 1.0])


# ---------- argmax / argmin ----------


def test_argmax_tensor_method():
    a = Tensor(np.array([[1, 3, 2], [5, 4, 0]], dtype=np.float32))
    assert np.array_equal(a.argmax(axis=-1), [1, 0])
    assert a.argmax() == 3  # flat index of 5


def test_argmin_tensor_method():
    a = Tensor(np.array([[1, 3, 2], [5, 4, 0]], dtype=np.float32))
    assert np.array_equal(a.argmin(axis=-1), [0, 2])


# ---------- topk ----------


def test_topk_values_and_indices():
    a = Tensor(np.array([1.0, 5.0, 3.0, 8.0, 2.0], dtype=np.float32))
    vals, idx = ng.topk(a, k=3, axis=0)
    assert np.array_equal(vals.data, [8.0, 5.0, 3.0])
    assert np.array_equal(idx, [3, 1, 2])


def test_topk_2d_axis():
    a = Tensor(np.array([[1, 5, 3], [8, 2, 7]], dtype=np.float32))
    vals, idx = ng.topk(a, k=2, axis=1)
    assert np.array_equal(vals.data, [[5, 3], [8, 7]])
    assert np.array_equal(idx, [[1, 2], [0, 2]])
