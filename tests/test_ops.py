import numpy as np
import pytest

from nanograd import Tensor
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _rt(shape, rng, lo=-1.0, hi=1.0):
    return Tensor(rng.uniform(lo, hi, size=shape).astype(np.float32), requires_grad=True)


# ---------- reductions ----------


def test_sum_all(rng):
    a = _rt((3, 4), rng)
    gradcheck(lambda x: x.sum(), [a])


def test_sum_axis(rng):
    a = _rt((2, 3, 4), rng)
    gradcheck(lambda x: x.sum(axis=1), [a])


def test_sum_axis_keepdims(rng):
    a = _rt((2, 3, 4), rng)
    gradcheck(lambda x: x.sum(axis=1, keepdims=True), [a])


def test_mean_all(rng):
    a = _rt((3, 4), rng)
    gradcheck(lambda x: x.mean(), [a])


def test_mean_axis(rng):
    a = _rt((2, 3, 4), rng)
    gradcheck(lambda x: x.mean(axis=2, keepdims=False), [a])


def test_max_axis(rng):
    a = _rt((3, 4), rng)
    # ensure unique max per row for stable gradient check (no ties)
    for i in range(3):
        a.data[i, np.argmax(a.data[i])] += 2.0
    gradcheck(lambda x: x.max(axis=1), [a], atol=1e-2, rtol=1e-2)


# ---------- matmul ----------


def test_matmul_2d(rng):
    a = _rt((3, 4), rng)
    b = _rt((4, 5), rng)
    gradcheck(lambda x, y: x @ y, [a, b])


def test_matmul_batched(rng):
    a = _rt((2, 3, 4), rng)
    b = _rt((2, 4, 5), rng)
    gradcheck(lambda x, y: x @ y, [a, b])


def test_matmul_shape_correctness(rng):
    a = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    b = Tensor(rng.standard_normal((4, 5)).astype(np.float32))
    c = a @ b
    assert c.shape == (3, 5)
    assert np.allclose(c.data, a.data @ b.data)


# ---------- shape ----------


def test_reshape(rng):
    a = _rt((2, 6), rng)
    gradcheck(lambda x: x.reshape(3, 4), [a])


def test_reshape_tuple(rng):
    a = _rt((2, 6), rng)
    gradcheck(lambda x: x.reshape((3, 4)), [a])


def test_transpose_default(rng):
    a = _rt((3, 4), rng)
    gradcheck(lambda x: x.T, [a])


def test_transpose_explicit(rng):
    a = _rt((2, 3, 4), rng)
    gradcheck(lambda x: x.transpose(2, 0, 1), [a])


def test_expand(rng):
    a = _rt((1, 4), rng)
    gradcheck(lambda x: x.expand(3, 4), [a])


# ---------- indexing ----------


def test_getitem_slice(rng):
    a = _rt((4, 5), rng)
    gradcheck(lambda x: x[:, 1:3], [a])


def test_getitem_int(rng):
    a = _rt((4, 5), rng)
    gradcheck(lambda x: x[2], [a])


def test_getitem_intarray(rng):
    a = _rt((5, 3), rng)
    idx = np.array([0, 2, 4])
    gradcheck(lambda x: x[idx], [a])


# ---------- unary extra ----------


def test_sqrt(rng):
    a = Tensor(rng.uniform(0.5, 2.0, size=(3, 4)).astype(np.float32), requires_grad=True)
    gradcheck(lambda x: x.sqrt(), [a])


def test_abs(rng):
    a = _rt((3, 4), rng)
    # ensure no zeros (grad of abs at 0 is undefined)
    a.data = np.where(np.abs(a.data) < 0.1, 0.5, a.data)
    gradcheck(lambda x: x.abs(), [a])


# ---------- composed ----------


def test_composed_linear_like(rng):
    # y = (x @ W + b).sum()
    x = _rt((4, 3), rng)
    W = _rt((3, 5), rng)
    b = _rt((5,), rng)
    gradcheck(lambda x, W, b: (x @ W + b).sum(), [x, W, b])


def test_composed_softmax_like(rng):
    a = _rt((3, 4), rng)
    # logsoftmax-ish: x - log(exp(x).sum(axis=1, keepdims=True))
    def f(x):
        return (x - x.exp().sum(axis=1, keepdims=True).log()).sum()

    gradcheck(f, [a], atol=1e-2, rtol=1e-2)
