import numpy as np
import pytest

from nanograd import Tensor
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _rand_tensor(shape, rng, requires_grad=True):
    return Tensor(rng.standard_normal(shape).astype(np.float32), requires_grad=requires_grad)


# ---------- scalar ----------


def test_scalar_add():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    y = a + b
    y.backward()
    assert y.item() == 5.0
    assert a.grad.item() == 1.0
    assert b.grad.item() == 1.0


def test_scalar_mul_chain():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    y = a * b + a  # dy/da = b+1 = 4, dy/db = a = 2
    y.backward()
    assert np.isclose(a.grad.item(), 4.0)
    assert np.isclose(b.grad.item(), 2.0)


def test_scalar_pow():
    a = Tensor(3.0, requires_grad=True)
    y = a ** 2  # dy/da = 2a = 6
    y.backward()
    assert np.isclose(a.grad.item(), 6.0)


def test_scalar_div():
    a = Tensor(6.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)
    y = a / b  # dy/da = 1/b = 0.5, dy/db = -a/b^2 = -1.5
    y.backward()
    assert np.isclose(a.grad.item(), 0.5)
    assert np.isclose(b.grad.item(), -1.5)


# ---------- gradcheck ----------


def test_gradcheck_add(rng):
    a = _rand_tensor((3, 4), rng)
    b = _rand_tensor((3, 4), rng)
    gradcheck(lambda x, y: x + y, [a, b])


def test_gradcheck_mul(rng):
    a = _rand_tensor((3, 4), rng)
    b = _rand_tensor((3, 4), rng)
    gradcheck(lambda x, y: x * y, [a, b])


def test_gradcheck_sub(rng):
    a = _rand_tensor((3, 4), rng)
    b = _rand_tensor((3, 4), rng)
    gradcheck(lambda x, y: x - y, [a, b])


def test_gradcheck_div(rng):
    a = _rand_tensor((3, 4), rng)
    # avoid b near zero
    b = Tensor(rng.uniform(0.5, 2.0, size=(3, 4)).astype(np.float32), requires_grad=True)
    gradcheck(lambda x, y: x / y, [a, b])


def test_gradcheck_neg(rng):
    a = _rand_tensor((3, 4), rng)
    gradcheck(lambda x: -x, [a])


def test_gradcheck_exp(rng):
    a = _rand_tensor((3,), rng)
    # bound to avoid overflow
    a.data = np.clip(a.data, -2, 2)
    gradcheck(lambda x: x.exp(), [a])


def test_gradcheck_log(rng):
    a = Tensor(rng.uniform(0.5, 2.0, size=(3, 4)).astype(np.float32), requires_grad=True)
    gradcheck(lambda x: x.log(), [a])


def test_gradcheck_composed(rng):
    a = _rand_tensor((3,), rng)
    b = _rand_tensor((3,), rng)
    # (a*b + a).exp() — reduce magnitudes
    a.data = np.clip(a.data, -1, 1)
    b.data = np.clip(b.data, -1, 1)
    gradcheck(lambda x, y: (x * y + x).exp(), [a, b])


# ---------- broadcasting grad ----------


def test_broadcast_scalar_vector(rng):
    a = Tensor(rng.standard_normal(()).astype(np.float32), requires_grad=True)
    b = _rand_tensor((3, 4), rng)
    y = a * b
    g = np.ones_like(y.data)
    y.backward(g)
    # grad of a must be sum of b (scalar broadcast)
    assert np.allclose(a.grad, b.data.sum())
    assert np.allclose(b.grad, a.data)


def test_broadcast_row_vector(rng):
    a = _rand_tensor((1, 4), rng)
    b = _rand_tensor((3, 4), rng)
    y = a + b
    g = np.ones_like(y.data)
    y.backward(g)
    # a was broadcast along axis 0, grad must be summed
    assert a.grad.shape == (1, 4)
    assert np.allclose(a.grad, np.ones((1, 4)) * 3)
    assert np.allclose(b.grad, np.ones((3, 4)))


# ---------- grad accumulation ----------


def test_grad_accumulation():
    a = Tensor(2.0, requires_grad=True)
    y1 = a * a
    y1.backward()
    y2 = a * a
    y2.backward()
    # grads should accumulate (no zero_grad between)
    assert np.isclose(a.grad.item(), 8.0)


def test_zero_grad():
    a = Tensor(2.0, requires_grad=True)
    y = a * a
    y.backward()
    a.zero_grad()
    assert a.grad is None


def test_detach():
    a = Tensor(2.0, requires_grad=True)
    b = a.detach()
    assert b.requires_grad is False
    assert b.data == 2.0
    assert b._ctx is None


def test_requires_grad_false_no_backward():
    a = Tensor(1.0, requires_grad=False)
    with pytest.raises(RuntimeError):
        a.backward()
