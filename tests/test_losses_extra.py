import numpy as np
import pytest

from nanograd import Tensor
from nanograd.nn import functional as F
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(53)


def test_l1_loss_value(rng):
    p = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
    t = Tensor(np.array([[0.0, 3.0]], dtype=np.float32))
    loss = F.l1_loss(p, t).item()
    assert np.isclose(loss, (1.0 + 1.0) / 2)


def test_l1_loss_gradcheck(rng):
    p = Tensor(rng.standard_normal((3, 4)).astype(np.float32), requires_grad=True)
    # avoid zeros (grad of L1 at 0 is undefined)
    p.data = np.where(np.abs(p.data) < 0.1, 0.5, p.data)
    t = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    gradcheck(lambda x: F.l1_loss(x, t), [p], atol=1e-2, rtol=1e-2)


def test_huber_loss_quadratic_regime():
    p = Tensor(np.array([[0.5]], dtype=np.float32))
    t = Tensor(np.array([[0.0]], dtype=np.float32))
    # |diff|=0.5 < delta=1 -> 0.5*0.5^2 = 0.125
    loss = F.huber_loss(p, t, delta=1.0).item()
    assert np.isclose(loss, 0.125)


def test_huber_loss_linear_regime():
    p = Tensor(np.array([[3.0]], dtype=np.float32))
    t = Tensor(np.array([[0.0]], dtype=np.float32))
    # |diff|=3 > delta=1 -> 0.5*1 + 1*(3-1) = 2.5
    loss = F.huber_loss(p, t, delta=1.0).item()
    assert np.isclose(loss, 2.5)


def test_huber_loss_gradcheck(rng):
    p = Tensor(rng.standard_normal((3, 4)).astype(np.float32), requires_grad=True)
    t = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    gradcheck(lambda x: F.huber_loss(x, t, 1.0), [p], atol=1e-2, rtol=1e-2)
