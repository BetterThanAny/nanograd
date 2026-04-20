import numpy as np
import pytest

from nanograd import Tensor
from nanograd.nn import functional as F
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def _rt(shape, rng, lo=-1.0, hi=1.0):
    return Tensor(rng.uniform(lo, hi, size=shape).astype(np.float32), requires_grad=True)


# ---------- activations ----------


def test_relu(rng):
    a = _rt((3, 4), rng)
    # avoid zeros (undefined grad)
    a.data = np.where(np.abs(a.data) < 0.1, 0.5, a.data)
    gradcheck(lambda x: F.relu(x), [a])


def test_sigmoid(rng):
    a = _rt((3, 4), rng)
    gradcheck(lambda x: F.sigmoid(x), [a])


def test_tanh(rng):
    a = _rt((3, 4), rng)
    gradcheck(lambda x: F.tanh(x), [a])


def test_leaky_relu(rng):
    a = _rt((3, 4), rng)
    a.data = np.where(np.abs(a.data) < 0.1, 0.5, a.data)
    gradcheck(lambda x: F.leaky_relu(x, 0.1), [a])


def test_gelu(rng):
    a = _rt((3, 4), rng)
    gradcheck(lambda x: F.gelu(x), [a], atol=1e-2, rtol=1e-2)


def test_softmax(rng):
    a = _rt((3, 4), rng)
    out = F.softmax(a)
    # row sums should be 1
    assert np.allclose(out.data.sum(axis=-1), 1.0, atol=1e-6)
    gradcheck(lambda x: F.softmax(x).sum(), [a])


def test_log_softmax(rng):
    a = _rt((3, 4), rng)
    out = F.log_softmax(a)
    # exp(logsoftmax).sum == 1
    assert np.allclose(np.exp(out.data).sum(axis=-1), 1.0, atol=1e-5)
    gradcheck(lambda x: F.log_softmax(x).sum(), [a])


def test_softmax_numerical_stability():
    a = Tensor(np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float32), requires_grad=True)
    out = F.softmax(a)
    assert np.allclose(out.data, 1.0 / 3)
    assert not np.any(np.isnan(out.data))


# ---------- losses ----------


def test_mse(rng):
    p = _rt((3, 4), rng)
    t = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    gradcheck(lambda x: F.mse_loss(x, t), [p])


def test_bce_probs(rng):
    p = Tensor(rng.uniform(0.2, 0.8, size=(3, 4)).astype(np.float32), requires_grad=True)
    t = Tensor(rng.integers(0, 2, size=(3, 4)).astype(np.float32))
    gradcheck(lambda x: F.bce_loss(x, t), [p], atol=1e-2, rtol=1e-2)


def test_bce_with_logits(rng):
    logits = _rt((3, 4), rng, lo=-2, hi=2)
    t = Tensor(rng.integers(0, 2, size=(3, 4)).astype(np.float32))
    gradcheck(lambda x: F.bce_with_logits_loss(x, t), [logits])


def test_bce_logits_stability():
    # large logits shouldn't nan
    logits = Tensor(np.array([[1e3, -1e3]], dtype=np.float32), requires_grad=True)
    target = Tensor(np.array([[1.0, 0.0]], dtype=np.float32))
    loss = F.bce_with_logits_loss(logits, target)
    assert np.isfinite(loss.data)


def test_cross_entropy(rng):
    logits = _rt((3, 4), rng)
    target = Tensor(np.array([0, 2, 1], dtype=np.int64))
    # gradcheck only on logits (target has no grad)
    gradcheck(lambda x: F.cross_entropy(x, target), [logits])


def test_cross_entropy_stability():
    logits = Tensor(np.array([[1e3, -1e3, 0.0]], dtype=np.float32), requires_grad=True)
    target = Tensor(np.array([0], dtype=np.int64))
    loss = F.cross_entropy(logits, target)
    assert np.isfinite(loss.data)


def test_cross_entropy_matches_manual(rng):
    logits = _rt((5, 3), rng)
    target = Tensor(np.array([0, 1, 2, 1, 0], dtype=np.int64))
    # compare to logsoftmax + gather
    lsm = F.log_softmax(logits)
    manual = -lsm.data[np.arange(5), target.data.astype(np.int64)].mean()
    loss = F.cross_entropy(logits, target)
    assert np.isclose(loss.data, manual, atol=1e-5)
