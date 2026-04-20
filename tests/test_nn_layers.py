import numpy as np
import pytest

from nanograd import Tensor
from nanograd import nn
from nanograd.nn import functional as F
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(11)


# ---------- Module basics ----------


def test_parameter_registration():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(np.zeros(3, dtype=np.float32))
            self.b = nn.Parameter(np.ones(2, dtype=np.float32))

    m = M()
    names = [n for n, _ in m.named_parameters()]
    assert names == ["a", "b"]
    assert m.num_params() == 5


def test_submodule_registration():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(3, 4)
            self.l2 = nn.Linear(4, 2)

    m = M()
    names = sorted([n for n, _ in m.named_parameters()])
    assert names == ["l1.bias", "l1.weight", "l2.bias", "l2.weight"]


def test_train_eval_mode():
    m = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.5), nn.Linear(4, 2))
    assert m.training is True
    m.eval()
    for sub in m.modules():
        assert sub.training is False
    m.train()
    for sub in m.modules():
        assert sub.training is True


def test_zero_grad():
    m = nn.Linear(3, 2)
    x = Tensor(np.ones((1, 3), dtype=np.float32), requires_grad=True)
    y = m(x).sum()
    y.backward()
    for p in m.parameters():
        assert p.grad is not None
    m.zero_grad()
    for p in m.parameters():
        assert p.grad is None


def test_state_dict_roundtrip():
    m = nn.Linear(3, 2, seed=0)
    sd = m.state_dict()
    m2 = nn.Linear(3, 2, seed=1)  # different init
    # shouldn't match before loading
    assert not np.allclose(m.weight.data, m2.weight.data)
    m2.load_state_dict(sd)
    assert np.allclose(m.weight.data, m2.weight.data)
    assert np.allclose(m.bias.data, m2.bias.data)


# ---------- Linear ----------


def test_linear_shape(rng):
    m = nn.Linear(4, 3)
    x = Tensor(rng.standard_normal((5, 4)).astype(np.float32))
    y = m(x)
    assert y.shape == (5, 3)


def test_linear_gradcheck(rng):
    m = nn.Linear(4, 3, seed=0)
    x = Tensor(rng.standard_normal((2, 4)).astype(np.float32), requires_grad=True)
    gradcheck(lambda x: m(x).sum(), [x])
    # also check weight gradient
    gradcheck(lambda w: (x @ w + m.bias).sum(), [m.weight])


def test_linear_no_bias(rng):
    m = nn.Linear(4, 3, bias=False, seed=0)
    assert m.bias is None
    x = Tensor(rng.standard_normal((2, 4)).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 3)


# ---------- Sequential ----------


def test_sequential_forward(rng):
    m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = Tensor(rng.standard_normal((3, 4)).astype(np.float32))
    y = m(x)
    assert y.shape == (3, 2)
    # all params should be registered
    assert len(list(m.parameters())) == 4  # 2 weights + 2 biases


def test_sequential_iteration():
    m = nn.Sequential(nn.ReLU(), nn.Sigmoid())
    items = list(m)
    assert isinstance(items[0], nn.ReLU)
    assert isinstance(items[1], nn.Sigmoid)
    assert len(m) == 2


# ---------- Dropout ----------


def test_dropout_train_mode(rng):
    m = nn.Dropout(p=0.5, seed=0)
    x = Tensor(np.ones((1000,), dtype=np.float32))
    m.train()
    y = m(x)
    # about half zero, rest scaled by 2
    zero_frac = (y.data == 0).mean()
    assert 0.4 < zero_frac < 0.6
    # non-zero values should be ~2x
    nonzero = y.data[y.data != 0]
    assert np.allclose(nonzero, 2.0)


def test_dropout_eval_mode():
    m = nn.Dropout(p=0.5, seed=0)
    x = Tensor(np.ones((100,), dtype=np.float32))
    m.eval()
    y = m(x)
    assert np.allclose(y.data, 1.0)


def test_dropout_gradcheck(rng):
    # with fixed mask, should be differentiable
    m = nn.Dropout(p=0.3, seed=0)
    x = Tensor(rng.standard_normal((3, 4)).astype(np.float32), requires_grad=True)
    m.train()
    # gradcheck needs deterministic fn — reset the rng state before each call
    seed_state = m._rng.bit_generator.state

    def f(x):
        m._rng.bit_generator.state = seed_state
        return m(x).sum()

    gradcheck(f, [x])


# ---------- LayerNorm ----------


def test_layernorm_stats(rng):
    m = nn.LayerNorm(5)
    x = Tensor(rng.standard_normal((3, 5)).astype(np.float32))
    y = m(x)
    # mean ~ 0, var ~ 1
    assert np.allclose(y.data.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(y.data.var(axis=-1), 1.0, atol=1e-4)


def test_layernorm_gradcheck(rng):
    m = nn.LayerNorm(4)
    x = Tensor(rng.standard_normal((2, 4)).astype(np.float32), requires_grad=True)
    gradcheck(lambda x: m(x).sum(), [x], atol=1e-2, rtol=1e-2)
    # weight and bias grads
    gradcheck(lambda w: _ln_with_weight(x, w, m.bias, 1e-5).sum(), [m.weight], atol=1e-2, rtol=1e-2)
    gradcheck(lambda b: _ln_with_weight(x, m.weight, b, 1e-5).sum(), [m.bias], atol=1e-2, rtol=1e-2)


def _ln_with_weight(x, w, b, eps):
    from nanograd.nn.layers import _LayerNormFn
    return _LayerNormFn.apply(x, w, b, eps=eps)
