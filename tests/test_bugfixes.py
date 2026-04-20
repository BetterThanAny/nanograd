"""Regression tests for bugs found in the audit. Each test name maps to the audit ID."""
import numpy as np
import pytest

import nanograd as ng
from nanograd import Tensor, nn, optim


# ---------------------------------------------------------------------------
# C1: MaxPool2d padding must use -inf, not 0
# ---------------------------------------------------------------------------


def test_C1_maxpool_negative_inputs_with_padding():
    # all -1s: padded 0s must NOT beat the real -1s
    x = Tensor(-np.ones((1, 1, 3, 3), dtype=np.float32), requires_grad=True)
    m = nn.MaxPool2d(2, stride=2, padding=1)
    y = m(x).data
    # every output must be -1.0 (the real max among real values)
    assert np.all(y == -1.0), f"got {y}"


def test_C1_maxpool_grad_not_on_phantom_pad():
    x = Tensor(-np.ones((1, 1, 3, 3), dtype=np.float32), requires_grad=True)
    m = nn.MaxPool2d(2, stride=2, padding=1)
    m(x).sum().backward()
    # grad must flow back into real positions; total grad should equal num outputs
    # (each output takes grad 1.0 and assigns to exactly one real position)
    assert x.grad.sum() == 4.0


# ---------------------------------------------------------------------------
# C2: tied / shared parameters deduped in Optimizer
# ---------------------------------------------------------------------------


def test_C2_optimizer_dedupes_tied_params():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(4, 4, bias=False)
            self.dec = nn.Linear(4, 4, bias=False)
            # tie weights
            self.dec.weight = self.enc.weight

    m = M()
    params = list(m.parameters())
    # parameters() yields the tied weight twice, but optimizer dedupes
    opt = optim.SGD(params, lr=0.1)
    assert len(opt.params) == 1


def test_C2_tied_adam_t_advances_once():
    p = nn.Parameter(np.zeros(3, dtype=np.float32))
    opt = optim.Adam([p, p, p], lr=0.1)  # same Parameter 3 times
    p.grad = np.ones_like(p.data)
    opt.step()
    # state["t"] should be 1, not 3
    assert opt.state[id(p)]["t"] == 1


# ---------------------------------------------------------------------------
# C3: double backward on same graph doesn't double-count intermediate grads
# ---------------------------------------------------------------------------


def test_C3_double_backward_scalar():
    a = Tensor([3.0], requires_grad=True)
    y = a * a
    y.backward(np.ones(1))
    assert np.isclose(a.grad[0], 6.0)
    # second backward — leaf grad accumulates, but intermediate grads are reset
    y.backward(np.ones(1))
    assert np.isclose(a.grad[0], 12.0), f"expected 12, got {a.grad}"


def test_C3_zero_grad_between_still_works():
    a = Tensor([3.0], requires_grad=True)
    y = a * a
    y.backward(np.ones(1))
    a.zero_grad()
    y.backward(np.ones(1))
    assert np.isclose(a.grad[0], 6.0)


# ---------------------------------------------------------------------------
# C4: Module.__setattr__ clears stale Parameters on non-Parameter reassignment
# ---------------------------------------------------------------------------


def test_C4_bias_reassign_to_none_clears_registration():
    l = nn.Linear(3, 4, bias=True)
    assert "bias" in l._parameters
    l.bias = None
    assert "bias" not in l._parameters
    assert l.num_params() == 3 * 4  # just weight now


def test_C4_submodule_reassign_to_none_clears_registration():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(3, 2)

    m = M()
    assert "head" in m._modules
    m.head = None
    assert "head" not in m._modules


# ---------------------------------------------------------------------------
# I1: RNN / LSTM / GRU handle T=0
# ---------------------------------------------------------------------------


def test_I1_rnn_empty_sequence():
    m = nn.RNN(4, 8)
    x = Tensor(np.zeros((2, 0, 4), dtype=np.float32))
    out, h = m(x)
    assert out.shape == (2, 0, 8)
    assert h.shape == (2, 8)


def test_I1_lstm_empty_sequence():
    m = nn.LSTM(4, 8)
    x = Tensor(np.zeros((2, 0, 4), dtype=np.float32))
    out, (h, c) = m(x)
    assert out.shape == (2, 0, 8)
    assert h.shape == (2, 8) and c.shape == (2, 8)


def test_I1_gru_empty_sequence():
    m = nn.GRU(4, 8)
    x = Tensor(np.zeros((2, 0, 4), dtype=np.float32))
    out, h = m(x)
    assert out.shape == (2, 0, 8)
    assert h.shape == (2, 8)


# ---------------------------------------------------------------------------
# I2: cat / stack with empty list raise clean errors
# ---------------------------------------------------------------------------


def test_I2_cat_empty_raises_cleanly():
    with pytest.raises(ValueError, match="at least one"):
        ng.cat([], axis=0)


def test_I2_stack_empty_raises_cleanly():
    with pytest.raises(ValueError, match="at least one"):
        ng.stack([], axis=0)


# ---------------------------------------------------------------------------
# I3: Tensor-as-index in Getitem works
# ---------------------------------------------------------------------------


def test_I3_tensor_index_integer():
    x = Tensor(np.arange(10, dtype=np.float32), requires_grad=True)
    idx = Tensor(np.array([1, 3, 5], dtype=np.int64))
    y = x[idx]
    assert y.shape == (3,)
    assert np.array_equal(y.data, [1.0, 3.0, 5.0])


def test_I3_tensor_index_backward():
    x = Tensor(np.arange(5, dtype=np.float32), requires_grad=True)
    idx = Tensor(np.array([0, 2, 2, 4], dtype=np.int64))
    y = x[idx]
    y.sum().backward()
    # index 0 used once, 2 used twice, 4 once
    assert np.array_equal(x.grad, [1.0, 0.0, 2.0, 0.0, 1.0])


def test_I3_tensor_index_with_float_dtype_coerced():
    x = Tensor(np.arange(5, dtype=np.float32))
    idx = Tensor(np.array([0.0, 2.0, 4.0], dtype=np.float32))
    y = x[idx]
    assert np.array_equal(y.data, [0.0, 2.0, 4.0])
