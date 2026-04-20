"""Optimizer & scheduler tests. Verify convergence on a convex quadratic."""
import numpy as np
import pytest

from nanograd import Tensor
from nanograd import nn, optim


def _quadratic_target(opt_factory, steps: int = 500):
    """Minimize f(x) = ||x - 3||^2 starting from x=0. Return final ||x-3||."""
    x = nn.Parameter(np.zeros(4, dtype=np.float32))
    target = np.full(4, 3.0, dtype=np.float32)
    opt = opt_factory([x])
    for _ in range(steps):
        diff = x - Tensor(target)
        loss = (diff * diff).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(np.linalg.norm(x.data - target))


def test_sgd_converges():
    err = _quadratic_target(lambda p: optim.SGD(p, lr=0.1), steps=100)
    assert err < 1e-3


def test_sgd_momentum():
    err = _quadratic_target(lambda p: optim.SGD(p, lr=0.01, momentum=0.9), steps=500)
    assert err < 1e-3


def test_sgd_nesterov():
    err = _quadratic_target(lambda p: optim.SGD(p, lr=0.01, momentum=0.9, nesterov=True), steps=500)
    assert err < 1e-3


def test_sgd_weight_decay_pulls_toward_zero():
    x = nn.Parameter(np.ones(3, dtype=np.float32))
    opt = optim.SGD([x], lr=0.1, weight_decay=1.0)
    # no loss gradient — only weight decay
    for _ in range(20):
        x.grad = np.zeros_like(x.data)  # simulate loss.backward() with 0 loss
        opt.step()
    assert np.all(np.abs(x.data) < 0.5)


def test_adam_converges():
    err = _quadratic_target(lambda p: optim.Adam(p, lr=0.1), steps=200)
    assert err < 1e-2


def test_adamw_converges():
    err = _quadratic_target(lambda p: optim.AdamW(p, lr=0.1, weight_decay=0.0), steps=200)
    assert err < 1e-2


def test_rmsprop_converges():
    err = _quadratic_target(lambda p: optim.RMSProp(p, lr=0.05), steps=300)
    assert err < 1e-2


def test_zero_grad_clears():
    x = nn.Parameter(np.zeros(3, dtype=np.float32))
    opt = optim.SGD([x], lr=0.01)
    x.grad = np.ones_like(x.data)
    opt.zero_grad()
    assert x.grad is None


def test_step_skips_none_grad():
    x = nn.Parameter(np.ones(3, dtype=np.float32))
    y = nn.Parameter(np.ones(3, dtype=np.float32))
    opt = optim.SGD([x, y], lr=0.1)
    x.grad = np.ones_like(x.data)
    # y has no grad — should be skipped
    opt.step()
    assert np.allclose(x.data, 1.0 - 0.1)
    assert np.allclose(y.data, 1.0)


# ---------- schedulers ----------


def test_step_lr():
    x = nn.Parameter(np.zeros(1, dtype=np.float32))
    opt = optim.SGD([x], lr=1.0)
    sched = optim.StepLR(opt, step_size=3, gamma=0.1)
    # after __init__, last_step=0 → lr = 1.0 * 0.1^0 = 1.0
    assert np.isclose(opt.lr, 1.0)
    for _ in range(3):
        sched.step()
    # last_step=3 → lr = 1.0 * 0.1^1 = 0.1
    assert np.isclose(opt.lr, 0.1)


def test_cosine_lr_endpoints():
    x = nn.Parameter(np.zeros(1, dtype=np.float32))
    opt = optim.SGD([x], lr=1.0)
    sched = optim.CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    # start at base_lr
    assert np.isclose(opt.lr, 1.0)
    for _ in range(10):
        sched.step()
    # after T_max steps → eta_min
    assert np.isclose(opt.lr, 0.0, atol=1e-6)


def test_exponential_lr():
    x = nn.Parameter(np.zeros(1, dtype=np.float32))
    opt = optim.SGD([x], lr=1.0)
    sched = optim.ExponentialLR(opt, gamma=0.9)
    lrs = [opt.lr]
    for _ in range(3):
        sched.step()
        lrs.append(opt.lr)
    assert np.allclose(lrs, [1.0, 0.9, 0.81, 0.729])


def test_warmup_cosine():
    x = nn.Parameter(np.zeros(1, dtype=np.float32))
    opt = optim.SGD([x], lr=1.0)
    sched = optim.WarmupCosine(opt, warmup=5, T_max=15, eta_min=0.0)
    # step 0 during warmup
    assert opt.lr < 1.0
    # after warmup steps, should reach ~base_lr
    for _ in range(4):
        sched.step()
    assert np.isclose(opt.lr, 1.0, atol=0.01)
    # at end → eta_min
    for _ in range(10):
        sched.step()
    assert opt.lr < 0.1


# ---------- integration: train a Linear layer ----------


def test_adam_trains_linear_layer():
    """y = W_true x + b_true. Fit Linear with Adam."""
    rng = np.random.default_rng(0)
    W_true = rng.standard_normal((4, 2)).astype(np.float32)
    b_true = rng.standard_normal((2,)).astype(np.float32)
    X = rng.standard_normal((64, 4)).astype(np.float32)
    Y = X @ W_true + b_true

    model = nn.Linear(4, 2, seed=0)
    opt = optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        pred = model(Tensor(X))
        loss = ((pred - Tensor(Y)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # loss should be tiny
    assert loss.item() < 1e-3
