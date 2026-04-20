import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F
from nanograd.training import EMA


def test_ema_shadow_initialized_to_current_params():
    m = nn.Linear(4, 3, seed=0)
    ema = EMA(m, decay=0.9)
    for name, p in m.named_parameters():
        np.testing.assert_array_equal(ema._shadow[name], p.data)


def test_ema_update_tracks_running_average():
    m = nn.Linear(2, 2, seed=0)
    ema = EMA(m, decay=0.8)
    w0 = m.weight.data.copy()
    m.weight.data[...] = w0 + 1.0
    ema.update()
    expected = 0.8 * w0 + 0.2 * (w0 + 1.0)
    np.testing.assert_allclose(ema._shadow["weight"], expected, atol=1e-6)


def test_ema_swap_context_restores_weights():
    m = nn.Linear(3, 2, seed=0)
    ema = EMA(m, decay=0.5)
    # force shadow to a known value
    for name in ema._shadow:
        ema._shadow[name] = np.zeros_like(ema._shadow[name])
    before = {name: p.data.copy() for name, p in m.named_parameters()}
    with ema.swap_into(m):
        for _name, p in m.named_parameters():
            assert np.abs(p.data).sum() == 0, "swap did not install zeros"
    for name, p in m.named_parameters():
        np.testing.assert_array_equal(p.data, before[name]), f"not restored: {name}"


def test_ema_apply_to_copies_shadow():
    m = nn.Linear(3, 2, seed=0)
    ema = EMA(m, decay=0.5)
    new_shadow = {name: np.ones_like(arr) * 2.5 for name, arr in ema._shadow.items()}
    ema._shadow = new_shadow
    ema.apply_to()
    for _name, p in m.named_parameters():
        assert np.all(p.data == 2.5)


def test_ema_buffers_tracked_when_requested():
    bn = nn.BatchNorm2d(4)
    ema = EMA(bn, decay=0.5, include_buffers=True)
    # BatchNorm has running_mean / running_var buffers
    bufs = dict(bn.named_buffers())
    assert len(bufs) >= 1
    bufs_shadow = {k.replace("_buf_:", ""): v for k, v in ema._shadow.items() if k.startswith("_buf_:")}
    assert set(bufs_shadow.keys()) == set(bufs.keys())


def test_ema_smooths_noisy_training():
    """Targets have per-batch noise — raw params oscillate, EMA should be smoother and at least as accurate."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    W_true = rng.standard_normal((3, 2)).astype(np.float32)
    y_clean = X @ W_true

    m = nn.Linear(3, 2, seed=0, bias=False)
    opt = optim.SGD(m.parameters(), lr=0.3)
    ema = EMA(m, decay=0.9)

    for step in range(300):
        i = step % 8
        xb = Tensor(X[i * 5 : (i + 1) * 5])
        noise = rng.standard_normal((5, 2)).astype(np.float32) * 0.5
        yb = Tensor(y_clean[i * 5 : (i + 1) * 5] + noise)
        loss = F.mse_loss(m(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update()

    raw_err = float(((m(Tensor(X)).data - y_clean) ** 2).mean())
    with ema.swap_into(m):
        ema_err = float(((m(Tensor(X)).data - y_clean) ** 2).mean())
    # EMA should not be worse than raw (within a small tolerance)
    assert ema_err <= raw_err + 1e-3, f"EMA meaningfully worse: ema={ema_err:.4f} raw={raw_err:.4f}"
