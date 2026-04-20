import numpy as np
import pytest

from nanograd import Tensor, nn
from nanograd.nn.conv import col2im, im2col
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(17)


def _rt(shape, rng, lo=-1.0, hi=1.0):
    return Tensor(rng.uniform(lo, hi, size=shape).astype(np.float32), requires_grad=True)


# ---------- im2col / col2im ----------


def test_im2col_shape(rng):
    x = rng.standard_normal((2, 3, 5, 5)).astype(np.float32)
    cols = im2col(x, 3, 3, stride=1, pad=0)
    # H_out = W_out = 3
    assert cols.shape == (2, 3, 3, 3, 3, 3)


def test_im2col_identity_roundtrip_noverlap():
    # 2x2 non-overlapping kernel on 4x4 input: each element appears in exactly one col
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    cols = im2col(x, 2, 2, stride=2, pad=0)
    restored = col2im(cols, x.shape, stride=2, pad=0)
    assert np.array_equal(restored, x)


def test_col2im_sums_overlaps():
    x = np.ones((1, 1, 3, 3), dtype=np.float32)
    # stride=1, k=2 — each interior element appears in up to 4 cols
    cols = im2col(x, 2, 2, stride=1, pad=0)
    restored = col2im(cols, x.shape, stride=1, pad=0)
    # center element appears 4 times
    assert restored[0, 0, 1, 1] == 4.0


# ---------- Conv2d ----------


def test_conv2d_shape(rng):
    m = nn.Conv2d(3, 8, 3, stride=1, padding=1, seed=0)
    x = Tensor(rng.standard_normal((2, 3, 10, 10)).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 8, 10, 10)


def test_conv2d_stride_pad_shape(rng):
    m = nn.Conv2d(3, 8, 3, stride=2, padding=1, seed=0)
    x = Tensor(rng.standard_normal((2, 3, 10, 10)).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 8, 5, 5)


def test_conv2d_gradcheck_input(rng):
    m = nn.Conv2d(2, 3, 3, stride=1, padding=1, seed=0)
    x = _rt((1, 2, 4, 4), rng)
    gradcheck(lambda x: m(x).sum(), [x], atol=1e-2, rtol=1e-2)


def test_conv2d_gradcheck_weight(rng):
    m = nn.Conv2d(2, 3, 3, stride=1, padding=0, seed=0)
    x = Tensor(np.random.default_rng(1).standard_normal((1, 2, 5, 5)).astype(np.float32))
    gradcheck(lambda w: _conv_with_w(x, w, m.bias, stride=1, padding=0).sum(), [m.weight], atol=1e-2, rtol=1e-2)


def test_conv2d_gradcheck_bias(rng):
    m = nn.Conv2d(2, 3, 3, stride=1, padding=0, seed=0)
    x = Tensor(np.random.default_rng(2).standard_normal((1, 2, 5, 5)).astype(np.float32))
    gradcheck(lambda b: _conv_with_w(x, m.weight, b, stride=1, padding=0).sum(), [m.bias], atol=1e-2, rtol=1e-2)


def test_conv2d_matches_numpy_manual(rng):
    """Compare to a manual numpy conv2d result."""
    m = nn.Conv2d(2, 1, 3, stride=1, padding=0, bias=False, seed=0)
    x = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    out = m(Tensor(x)).data

    w = m.weight.data  # (1, 2, 3, 3)
    ref = np.zeros((1, 1, 2, 2), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            patch = x[0, :, i : i + 3, j : j + 3]
            ref[0, 0, i, j] = (patch * w[0]).sum()
    assert np.allclose(out, ref, atol=1e-5)


def _conv_with_w(x, w, b, stride, padding):
    from nanograd.nn.conv import _Conv2dFn
    out = _Conv2dFn.apply(x, w, stride=stride, padding=padding)
    if b is not None:
        out = out + b.reshape(1, -1, 1, 1)
    return out


# ---------- MaxPool2d / AvgPool2d ----------


def test_maxpool2d_shape(rng):
    m = nn.MaxPool2d(2)
    x = Tensor(rng.standard_normal((2, 3, 6, 6)).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 3, 3, 3)


def test_maxpool2d_values():
    x = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=np.float32)
    m = nn.MaxPool2d(2)
    y = m(Tensor(x)).data
    assert np.array_equal(y, [[[[6, 8], [14, 16]]]])


def test_maxpool2d_gradcheck(rng):
    m = nn.MaxPool2d(2)
    x = _rt((1, 2, 4, 4), rng)
    # ensure no ties by jittering
    x.data += 0.01 * np.arange(x.data.size).reshape(x.data.shape).astype(np.float32)
    gradcheck(lambda x: m(x).sum(), [x], atol=1e-2, rtol=1e-2)


def test_avgpool2d_values():
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    m = nn.AvgPool2d(2)
    y = m(Tensor(x)).data
    assert np.allclose(y, 2.5)


def test_avgpool2d_gradcheck(rng):
    m = nn.AvgPool2d(2)
    x = _rt((1, 2, 4, 4), rng)
    gradcheck(lambda x: m(x).sum(), [x], atol=1e-2, rtol=1e-2)


# ---------- BatchNorm2d ----------


def test_batchnorm2d_train_stats(rng):
    m = nn.BatchNorm2d(4)
    x = Tensor(rng.standard_normal((8, 4, 5, 5)).astype(np.float32) * 3 + 2)
    m.train()
    y = m(x)
    assert y.shape == x.shape
    # per-channel mean of output near 0, var near 1
    mean = y.data.mean(axis=(0, 2, 3))
    var = y.data.var(axis=(0, 2, 3))
    assert np.allclose(mean, 0.0, atol=1e-5)
    assert np.allclose(var, 1.0, atol=1e-3)


def test_batchnorm2d_running_stats_update(rng):
    m = nn.BatchNorm2d(2, momentum=0.5)
    x = Tensor(rng.standard_normal((4, 2, 3, 3)).astype(np.float32))
    m.train()
    _ = m(x)
    # running_mean should have moved from 0
    assert not np.allclose(m.running_mean, 0.0)


def test_batchnorm2d_eval_uses_running(rng):
    m = nn.BatchNorm2d(2)
    m.running_mean[:] = np.array([1.0, -1.0])
    m.running_var[:] = np.array([4.0, 4.0])
    x = Tensor(np.ones((1, 2, 2, 2), dtype=np.float32))
    m.eval()
    y = m(x).data
    # (1-1)/sqrt(4+eps) = 0 for channel 0, (1-(-1))/2 = 1 for channel 1
    assert np.allclose(y[0, 0], 0.0, atol=1e-4)
    assert np.allclose(y[0, 1], 1.0, atol=1e-4)


def test_batchnorm2d_gradcheck(rng):
    m = nn.BatchNorm2d(3)
    x = _rt((4, 3, 3, 3), rng)
    m.train()

    # gradcheck is a bit tricky because running stats mutate. Snapshot & restore.
    def f(x):
        m.running_mean[:] = 0.0
        m.running_var[:] = 1.0
        return m(x).sum()

    gradcheck(f, [x], atol=1e-1, rtol=1e-1)


# ---------- Flatten ----------


def test_flatten(rng):
    m = nn.Flatten()
    x = Tensor(rng.standard_normal((2, 3, 4, 4)).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 48)
