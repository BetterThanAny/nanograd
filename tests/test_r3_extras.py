"""Tests for positional encoding, init helpers, gradient clipping."""
import math

import numpy as np
import pytest

from nanograd import Tensor, nn, optim
from nanograd.nn import init
from nanograd.nn.attention import sinusoidal_positional_encoding


# ---------------------------------------------------------------------------
# positional encoding
# ---------------------------------------------------------------------------


def test_sinusoidal_pe_shape():
    pe = sinusoidal_positional_encoding(10, 16)
    assert pe.shape == (10, 16)


def test_sinusoidal_pe_odd_dim_shape():
    """Regression test: odd dim must not crash or produce wrong shapes."""
    pe = sinusoidal_positional_encoding(10, 5)
    assert pe.shape == (10, 5)


def test_sinusoidal_pe_values():
    pe = sinusoidal_positional_encoding(4, 4)
    # position 0 should be all sin(0)=0, cos(0)=1
    expected_pos0 = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    assert np.allclose(pe[0], expected_pos0)


def test_sinusoidal_module_adds_to_input(rng=np.random.default_rng(0)):
    pe = nn.SinusoidalPositionalEncoding(16, 8)
    x = Tensor(np.zeros((1, 4, 8), dtype=np.float32))
    y = pe(x)
    assert y.shape == (1, 4, 8)
    # since x=0, y equals pe[:4]
    assert np.allclose(y.data[0], pe.pe[:4])


def test_learned_positional_encoding_shape():
    pe = nn.LearnedPositionalEncoding(16, 8, seed=0)
    x = Tensor(np.zeros((1, 4, 8), dtype=np.float32))
    y = pe(x)
    assert y.shape == (1, 4, 8)


def test_learned_pe_has_params():
    pe = nn.LearnedPositionalEncoding(10, 4)
    assert pe.num_params() == 40


# ---------------------------------------------------------------------------
# init helpers
# ---------------------------------------------------------------------------


def test_kaiming_normal_stats():
    w = nn.Parameter(np.zeros((100, 100), dtype=np.float32))
    init.kaiming_normal_(w, nonlinearity="relu")
    # std should be ~ sqrt(2/fan_in) = sqrt(2/100) ≈ 0.1414
    expected_std = math.sqrt(2.0 / 100)
    assert abs(w.data.std() - expected_std) < 0.02


def test_xavier_normal_stats():
    w = nn.Parameter(np.zeros((100, 100), dtype=np.float32))
    init.xavier_normal_(w)
    expected_std = math.sqrt(2.0 / (100 + 100))
    assert abs(w.data.std() - expected_std) < 0.02


def test_zeros_ones():
    w = nn.Parameter(np.random.default_rng(0).standard_normal((5, 5)).astype(np.float32))
    init.zeros_(w)
    assert np.all(w.data == 0)
    init.ones_(w)
    assert np.all(w.data == 1)


def test_init_on_conv_weight():
    """Conv weight shape is (out, in, kh, kw)."""
    conv = nn.Conv2d(3, 8, 3, seed=0)
    init.kaiming_normal_(conv.weight)
    # std ≈ sqrt(2 / (3*3*3)) ≈ 0.272
    expected = math.sqrt(2.0 / (3 * 3 * 3))
    assert abs(conv.weight.data.std() - expected) < 0.1


# ---------------------------------------------------------------------------
# gradient clipping
# ---------------------------------------------------------------------------


def test_clip_grad_norm_clips_when_exceeded():
    p = nn.Parameter(np.zeros(5, dtype=np.float32))
    p.grad = np.array([3.0, 4.0, 0.0, 0.0, 0.0], dtype=np.float32)  # norm=5
    total = optim.clip_grad_norm_([p], max_norm=1.0)
    assert np.isclose(total, 5.0, atol=1e-3)
    # clipped norm should now be ~1
    new_norm = np.linalg.norm(p.grad)
    assert np.isclose(new_norm, 1.0, atol=1e-3)


def test_clip_grad_norm_no_clip_when_below():
    p = nn.Parameter(np.zeros(3, dtype=np.float32))
    p.grad = np.array([0.3, 0.4, 0.0], dtype=np.float32)  # norm=0.5
    total = optim.clip_grad_norm_([p], max_norm=1.0)
    assert np.isclose(total, 0.5, atol=1e-3)
    # unchanged
    assert np.allclose(p.grad, [0.3, 0.4, 0.0])


def test_clip_grad_norm_multi_param():
    p1 = nn.Parameter(np.zeros(2, dtype=np.float32))
    p2 = nn.Parameter(np.zeros(2, dtype=np.float32))
    p1.grad = np.array([3.0, 0.0], dtype=np.float32)
    p2.grad = np.array([0.0, 4.0], dtype=np.float32)
    total = optim.clip_grad_norm_([p1, p2], max_norm=1.0)
    # total = sqrt(9 + 16) = 5
    assert np.isclose(total, 5.0)
    # each param scaled by 1/5
    assert np.isclose(np.linalg.norm(np.concatenate([p1.grad, p2.grad])), 1.0, atol=1e-3)


def test_clip_grad_value():
    p = nn.Parameter(np.zeros(5, dtype=np.float32))
    p.grad = np.array([-10.0, -0.5, 0.0, 0.5, 10.0], dtype=np.float32)
    optim.clip_grad_value_([p], clip_value=1.0)
    assert np.allclose(p.grad, [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_clip_grad_norm_p1_handles_negative():
    """Regression: 1-norm was computing sum(x) instead of sum(|x|)."""
    p = nn.Parameter(np.zeros(3, dtype=np.float32))
    p.grad = np.array([-3.0, -2.0, 1.0], dtype=np.float32)
    # 1-norm = |-3| + |-2| + |1| = 6
    total = optim.clip_grad_norm_([p], max_norm=100.0, norm_type=1)
    assert np.isclose(total, 6.0)


def test_clip_grad_norm_inf():
    p = nn.Parameter(np.zeros(4, dtype=np.float32))
    p.grad = np.array([0.5, -3.0, 1.5, 2.0], dtype=np.float32)
    total = optim.clip_grad_norm_([p], max_norm=100.0, norm_type=float("inf"))
    assert np.isclose(total, 3.0)


def test_clip_grad_skips_none():
    p1 = nn.Parameter(np.zeros(2, dtype=np.float32))  # no grad
    p2 = nn.Parameter(np.zeros(2, dtype=np.float32))
    p2.grad = np.ones(2, dtype=np.float32)
    # should not crash
    total = optim.clip_grad_norm_([p1, p2], max_norm=10.0)
    assert total > 0
