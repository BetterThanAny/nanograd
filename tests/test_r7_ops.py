"""Tests for R7 op additions: std/var, flip/roll, gather/scatter_add, F.normalize."""
from __future__ import annotations

import numpy as np

from nanograd import Tensor, flip, gather, roll, scatter_add
from nanograd.nn import functional as F
from nanograd.utils.gradcheck import gradcheck


# ---------- std / var ----------


def test_var_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    t = Tensor(a, requires_grad=True)
    np.testing.assert_allclose(t.var(axis=1).data, a.var(axis=1), atol=1e-5)
    np.testing.assert_allclose(t.var().data, a.var(), atol=1e-5)
    np.testing.assert_allclose(
        t.var(axis=1, unbiased=True).data, a.var(axis=1, ddof=1), atol=1e-5
    )


def test_std_matches_numpy():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    t = Tensor(a, requires_grad=True)
    np.testing.assert_allclose(t.std(axis=1).data, a.std(axis=1), atol=1e-5)


def test_var_gradcheck():
    rng = np.random.default_rng(2)
    a = Tensor(rng.standard_normal((3, 4)).astype(np.float64), requires_grad=True)
    gradcheck(lambda x: x.var(axis=1), [a], atol=1e-2, rtol=1e-2)


# ---------- flip ----------


def test_flip_forward():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = Tensor(a, requires_grad=True)
    np.testing.assert_array_equal(flip(t, axis=1).data, np.flip(a, axis=1))
    np.testing.assert_array_equal(t.flip(axis=0).data, np.flip(a, axis=0))


def test_flip_gradcheck():
    a = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
    gradcheck(lambda x: flip(x, axis=1), [a], atol=1e-2, rtol=1e-2)


# ---------- roll ----------


def test_roll_forward():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = Tensor(a, requires_grad=True)
    np.testing.assert_array_equal(roll(t, 2, axis=1).data, np.roll(a, 2, axis=1))


def test_roll_gradcheck():
    a = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
    gradcheck(lambda x: roll(x, 1, axis=1), [a], atol=1e-2, rtol=1e-2)


# ---------- gather ----------


def test_gather_forward():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    idx = np.array([[0, 2, 1, 3], [3, 0, 1, 2], [1, 1, 1, 1]])
    t = Tensor(a, requires_grad=True)
    out = gather(t, idx, axis=1).data
    np.testing.assert_array_equal(out, np.take_along_axis(a, idx, axis=1))


def test_gather_gradcheck():
    a = Tensor(np.random.randn(3, 5).astype(np.float64), requires_grad=True)
    idx = np.array([[0, 2, 1], [3, 0, 4], [1, 1, 2]], dtype=np.int64)
    gradcheck(lambda x: gather(x, idx, axis=1), [a], atol=1e-2, rtol=1e-2)


def test_gather_with_duplicate_indices_grad_accumulates():
    a = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
    idx = np.array([[0, 0, 1]])
    out = gather(a, idx, axis=1)
    out.sum().backward()
    # positions 0 got 2 gradients, position 1 got 1, position 2 got 0
    np.testing.assert_array_equal(a.grad, [[2.0, 1.0, 0.0]])


# ---------- scatter_add ----------


def test_scatter_add_forward():
    base = np.zeros((3, 4), dtype=np.float32)
    src = np.ones((3, 4), dtype=np.float32)
    idx = np.array([[0, 1, 2, 3], [1, 1, 2, 2], [0, 0, 0, 0]])
    out = scatter_add(Tensor(base), idx, Tensor(src), axis=1).data
    expect = np.zeros((3, 4))
    np.add.at(expect, (np.indices(idx.shape)[0], idx), src)
    np.testing.assert_array_equal(out, expect)


def test_scatter_add_gradcheck():
    base = Tensor(np.random.randn(3, 4).astype(np.float64), requires_grad=True)
    src = Tensor(np.random.randn(3, 2).astype(np.float64), requires_grad=True)
    idx = np.array([[0, 1], [2, 2], [1, 3]], dtype=np.int64)
    gradcheck(
        lambda b, s: scatter_add(b, idx, s, axis=1),
        [base, src],
        atol=1e-2,
        rtol=1e-2,
    )


# ---------- F.normalize ----------


def test_normalize_l2_matches_numpy():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((3, 4)).astype(np.float32)
    t = Tensor(a, requires_grad=True)
    out = F.normalize(t, p=2.0, axis=1).data
    expect = a / np.linalg.norm(a, axis=1, keepdims=True)
    np.testing.assert_allclose(out, expect, atol=1e-5)
    # unit norm
    np.testing.assert_allclose(
        np.linalg.norm(out, axis=1), np.ones(3), atol=1e-5
    )


def test_normalize_gradcheck():
    a = Tensor(np.random.randn(2, 3).astype(np.float64) + 0.5, requires_grad=True)
    gradcheck(lambda x: F.normalize(x, axis=1), [a], atol=1e-2, rtol=1e-2)
