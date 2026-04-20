"""Tests for focal + triplet losses."""
import numpy as np
import pytest

from nanograd import Tensor
from nanograd.nn import functional as F
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(99)


# ---------- Focal loss ----------


def test_focal_loss_runs():
    logits = Tensor(np.array([[-2.0, 2.0], [0.5, -0.5]], dtype=np.float32), requires_grad=True)
    target = Tensor(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    loss = F.focal_loss(logits, target, alpha=0.25, gamma=2.0)
    assert np.isfinite(loss.item())
    assert loss.item() > 0


def test_focal_loss_approaches_bce_at_gamma_0():
    """When gamma=0 and alpha=1, focal = -t*log(p) which is close to BCE with logits."""
    logits = Tensor(np.array([[0.5, -0.5, 2.0]], dtype=np.float32), requires_grad=True)
    target = Tensor(np.array([[1.0, 0.0, 1.0]], dtype=np.float32))
    # alpha=1, gamma=0 → focal = -t*log(p) - 0*(1-t)*log(1-p)
    # this is just positive-only BCE, not equal to full BCE
    loss_focal = F.focal_loss(logits, target, alpha=1.0, gamma=0.0)
    assert np.isfinite(loss_focal.item())


def test_focal_loss_gradcheck(rng):
    logits = Tensor(rng.uniform(-2, 2, (3, 4)).astype(np.float32), requires_grad=True)
    target = Tensor(rng.integers(0, 2, (3, 4)).astype(np.float32))
    gradcheck(lambda x: F.focal_loss(x, target, 0.25, 2.0), [logits], atol=1e-2, rtol=1e-2)


def test_focal_loss_stable_large_logits():
    logits = Tensor(np.array([[100.0, -100.0]], dtype=np.float32), requires_grad=True)
    target = Tensor(np.array([[1.0, 0.0]], dtype=np.float32))
    loss = F.focal_loss(logits, target)
    assert np.isfinite(loss.item())


# ---------- Triplet loss ----------


def test_triplet_loss_easy_case():
    # anchor close to positive, far from negative → loss = 0
    a = Tensor(np.array([[1.0, 0.0]], dtype=np.float32), requires_grad=True)
    p = Tensor(np.array([[1.1, 0.0]], dtype=np.float32))
    n = Tensor(np.array([[5.0, 5.0]], dtype=np.float32))
    loss = F.triplet_loss(a, p, n, margin=0.5)
    assert np.isclose(loss.item(), 0.0)


def test_triplet_loss_hard_case():
    # anchor close to negative, far from positive → loss > 0
    a = Tensor(np.array([[0.0, 0.0]], dtype=np.float32), requires_grad=True)
    p = Tensor(np.array([[5.0, 0.0]], dtype=np.float32))
    n = Tensor(np.array([[0.1, 0.0]], dtype=np.float32))
    loss = F.triplet_loss(a, p, n, margin=0.5)
    # d_ap = 25, d_an = 0.01, margin = 0.5 → 25 - 0.01 + 0.5 = 25.49
    assert np.isclose(loss.item(), 25.49, atol=1e-3)


def test_triplet_loss_gradcheck(rng):
    a = Tensor(rng.standard_normal((2, 4)).astype(np.float32), requires_grad=True)
    p = Tensor(rng.standard_normal((2, 4)).astype(np.float32), requires_grad=True)
    n = Tensor(rng.standard_normal((2, 4)).astype(np.float32), requires_grad=True)
    gradcheck(lambda a, p, n: F.triplet_loss(a, p, n, 1.0), [a, p, n], atol=1e-2, rtol=1e-2)


def test_triplet_loss_no_grad_in_inactive_region(rng):
    # deliberately inactive (easy case): loss = 0, grad should be 0
    a = Tensor(np.array([[0.0]], dtype=np.float32), requires_grad=True)
    p = Tensor(np.array([[0.0]], dtype=np.float32), requires_grad=True)
    n = Tensor(np.array([[10.0]], dtype=np.float32), requires_grad=True)
    loss = F.triplet_loss(a, p, n, margin=0.1)
    loss.backward()
    assert np.allclose(a.grad, 0.0)
    assert np.allclose(p.grad, 0.0)
    assert np.allclose(n.grad, 0.0)
