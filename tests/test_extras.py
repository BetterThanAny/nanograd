"""Tests for extensions: Embedding, checkpoint, Adagrad."""
import tempfile

import numpy as np
import pytest

from nanograd import Tensor, nn, optim, utils
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(41)


def test_embedding_shape(rng):
    e = nn.Embedding(100, 16, seed=0)
    idx = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    out = e(idx)
    assert out.shape == (2, 3, 16)


def test_embedding_grad(rng):
    e = nn.Embedding(5, 3, seed=0)
    idx = Tensor(np.array([0, 2, 2, 4], dtype=np.int64))
    out = e(idx)
    out.sum().backward()
    # row 0 used once, row 2 used twice, row 4 used once
    assert np.allclose(e.weight.grad[0], 1.0)
    assert np.allclose(e.weight.grad[1], 0.0)
    assert np.allclose(e.weight.grad[2], 2.0)
    assert np.allclose(e.weight.grad[4], 1.0)


def test_adagrad_converges():
    x = nn.Parameter(np.zeros(3, dtype=np.float32))
    target = np.ones(3, dtype=np.float32)
    opt = optim.Adagrad([x], lr=0.5)
    for _ in range(200):
        diff = x - Tensor(target)
        loss = (diff * diff).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert np.linalg.norm(x.data - target) < 1e-2


def test_checkpoint_roundtrip(tmp_path):
    m = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    # set parameter values to known markers
    for i, p in enumerate(m.parameters()):
        p.data[:] = i + 1

    path = tmp_path / "ckpt.npz"
    utils.save(m, path)

    m2 = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    utils.load(m2, path)
    for p, p2 in zip(m.parameters(), m2.parameters()):
        assert np.allclose(p.data, p2.data)
