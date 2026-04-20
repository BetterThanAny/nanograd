"""Tests for the DQN example's reusable building blocks."""
from __future__ import annotations

import numpy as np

from examples.dqn_gridworld import QNet, Replay, copy_params, run_episode


def test_replay_buffer_capacity_and_sampling():
    r = Replay(capacity=10, seed=0)
    for i in range(20):
        r.push(np.array([i], dtype=np.float32), 0, float(i), np.array([i], dtype=np.float32), False)
    assert len(r) == 10
    s, a, rw, sp, d = r.sample(5)
    assert s.shape == (5, 1) and a.shape == (5,) and rw.shape == (5,) and sp.shape == (5, 1)


def test_qnet_forward_shape():
    q = QNet(seed=0)
    x = np.random.randn(3, 25).astype(np.float32)
    from nanograd import Tensor
    out = q(Tensor(x))
    assert out.shape == (3, 4)


def test_copy_params_deep_copy():
    a = QNet(seed=0)
    b = QNet(seed=1)
    copy_params(a, b)
    for pa, pb in zip(a.parameters(), b.parameters()):
        np.testing.assert_array_equal(pa.data, pb.data)
    # mutating b should not affect a
    for pb in b.parameters():
        pb.data += 1.0
    any_diff = any(not np.allclose(pa.data, pb.data) for pa, pb in zip(a.parameters(), b.parameters()))
    assert any_diff, "copy_params did not deep-copy"


def test_run_episode_produces_transitions():
    q = QNet(seed=0)
    rng = np.random.default_rng(0)
    traj = run_episode(q, rng, epsilon=1.0, max_steps=10)
    assert 1 <= len(traj) <= 10
    for s, a, r, sp, d in traj:
        assert s.shape == (25,) and sp.shape == (25,)
        assert 0 <= a < 4
