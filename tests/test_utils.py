import numpy as np

from nanograd import Tensor, nn
from nanograd.utils import param_summary, profile, summary, to_dot


def test_to_dot_contains_ops():
    x = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
    y = x * 2 + 1
    z = y.sum()
    dot = to_dot(z)
    assert "digraph G" in dot
    assert "Mul" in dot
    assert "Add" in dot
    assert "Sum" in dot


def test_to_dot_includes_shapes():
    x = Tensor(np.ones((3, 4), dtype=np.float32), requires_grad=True)
    dot = to_dot(x)
    assert "3×4" in dot


def test_param_summary():
    m = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    s = param_summary(m)
    assert "total" in s
    # 3*4 + 4 + 4*2 + 2 = 26
    assert "26" in s


def test_profile_records_ops():
    x = Tensor(np.ones((5, 5), dtype=np.float32), requires_grad=True)
    with profile() as state:
        for _ in range(10):
            y = (x * 2 + 1).sum()
    assert state.counts["Mul"] == 10
    assert state.counts["Sum"] == 10
    s = summary(state)
    assert "Mul" in s
    assert "Sum" in s


def test_profile_inactive_after_exit():
    from nanograd.utils.profile import _TimingState

    x = Tensor(np.ones((3,), dtype=np.float32), requires_grad=True)
    with profile():
        _ = x * 2
    # after exit, ops should not be recorded
    _ = x * 2
    assert _TimingState.active is False
