"""DQN on the 5x5 gridworld used by reinforce_gridworld.

Network: MLP state → Q(s, a). Target net updated every ``target_sync`` steps.
Replay buffer: uniform sample of recent transitions. Epsilon-greedy exploration.
Loss: 1-step TD with target-network bootstrap:
    L = mean((r + gamma * max_a' Q_target(s', a')*(1-done) - Q(s,a))^2).
"""
from __future__ import annotations

from collections import deque
import random

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F

from examples.reinforce_gridworld import (
    N_ACTIONS,
    N_STATES,
    ROWS,
    COLS,
    GOAL,
    state_onehot,
    step_env,
)


class QNet(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.fc1 = nn.Linear(N_STATES, 64, seed=seed)
        self.fc2 = nn.Linear(64, N_ACTIONS, seed=seed + 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def copy_params(src: nn.Module, dst: nn.Module) -> None:
    for sp, dp in zip(src.parameters(), dst.parameters()):
        dp.data[...] = sp.data


class Replay:
    def __init__(self, capacity: int = 5000, seed: int = 0):
        self.buf: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, s, a, r, sp, done):
        self.buf.append((s, a, r, sp, done))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buf, batch_size)
        s, a, r, sp, d = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(sp).astype(np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


def run_episode(qnet, rng, epsilon: float, greedy: bool = False, max_steps: int = 50):
    pos = (0, 0)
    traj = []
    for _ in range(max_steps):
        s = state_onehot(pos)
        if not greedy and rng.random() < epsilon:
            a = rng.integers(0, N_ACTIONS)
        else:
            q = qnet(Tensor(s[None, :])).data[0]
            a = int(np.argmax(q))
        new_pos, r, done = step_env(pos, a)
        sp = state_onehot(new_pos)
        traj.append((s, int(a), float(r), sp, bool(done)))
        pos = new_pos
        if done:
            break
    return traj


def main():
    rng = np.random.default_rng(0)
    qnet = QNet(seed=0)
    tgt = QNet(seed=0)
    copy_params(qnet, tgt)
    opt = optim.Adam(qnet.parameters(), lr=5e-3)
    replay = Replay(capacity=5000, seed=0)

    gamma = 0.99
    batch_size = 64
    target_sync = 50
    eps_start, eps_end = 1.0, 0.05
    n_episodes = 300
    step_count = 0
    returns_log = []

    for ep in range(1, n_episodes + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (n_episodes * 0.6))
        traj = run_episode(qnet, rng, eps)
        returns_log.append(sum(t[2] for t in traj))
        for t in traj:
            replay.push(*t)

        if len(replay) >= batch_size:
            for _ in range(len(traj)):
                s, a, r, sp, d = replay.sample(batch_size)
                q_all = qnet(Tensor(s))                     # (B, A)
                q_sa = q_all[np.arange(batch_size), a]      # (B,)
                q_next = tgt(Tensor(sp)).data.max(axis=-1)  # (B,) no grad
                target = r + gamma * q_next * (1.0 - d)
                loss = ((q_sa - Tensor(target)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                optim.clip_grad_norm_(qnet.parameters(), max_norm=10.0)
                opt.step()

                step_count += 1
                if step_count % target_sync == 0:
                    copy_params(qnet, tgt)

        if ep % 30 == 0:
            recent = np.mean(returns_log[-30:])
            print(f"ep {ep:4d}  eps={eps:.2f}  avg_return(30)={recent:+.3f}", flush=True)

    greedy_traj = run_episode(qnet, rng, epsilon=0.0, greedy=True)
    greedy_ret = sum(t[2] for t in greedy_traj)
    print(f"\ngreedy return: {greedy_ret:.3f}  steps: {len(greedy_traj)}", flush=True)
    assert greedy_ret > 0.5, f"DQN did not learn policy: return={greedy_ret:.3f}"


if __name__ == "__main__":
    main()
