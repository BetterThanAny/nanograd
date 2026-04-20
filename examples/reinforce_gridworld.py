"""REINFORCE (policy gradient) on a tiny gridworld.

Env: 5x5 grid. Agent starts at (0,0), goal at (4,4). Reward +1 at goal, -0.01 per step.
Episode ends at goal or after 50 steps.

Action space: 4 directions (up/down/left/right).
State: one-hot of (row * 5 + col).

Policy: MLP that maps state → softmax over 4 actions.
Loss: -sum_t log_pi(a_t | s_t) * G_t   where G_t is the return-to-go.
"""
from __future__ import annotations

import numpy as np

from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F


ROWS, COLS = 5, 5
N_STATES = ROWS * COLS
N_ACTIONS = 4
GOAL = (ROWS - 1, COLS - 1)


def step_env(pos, action):
    r, c = pos
    if action == 0:
        r = max(0, r - 1)
    elif action == 1:
        r = min(ROWS - 1, r + 1)
    elif action == 2:
        c = max(0, c - 1)
    elif action == 3:
        c = min(COLS - 1, c + 1)
    new_pos = (r, c)
    done = new_pos == GOAL
    reward = 1.0 if done else -0.01
    return new_pos, reward, done


def state_onehot(pos) -> np.ndarray:
    v = np.zeros(N_STATES, dtype=np.float32)
    v[pos[0] * COLS + pos[1]] = 1.0
    return v


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_STATES, 32, seed=0)
        self.fc2 = nn.Linear(32, N_ACTIONS, seed=1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))  # logits


def run_episode(policy, rng, max_steps=50, greedy=False):
    pos = (0, 0)
    states, actions, rewards = [], [], []
    for _ in range(max_steps):
        s = state_onehot(pos)
        logits = policy(Tensor(s[None, :]))
        probs = F.softmax(logits).data[0]
        if greedy:
            a = int(np.argmax(probs))
        else:
            a = int(rng.choice(N_ACTIONS, p=probs))
        pos, r, done = step_env(pos, a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        if done:
            break
    return np.stack(states), np.array(actions), np.array(rewards)


def returns_to_go(rewards, gamma=0.99):
    G = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        G[t] = running
    # normalize for stable gradients
    return (G - G.mean()) / (G.std() + 1e-8)


def main():
    rng = np.random.default_rng(0)
    policy = Policy()
    opt = optim.Adam(policy.parameters(), lr=1e-2)

    returns_log = []
    for ep in range(1, 301):
        states, actions, rewards = run_episode(policy, rng)
        G = returns_to_go(rewards)

        logits = policy(Tensor(states))               # (T, 4)
        log_probs = F.log_softmax(logits)             # (T, 4)
        # select the log-prob of chosen action per step
        log_pa = log_probs[np.arange(len(actions)), actions]  # (T,)
        loss = -(log_pa * Tensor(G)).mean()

        opt.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

        returns_log.append(rewards.sum())
        if ep % 50 == 0:
            recent = np.mean(returns_log[-50:])
            print(f"episode {ep:4d}  avg return (last 50) = {recent:+.3f}  steps = {len(rewards)}", flush=True)

    # greedy evaluation
    _, _, greedy_rewards = run_episode(policy, rng, greedy=True)
    greedy_return = greedy_rewards.sum()
    print(f"\ngreedy return: {greedy_return:.3f}  steps: {len(greedy_rewards)}", flush=True)
    # optimal path length is 8 (Manhattan distance 5+5-? actually 4+4=8 with cost -0.07 + 1 = 0.93)
    assert greedy_return > 0.5, f"policy did not learn shortest path: return={greedy_return:.3f}"


if __name__ == "__main__":
    main()
