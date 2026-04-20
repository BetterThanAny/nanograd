from __future__ import annotations

import math

from nanograd.optim.optimizer import Optimizer


class _LRScheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_step = -1
        self.step()

    def step(self) -> None:
        self.last_step += 1
        self.optimizer.lr = self._compute_lr()

    def _compute_lr(self) -> float:  # pragma: no cover
        raise NotImplementedError


class StepLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def _compute_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self.last_step // self.step_size))


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def _compute_lr(self) -> float:
        t = min(self.last_step, self.T_max)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * t / self.T_max)
        )


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float):
        self.gamma = gamma
        super().__init__(optimizer)

    def _compute_lr(self) -> float:
        return self.base_lr * (self.gamma ** self.last_step)


class WarmupCosine(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup: int, T_max: int, eta_min: float = 0.0):
        self.warmup = warmup
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def _compute_lr(self) -> float:
        s = self.last_step
        if s < self.warmup:
            return self.base_lr * (s + 1) / max(1, self.warmup)
        t = min(s - self.warmup, self.T_max - self.warmup)
        progress = t / max(1, self.T_max - self.warmup)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
