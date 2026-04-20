"""Small training utilities: EarlyStopping, ModelCheckpoint, MetricTracker."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from nanograd.nn.module import Module
from nanograd.utils.checkpoint import save


class EarlyStopping:
    """Tracks a monitored metric and signals when training should stop."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def _is_better(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def step(self, metric: float) -> bool:
        """Call once per validation. Returns True if training should stop."""
        if self._is_better(metric):
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelCheckpoint:
    """Save the best model by monitored metric."""

    def __init__(
        self,
        path: str | Path,
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.path = Path(path)
        self.mode = mode
        self.best: Optional[float] = None

    def step(self, metric: float, module: Module) -> bool:
        """Save if metric improved. Returns True if saved."""
        is_better = (
            self.best is None
            or (self.mode == "min" and metric < self.best)
            or (self.mode == "max" and metric > self.best)
        )
        if is_better:
            self.best = metric
            save(module, self.path)
            return True
        return False


class MetricTracker:
    """Keeps running averages of metrics across a loop."""

    def __init__(self):
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        self._sums[name] = self._sums.get(name, 0.0) + float(value) * n
        self._counts[name] = self._counts.get(name, 0) + n

    def avg(self, name: str) -> float:
        c = self._counts.get(name, 0)
        return self._sums.get(name, 0.0) / c if c else 0.0

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def summary(self) -> dict[str, float]:
        return {k: self.avg(k) for k in self._sums}
