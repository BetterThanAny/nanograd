"""Small training utilities: EarlyStopping, ModelCheckpoint, MetricTracker, Trainer, EMA."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from nanograd.nn.module import Module
from nanograd.optim.grad_clip import clip_grad_norm_
from nanograd.optim.optimizer import Optimizer
from nanograd.tensor import Tensor
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


class Trainer:
    """Minimal training loop orchestrator.

    ``step_fn`` takes a batch (the DataLoader yield) and returns a scalar Tensor loss.
    Optional ``eval_fn`` runs in eval mode on a validation loader and returns a dict of metrics.

    Example:
        def step(batch):
            X, y = batch
            logits = model(Tensor(X))
            return F.cross_entropy(logits, Tensor(y))

        trainer = Trainer(model, optimizer, step)
        trainer.fit(train_loader, epochs=3)
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        step_fn: Callable,
        eval_fn: Optional[Callable] = None,
        grad_clip: Optional[float] = None,
        on_epoch_end: Optional[Callable] = None,
        callbacks: Sequence = (),
    ):
        self.model = model
        self.optimizer = optimizer
        self.step_fn = step_fn
        self.eval_fn = eval_fn
        self.grad_clip = grad_clip
        self.on_epoch_end = on_epoch_end
        self.callbacks = list(callbacks)  # EarlyStopping / ModelCheckpoint etc.

    def fit(
        self,
        train_loader: Iterable,
        epochs: int = 1,
        val_loader: Optional[Iterable] = None,
        verbose: bool = True,
    ) -> dict:
        history: dict[str, list] = {"train_loss": [], "val_loss": []}

        for ep in range(1, epochs + 1):
            self.model.train()
            t0 = time.time()
            tracker = MetricTracker()
            for batch in train_loader:
                loss = self.step_fn(batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                # batch size: try to infer
                n = self._batch_size(batch)
                tracker.update("loss", loss.item(), n=n)
            train_loss = tracker.avg("loss")
            history["train_loss"].append(train_loss)

            val_metrics: dict = {}
            if val_loader is not None and self.eval_fn is not None:
                self.model.eval()
                val_metrics = self.eval_fn(val_loader)
                if "loss" in val_metrics:
                    history["val_loss"].append(val_metrics["loss"])

            dt = time.time() - t0
            if verbose:
                msg = f"epoch {ep}/{epochs}  train_loss={train_loss:.4f}"
                if val_metrics:
                    msg += "  " + "  ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items())
                msg += f"  ({dt:.1f}s)"
                print(msg, flush=True)

            if self.on_epoch_end is not None:
                self.on_epoch_end(ep, train_loss, val_metrics)

            # run callbacks; ES can stop the loop
            stop = False
            for cb in self.callbacks:
                monitor_val = val_metrics.get("loss", train_loss) if isinstance(cb, EarlyStopping) else None
                if isinstance(cb, EarlyStopping):
                    if cb.step(monitor_val):
                        stop = True
                elif isinstance(cb, ModelCheckpoint):
                    cb.step(val_metrics.get("loss", train_loss), self.model)
            if stop:
                if verbose:
                    print(f"[trainer] early stopping at epoch {ep}", flush=True)
                break

        return history

    @staticmethod
    def _batch_size(batch) -> int:
        if isinstance(batch, tuple):
            b = batch[0]
        else:
            b = batch
        if hasattr(b, "shape"):
            return b.shape[0] if len(b.shape) else 1
        return 1


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


class EMA:
    """Exponential moving average of a model's parameters.

    Keeps a shadow copy of every parameter (and, on request, registered buffers
    such as BatchNorm running stats) with decay ``decay``:
        shadow = decay * shadow + (1 - decay) * param

    Usage:
        ema = EMA(model, decay=0.999)
        # training loop:
        loss.backward(); opt.step(); ema.update()
        # eval:
        with ema.swap_into(model):
            evaluate(model)
    """

    def __init__(self, model: Module, decay: float = 0.999, include_buffers: bool = False):
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")
        self.model = model
        self.decay = decay
        self.include_buffers = include_buffers
        self._shadow: dict[str, np.ndarray] = {}
        for name, p in model.named_parameters():
            self._shadow[name] = p.data.copy()
        if include_buffers:
            for name, b in self._buffer_items():
                self._shadow[f"_buf_:{name}"] = b.copy()

    def _buffer_items(self):
        # Module exposes named_buffers() (R3 adds state_dict buffer support)
        if hasattr(self.model, "named_buffers"):
            yield from self.model.named_buffers()

    def update(self) -> None:
        d = self.decay
        for name, p in self.model.named_parameters():
            s = self._shadow[name]
            s *= d
            s += (1.0 - d) * p.data
        if self.include_buffers:
            for name, b in self._buffer_items():
                s = self._shadow[f"_buf_:{name}"]
                s *= d
                s += (1.0 - d) * b

    def apply_to(self, model: Optional[Module] = None) -> None:
        """Copy EMA shadow weights into ``model`` (defaults to tracked model)."""
        m = model or self.model
        for name, p in m.named_parameters():
            if name in self._shadow:
                p.data[...] = self._shadow[name]
        if self.include_buffers and hasattr(m, "named_buffers"):
            for name, b in m.named_buffers():
                key = f"_buf_:{name}"
                if key in self._shadow:
                    b[...] = self._shadow[key]

    class _Swap:
        def __init__(self, ema: "EMA", model: Module):
            self.ema = ema
            self.model = model
            self._backup: dict[str, np.ndarray] = {}
            self._buf_backup: dict[str, np.ndarray] = {}

        def __enter__(self):
            for name, p in self.model.named_parameters():
                self._backup[name] = p.data.copy()
                if name in self.ema._shadow:
                    p.data[...] = self.ema._shadow[name]
            if self.ema.include_buffers and hasattr(self.model, "named_buffers"):
                for name, b in self.model.named_buffers():
                    self._buf_backup[name] = b.copy()
                    key = f"_buf_:{name}"
                    if key in self.ema._shadow:
                        b[...] = self.ema._shadow[key]
            return self.model

        def __exit__(self, *exc):
            for name, p in self.model.named_parameters():
                if name in self._backup:
                    p.data[...] = self._backup[name]
            if self.ema.include_buffers and hasattr(self.model, "named_buffers"):
                for name, b in self.model.named_buffers():
                    if name in self._buf_backup:
                        b[...] = self._buf_backup[name]

    def swap_into(self, model: Optional[Module] = None):
        """Context manager that installs EMA weights for ``model`` for its duration."""
        return EMA._Swap(self, model or self.model)
