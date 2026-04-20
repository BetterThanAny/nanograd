from __future__ import annotations

from typing import Iterable, List

import numpy as np

from nanograd.tensor import Tensor


class Optimizer:
    """Base class. Subclasses override `step_param`."""

    def __init__(self, params: Iterable[Tensor]):
        # dedupe by id so tied / weight-shared parameters step only once
        seen: dict[int, Tensor] = {}
        for p in params:
            seen.setdefault(id(p), p)
        self.params: List[Tensor] = list(seen.values())
        self.state: dict = {}
        for p in self.params:
            self.state[id(p)] = {}

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            self._step_param(p, self.state[id(p)])

    def _step_param(self, p: Tensor, state: dict) -> None:  # pragma: no cover
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")

    def _step_param(self, p: Tensor, state: dict) -> None:
        g = p.grad
        if self.weight_decay != 0:
            g = g + self.weight_decay * p.data
        if self.momentum != 0:
            if "buf" not in state:
                state["buf"] = np.zeros_like(p.data)
            buf = state["buf"]
            buf *= self.momentum
            buf += g
            if self.nesterov:
                g = g + self.momentum * buf
            else:
                g = buf
        p.data -= self.lr * g


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def _step_param(self, p: Tensor, state: dict) -> None:
        g = p.grad
        if self.weight_decay != 0:
            g = g + self.weight_decay * p.data
        if "m" not in state:
            state["m"] = np.zeros_like(p.data)
            state["v"] = np.zeros_like(p.data)
            state["t"] = 0
        state["t"] += 1
        t = state["t"]
        m, v = state["m"], state["v"]
        m[...] = self.b1 * m + (1 - self.b1) * g
        v[...] = self.b2 * v + (1 - self.b2) * g * g
        m_hat = m / (1 - self.b1 ** t)
        v_hat = v / (1 - self.b2 ** t)
        p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """Decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def _step_param(self, p: Tensor, state: dict) -> None:
        g = p.grad
        if "m" not in state:
            state["m"] = np.zeros_like(p.data)
            state["v"] = np.zeros_like(p.data)
            state["t"] = 0
        state["t"] += 1
        t = state["t"]
        m, v = state["m"], state["v"]
        m[...] = self.b1 * m + (1 - self.b1) * g
        v[...] = self.b2 * v + (1 - self.b2) * g * g
        m_hat = m / (1 - self.b1 ** t)
        v_hat = v / (1 - self.b2 ** t)
        # decoupled weight decay
        if self.weight_decay != 0:
            p.data -= self.lr * self.weight_decay * p.data
        p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Adagrad(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay

    def _step_param(self, p: Tensor, state: dict) -> None:
        g = p.grad
        if self.weight_decay != 0:
            g = g + self.weight_decay * p.data
        if "sum_sq" not in state:
            state["sum_sq"] = np.zeros_like(p.data)
        state["sum_sq"] += g * g
        p.data -= self.lr * g / (np.sqrt(state["sum_sq"]) + self.eps)


class RMSProp(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

    def _step_param(self, p: Tensor, state: dict) -> None:
        g = p.grad
        if self.weight_decay != 0:
            g = g + self.weight_decay * p.data
        if "square_avg" not in state:
            state["square_avg"] = np.zeros_like(p.data)
            if self.momentum != 0:
                state["momentum_buf"] = np.zeros_like(p.data)
        sa = state["square_avg"]
        sa[...] = self.alpha * sa + (1 - self.alpha) * g * g
        avg = np.sqrt(sa) + self.eps
        if self.momentum != 0:
            buf = state["momentum_buf"]
            buf[...] = self.momentum * buf + g / avg
            p.data -= self.lr * buf
        else:
            p.data -= self.lr * g / avg
