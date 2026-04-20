from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np


ArrayLike = Union["Tensor", np.ndarray, float, int, list]


def _as_array(x: ArrayLike, dtype=np.float32) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.data
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False) if x.dtype != dtype else x
    return np.asarray(x, dtype=dtype)


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_ctx")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        dtype=np.float32,
    ) -> None:
        self.data: np.ndarray = _as_array(data, dtype=dtype)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = bool(requires_grad)
        self._ctx: Optional["Function"] = None

    # ---- dunder / meta ----

    def __repr__(self) -> str:
        g = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{g})"

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    # ---- factory ----

    @staticmethod
    def zeros(*shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad: bool = False) -> "Tensor":
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad: bool = False, seed: Optional[int] = None) -> "Tensor":
        rng = np.random.default_rng(seed)
        return Tensor(rng.standard_normal(shape).astype(np.float32), requires_grad=requires_grad)

    @staticmethod
    def uniform(*shape, low: float = -1.0, high: float = 1.0, requires_grad: bool = False, seed: Optional[int] = None) -> "Tensor":
        rng = np.random.default_rng(seed)
        return Tensor(rng.uniform(low, high, size=shape).astype(np.float32), requires_grad=requires_grad)

    # ---- grad utilities ----

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.item())

    # ---- backward ----

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("called backward on a tensor that does not require grad")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be provided for non-scalar Tensor")
            grad_arr = np.ones_like(self.data)
        else:
            grad_arr = _as_array(grad)

        # topological sort
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(t: Tensor) -> None:
            if id(t) in visited:
                return
            visited.add(id(t))
            if t._ctx is not None:
                for p in t._ctx.parents:
                    if p.requires_grad:
                        build(p)
                topo.append(t)

        build(self)

        # accumulate input grad on self
        self.grad = grad_arr if self.grad is None else self.grad + grad_arr

        for t in reversed(topo):
            ctx = t._ctx
            assert ctx is not None
            grads = ctx.backward(t.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for p, g in zip(ctx.parents, grads):
                if not p.requires_grad or g is None:
                    continue
                g = _unbroadcast(g, p.data.shape)
                p.grad = g if p.grad is None else p.grad + g


def _unbroadcast(grad: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Reduce `grad` back to `shape` by summing broadcasted axes."""
    # sum over leading extra dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # sum over broadcast dims (size 1)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# local import to avoid cycle; bound at module load
from nanograd.function import Function  # noqa: E402
