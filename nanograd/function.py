from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from nanograd.tensor import Tensor


class Function:
    """Base class for autograd ops.

    Subclass defines:
        forward(ctx, *inputs_ndarray) -> ndarray
        backward(ctx, grad_output_ndarray) -> tuple of ndarray

    ctx stores tensors via ctx.save_for_backward, scalars via attribute set.
    """

    def __init__(self) -> None:
        self.parents: Tuple[Tensor, ...] = ()
        self._saved: Tuple[Any, ...] = ()

    def save_for_backward(self, *items: Any) -> None:
        self._saved = items

    @property
    def saved(self) -> Tuple[Any, ...]:
        return self._saved

    # --- subclass overrides ---
    def forward(self, *args, **kwargs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray):  # pragma: no cover
        raise NotImplementedError

    # --- invocation ---
    @classmethod
    def apply(cls, *tensors: Tensor, **kwargs) -> Tensor:
        ctx = cls()
        ctx.parents = tensors
        raw_inputs = tuple(t.data for t in tensors)
        out_data = ctx.forward(*raw_inputs, **kwargs)
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(out_data, requires_grad=requires_grad)
        if requires_grad:
            out._ctx = ctx
        return out
