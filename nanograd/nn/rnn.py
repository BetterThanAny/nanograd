"""RNN and LSTM cells.

Shape conventions:
  input x:  (B, T, D_in) if batched, (T, D_in) if unbatched
  hidden h: (B, D_hid)   if batched, (D_hid,)  if unbatched

For RNNCell/LSTMCell, inputs are the current time step only (no T dim).
For RNN/LSTM modules, outputs include the full sequence.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from nanograd.nn import functional as F
from nanograd.nn.module import Module, Parameter
from nanograd.tensor import Tensor


# ---------------------------------------------------------------------------
# RNNCell (tanh)
# ---------------------------------------------------------------------------


class RNNCell(Module):
    """h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)"""

    def __init__(self, input_size: int, hidden_size: int, seed: Optional[int] = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        bound = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(rng.uniform(-bound, bound, size=(input_size, hidden_size)).astype(np.float32))
        self.W_hh = Parameter(rng.uniform(-bound, bound, size=(hidden_size, hidden_size)).astype(np.float32))
        self.b_ih = Parameter(rng.uniform(-bound, bound, size=(hidden_size,)).astype(np.float32))
        self.b_hh = Parameter(rng.uniform(-bound, bound, size=(hidden_size,)).astype(np.float32))

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return F.tanh(x @ self.W_ih + self.b_ih + h @ self.W_hh + self.b_hh)


class RNN(Module):
    """Processes a sequence step-by-step using an RNNCell."""

    def __init__(self, input_size: int, hidden_size: int, seed: Optional[int] = None):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size, seed=seed)
        self.hidden_size = hidden_size

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # x: (B, T, D) -> iterate over T
        B, T, _ = x.shape
        if h0 is None:
            h = Tensor(np.zeros((B, self.hidden_size), dtype=np.float32))
        else:
            h = h0
        outs = []
        for t in range(T):
            xt = x[:, t, :]
            h = self.cell(xt, h)
            outs.append(h)
        # stack outs along time: use concat via reshape — simpler: build via np on the data side won't capture grad.
        # We need a Stack function.
        out = _stack(outs, axis=1)
        return out, h


# ---------------------------------------------------------------------------
# LSTMCell
# ---------------------------------------------------------------------------


class LSTMCell(Module):
    """Standard LSTM. i, f, g, o gates concatenated for efficiency."""

    def __init__(self, input_size: int, hidden_size: int, seed: Optional[int] = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        bound = 1 / math.sqrt(hidden_size)
        # single matmul with weights shape (input, 4*hidden) and (hidden, 4*hidden)
        self.W_ih = Parameter(rng.uniform(-bound, bound, size=(input_size, 4 * hidden_size)).astype(np.float32))
        self.W_hh = Parameter(rng.uniform(-bound, bound, size=(hidden_size, 4 * hidden_size)).astype(np.float32))
        self.b_ih = Parameter(rng.uniform(-bound, bound, size=(4 * hidden_size,)).astype(np.float32))
        self.b_hh = Parameter(rng.uniform(-bound, bound, size=(4 * hidden_size,)).astype(np.float32))

    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        h, c = state
        gates = x @ self.W_ih + self.b_ih + h @ self.W_hh + self.b_hh  # (B, 4H)
        H = self.hidden_size
        i = F.sigmoid(gates[:, 0 * H : 1 * H])
        f = F.sigmoid(gates[:, 1 * H : 2 * H])
        g = F.tanh(gates[:, 2 * H : 3 * H])
        o = F.sigmoid(gates[:, 3 * H : 4 * H])
        c_new = f * c + i * g
        h_new = o * F.tanh(c_new)
        return h_new, c_new


class GRUCell(Module):
    """Gated Recurrent Unit cell. Slightly cheaper than LSTM."""

    def __init__(self, input_size: int, hidden_size: int, seed: Optional[int] = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        bound = 1 / math.sqrt(hidden_size)
        # 3 gates: r (reset), z (update), n (candidate)
        self.W_ih = Parameter(rng.uniform(-bound, bound, size=(input_size, 3 * hidden_size)).astype(np.float32))
        self.W_hh = Parameter(rng.uniform(-bound, bound, size=(hidden_size, 3 * hidden_size)).astype(np.float32))
        self.b_ih = Parameter(rng.uniform(-bound, bound, size=(3 * hidden_size,)).astype(np.float32))
        self.b_hh = Parameter(rng.uniform(-bound, bound, size=(3 * hidden_size,)).astype(np.float32))

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        gi = x @ self.W_ih + self.b_ih
        gh = h @ self.W_hh + self.b_hh
        H = self.hidden_size
        r = F.sigmoid(gi[:, 0:H] + gh[:, 0:H])
        z = F.sigmoid(gi[:, H : 2 * H] + gh[:, H : 2 * H])
        n = F.tanh(gi[:, 2 * H : 3 * H] + r * gh[:, 2 * H : 3 * H])
        return (Tensor(np.ones_like(z.data)) - z) * n + z * h


class GRU(Module):
    def __init__(self, input_size: int, hidden_size: int, seed: Optional[int] = None):
        super().__init__()
        self.cell = GRUCell(input_size, hidden_size, seed=seed)
        self.hidden_size = hidden_size

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, T, _ = x.shape
        if h0 is None:
            h = Tensor(np.zeros((B, self.hidden_size), dtype=np.float32))
        else:
            h = h0
        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outs.append(h)
        return _stack(outs, axis=1), h


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, seed: Optional[int] = None):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size, seed=seed)
        self.hidden_size = hidden_size

    def forward(
        self,
        x: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, T, _ = x.shape
        if state is None:
            h = Tensor(np.zeros((B, self.hidden_size), dtype=np.float32))
            c = Tensor(np.zeros((B, self.hidden_size), dtype=np.float32))
        else:
            h, c = state
        outs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
            outs.append(h)
        out = _stack(outs, axis=1)
        return out, (h, c)


# ---------------------------------------------------------------------------
# stack helper (autograd-aware)
# ---------------------------------------------------------------------------


from nanograd.function import Function


class _Stack(Function):
    """Stack a list of tensors along a new axis."""

    def forward(self, *arrs, axis):
        self.axis = axis
        self.n = len(arrs)
        return np.stack(arrs, axis=axis)

    def backward(self, g):
        # slice g along axis
        return tuple(
            np.take(g, i, axis=self.axis) for i in range(self.n)
        )


def _stack(tensors: list, axis: int = 0) -> Tensor:
    return _Stack.apply(*tensors, axis=axis)
