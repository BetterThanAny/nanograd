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
    """Multi-layer RNN. Stacks ``num_layers`` RNNCells feeding forward hidden states."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        base = seed or 0
        for i in range(num_layers):
            inp = input_size if i == 0 else hidden_size
            setattr(self, f"cell_{i}", RNNCell(inp, hidden_size, seed=base + i * 100))

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # x: (B, T, D); h0: (num_layers, B, H) or None
        B, T, _ = x.shape
        if h0 is None:
            h_layers = [Tensor(np.zeros((B, self.hidden_size), dtype=np.float32)) for _ in range(self.num_layers)]
        else:
            h_layers = [h0[l] for l in range(self.num_layers)]
        if T == 0:
            empty = Tensor(np.zeros((B, 0, self.hidden_size), dtype=np.float32))
            return empty, _stack(h_layers, axis=0)
        outs = []
        for t in range(T):
            xt = x[:, t, :]
            for l in range(self.num_layers):
                h_layers[l] = getattr(self, f"cell_{l}")(xt, h_layers[l])
                xt = h_layers[l]
            outs.append(xt)
        out = _stack(outs, axis=1)
        return out, _stack(h_layers, axis=0)


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
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        base = seed or 0
        for i in range(num_layers):
            inp = input_size if i == 0 else hidden_size
            setattr(self, f"cell_{i}", GRUCell(inp, hidden_size, seed=base + i * 100))

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, T, _ = x.shape
        if h0 is None:
            h_layers = [Tensor(np.zeros((B, self.hidden_size), dtype=np.float32)) for _ in range(self.num_layers)]
        else:
            h_layers = [h0[l] for l in range(self.num_layers)]
        if T == 0:
            return Tensor(np.zeros((B, 0, self.hidden_size), dtype=np.float32)), _stack(h_layers, axis=0)
        outs = []
        for t in range(T):
            xt = x[:, t, :]
            for l in range(self.num_layers):
                h_layers[l] = getattr(self, f"cell_{l}")(xt, h_layers[l])
                xt = h_layers[l]
            outs.append(xt)
        return _stack(outs, axis=1), _stack(h_layers, axis=0)


class LSTM(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        base = seed or 0
        for i in range(num_layers):
            inp = input_size if i == 0 else hidden_size
            setattr(self, f"cell_{i}", LSTMCell(inp, hidden_size, seed=base + i * 100))

    def forward(
        self,
        x: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, T, _ = x.shape
        if state is None:
            h_layers = [Tensor(np.zeros((B, self.hidden_size), dtype=np.float32)) for _ in range(self.num_layers)]
            c_layers = [Tensor(np.zeros((B, self.hidden_size), dtype=np.float32)) for _ in range(self.num_layers)]
        else:
            h, c = state
            h_layers = [h[l] for l in range(self.num_layers)]
            c_layers = [c[l] for l in range(self.num_layers)]
        if T == 0:
            empty = Tensor(np.zeros((B, 0, self.hidden_size), dtype=np.float32))
            return empty, (_stack(h_layers, axis=0), _stack(c_layers, axis=0))
        outs = []
        for t in range(T):
            xt = x[:, t, :]
            for l in range(self.num_layers):
                h_layers[l], c_layers[l] = getattr(self, f"cell_{l}")(xt, (h_layers[l], c_layers[l]))
                xt = h_layers[l]
            outs.append(xt)
        out = _stack(outs, axis=1)
        return out, (_stack(h_layers, axis=0), _stack(c_layers, axis=0))


# ---------------------------------------------------------------------------
# stack helper (autograd-aware)
# ---------------------------------------------------------------------------


from nanograd.function import Function


class Bidirectional(Module):
    """Wraps an RNN/LSTM/GRU and runs it on both the forward and reversed sequence,
    then concatenates the outputs along the hidden dim.

    Given an inner module with hidden_size H, output has hidden_size 2*H.
    """

    def __init__(self, module_fw: Module, module_bw: Module):
        super().__init__()
        self.fw = module_fw
        self.bw = module_bw
        # expose hidden size of concatenated output
        inner = getattr(module_fw, "hidden_size", None)
        self.hidden_size = 2 * inner if inner is not None else None

    def forward(self, x: Tensor):
        # fw on x; bw on reversed x, then reverse the output back
        out_fw, *rest_fw = _as_tuple(self.fw(x))
        x_rev = _reverse_time(x)
        out_bw, *rest_bw = _as_tuple(self.bw(x_rev))
        out_bw = _reverse_time(out_bw)
        # concat along last dim (hidden)
        out = _ConcatLastDim.apply(out_fw, out_bw)
        return out


def _as_tuple(v):
    if isinstance(v, tuple):
        return v
    return (v,)


def _reverse_time(x: Tensor) -> Tensor:
    # reverse along T axis (axis=1 for (B, T, D))
    from nanograd.ops import Getitem

    B, T, D = x.shape
    idx = np.arange(T - 1, -1, -1, dtype=np.int64)
    # numpy-style advanced indexing on axis 1: x[:, idx, :]
    return Getitem.apply(x, idx=(slice(None), idx, slice(None)))


class _ConcatLastDim(Function):
    """Concatenate two tensors along the last dim."""

    def forward(self, a, b):
        self.split = a.shape[-1]
        return np.concatenate([a, b], axis=-1)

    def backward(self, g):
        return g[..., : self.split], g[..., self.split :]


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
