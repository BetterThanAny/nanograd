"""Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d, Flatten.

Implemented via im2col for generality (any stride/padding/dilation=1).
Weight layout: (C_out, C_in, k, k). Input: (N, C, H, W).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

from nanograd.function import Function
from nanograd.nn.module import Module, Parameter
from nanograd.tensor import Tensor


# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


def _conv_out_size(h: int, k: int, stride: int, pad: int) -> int:
    return (h + 2 * pad - k) // stride + 1


def im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int, fill_value: float = 0.0) -> np.ndarray:
    """(N,C,H,W) -> (N, C, kh, kw, H_out, W_out). Padded with ``fill_value``."""
    N, C, H, W = x.shape
    H_out = _conv_out_size(H, kh, stride, pad)
    W_out = _conv_out_size(W, kw, stride, pad)
    if pad:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_values=fill_value)
    else:
        x_pad = x
    cols = np.empty((N, C, kh, kw, H_out, W_out), dtype=x.dtype)
    for i in range(kh):
        i_end = i + stride * H_out
        for j in range(kw):
            j_end = j + stride * W_out
            cols[:, :, i, j, :, :] = x_pad[:, :, i:i_end:stride, j:j_end:stride]
    return cols


def col2im(cols: np.ndarray, x_shape: Tuple[int, int, int, int], stride: int, pad: int) -> np.ndarray:
    """Inverse of im2col with summation on overlaps.

    cols: (N, C, kh, kw, H_out, W_out) -> (N, C, H, W)
    """
    N, C, H, W = x_shape
    _, _, kh, kw, H_out, W_out = cols.shape
    H_pad = H + 2 * pad
    W_pad = W + 2 * pad
    x_pad = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    for i in range(kh):
        i_end = i + stride * H_out
        for j in range(kw):
            j_end = j + stride * W_out
            x_pad[:, :, i:i_end:stride, j:j_end:stride] += cols[:, :, i, j, :, :]
    if pad == 0:
        return x_pad
    return x_pad[:, :, pad:-pad, pad:-pad]


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------


class _Conv2dFn(Function):
    def forward(self, x, w, *, stride, padding):
        N, C_in, H, W = x.shape
        C_out, _, kh, kw = w.shape
        H_out = _conv_out_size(H, kh, stride, padding)
        W_out = _conv_out_size(W, kw, stride, padding)
        cols = im2col(x, kh, kw, stride, padding)                       # (N, C_in, kh, kw, H_out, W_out)
        cols_2d = cols.reshape(N, C_in * kh * kw, H_out * W_out)        # per-batch cols
        w_2d = w.reshape(C_out, C_in * kh * kw)
        out = w_2d @ cols_2d                                            # (N, C_out, H_out*W_out) broadcast
        out = out.reshape(N, C_out, H_out, W_out)
        self.save_for_backward(x.shape, w, cols_2d)
        self.stride, self.padding, self.kh, self.kw = stride, padding, kh, kw
        return out

    def backward(self, grad_out):
        x_shape, w, cols_2d = self.saved
        N, C_in, H, W = x_shape
        C_out, _, kh, kw = w.shape
        H_out, W_out = grad_out.shape[-2], grad_out.shape[-1]
        g = grad_out.reshape(N, C_out, H_out * W_out)
        # weight grad: sum over batch of g @ cols.T
        w_2d = w.reshape(C_out, -1)
        # dw_2d[o, i] = sum_n sum_j g[n,o,j] * cols[n,i,j]
        dw_2d = np.einsum("noj,nij->oi", g, cols_2d)
        dw = dw_2d.reshape(w.shape)
        # dcols: for each n, w_2d.T @ g[n]
        dcols_2d = np.einsum("oi,noj->nij", w_2d, g)
        dcols = dcols_2d.reshape(N, C_in, kh, kw, H_out, W_out)
        dx = col2im(dcols, x_shape, self.stride, self.padding)
        return dx, dw


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = kh, kw
        self.stride = stride
        self.padding = padding
        rng = np.random.default_rng(seed)
        # Kaiming uniform (fan_in = in_channels * kh * kw)
        fan_in = in_channels * kh * kw
        bound = 1 / math.sqrt(fan_in)
        w = rng.uniform(-bound, bound, size=(out_channels, in_channels, kh, kw)).astype(np.float32)
        self.weight = Parameter(w)
        if bias:
            self.bias = Parameter(rng.uniform(-bound, bound, size=(out_channels,)).astype(np.float32))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = _Conv2dFn.apply(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            # broadcast (C_out,) to (1, C_out, 1, 1)
            out = out + self.bias.reshape(1, self.out_channels, 1, 1)
        return out


# ---------------------------------------------------------------------------
# MaxPool2d
# ---------------------------------------------------------------------------


class _MaxPool2dFn(Function):
    def forward(self, x, *, kh, kw, stride, padding):
        # pad with -inf so padded positions never win max comparisons
        fill = np.finfo(x.dtype).min if np.issubdtype(x.dtype, np.floating) else np.iinfo(x.dtype).min
        cols = im2col(x, kh, kw, stride, padding, fill_value=fill)  # (N, C, kh, kw, H_out, W_out)
        N, C, _, _, H_out, W_out = cols.shape
        cols_flat = cols.reshape(N, C, kh * kw, H_out, W_out)
        out = cols_flat.max(axis=2)
        idx = cols_flat.argmax(axis=2)                     # (N, C, H_out, W_out)
        self.save_for_backward(idx)
        self.x_shape = x.shape
        self.kh, self.kw = kh, kw
        self.stride, self.padding = stride, padding
        return out

    def backward(self, g):
        (idx,) = self.saved
        N, C, H_out, W_out = g.shape
        kh, kw = self.kh, self.kw
        dcols_flat = np.zeros((N, C, kh * kw, H_out, W_out), dtype=g.dtype)
        n_idx, c_idx, h_idx, w_idx = np.ix_(np.arange(N), np.arange(C), np.arange(H_out), np.arange(W_out))
        dcols_flat[n_idx, c_idx, idx, h_idx, w_idx] = g
        dcols = dcols_flat.reshape(N, C, kh, kw, H_out, W_out)
        dx = col2im(dcols, self.x_shape, self.stride, self.padding)
        return (dx,)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding: int = 0):
        super().__init__()
        self.kh, self.kw = _pair(kernel_size)
        self.stride = stride if stride is not None else self.kh
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return _MaxPool2dFn.apply(x, kh=self.kh, kw=self.kw, stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# AvgPool2d
# ---------------------------------------------------------------------------


class _AvgPool2dFn(Function):
    def forward(self, x, *, kh, kw, stride, padding):
        cols = im2col(x, kh, kw, stride, padding)
        out = cols.mean(axis=(2, 3))
        self.x_shape = x.shape
        self.kh, self.kw = kh, kw
        self.stride, self.padding = stride, padding
        return out

    def backward(self, g):
        kh, kw = self.kh, self.kw
        # distribute grad equally across k*k
        N, C, H_out, W_out = g.shape
        dcols = np.broadcast_to(
            (g / (kh * kw))[:, :, None, None, :, :],
            (N, C, kh, kw, H_out, W_out),
        ).copy()
        dx = col2im(dcols, self.x_shape, self.stride, self.padding)
        return (dx,)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding: int = 0):
        super().__init__()
        self.kh, self.kw = _pair(kernel_size)
        self.stride = stride if stride is not None else self.kh
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return _AvgPool2dFn.apply(x, kh=self.kh, kw=self.kw, stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# BatchNorm2d
# ---------------------------------------------------------------------------


class _BN2dTrainFn(Function):
    def forward(self, x, gamma, beta, *, eps, running_mean, running_var, momentum):
        # x: (N, C, H, W) -> normalize per C across (N, H, W)
        axes = (0, 2, 3)
        mean = x.mean(axis=axes, keepdims=True)          # (1, C, 1, 1)
        var = x.var(axis=axes, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        xhat = (x - mean) * inv
        g_b = gamma.reshape(1, -1, 1, 1)
        bt = beta.reshape(1, -1, 1, 1)
        out = xhat * g_b + bt
        # update running stats (in-place)
        running_mean *= 1 - momentum
        running_mean += momentum * mean.reshape(-1)
        running_var *= 1 - momentum
        running_var += momentum * var.reshape(-1)
        self.save_for_backward(xhat, gamma, inv, axes)
        return out.astype(x.dtype)

    def backward(self, g):
        xhat, gamma, inv, axes = self.saved
        N = 1
        for a in axes:
            N *= xhat.shape[a]
        g_b = gamma.reshape(1, -1, 1, 1)
        dxhat = g * g_b
        dx = (1.0 / N) * inv * (
            N * dxhat
            - dxhat.sum(axis=axes, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=axes, keepdims=True)
        )
        dgamma = (g * xhat).sum(axis=axes)
        dbeta = g.sum(axis=axes)
        return dx, dgamma, dbeta


class _BN2dEvalFn(Function):
    def forward(self, x, gamma, beta, *, eps, running_mean, running_var):
        mean = running_mean.reshape(1, -1, 1, 1)
        var = running_var.reshape(1, -1, 1, 1)
        inv = 1.0 / np.sqrt(var + eps)
        xhat = (x - mean) * inv
        g_b = gamma.reshape(1, -1, 1, 1)
        bt = beta.reshape(1, -1, 1, 1)
        self.save_for_backward(xhat, g_b, inv)
        return (xhat * g_b + bt).astype(x.dtype)

    def backward(self, g):
        xhat, g_b, inv = self.saved
        dx = g * g_b * inv
        axes = (0, 2, 3)
        dgamma = (g * xhat).sum(axis=axes)
        dbeta = g.sum(axis=axes)
        return dx, dgamma, dbeta


class _GroupNormFn(Function):
    def forward(self, x, gamma, beta, *, num_groups, eps):
        N, C, H, W = x.shape
        G = num_groups
        assert C % G == 0, f"channels {C} must be divisible by groups {G}"
        Cg = C // G
        xg = x.reshape(N, G, Cg, H, W)
        # normalize over (Cg, H, W) per (N, G)
        axes = (2, 3, 4)
        mean = xg.mean(axis=axes, keepdims=True)
        var = xg.var(axis=axes, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        xhat = (xg - mean) * inv
        xhat = xhat.reshape(N, C, H, W)
        out = xhat * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)
        self.save_for_backward(xhat, gamma, inv, G, Cg, N, C, H, W)
        return out.astype(x.dtype)

    def backward(self, g):
        xhat, gamma, inv, G, Cg, N, C, H, W = self.saved
        g_b = gamma.reshape(1, C, 1, 1)
        dxhat = (g * g_b).reshape(N, G, Cg, H, W)
        xhat_g = xhat.reshape(N, G, Cg, H, W)
        axes = (2, 3, 4)
        M = Cg * H * W
        dx = (1.0 / M) * inv * (
            M * dxhat
            - dxhat.sum(axis=axes, keepdims=True)
            - xhat_g * (dxhat * xhat_g).sum(axis=axes, keepdims=True)
        )
        dx = dx.reshape(N, C, H, W)
        dgamma = (g * xhat).sum(axis=(0, 2, 3))
        dbeta = g.sum(axis=(0, 2, 3))
        return dx, dgamma, dbeta


class GroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = Parameter(np.ones(num_channels, dtype=np.float32))
        self.bias = Parameter(np.zeros(num_channels, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return _GroupNormFn.apply(x, self.weight, self.bias, num_groups=self.num_groups, eps=self.eps)


class InstanceNorm2d(Module):
    """Equivalent to GroupNorm with num_groups = num_channels."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        if affine:
            self.weight = Parameter(np.ones(num_features, dtype=np.float32))
            self.bias = Parameter(np.zeros(num_features, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        C = x.shape[1]
        w = self.weight if self.weight is not None else Tensor(np.ones(C, dtype=np.float32))
        b = self.bias if self.bias is not None else Tensor(np.zeros(C, dtype=np.float32))
        return _GroupNormFn.apply(x, w, b, num_groups=C, eps=self.eps)


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones(num_features, dtype=np.float32))
        self.bias = Parameter(np.zeros(num_features, dtype=np.float32))
        self.register_buffer("running_mean", np.zeros(num_features, dtype=np.float32))
        self.register_buffer("running_var", np.ones(num_features, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return _BN2dTrainFn.apply(
                x,
                self.weight,
                self.bias,
                eps=self.eps,
                running_mean=self.running_mean,
                running_var=self.running_var,
                momentum=self.momentum,
            )
        return _BN2dEvalFn.apply(
            x,
            self.weight,
            self.bias,
            eps=self.eps,
            running_mean=self.running_mean,
            running_var=self.running_var,
        )


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


class Flatten(Module):
    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        tail = 1
        for s in shape[self.start_dim:]:
            tail *= s
        new_shape = shape[: self.start_dim] + (tail,)
        return x.reshape(*new_shape)


class AdaptiveAvgPool2d(Module):
    """Adaptive average pool to a target spatial size.

    For target_size=1 (global average), computed as mean over H and W.
    For other sizes, falls back to computing pool kernel/stride that approximates it.
    """

    def __init__(self, output_size):
        super().__init__()
        self.output_size = _pair(output_size)

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2], x.shape[-1]
        oh, ow = self.output_size
        if oh == 1 and ow == 1:
            # global average pool: mean over H and W
            return x.mean(axis=(-2, -1), keepdims=True)
        # uniform grid: require divisibility
        if H % oh != 0 or W % ow != 0:
            raise ValueError(
                f"AdaptiveAvgPool2d: input {H}x{W} not divisible by output {oh}x{ow}"
            )
        kh, kw = H // oh, W // ow
        return _AvgPool2dFn.apply(x, kh=kh, kw=kw, stride=kh, padding=0)
