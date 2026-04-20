# nanograd architecture

A short tour of the internals, aimed at readers who already know PyTorch-level concepts and want to see how they're implemented without any C++ / CUDA / JIT magic.

## The autograd core (100 lines)

`nanograd.tensor.Tensor` stores:
- `.data` — a `numpy.ndarray` (float32 by default)
- `.grad` — gradient `ndarray` (or `None`)
- `.requires_grad` — bool
- `._ctx` — the `Function` instance that produced this tensor (or `None` if it's a leaf)

Every operation lives in a `Function` subclass:
```python
class Mul(Function):
    def forward(self, a, b): ...          # pure numpy arrays
    def backward(self, g): ...            # returns tuple of upstream grads
```

`Function.apply(*tensors, **kwargs)` is the entry point: it unwraps Tensor → ndarray, calls `forward`, and wraps the result in a new Tensor that remembers the `Function` instance as its `_ctx`.

### Backward pass

`Tensor.backward()` does a topological sort from the output:
1. Walk the graph via `_ctx.parents`, collect nodes in topo order.
2. Reset intermediate-node grads to `None` (so repeat `.backward()` doesn't double-count — see bug fix C3).
3. Seed the output's grad.
4. Walk in reverse, calling `_ctx.backward(grad)` on each node and distributing gradients to parents.
5. For broadcasting, `_unbroadcast` reduces summed axes before adding to parent grads.

### Why this is simple

- Dynamic graph: every forward rebuilds the DAG.
- No "backward functions" registered separately — each `Function` owns both directions.
- Gradients are plain ndarrays, not Tensors: autograd only goes one level deep. Second-order grads would require additional machinery.

## The Module system

`nanograd.nn.Module` uses `__setattr__` introspection: assigning a `Parameter` or another `Module` auto-registers it. `parameters()` walks the tree recursively by name.

Features:
- `state_dict()` / `load_state_dict()` include both parameters and buffers (so `BatchNorm2d`'s `running_mean`/`running_var` round-trip correctly).
- `train()` / `eval()` propagate to all submodules.
- `Module.__setattr__` cleans up stale registrations when you reassign to a non-Parameter/non-Module value (bug fix C4).

## Conv2d via im2col

Naive convolution is 7 nested loops. `im2col` flattens the input's sliding windows into a matrix so conv becomes a single GEMM:

```
im2col(x[N,C,H,W], kh, kw) → cols[N, C, kh, kw, H_out, W_out]
conv output = weight.reshape(C_out, C_in*kh*kw) @ cols.reshape(N, C_in*kh*kw, H_out*W_out)
```

`col2im` is the transpose, needed both for `Conv2d` backward and for `ConvTranspose2d` forward. Overlapping windows (stride < kernel) require accumulation via `+=`.

`MaxPool2d` uses im2col too, but with `-inf` padding so padded positions never win max comparisons (bug fix C1).

## Layer norm family

All family members share one kernel:
```
mean = x.mean(over some axes)
var  = x.var(over same axes)
xhat = (x - mean) / sqrt(var + eps)
y    = xhat * gamma + beta
```
Differ only in which axes are reduced:
- `LayerNorm`: last N axes matching `normalized_shape`
- `BatchNorm2d`: `(N, H, W)` per channel
- `GroupNorm`: `(Cg, H, W)` per `(N, group)`
- `InstanceNorm2d` = `GroupNorm(num_groups=C)`

Backward is the standard chain-rule expansion:
```
dxhat = g * gamma
dx = (1/M) * inv * (M*dxhat - dxhat.sum() - xhat * (dxhat*xhat).sum())
```

## Optimizer

`Optimizer.__init__` dedupes by `id(param)` — critical for weight-tied models (bug fix C2). Each `.step()` walks the params, skips `None` grads, and calls a per-optimizer `_step_param`. State (moments, momentum buffers) is keyed by `id(p)`.

`Adam` vs `AdamW`: only diff is where weight-decay is applied. `Adam` adds `wd*p` to the gradient first (L2-regularization equivalent). `AdamW` decouples: `p *= (1 - lr*wd)` before the moment-based update.

## JIT fusion

`nanograd.jit.FusedChain` is the smallest useful "kernel fusion." It takes a list of elementwise ops and evaluates them on a single scratch buffer, using numpy's `out=` parameter:

```python
buf = np.empty_like(x)
_apply_forward(x, buf, ops[0])
for op in ops[1:]:
    _apply_forward(buf, buf, op)   # in-place; reuses the same memory
```

For chains like `relu ∘ (mul 2) ∘ (add 1) ∘ pow(2) ∘ abs ∘ exp` on a 500×500 array this is ~12× faster than the eager path because we eliminate (n-1) allocations.

Backward recomputes intermediates into fresh buffers (since the forward overwrote them) and composes per-op VJPs.

## Dataset / DataLoader

- `Dataset` is a duck-typed protocol: `__len__` + `__getitem__`.
- `DataLoader` yields batches by slicing indices and calling `dataset[idx]`.
- `default_collate` stacks tuples element-wise.
- Transforms (`Compose`, `RandomCrop`, `RandomHorizontalFlip`, `Normalize`) operate on raw ndarrays inside `SampleTransform`, which wraps a Dataset.

MNIST / CIFAR loaders download + cache; no torchvision dependency.

## Models

`nanograd.models` provides a `ResNet` that composes existing layers. `BasicBlock` is two-conv + BN + residual; `_make_layer` stacks them; `_make_layer` feeds into an `AdaptiveAvgPool2d(1)` + `Linear`. Variants:
- `resnet18`: 64→128→256→512 channels, 7×7 stem, MaxPool, 4 stages.
- `resnet_cifar`: 16→32→64 channels, 3×3 stem, no MaxPool, 3 stages — standard CIFAR layout.

## Why pure numpy?

- Keeps every operation inspectable with `print()`.
- Zero binary dependencies — easy to run anywhere Python runs.
- Slow, but the point was learning how each piece works, not speed.

Typical throughput on a laptop CPU: 15-100 steps/s for small models (see `benchmarks/throughput.py`).

## What's deliberately missing

- GPU / CUDA / BLAS tuning
- PyTorch API compatibility (convenience borrowed, but no promise)
- Double-precision paths
- Distributed / multi-device
- Higher-order gradients (backward-of-backward)
- True JIT with graph rewriting (the "fusion" is a small memory optimization only)

Each missing piece would be 2-5× more code; the current tree is ~7k lines.
