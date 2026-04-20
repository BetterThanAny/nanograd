# Changelog

## Round 2 — bug fixes & extensions

### Bug fixes (audit round)
- **C1** `MaxPool2d` padding now uses `-inf` instead of zero (was silently beating negative inputs).
- **C2** `Optimizer` dedupes tied parameters by `id` — shared weights step once per `.step()`.
- **C3** `Tensor.backward()` resets intermediate (non-leaf) grads at the start of each call. Repeated backward on the same graph no longer double-counts.
- **C4** `Module.__setattr__` removes stale `_parameters` / `_modules` entries when a registered name is reassigned to a non-Parameter / non-Module value.
- **I1** `RNN` / `LSTM` / `GRU` handle `T=0` (empty sequence) without crashing.
- **I2** `ops.cat([])` and `ops.stack([])` raise clean `ValueError` instead of bubbling from numpy.
- **I3** `Tensor.__getitem__` accepts a `Tensor` as index (auto-unwraps and coerces float → int64).

Regression tests in `tests/test_bugfixes.py`.

### New features
- `GroupNorm`, `InstanceNorm2d`
- `ELU`, `SiLU` / `Swish`, `Mish` activations
- `AdaptiveAvgPool2d` (global + uniform-grid)
- `cat`, `stack`, `pad` ops
- Image transforms: `Compose`, `Normalize`, `RandomHorizontalFlip`, `RandomCrop`, `ToFloat`, `SampleTransform`
- Training utilities: `EarlyStopping`, `ModelCheckpoint`, `MetricTracker`
- `nanograd.models`: `BasicBlock`, `ResNet`, `resnet18`, `resnet_cifar`

## Round 1 — original 12 milestones

| # | Milestone | Commit |
|---|-----------|--------|
| M1 | Tensor & autograd | elementwise ops + broadcasting + gradcheck |
| M2 | Tensor ops | matmul / sum / mean / max / reshape / transpose / indexing |
| M3 | Activations & losses | ReLU / Sigmoid / Tanh / GELU / Softmax / LogSoftmax + MSE / BCE / CE |
| M4 | nn.Module system | auto param registration, Linear / Sequential / Dropout / LayerNorm |
| M5 | Optimizers | SGD + Adam + AdamW + RMSProp + schedulers |
| M6 | Data pipeline | Dataset / DataLoader / MNIST loader |
| M7 | Training validation | XOR + MNIST MLP 97.39% |
| M8 | Conv & pool | Conv2d / MaxPool2d / BatchNorm2d (im2col) |
| M9 | Debug & viz | DOT graph + param summary + op profiler |
| M10 | Sequence models | RNN / LSTM / Attention / TransformerBlock |
| M11 | JIT & fusion | 12.7× speedup on elementwise chains |
| M12 | CIFAR-10 + benchmarks | CIFAR CNN 41.96%, throughput numbers |

Plus extras (Embedding, Adagrad, checkpoint, L1 / Huber, char-level LSTM LM, GRU).
