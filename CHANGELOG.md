# Changelog

## Round 7 — ViT / U-Net / DQN / EMA / op-surface expansion

### New models
- `nanograd.models.ViT` — patch embed (Conv2d kernel=patch, stride=patch) + CLS token + learned positional encoding + N TransformerBlocks. MNIST: **95.9%** test acc on 10k-sample subset with a 139k-param / 4-block config (6 epochs, cosine LR).
- `nanograd.models.UNet` — encoder/bottleneck/decoder with skip concat via `ConvTranspose2d(stride=2)` and `_cat` along the channel axis. MNIST autoencoder: **MSE 0.0048** on held-out subset (29k params, 2 epochs).

### New example
- `examples/dqn_gridworld.py` — Q-net + target net (copied every 50 gradient steps) + uniform replay buffer (cap 5000) + epsilon-greedy (1.0 → 0.05). Reuses the gridworld env from `reinforce_gridworld.py`. Greedy return **0.930** on the optimal 8-step path.

### New training utility
- `training.EMA` — shadow-copy EMA of all parameters (and optionally buffers) with `.update()` after `opt.step()`, `.apply_to()` for permanent copy, and `.swap_into()` as a context manager for temporary eval-time swap. Wired into `mnist_vae.py`.

### New ops
- `Tensor.std` / `Tensor.var` (pure compositions over mean/sum/sqrt)
- `flip`, `roll` (forwards via `np.flip`/`np.roll`; backward is the same op with inverted args)
- `gather` — `np.take_along_axis`; backward scatter-adds `g` at the indices with `np.add.at`
- `scatter_add` — returns `base + scatter(src)` along axis; `grad_base = g`, `grad_src = gather(g, index)`
- `F.normalize` — L_p normalize along axis with eps

### Bug fixes (audit)
- **EMA.swap_into ignored buffers** when `include_buffers=True` — only params were swapped/restored, so BatchNorm running stats stayed on the live values inside the context. Fixed and regression-tested.

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
