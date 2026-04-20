# nanograd — 从零实现的深度学习框架

## 目标

用纯 Python + NumPy 实现一个功能完整、可训练真实模型（MNIST / CIFAR / 小型 Transformer）的深度学习框架，包含：

- 动态计算图与反向传播
- 完整的张量算子 + broadcasting
- nn.Module 体系与常用层（MLP / CNN / RNN / Attention）
- 优化器与学习率调度
- 数据管道
- 计算图可视化与调试工具
- lazy 执行 + kernel fusion 的小型 JIT

## 非目标

- GPU / CUDA 后端（时间允许再说）
- 与 PyTorch API 严格兼容（借鉴但不复刻）
- 生产级性能（以正确性优先，速度 benchmark 只用来对比）

## 里程碑（每个一个 commit，含测试）

## Round 1 — 12 core milestones (ALL DONE)

| # | 主题 | 验收 | 实际结果 |
|---|------|------|---------|
| M1 | Tensor & autograd 核心 | 数值梯度误差 < 1e-5 | ✅ gradcheck pass |
| M2 | 张量算子扩展 | matmul/reduce/shape/broadcast 梯度测试 | ✅ |
| M3 | 激活 & 损失 | ReLU/Sigmoid/Tanh/Softmax + MSE/CE/BCE | ✅ with stability |
| M4 | nn.Module 体系 | Linear/Sequential/Dropout/LayerNorm | ✅ |
| M5 | 优化器 | SGD/Adam/AdamW/RMSProp + 调度器 | ✅ |
| M6 | 数据管道 | Dataset/DataLoader/MNIST | ✅ |
| M7 | XOR + MNIST MLP | loss→0; MNIST≥95% | ✅ **97.39%** |
| M8 | 卷积 & 池化 | MNIST CNN ≥ 98% | ✅ **98.25%** |
| M9 | 可视化 & 调试 | dot 图 / 参数 / profiler | ✅ |
| M10 | RNN / LSTM / Attention | toy seq 收敛 | ✅ |
| M11 | JIT & kernel fusion | elementwise 融合 | ✅ **12.7× speedup** |
| M12 | 真实任务 benchmark | MNIST + CIFAR-10 CNN | ✅ CIFAR **41.96%** |

## Round 2 — bug audit + feature expansion

- Independent audit found 4 critical + 4 important bugs — all fixed, 16 regression tests added.
- Added: `cat`/`stack`/`pad`/`AdaptiveAvgPool2d`/`GroupNorm`/`InstanceNorm2d`
- Added: `ELU`/`SiLU`/`Mish` activations
- Added: image transforms (`Normalize`, `RandomCrop`, `RandomHorizontalFlip`, `Compose`)
- Added: training utils (`EarlyStopping`, `ModelCheckpoint`, `MetricTracker`)
- Added: `BasicBlock` / `ResNet` / `resnet18` / `resnet_cifar`
- Added: `Bidirectional` wrapper for RNNs
- ResNet-CIFAR-8 (78k params, 1 ep, 5k subset, w/ aug) → **32.54%** (3.2× random)

## Round 3 — advanced features

- `ConvTranspose2d` (im2col/col2im via einsum)
- `SinusoidalPositionalEncoding`, `LearnedPositionalEncoding`
- Weight inits (`kaiming_uniform_`, `kaiming_normal_`, `xavier_uniform_`, `xavier_normal_`)
- Gradient clipping (`clip_grad_norm_`, `clip_grad_value_`)
- Examples: MNIST autoencoder (MSE 0.25 → **0.013**), Transformer char LM (generates "lazy dog" corpus)

Additional bugs fixed this round (self-audit):
- Sinusoidal PE shape was wrong for odd `dim`
- `state_dict` didn't include buffers (BN running stats lost across save/load)
- `clip_grad_norm_` missed `abs()` for non-2 norms

## Round 4 — demos

- GAN (DCGAN-style) on MNIST — ConvTranspose + BN + BCE-logits + two-optimizer loop

## 目录结构

```
nanograd/
├── nanograd/
│   ├── __init__.py
│   ├── tensor.py          # Tensor + autograd
│   ├── functional.py      # 算子实现
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── module.py
│   │   ├── linear.py
│   │   ├── conv.py
│   │   ├── norm.py
│   │   ├── activation.py
│   │   ├── rnn.py
│   │   └── attention.py
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── sgd.py
│   │   ├── adam.py
│   │   └── lr_scheduler.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── mnist.py
│   ├── utils/
│   │   ├── viz.py
│   │   └── gradcheck.py
│   └── jit/
│       ├── lazy.py
│       └── fuser.py
├── tests/
│   ├── test_autograd.py
│   ├── test_ops.py
│   ├── test_nn.py
│   ├── test_optim.py
│   └── test_integration.py
├── examples/
│   ├── xor.py
│   ├── mnist_mlp.py
│   ├── mnist_cnn.py
│   └── cifar_cnn.py
├── benchmarks/
│   └── vs_pytorch.py
├── PLAN.md
├── CLAUDE.md
└── pyproject.toml
```

## 测试策略

- **单元测试**：每个算子 + 每层 + 每优化器一份 `test_*`
- **数值梯度检查**：所有带参数的层，`gradcheck` 误差 < 1e-5
- **集成测试**：XOR/MNIST/CIFAR 跑到目标指标
- **CI 命令**：`uv run pytest -q`（核心测试 < 30s）+ 单独的 `uv run pytest -m slow`（集成）

## 已安装工具

| 工具 | 安装命令 | 时间 | 原因 | 卸载命令 |
|------|---------|------|------|---------|
| uv | 已有 | - | Python 项目管理 | - |
| numpy | `uv add numpy` | M1 | 张量底层存储 | `uv remove numpy` |
| pytest | `uv add --dev pytest` | M1 | 测试 | `uv remove --dev pytest` |

（后续工具按需追加）

## 风险与备注

- **纯 Python + NumPy 训练 CNN 会很慢** → MNIST 用小网络 + 少量 epoch 保证能跑完；CIFAR 可能跑不到 SOTA，只保证能 train 不 NaN
- **避免过度设计** → 每个 milestone 只实现本 milestone 需要的东西，下个 milestone 再扩
- **与 PyTorch API 不强制一致** → 便利优先，怪就怪
