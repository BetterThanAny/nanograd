# nanograd

A from-scratch deep learning framework in pure Python + NumPy. Built to learn
the internals: autograd, tensor ops, layers, optimizers, data, and a tiny
kernel fuser.

## What's inside

| Module | Contents |
|--------|----------|
| `nanograd.tensor` | `Tensor` class, dynamic autograd, broadcasting-aware backward |
| `nanograd.ops` | Elementwise, reductions, `matmul`, shape ops, indexing |
| `nanograd.nn` | `Module`, `Linear`, `Sequential`, `Dropout`, `LayerNorm`, `Embedding`, `Conv2d`/`Max/AvgPool2d`, `BatchNorm2d`, `RNN`/`LSTM`/`GRU`, `MultiHeadAttention`, `TransformerBlock` |
| `nanograd.nn.functional` | Stable activations (ReLU/Sigmoid/Tanh/GELU/Softmax/LogSoftmax) and losses (MSE/BCE/BCEWithLogits/CrossEntropy/L1/Huber) |
| `nanograd.optim` | `SGD` (+momentum/nesterov), `Adam`, `AdamW`, `Adagrad`, `RMSProp`, LR schedulers |
| `nanograd.data` | `Dataset`, `DataLoader`, `MNIST`, `CIFAR10` loaders |
| `nanograd.utils` | `gradcheck`, DOT graph viz, param summary, op profiler, checkpoint save/load |
| `nanograd.jit` | Elementwise op fusion — chain multiple ops into a single buffer |

## Quickstart

```python
from nanograd import Tensor, nn, optim
from nanograd.nn import functional as F

model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
opt = optim.Adam(model.parameters(), lr=1e-3)

for X, y in loader:
    logits = model(Tensor(X))
    loss = F.cross_entropy(logits, Tensor(y))
    opt.zero_grad()
    loss.backward()
    opt.step()
```

## Verification metrics

All milestones are test-verified. Non-slow tests: **138 passing in <1s**.

| Task | Metric | Achieved |
|------|--------|----------|
| XOR | loss → 0 | 7e-6 |
| MNIST MLP (5 ep, Adam) | test accuracy | **97.39%** |
| MNIST CNN (2 ep, Adam) | test accuracy | **98.25%** |
| CIFAR-10 CNN (1 ep, 5k subset) | test accuracy | **41.96%** (random 10%) |
| Char-level LSTM LM | memorizes tiny corpus | loss < 0.05, generates correctly |
| Numerical gradient check | max abs diff | < 1e-3 across all ops |
| Elementwise fusion | 500×500 chain, 5 ops | **12.7× vs eager** |

## Throughput (laptop CPU, numpy 2.4)

```
MLP (b=64, 784)            params= 109,386    21.6 steps/s  (  46.4 ms/step)
CNN (b=32, 1x28)           params= 105,866    14.6 steps/s  (  68.6 ms/step)
LSTM (b=32, T=16)          params=  25,738   114.4 steps/s  (   8.7 ms/step)
Transformer (b=8,T=16)     params=  13,034   513.7 steps/s  (   1.9 ms/step)
```

## Running the tests

```bash
uv run pytest -q              # fast unit tests (~1s)
uv run pytest -m slow         # integration: MNIST MLP reaches 95%
```

## Running the examples

```bash
uv run python examples/xor.py
uv run python examples/mnist_mlp.py
uv run python examples/mnist_cnn.py
uv run python examples/cifar_cnn.py 5000 1   # subset of 5000, 1 epoch
uv run python benchmarks/throughput.py
```

## Non-goals

- GPU / CUDA
- PyTorch API compatibility
- Production-grade performance

## Design notes

- **Autograd engine**: `Tensor.backward()` does a topological sort from the
  output, calling each `Function.backward` once. Gradients are accumulated;
  broadcast axes are reduced automatically via `_unbroadcast`.
- **Numerical stability**: softmax subtracts max; log-softmax uses logsumexp;
  BCE-with-logits uses the stable `max(x,0) - x*t + log(1+exp(-|x|))`.
- **Conv2d**: im2col via explicit loop over kernel positions (vectorizes over
  N/C). `col2im` uses accumulation to handle overlapping windows.
- **Fusion**: naive "chain of elementwise ops on the same buffer via numpy's
  `out=` parameter." Not a real JIT — but eliminates (n-1) allocations for
  n-op chains and gives ~12× speedup on 500×500 arrays.
