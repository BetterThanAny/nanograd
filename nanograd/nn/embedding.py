from __future__ import annotations

from typing import Optional

import numpy as np

from nanograd.function import Function
from nanograd.nn.module import Module, Parameter
from nanograd.tensor import Tensor


class _EmbeddingFn(Function):
    def forward(self, w, *, idx):
        self.save_for_backward(idx, w.shape)
        return w[idx]

    def backward(self, g):
        idx, shape = self.saved
        dw = np.zeros(shape, dtype=g.dtype)
        np.add.at(dw, idx, g)
        return (dw,)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, seed: Optional[int] = None):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.weight = Parameter(rng.standard_normal((num_embeddings, embedding_dim)).astype(np.float32))

    def forward(self, idx):
        if isinstance(idx, Tensor):
            arr = idx.data.astype(np.int64)
        else:
            arr = np.asarray(idx, dtype=np.int64)
        return _EmbeddingFn.apply(self.weight, idx=arr)
