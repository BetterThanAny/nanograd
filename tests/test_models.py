"""Tests for ResNet variants."""
import numpy as np
import pytest

from nanograd import Tensor
from nanograd.models import BasicBlock, ResNet, resnet18, resnet_cifar


def test_basic_block_identity_shape():
    block = BasicBlock(16, 16, stride=1)
    x = Tensor(np.random.default_rng(0).standard_normal((1, 16, 8, 8)).astype(np.float32))
    y = block(x)
    assert y.shape == (1, 16, 8, 8)


def test_basic_block_downsample_shape():
    block = BasicBlock(16, 32, stride=2)
    x = Tensor(np.random.default_rng(0).standard_normal((1, 16, 8, 8)).astype(np.float32))
    y = block(x)
    assert y.shape == (1, 32, 4, 4)
    assert block.downsample is not None


def test_resnet_cifar_shape():
    model = resnet_cifar(num_blocks_per_stage=1, num_classes=10)
    x = Tensor(np.random.default_rng(0).standard_normal((2, 3, 32, 32)).astype(np.float32))
    y = model(x)
    assert y.shape == (2, 10)


def test_resnet18_shape():
    # tiny imagenet input to keep fast
    model = resnet18(num_classes=1000)
    x = Tensor(np.random.default_rng(0).standard_normal((1, 3, 64, 64)).astype(np.float32))
    y = model(x)
    assert y.shape == (1, 1000)


def test_resnet_cifar_backprop():
    """Single backward pass end-to-end to confirm the graph works."""
    from nanograd.nn import functional as F

    model = resnet_cifar(num_blocks_per_stage=1, num_classes=5)
    x = Tensor(np.random.default_rng(0).standard_normal((2, 3, 32, 32)).astype(np.float32))
    y = Tensor(np.array([0, 1], dtype=np.int64))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    # every parameter should have a grad
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert not np.any(np.isnan(p.grad)), f"NaN grad on {name}"


def test_resnet_cifar_param_count():
    # n=3 is ResNet-20 (~270k)
    model = resnet_cifar(num_blocks_per_stage=3)
    n = model.num_params()
    assert 200_000 < n < 500_000, f"got {n}"


def test_resnet_cifar_small_param_count():
    model = resnet_cifar(num_blocks_per_stage=1)
    # n=1 (~77k)
    n = model.num_params()
    assert 50_000 < n < 150_000, f"got {n}"
