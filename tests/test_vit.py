import numpy as np
import pytest

from nanograd.models.vit import PatchEmbed, ViT
from nanograd.tensor import Tensor


def test_patch_embed_shape():
    pe = PatchEmbed(image_size=28, patch_size=4, in_channels=1, embed_dim=32, seed=0)
    x = Tensor(np.random.randn(2, 1, 28, 28).astype(np.float32))
    out = pe(x)
    assert out.shape == (2, 49, 32)


def test_vit_forward_shape():
    m = ViT(image_size=28, patch_size=7, embed_dim=16, depth=2, num_heads=2, seed=0)
    x = Tensor(np.random.randn(3, 1, 28, 28).astype(np.float32))
    y = m(x)
    assert y.shape == (3, 10)


def test_vit_backward_grad_flows():
    m = ViT(image_size=16, patch_size=4, embed_dim=16, depth=2, num_heads=2, seed=0)
    x = Tensor(np.random.randn(2, 1, 16, 16).astype(np.float32))
    y = m(x)
    loss = (y * y).sum()
    loss.backward()
    # every parameter should receive some gradient
    for name, p in [("cls", m.cls_token), ("pos", m.pos_enc.weight), ("head_w", m.head.weight)]:
        assert p.grad is not None, f"{name} grad is None"
        assert np.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"
        assert np.abs(p.grad).sum() > 0, f"{name} grad is zero"


def test_vit_overfits_tiny_batch():
    """ViT should be able to memorize 16 samples — sanity check training path."""
    from nanograd import nn, optim
    from nanograd.nn import functional as F

    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 1, 16, 16)).astype(np.float32)
    y = rng.integers(0, 4, size=16)

    m = ViT(image_size=16, patch_size=4, num_classes=4, embed_dim=32, depth=2, num_heads=4, seed=0)
    opt = optim.Adam(m.parameters(), lr=3e-3)
    final_loss = None
    for _ in range(100):
        logits = m(Tensor(X))
        loss = F.cross_entropy(logits, Tensor(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()
    acc = (m(Tensor(X)).data.argmax(-1) == y).mean()
    assert final_loss < 0.1, f"ViT failed to overfit: loss={final_loss:.4f}"
    assert acc == 1.0, f"ViT failed to memorize: acc={acc:.4f}"
