import numpy as np

from nanograd import Tensor, optim
from nanograd.models.unet import DoubleConv, UNet
from nanograd.nn import functional as F


def test_unet_shape_preserved():
    m = UNet(in_channels=1, out_channels=1, base=8, seed=0)
    x = Tensor(np.random.randn(2, 1, 32, 32).astype(np.float32))
    y = m(x)
    assert y.shape == (2, 1, 32, 32)


def test_unet_multichannel():
    m = UNet(in_channels=3, out_channels=2, base=4, seed=0)
    x = Tensor(np.random.randn(1, 3, 16, 16).astype(np.float32))
    y = m(x)
    assert y.shape == (1, 2, 16, 16)


def test_unet_grads_flow():
    m = UNet(in_channels=1, out_channels=1, base=8, seed=0)
    x = Tensor(np.random.randn(1, 1, 16, 16).astype(np.float32))
    y = m(x)
    loss = (y * y).sum()
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and np.isfinite(p.grad).all() and np.abs(p.grad).sum() > 0, name


def test_double_conv_shape():
    d = DoubleConv(4, 8, seed=0)
    x = Tensor(np.random.randn(2, 4, 16, 16).astype(np.float32))
    assert d(x).shape == (2, 8, 16, 16)


def test_unet_overfits_single_image():
    """Sanity check: U-Net should be able to reconstruct a single image."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1, 1, 16, 16)).astype(np.float32)
    m = UNet(1, 1, base=8, seed=0)
    opt = optim.Adam(m.parameters(), lr=5e-3)
    final = None
    for _ in range(100):
        y = m(Tensor(X))
        loss = F.mse_loss(y, Tensor(X))
        opt.zero_grad()
        loss.backward()
        opt.step()
        final = loss.item()
    assert final < 0.05, f"U-Net failed to reconstruct: {final:.4f}"
