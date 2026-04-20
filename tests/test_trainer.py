"""Tests for the Trainer orchestration class."""
import numpy as np
import pytest

from nanograd import Tensor, nn, optim
from nanograd.data import DataLoader, TensorDataset
from nanograd.nn import functional as F
from nanograd.training import EarlyStopping, ModelCheckpoint, Trainer


def _make_regression_setup(n=128):
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 1)).astype(np.float32)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = X @ W + 0.1 * rng.standard_normal((n, 1)).astype(np.float32)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=16, shuffle=True, seed=0)
    model = nn.Linear(4, 1, seed=0)
    opt = optim.Adam(model.parameters(), lr=0.05)

    def step(batch):
        X, y = batch
        pred = model(Tensor(X))
        return F.mse_loss(pred, Tensor(y))

    return model, opt, step, loader, ds


def test_trainer_reduces_loss():
    model, opt, step, loader, _ = _make_regression_setup()
    trainer = Trainer(model, opt, step)
    history = trainer.fit(loader, epochs=5, verbose=False)
    assert len(history["train_loss"]) == 5
    assert history["train_loss"][-1] < history["train_loss"][0] / 2


def test_trainer_with_eval():
    model, opt, step, loader, _ = _make_regression_setup()

    def eval_fn(val_loader):
        total, n = 0.0, 0
        for X, y in val_loader:
            pred = model(Tensor(X))
            loss = F.mse_loss(pred, Tensor(y))
            total += loss.item() * len(X)
            n += len(X)
        return {"loss": total / n}

    trainer = Trainer(model, opt, step, eval_fn=eval_fn)
    history = trainer.fit(loader, epochs=2, val_loader=loader, verbose=False)
    assert len(history["val_loss"]) == 2


def test_trainer_with_early_stopping():
    model, opt, step, loader, _ = _make_regression_setup()

    def eval_fn(val_loader):
        total, n = 0.0, 0
        for X, y in val_loader:
            pred = model(Tensor(X))
            total += F.mse_loss(pred, Tensor(y)).item() * len(X)
            n += len(X)
        return {"loss": total / n}

    es = EarlyStopping(patience=2, mode="min")
    trainer = Trainer(model, opt, step, eval_fn=eval_fn, callbacks=[es])
    # fit for many epochs; ES should trigger before 50
    history = trainer.fit(loader, epochs=50, val_loader=loader, verbose=False)
    assert len(history["train_loss"]) < 50  # stopped early


def test_trainer_with_grad_clip():
    model, opt, step, loader, _ = _make_regression_setup()
    trainer = Trainer(model, opt, step, grad_clip=1.0)
    trainer.fit(loader, epochs=2, verbose=False)
    # if grad_clip works, training completes without NaN
    for p in model.parameters():
        assert np.all(np.isfinite(p.data))


def test_trainer_with_checkpoint(tmp_path):
    model, opt, step, loader, _ = _make_regression_setup()

    def eval_fn(val_loader):
        total, n = 0.0, 0
        for X, y in val_loader:
            pred = model(Tensor(X))
            total += F.mse_loss(pred, Tensor(y)).item() * len(X)
            n += len(X)
        return {"loss": total / n}

    cp = ModelCheckpoint(tmp_path / "best.npz", mode="min")
    trainer = Trainer(model, opt, step, eval_fn=eval_fn, callbacks=[cp])
    trainer.fit(loader, epochs=3, val_loader=loader, verbose=False)
    assert (tmp_path / "best.npz").exists()
    assert cp.best is not None
