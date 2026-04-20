import numpy as np
import pytest

from nanograd.data import DataLoader, Dataset, TensorDataset, TransformDataset


def test_tensor_dataset_basic():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    ds = TensorDataset(X, y)
    assert len(ds) == 5
    a, b = ds[2]
    assert np.array_equal(a, X[2])
    assert b == y[2]


def test_tensor_dataset_mismatched_lens():
    with pytest.raises(ValueError):
        TensorDataset(np.zeros(5), np.zeros(6))


def test_dataloader_basic():
    X = np.arange(10).reshape(5, 2).astype(np.float32)
    y = np.arange(5)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(dl)
    assert len(batches) == 3  # 2 + 2 + 1
    assert batches[0][0].shape == (2, 2)
    assert batches[-1][0].shape == (1, 2)


def test_dataloader_drop_last():
    ds = TensorDataset(np.arange(10))
    dl = DataLoader(ds, batch_size=3, drop_last=True)
    batches = list(dl)
    assert len(batches) == 3
    assert all(b[0].shape == (3,) for b in batches)


def test_dataloader_shuffle_is_reproducible():
    X = np.arange(20)
    ds = TensorDataset(X)
    dl1 = DataLoader(ds, batch_size=5, shuffle=True, seed=0)
    dl2 = DataLoader(ds, batch_size=5, shuffle=True, seed=0)
    a = np.concatenate([b[0] for b in dl1])
    b = np.concatenate([b[0] for b in dl2])
    assert np.array_equal(a, b)


def test_dataloader_shuffle_covers_all():
    X = np.arange(20)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=5, shuffle=True, seed=0)
    seen = np.concatenate([b[0] for b in dl])
    assert sorted(seen.tolist()) == list(range(20))


def test_dataloader_len():
    ds = TensorDataset(np.zeros(10))
    assert len(DataLoader(ds, batch_size=3)) == 4
    assert len(DataLoader(ds, batch_size=3, drop_last=True)) == 3


def test_transform_dataset():
    X = np.arange(5)
    ds = TensorDataset(X)

    def tf(sample):
        (x,) = sample
        return x * 2

    tds = TransformDataset(ds, tf)
    assert len(tds) == 5
    assert tds[3] == 6


def test_dataloader_custom_collate():
    ds = TensorDataset(np.arange(6).reshape(3, 2))

    def collate(samples):
        return np.sum([s[0] for s in samples])

    dl = DataLoader(ds, batch_size=2, collate_fn=collate)
    batches = list(dl)
    # samples[0]=(array([0,1]),), samples[1]=(array([2,3]),) → sum=6
    assert batches[0] == 6


def test_dataloader_multiarray():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    dl = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)
    b = next(iter(dl))
    assert b[0].shape == (4, 2)
    assert b[1].shape == (4,)
    assert b[1][0] == 0 and b[1][-1] == 3
