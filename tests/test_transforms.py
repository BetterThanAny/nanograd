import numpy as np
import pytest

from nanograd.data import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    SampleTransform,
    TensorDataset,
    ToFloat,
)


def test_normalize_per_channel():
    x = np.array([[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]], dtype=np.float32)
    norm = Normalize([1.0, 2.0], [2.0, 2.0])
    y = norm(x)
    assert np.allclose(y, 0.0)


def test_normalize_single_sample():
    x = np.ones((3, 4, 4), dtype=np.float32)
    norm = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    y = norm(x)
    assert np.allclose(y, 1.0)


def test_random_horizontal_flip_always():
    x = np.arange(12).reshape(1, 3, 4).astype(np.float32)
    flip = RandomHorizontalFlip(p=1.0)
    y = flip(x)
    assert np.allclose(y, x[..., ::-1])


def test_random_horizontal_flip_never():
    x = np.arange(12).reshape(1, 3, 4).astype(np.float32)
    flip = RandomHorizontalFlip(p=0.0)
    y = flip(x)
    assert np.allclose(y, x)


def test_random_crop_shape():
    x = np.ones((3, 8, 8), dtype=np.float32)
    crop = RandomCrop(4, padding=0, seed=0)
    y = crop(x)
    assert y.shape == (3, 4, 4)


def test_random_crop_with_padding_shape():
    x = np.ones((3, 4, 4), dtype=np.float32)
    crop = RandomCrop(4, padding=2, seed=0)
    y = crop(x)
    assert y.shape == (3, 4, 4)


def test_to_float():
    x = np.arange(10, dtype=np.uint8).reshape(1, 1, 10)
    y = ToFloat(scale=1 / 255.0)(x)
    assert y.dtype == np.float32
    assert np.allclose(y[0, 0], np.arange(10) / 255.0)


def test_compose():
    x = np.full((3, 4, 4), 255, dtype=np.uint8)
    tf = Compose([
        ToFloat(),
        Normalize([0.5] * 3, [0.5] * 3),
    ])
    y = tf(x)
    assert np.allclose(y, 1.0)


def test_sample_transform_tuple():
    X = np.zeros((5, 3, 4, 4), dtype=np.float32)
    y = np.arange(5)
    ds = TensorDataset(X, y)

    def tf(img):
        return img + 1

    tds = SampleTransform(ds, tf)
    sample = tds[0]
    assert sample[0].shape == (3, 4, 4)
    assert np.all(sample[0] == 1)
    assert sample[1] == 0
    assert len(tds) == 5


def test_sample_transform_non_tuple():
    # raw array dataset
    arr = np.zeros((3, 2, 2), dtype=np.float32)

    class D:
        def __len__(self):
            return 3

        def __getitem__(self, i):
            return arr[i]

    tds = SampleTransform(D(), lambda x: x + 1)
    assert np.all(tds[0] == 1)
