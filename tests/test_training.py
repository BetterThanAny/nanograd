import numpy as np
import pytest

from nanograd import nn
from nanograd.training import EarlyStopping, MetricTracker, ModelCheckpoint
from nanograd.utils import load


def test_earlystopping_min_mode():
    es = EarlyStopping(patience=3, mode="min")
    assert not es.step(1.0)   # best
    assert not es.step(0.9)   # better, reset
    assert not es.step(0.95)  # worse, counter=1
    assert not es.step(1.0)   # worse, counter=2
    assert es.step(1.5)       # worse, counter=3 → stop


def test_earlystopping_max_mode():
    es = EarlyStopping(patience=2, mode="max")
    assert not es.step(0.5)
    assert not es.step(0.6)   # better
    assert not es.step(0.55)  # worse, counter=1
    assert es.step(0.5)       # worse, counter=2 → stop


def test_earlystopping_min_delta():
    es = EarlyStopping(patience=1, mode="min", min_delta=0.1)
    es.step(1.0)
    # improvement smaller than delta counts as no improvement
    assert es.step(0.95)  # not < 1.0 - 0.1 = 0.9 → counter=1 → stop


def test_modelcheckpoint_saves_best(tmp_path):
    m = nn.Linear(3, 2)
    cp = ModelCheckpoint(tmp_path / "best.npz", mode="min")
    assert cp.step(1.0, m)   # first — saves
    assert not cp.step(1.5, m)  # worse — no save
    # modify parameters
    for p in m.parameters():
        p.data[:] = 99.0
    assert cp.step(0.5, m)   # better — saves new
    # reload into fresh module; should have 99.0 values
    m2 = nn.Linear(3, 2)
    load(m2, tmp_path / "best.npz")
    for p in m2.parameters():
        assert np.allclose(p.data, 99.0)


def test_metric_tracker():
    mt = MetricTracker()
    mt.update("loss", 1.0, n=4)
    mt.update("loss", 2.0, n=6)
    # weighted avg: (1*4 + 2*6) / 10 = 1.6
    assert np.isclose(mt.avg("loss"), 1.6)
    mt.update("acc", 0.8, n=10)
    assert np.isclose(mt.avg("acc"), 0.8)
    assert set(mt.summary().keys()) == {"loss", "acc"}


def test_metric_tracker_reset():
    mt = MetricTracker()
    mt.update("x", 1.0)
    mt.reset()
    assert mt.avg("x") == 0.0
    assert mt.summary() == {}
