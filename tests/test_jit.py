import numpy as np
import pytest

from nanograd import Tensor
from nanograd.jit import fused
from nanograd.utils import gradcheck


@pytest.fixture
def rng():
    return np.random.default_rng(31)


def test_fused_matches_eager_unary(rng):
    x = rng.standard_normal((10, 10)).astype(np.float32)
    a = Tensor(x)
    y_fused = fused(a, [("mul", 2.0), "relu", "exp", ("add", -1.0)]).data
    # eager equivalent
    y_eager = np.exp(np.maximum(x * 2.0, 0)) - 1.0
    assert np.allclose(y_fused, y_eager, atol=1e-5)


def test_fused_matches_composed_forward(rng):
    x = rng.standard_normal((50,)).astype(np.float32)
    a = Tensor(x)
    y_fused = fused(a, ["tanh", ("mul", 3.0), ("pow", 2.0)]).data
    y_ref = np.power(np.tanh(x) * 3.0, 2.0)
    assert np.allclose(y_fused, y_ref, atol=1e-5)


def test_fused_gradcheck_relu_chain(rng):
    # avoid zeros for relu
    x = Tensor((rng.uniform(0.2, 1.0, (10,)) * rng.choice([-1, 1], (10,))).astype(np.float32), requires_grad=True)
    gradcheck(lambda x: fused(x, [("mul", 2.0), "relu", ("add", 0.5)]).sum(), [x])


def test_fused_gradcheck_smooth(rng):
    x = Tensor(rng.uniform(-0.5, 0.5, (5,)).astype(np.float32), requires_grad=True)
    gradcheck(lambda x: fused(x, ["tanh", ("mul", 2.0), "exp"]).sum(), [x], atol=1e-2, rtol=1e-2)


def test_fused_empty_ops(rng):
    x = Tensor(rng.standard_normal((3,)).astype(np.float32))
    y = fused(x, [])
    assert np.array_equal(y.data, x.data)


def test_fused_gradcheck_pow(rng):
    x = Tensor(rng.uniform(0.5, 2.0, (4,)).astype(np.float32), requires_grad=True)
    gradcheck(lambda x: fused(x, [("pow", 3.0), "log"]).sum(), [x])


def test_fused_benchmark_not_slower_than_eager():
    """Fused chain should not regress vs eager on large arrays.

    Perf tests are noisy on laptops / CI, so we compare medians across multiple
    trial blocks with a generous upper bound. Strict speedup claims belong in
    ``benchmarks/``, not the unit test suite.
    """
    import time
    import statistics

    # scale inputs small so final exp(...) doesn't overflow
    x = np.random.default_rng(0).standard_normal((500, 500)).astype(np.float32) * 0.1
    a = Tensor(x)

    def eager():
        b = a * 2.0
        c = b + 1.0
        d = c * c
        e = d.abs()
        return e.exp()

    def jit():
        return fused(a, [("mul", 2.0), ("add", 1.0), ("pow", 2.0), "abs", "exp"])

    for _ in range(10):
        eager(); jit()

    def time_block(fn, iters):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return time.perf_counter() - t0

    trials, iters = 5, 30
    eager_ts = [time_block(eager, iters) for _ in range(trials)]
    jit_ts = [time_block(jit, iters) for _ in range(trials)]
    t_eager = statistics.median(eager_ts)
    t_jit = statistics.median(jit_ts)
    print(f"\n  eager: {t_eager*1000/iters:.2f} ms/iter  jit: {t_jit*1000/iters:.2f} ms/iter")
    assert t_jit <= t_eager * 1.5, (
        f"fused ({t_jit*1000/iters:.2f} ms/iter) slower than eager "
        f"({t_eager*1000/iters:.2f} ms/iter) beyond noise tolerance"
    )
