"""Parameter counter + simple per-op timing profiler."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable

import numpy as np

from nanograd.function import Function
from nanograd.nn.module import Module


# ---------------------------------------------------------------------------
# parameter summary
# ---------------------------------------------------------------------------


def param_summary(module: Module) -> str:
    lines = [f"{'name':40s} {'shape':25s} {'params':>10s}"]
    lines.append("-" * 80)
    total = 0
    for name, p in module.named_parameters():
        shape_s = "×".join(str(s) for s in p.shape)
        n = int(np.prod(p.shape))
        total += n
        lines.append(f"{name:40s} {shape_s:25s} {n:>10,}")
    lines.append("-" * 80)
    lines.append(f"{'total':>66s} {total:>10,}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# op timing profiler (monkey-patches Function.apply)
# ---------------------------------------------------------------------------


class _TimingState:
    times: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    active: bool = False


@contextmanager
def profile():
    """Context manager that records forward-pass time per Function class."""
    _TimingState.times.clear()
    _TimingState.counts.clear()
    _TimingState.active = True
    # grab classmethod descriptor (not bound method) so restore works
    orig_classmethod = Function.__dict__["apply"]
    orig_fn = orig_classmethod.__func__

    @classmethod
    def timed_apply(cls, *tensors, **kwargs):
        t0 = time.perf_counter()
        out = orig_fn(cls, *tensors, **kwargs)
        if _TimingState.active:
            dt = time.perf_counter() - t0
            _TimingState.times[cls.__name__] += dt
            _TimingState.counts[cls.__name__] += 1
        return out

    Function.apply = timed_apply
    try:
        yield _TimingState
    finally:
        _TimingState.active = False
        Function.apply = orig_classmethod


def summary(state: _TimingState) -> str:
    rows = [(name, state.times[name], state.counts[name]) for name in state.times]
    rows.sort(key=lambda r: r[1], reverse=True)
    lines = [f"{'op':25s} {'count':>8s} {'total_ms':>12s} {'avg_us':>12s}"]
    lines.append("-" * 60)
    for name, t, n in rows:
        lines.append(f"{name:25s} {n:>8d} {t*1000:>12.2f} {t/n*1e6:>12.1f}")
    total = sum(state.times.values())
    lines.append("-" * 60)
    lines.append(f"{'total':25s} {'':>8s} {total*1000:>12.2f}")
    return "\n".join(lines)
