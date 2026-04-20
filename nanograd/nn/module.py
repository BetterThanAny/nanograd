from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, Optional

import numpy as np

from nanograd.tensor import Tensor


class Parameter(Tensor):
    """Marker subclass of Tensor — auto-registered as a parameter by Module.__setattr__."""

    def __init__(self, data, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)


class Module:
    """Base class. Automatic parameter/submodule registration via __setattr__."""

    def __init__(self) -> None:
        self.__dict__["_parameters"]: OrderedDict[str, Parameter] = OrderedDict()
        self.__dict__["_modules"]: OrderedDict[str, "Module"] = OrderedDict()
        self.__dict__["_buffers"]: OrderedDict[str, np.ndarray] = OrderedDict()
        self.__dict__["training"]: bool = True

    # --- registration ---
    def __setattr__(self, name: str, value):
        # clear any prior registration under this name so reassignment doesn't leak
        if name in self._parameters and not isinstance(value, Parameter):
            del self._parameters[name]
        if name in self._modules and not isinstance(value, Module):
            del self._modules[name]
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, value: np.ndarray) -> None:
        self._buffers[name] = value
        object.__setattr__(self, name, value)

    # --- iteration ---
    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            yield (f"{prefix}{name}", p)
        for mname, mod in self._modules.items():
            yield from mod.named_parameters(prefix=f"{prefix}{mname}.")

    def modules(self) -> Iterator["Module"]:
        yield self
        for m in self._modules.values():
            yield from m.modules()

    # --- mode ---
    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # --- dispatch ---
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    # --- convenience ---
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def num_params(self) -> int:
        return sum(int(np.prod(p.shape)) for p in self.parameters())

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, np.ndarray]]:
        for name, b in self._buffers.items():
            yield (f"{prefix}{name}", b)
        for mname, mod in self._modules.items():
            yield from mod.named_buffers(prefix=f"{prefix}{mname}.")

    def state_dict(self) -> dict:
        out = {}
        for name, p in self.named_parameters():
            out[name] = p.data.copy()
        for name, b in self.named_buffers():
            out[name] = b.copy()
        return out

    def load_state_dict(self, state: dict) -> None:
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        all_keys = set(params) | set(buffers)
        missing = all_keys - set(state)
        extra = set(state) - all_keys
        if missing or extra:
            raise KeyError(f"state_dict mismatch: missing={missing} extra={extra}")
        for name, p in params.items():
            data = state[name]
            if data.shape != p.data.shape:
                raise ValueError(f"shape mismatch for {name}: {data.shape} vs {p.data.shape}")
            p.data[...] = data
        for name, b in buffers.items():
            data = state[name]
            if data.shape != b.shape:
                raise ValueError(f"shape mismatch for {name}: {data.shape} vs {b.shape}")
            b[...] = data
