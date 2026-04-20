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
        if isinstance(value, Parameter):
            self._parameters[name] = value
            object.__setattr__(self, name, value)
        elif isinstance(value, Module):
            self._modules[name] = value
            object.__setattr__(self, name, value)
        else:
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

    def state_dict(self) -> dict:
        out = {}
        for name, p in self.named_parameters():
            out[name] = p.data.copy()
        return out

    def load_state_dict(self, state: dict) -> None:
        params = dict(self.named_parameters())
        missing = set(params) - set(state)
        extra = set(state) - set(params)
        if missing or extra:
            raise KeyError(f"state_dict mismatch: missing={missing} extra={extra}")
        for name, p in params.items():
            data = state[name]
            if data.shape != p.data.shape:
                raise ValueError(f"shape mismatch for {name}: {data.shape} vs {p.data.shape}")
            p.data[...] = data
