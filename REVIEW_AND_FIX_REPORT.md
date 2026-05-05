# Review and Fix Report

## Changes
- Implemented differentiable `topk` values with backward scatter into the original tensor positions.
- Added `MatMul.backward` support for NumPy-style `1D@1D`, `1D@2D`, and `2D@1D` cases.
- Added regression tests for top-k gradients and 1D matmul gradients.

## Verification
- `uv run pytest tests/test_r5_ops.py tests/test_bugfixes.py -q` passed.
- Worker also ran full `uv run pytest -q`, which passed.
- `git diff --check` passed.

## Remaining
- Tensor dtype preservation was not changed because altering global dtype defaults is higher risk.
