from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dinoml import ops
from dinoml.frontend import Tensor


def one_hot(input: Any, num_classes: int) -> Tensor:
    return ops.one_hot(input, num_classes)


def pad(input: Any, pad: Any, mode: str = "constant", value: Any | None = None) -> Tensor:
    if mode != "constant":
        raise NotImplementedError(f"pad currently supports only mode='constant', got {mode!r}")
    return ops.pad(input, pad, value=0.0 if value is None else value)


def softmax(input: Any, dim: int | None = None, _stacklevel: int = 3, dtype: Any | None = None) -> Tensor:
    del _stacklevel
    if dtype is not None:
        raise NotImplementedError("softmax currently does not support dtype=")
    return ops.softmax(input, dim=-1 if dim is None else dim)


def silu(input: Any, inplace: bool = False) -> Tensor:
    if inplace:
        raise NotImplementedError("silu currently does not support inplace=True")
    return ops.silu(input)


def layer_norm(
    input: Any,
    normalized_shape: int | Sequence[int],
    weight: Any | None = None,
    bias: Any | None = None,
    eps: float = 1e-5,
) -> Tensor:
    if weight is None or bias is None:
        raise NotImplementedError("layer_norm currently requires weight and bias tensors")
    if isinstance(normalized_shape, int) and not isinstance(normalized_shape, bool):
        normalized_shape = [int(normalized_shape)]
    return ops.layer_norm(input, weight, bias, eps=eps, normalized_shape=normalized_shape)


def normalize(input: Any, p: float = 2.0, dim: int = -1, eps: float = 1e-12, out: Any | None = None) -> Tensor:
    return ops.normalize(input, p=p, dim=dim, eps=eps, out=out)


__all__ = ["one_hot", "pad", "softmax", "silu", "layer_norm", "normalize"]
