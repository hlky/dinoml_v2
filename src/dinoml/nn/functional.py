from __future__ import annotations

from typing import Any

from dinoml import ops
from dinoml.frontend import Tensor


def one_hot(input: Any, num_classes: int) -> Tensor:
    return ops.one_hot(input, num_classes)


def normalize(input: Any, p: float = 2.0, dim: int = -1, eps: float = 1e-12, out: Any | None = None) -> Tensor:
    return ops.normalize(input, p=p, dim=dim, eps=eps, out=out)


__all__ = ["one_hot", "normalize"]
