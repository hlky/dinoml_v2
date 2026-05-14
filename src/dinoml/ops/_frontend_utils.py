from __future__ import annotations

import builtins
from typing import Any


def infer_shape_spec(shape_specs: list[list[Any]], out_shape: list[int]) -> list[Any]:
    if not shape_specs:
        return list(out_shape)
    result: list[Any] = []
    max_rank = builtins.max(len(shape) for shape in shape_specs)
    aligned = [[1] * (max_rank - len(shape)) + list(shape) for shape in shape_specs]
    for dims in zip(*aligned):
        chosen = dims[0]
        for dim in dims[1:]:
            if _dim_is_one(chosen):
                chosen = dim
            elif _dim_is_one(dim):
                continue
            elif dim == chosen:
                continue
            else:
                return list(out_shape)
        result.append(chosen)
    return result


def _dim_is_one(dim: Any) -> bool:
    return isinstance(dim, int) and int(dim) == 1
