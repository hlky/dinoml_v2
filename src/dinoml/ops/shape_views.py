from __future__ import annotations

from math import prod
from typing import Any, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.shapes import is_dynamic_shape, shape_numel


def identity(x: Any) -> Tensor:
    tensor = as_tensor(x)
    return tensor.builder.emit_view("identity", tensor, tensor.shape, tensor.shape_spec)


def reshape(x: Any, shape: Sequence[int]) -> Tensor:
    tensor = as_tensor(x)
    if is_dynamic_shape(tensor.shape_spec):
        raise NotImplementedError("reshape currently supports only static input shapes")
    out_shape = _resolve_reshape_shape(tensor.shape, shape)
    return tensor.builder.emit_view("reshape", tensor, out_shape, out_shape)


def flatten(x: Any, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    tensor = as_tensor(x)
    rank = len(tensor.shape)
    start = _normalize_axis(start_dim, rank)
    end = _normalize_axis(end_dim, rank)
    if start > end:
        raise ValueError(f"flatten start_dim must be <= end_dim, got {start_dim} and {end_dim}")
    if is_dynamic_shape(tensor.shape_spec[start : end + 1]):
        raise NotImplementedError("flatten currently supports only static dimensions in the flattened range")
    out_shape_spec = [
        *tensor.shape_spec[:start],
        int(prod(int(dim) for dim in tensor.shape[start : end + 1])),
        *tensor.shape_spec[end + 1 :],
    ]
    out_shape = [
        *tensor.shape[:start],
        int(prod(int(dim) for dim in tensor.shape[start : end + 1])),
        *tensor.shape[end + 1 :],
    ]
    return tensor.builder.emit_view("flatten", tensor, out_shape, out_shape_spec)


def squeeze(x: Any, dim: int | Sequence[int] | None = None) -> Tensor:
    tensor = as_tensor(x)
    rank = len(tensor.shape)
    axes = _squeeze_axes(tensor.shape_spec, dim)
    out_shape_spec = [shape_dim for axis, shape_dim in enumerate(tensor.shape_spec) if axis not in axes]
    out_shape = [shape_dim for axis, shape_dim in enumerate(tensor.shape) if axis not in axes]
    if not out_shape:
        raise NotImplementedError("scalar shape-view tensors are not supported yet")
    return tensor.builder.emit_view("squeeze", tensor, out_shape, out_shape_spec)


def unsqueeze(x: Any, dim: int) -> Tensor:
    tensor = as_tensor(x)
    axis = _normalize_insert_axis(dim, len(tensor.shape))
    out_shape_spec = [*tensor.shape_spec[:axis], 1, *tensor.shape_spec[axis:]]
    out_shape = [*tensor.shape[:axis], 1, *tensor.shape[axis:]]
    return tensor.builder.emit_view("unsqueeze", tensor, out_shape, out_shape_spec)


def _resolve_reshape_shape(input_shape: Sequence[int], requested_shape: Sequence[int]) -> list[int]:
    if not requested_shape:
        raise NotImplementedError("scalar shape-view tensors are not supported yet")
    out_shape = [int(dim) for dim in requested_shape]
    inferred_axes = [idx for idx, dim in enumerate(out_shape) if dim == -1]
    if len(inferred_axes) > 1:
        raise ValueError("reshape can infer at most one -1 dimension")
    for dim in out_shape:
        if dim == -1:
            continue
        if dim <= 0:
            raise ValueError(f"reshape dimensions must be positive or -1, got {dim}")
    input_numel = shape_numel(input_shape)
    if inferred_axes:
        known_numel = int(prod(dim for dim in out_shape if dim != -1))
        if input_numel % known_numel != 0:
            raise ValueError(f"reshape cannot infer dimension for {list(requested_shape)} from input shape {list(input_shape)}")
        out_shape[inferred_axes[0]] = input_numel // known_numel
    if shape_numel(out_shape) != input_numel:
        raise ValueError(f"reshape must preserve element count: {list(input_shape)} -> {out_shape}")
    return out_shape


def _squeeze_axes(shape_spec: Sequence[Any], dim: int | Sequence[int] | None) -> set[int]:
    rank = len(shape_spec)
    if dim is None:
        return {axis for axis, shape_dim in enumerate(shape_spec) if _dim_is_known_one(shape_dim)}
    dims = [dim] if isinstance(dim, int) else list(dim)
    axes = {_normalize_axis(axis, rank) for axis in dims}
    for axis in axes:
        if not _dim_is_known_one(shape_spec[axis]):
            raise ValueError(f"Cannot squeeze axis {axis} with dimension {shape_spec[axis]!r}; expected a known size 1")
    return axes


def _dim_is_known_one(dim: Any) -> bool:
    if isinstance(dim, int):
        return int(dim) == 1
    return int(dim["min"]) == 1 and int(dim["max"]) == 1


def _normalize_axis(axis: int, rank: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise IndexError(f"axis {axis} is out of range for rank {rank}")
    return normalized


def _normalize_insert_axis(axis: int, rank: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += rank + 1
    if normalized < 0 or normalized > rank:
        raise IndexError(f"axis {axis} is out of range for unsqueeze rank {rank}")
    return normalized
