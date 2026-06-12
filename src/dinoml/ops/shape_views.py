from __future__ import annotations

from math import prod
from typing import Any, Mapping, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.shapes import (
    is_dynamic_shape,
    normalize_symbolic_int,
    shape_numel,
    symbolic_int_expr,
    symbolic_int_interval,
)


def identity(x: Any) -> Tensor:
    tensor = as_tensor(x)
    return tensor.builder.emit_view("identity", tensor, tensor.shape, tensor.shape_spec)


def reshape(x: Any, shape: Sequence[Any]) -> Tensor:
    tensor = as_tensor(x)
    out_shape, out_shape_spec = _resolve_reshape_shape(tensor.shape_spec, shape)
    return tensor.builder.emit_view("reshape", tensor, out_shape, out_shape_spec)


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


def unflatten(x: Any, dim: int, sizes: Sequence[Any]) -> Tensor:
    tensor = as_tensor(x)
    axis = _normalize_axis(dim, len(tensor.shape))
    replacement_shape, replacement_shape_spec = _resolve_unflatten_shape(tensor.shape_spec[axis], sizes)
    out_shape_spec = [
        *tensor.shape_spec[:axis],
        *replacement_shape_spec,
        *tensor.shape_spec[axis + 1 :],
    ]
    out_shape = [
        *tensor.shape[:axis],
        *replacement_shape,
        *tensor.shape[axis + 1 :],
    ]
    return tensor.builder.emit_view("reshape", tensor, out_shape, out_shape_spec)


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


def _resolve_reshape_shape(
    input_shape_spec: Sequence[int | Mapping[str, Any]],
    requested_shape: Sequence[Any],
) -> tuple[list[int], list[int | dict[str, Any]]]:
    if not requested_shape:
        raise NotImplementedError("scalar shape-view tensors are not supported yet")
    output_shape_spec: list[int | dict[str, Any]] = []
    inferred_axes = []
    for idx, dim in enumerate(requested_shape):
        if isinstance(dim, int) and dim == -1:
            inferred_axes.append(idx)
            output_shape_spec.append(-1)
            continue
        normalized = normalize_symbolic_int(dim, f"reshape dim {idx}")
        if isinstance(normalized, int) and normalized <= 0:
            raise ValueError(f"reshape dimensions must be positive or -1, got {dim}")
        if not isinstance(normalized, int) and symbolic_int_interval(normalized)[0] <= 0:
            raise ValueError(f"reshape symbolic dimension at axis {idx} must be positive, got {dim!r}")
        output_shape_spec.append(normalized if isinstance(normalized, int) else dict(normalized))
    if len(inferred_axes) > 1:
        raise ValueError("reshape can infer at most one -1 dimension")
    input_numel = _symbolic_numel(input_shape_spec)
    if inferred_axes:
        known_numel = _symbolic_numel([dim for dim in output_shape_spec if dim != -1])
        inferred = symbolic_int_expr("div", input_numel, known_numel)
        interval = symbolic_int_interval(inferred)
        if interval[0] <= 0:
            raise ValueError(f"reshape inferred a non-positive dimension {interval[0]} for {list(requested_shape)}")
        output_shape_spec[inferred_axes[0]] = inferred if isinstance(inferred, int) else dict(inferred)
    input_max_shape = [_symbolic_dim_max(dim) for dim in input_shape_spec]
    out_shape = [_symbolic_dim_max(dim) for dim in output_shape_spec]
    if shape_numel(out_shape) != shape_numel(input_max_shape):
        raise ValueError(f"reshape must preserve element count: {input_max_shape} -> {out_shape}")
    return out_shape, output_shape_spec


def _resolve_unflatten_shape(
    input_dim_spec: int | Mapping[str, Any],
    requested_sizes: Sequence[Any],
) -> tuple[list[int], list[int | dict[str, Any]]]:
    if isinstance(requested_sizes, (str, bytes, bytearray)) or not isinstance(requested_sizes, Sequence):
        raise TypeError(f"unflatten sizes must be a sequence, got {type(requested_sizes).__name__}")
    if not requested_sizes:
        raise ValueError("unflatten sizes must be non-empty")
    output_shape_spec: list[int | dict[str, Any]] = []
    inferred_axes: list[int] = []
    for idx, dim in enumerate(requested_sizes):
        if isinstance(dim, int) and dim == -1:
            inferred_axes.append(idx)
            output_shape_spec.append(-1)
            continue
        normalized = normalize_symbolic_int(dim, f"unflatten size {idx}")
        if isinstance(normalized, int) and normalized <= 0:
            raise ValueError(f"unflatten sizes must be positive or -1, got {dim}")
        if not isinstance(normalized, int) and symbolic_int_interval(normalized)[0] <= 0:
            raise ValueError(f"unflatten symbolic size at axis {idx} must be positive, got {dim!r}")
        output_shape_spec.append(normalized if isinstance(normalized, int) else dict(normalized))
    if len(inferred_axes) > 1:
        raise ValueError("unflatten can infer at most one -1 dimension")
    input_dim = normalize_symbolic_int(input_dim_spec, "unflatten input dim")
    if inferred_axes:
        known_numel = _symbolic_numel([dim for dim in output_shape_spec if dim != -1])
        inferred = symbolic_int_expr("div", input_dim, known_numel)
        interval = symbolic_int_interval(inferred)
        if interval[0] <= 0:
            raise ValueError(f"unflatten inferred a non-positive dimension {interval[0]} for {list(requested_sizes)}")
        output_shape_spec[inferred_axes[0]] = inferred if isinstance(inferred, int) else dict(inferred)
    input_dim_max = _symbolic_dim_max(input_dim)
    out_shape = [_symbolic_dim_max(dim) for dim in output_shape_spec]
    if shape_numel(out_shape) != input_dim_max:
        raise ValueError(f"unflatten target sizes must preserve dimension size: {input_dim_max} -> {out_shape}")
    return out_shape, output_shape_spec


def _symbolic_numel(shape_spec: Sequence[int | Mapping[str, Any]]) -> int | dict[str, Any]:
    numel: int | dict[str, Any] = 1
    for dim in shape_spec:
        numel = symbolic_int_expr("mul", numel, normalize_symbolic_int(dim))
    return numel


def _symbolic_dim_max(dim: int | Mapping[str, Any]) -> int:
    if isinstance(dim, int):
        return int(dim)
    return int(symbolic_int_interval(dim)[1])


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
