from __future__ import annotations

from math import isfinite, prod
from typing import Any, Mapping, Sequence

from dinoml.frontend import Parameter, Tensor, as_tensor
from dinoml.ops.elementwise import tanh
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def
from dinoml.ops.shape_views import reshape


COLLECTION_DTYPES = ("float16", "float32", "bfloat16", "bool")
GATHER_INDEX_DTYPES = ("int64", "int32")
SPECIALIZED_PERMUTE_DIMS: dict[str, tuple[int, ...]] = {
    "permute021": (0, 2, 1),
    "permute0213": (0, 2, 1, 3),
    "permute102": (1, 0, 2),
    "permute210": (2, 1, 0),
}


def infer_concatenate_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_concatenate_shape_with_attrs(input_shapes, {"dim": 0})


def infer_stack_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_stack_shape_with_attrs(input_shapes, {"dim": 0})


def infer_flip_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_flip_shape_with_attrs(input_shapes, {"dims": (0,)})


def infer_repeat_interleave_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_repeat_interleave_shape_with_attrs(input_shapes, {"repeats": 1, "dim": 0})


def infer_permute_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("permute expects one tensor input")
    return infer_permute_shape_with_attrs(input_shapes, {"dims": tuple(range(len(input_shapes[0])))})


def infer_dynamic_slice_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("dynamic_slice expects one tensor input")
    rank = len(input_shapes[0])
    return infer_dynamic_slice_shape_with_attrs(
        input_shapes,
        {"start_indices": (0,) * rank, "slice_sizes": tuple(input_shapes[0])},
    )


def infer_index_select_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("index_select expects one tensor input")
    return infer_index_select_shape_with_attrs(input_shapes, {"dim": 0, "indices": (0,)})


def infer_gather_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 2:
        raise ValueError("gather expects two tensor inputs")
    return infer_gather_shape_with_attrs(input_shapes, {"dim": 0})


def infer_batch_gather_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 2:
        raise ValueError("batch_gather expects two tensor inputs")
    return infer_batch_gather_shape_with_attrs(input_shapes, {})


def infer_slice_scatter_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 2:
        raise ValueError("slice_scatter expects two tensor inputs")
    rank = len(input_shapes[0])
    return infer_slice_scatter_shape_with_attrs(input_shapes, {"start_indices": (0,) * rank})


def infer_pad_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("pad expects one tensor input")
    return infer_pad_shape_with_attrs(input_shapes, {"pad": (0, 0)})


def infer_concatenate_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if not input_shapes:
        raise ValueError("concatenate expects a non-empty sequence of tensors")
    dim = normalize_concatenate_dim(attrs.get("dim", 0), len(input_shapes[0]))
    return resolve_concatenate_shape(input_shapes, dim)


def infer_stack_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if not input_shapes:
        raise ValueError("stack expects a non-empty sequence of tensors")
    dim = normalize_stack_dim(attrs.get("dim", 0), len(input_shapes[0]))
    return resolve_stack_shape(input_shapes, dim)


def infer_flip_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("flip expects one tensor input")
    return resolve_flip_shape(input_shapes[0], attrs.get("dims"))


def infer_repeat_interleave_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("repeat_interleave expects one tensor input")
    dim = normalize_repeat_interleave_dim(attrs.get("dim"), len(input_shapes[0]))
    repeats = normalize_repeat_interleave_repeats(attrs.get("repeats"))
    return resolve_repeat_interleave_shape(input_shapes[0], repeats, dim)


def infer_permute_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("permute expects one tensor input")
    return resolve_permute_shape(input_shapes[0], attrs.get("dims"))


def infer_specialized_permute_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
    *,
    op_name: str,
) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError(f"{op_name} expects one tensor input")
    fixed_dims = SPECIALIZED_PERMUTE_DIMS[op_name]
    rank = len(input_shapes[0])
    if rank != len(fixed_dims):
        raise ValueError(f"{op_name} expects rank-{len(fixed_dims)} input, got rank {rank}")
    attrs_dims = attrs.get("dims")
    if attrs_dims is not None:
        normalized_dims = normalize_permute_dims(attrs_dims, rank)
        if tuple(normalized_dims) != fixed_dims:
            raise ValueError(f"{op_name} uses fixed dims {list(fixed_dims)}, got {list(normalized_dims)}")
    return resolve_permute_shape(input_shapes[0], fixed_dims)


def infer_dynamic_slice_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("dynamic_slice expects one tensor input")
    return resolve_dynamic_slice_shape(input_shapes[0], attrs.get("start_indices"), attrs.get("slice_sizes"))


def infer_index_select_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("index_select expects one tensor input")
    return resolve_index_select_shape(input_shapes[0], attrs.get("dim", 0), attrs.get("indices"))


def infer_gather_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 2:
        raise ValueError("gather expects two tensor inputs")
    return resolve_gather_shape(input_shapes[0], input_shapes[1], attrs.get("dim", 0))


def infer_batch_gather_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    del attrs
    if len(input_shapes) != 2:
        raise ValueError("batch_gather expects two tensor inputs")
    return resolve_batch_gather_shape(input_shapes[0], input_shapes[1])


def infer_slice_scatter_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 2:
        raise ValueError("slice_scatter expects two tensor inputs")
    return resolve_slice_scatter_shape(input_shapes[0], input_shapes[1], attrs.get("start_indices"))


def infer_pad_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("pad expects one tensor input")
    return resolve_pad_shape(input_shapes[0], attrs.get("pad"))


def normalize_concatenate_dim(dim: Any, rank: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"concatenate dim must be an integer, got {dim!r}")
    if rank <= 0:
        raise ValueError("concatenate inputs must have rank >= 1")
    normalized = int(dim)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"concatenate dim {dim} is out of range for rank {rank}")
    return normalized


def normalize_stack_dim(dim: Any, rank: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"stack dim must be an integer, got {dim!r}")
    if rank < 0:
        raise ValueError("stack rank must be non-negative")
    normalized = int(dim)
    if normalized < 0:
        normalized += rank + 1
    if normalized < 0 or normalized > rank:
        raise ValueError(f"stack dim {dim} is out of range for rank {rank}")
    return normalized


def normalize_flip_dims(dims: Any, rank: int) -> list[int]:
    if isinstance(dims, int) and not isinstance(dims, bool):
        requested = [int(dims)]
    elif isinstance(dims, Sequence) and not isinstance(dims, (str, bytes, bytearray)):
        requested = []
        for dim in dims:
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise ValueError(f"flip dims must be integers, got {dims!r}")
            requested.append(int(dim))
    else:
        raise ValueError(f"flip dims must be an integer or non-empty sequence of integers, got {dims!r}")
    if not requested:
        raise ValueError("flip dims must be non-empty")
    if rank <= 0:
        raise ValueError("flip input must have rank >= 1")
    normalized_dims: list[int] = []
    seen: set[int] = set()
    for dim in requested:
        normalized = dim + rank if dim < 0 else dim
        if normalized < 0 or normalized >= rank:
            raise ValueError(f"flip dim {dim} is out of range for rank {rank}")
        if normalized in seen:
            raise ValueError(f"flip dims must not contain duplicates: {requested!r}")
        seen.add(normalized)
        normalized_dims.append(normalized)
    return normalized_dims


def normalize_repeat_interleave_dim(dim: Any, rank: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"repeat_interleave dim must be an integer, got {dim!r}")
    if rank <= 0:
        raise ValueError("repeat_interleave input must have rank >= 1")
    normalized = int(dim)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"repeat_interleave dim {dim} is out of range for rank {rank}")
    return normalized


def normalize_repeat_interleave_repeats(repeats: Any) -> int:
    if not isinstance(repeats, int) or isinstance(repeats, bool):
        raise ValueError(f"repeat_interleave repeats must be a positive integer scalar, got {repeats!r}")
    normalized = int(repeats)
    if normalized <= 0:
        raise ValueError(f"repeat_interleave repeats must be positive, got {repeats!r}")
    return normalized


def normalize_permute_dims(dims: Any, rank: int) -> list[int]:
    if not isinstance(dims, Sequence) or isinstance(dims, (str, bytes, bytearray)):
        raise ValueError(f"permute dims must be a sequence of integers, got {dims!r}")
    requested: list[int] = []
    for dim in dims:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"permute dims must be integers, got {dims!r}")
        requested.append(int(dim))
    if rank <= 0:
        raise ValueError("permute input must have rank >= 1")
    if len(requested) != rank:
        raise ValueError(f"permute dims length {len(requested)} must match rank {rank}")
    normalized_dims: list[int] = []
    seen: set[int] = set()
    for dim in requested:
        normalized = dim + rank if dim < 0 else dim
        if normalized < 0 or normalized >= rank:
            raise ValueError(f"permute dim {dim} is out of range for rank {rank}")
        if normalized in seen:
            raise ValueError(f"permute dims must not contain duplicates: {requested!r}")
        seen.add(normalized)
        normalized_dims.append(normalized)
    return normalized_dims


def normalize_transpose_dims(dim0: Any, dim1: Any, rank: int) -> tuple[int, int]:
    if not isinstance(dim0, int) or isinstance(dim0, bool):
        raise ValueError(f"transpose dim0 must be an integer, got {dim0!r}")
    if not isinstance(dim1, int) or isinstance(dim1, bool):
        raise ValueError(f"transpose dim1 must be an integer, got {dim1!r}")
    if rank <= 0:
        raise ValueError("transpose input must have rank >= 1")
    normalized0 = int(dim0)
    normalized1 = int(dim1)
    if normalized0 < 0:
        normalized0 += rank
    if normalized1 < 0:
        normalized1 += rank
    if normalized0 < 0 or normalized0 >= rank:
        raise ValueError(f"transpose dim0 {dim0} is out of range for rank {rank}")
    if normalized1 < 0 or normalized1 >= rank:
        raise ValueError(f"transpose dim1 {dim1} is out of range for rank {rank}")
    return normalized0, normalized1


def normalize_dynamic_slice_attrs(
    start_indices: Any,
    slice_sizes: Any,
    input_shape: Sequence[int],
) -> tuple[list[int], list[int]]:
    rank = len(input_shape)
    starts = _normalize_dynamic_slice_int_sequence(start_indices, rank, "start_indices")
    sizes = _normalize_dynamic_slice_int_sequence(slice_sizes, rank, "slice_sizes")
    for axis, (start, size, extent) in enumerate(zip(starts, sizes, input_shape)):
        if start < 0:
            raise ValueError(f"dynamic_slice start_indices[{axis}] must be non-negative, got {start}")
        if size <= 0:
            raise ValueError(f"dynamic_slice slice_sizes[{axis}] must be positive, got {size}")
        if start + size > int(extent):
            raise ValueError(
                f"dynamic_slice axis {axis} start {start} plus size {size} exceeds input dim {int(extent)}"
            )
    return starts, sizes


def normalize_index_select_dim(dim: Any, rank: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"index_select dim must be an integer, got {dim!r}")
    if rank <= 0:
        raise ValueError("index_select input must have rank >= 1")
    normalized = int(dim)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"index_select dim {dim} is out of range for rank {rank}")
    return normalized


def normalize_index_select_indices(indices: Any, dim_extent: int) -> list[int]:
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes, bytearray)):
        raise ValueError(f"index_select indices must be a non-empty sequence of integers, got {indices!r}")
    normalized: list[int] = []
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"index_select indices must contain only non-bool integers, got {indices!r}")
        value = int(index)
        if value < 0 or value >= int(dim_extent):
            raise ValueError(f"index_select index {value} is out of bounds for dim size {int(dim_extent)}")
        normalized.append(value)
    if not normalized:
        raise ValueError("index_select indices must be a non-empty sequence of integers")
    return normalized


def normalize_index_select_attrs(dim: Any, indices: Any, input_shape: Sequence[int]) -> tuple[int, list[int]]:
    normalized_dim = normalize_index_select_dim(dim, len(input_shape))
    normalized_indices = normalize_index_select_indices(indices, int(input_shape[normalized_dim]))
    return normalized_dim, normalized_indices


def normalize_gather_dim(dim: Any, rank: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"gather dim must be an integer, got {dim!r}")
    if rank <= 0:
        raise ValueError("gather input must have rank >= 1")
    normalized = int(dim)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"gather dim {dim} is out of range for rank {rank}")
    return normalized


def normalize_gather_attrs(dim: Any, input_shape: Sequence[int], index_shape: Sequence[int]) -> int:
    rank = len(input_shape)
    normalized_dim = normalize_gather_dim(dim, rank)
    if len(index_shape) != rank:
        raise ValueError(f"gather index rank {len(index_shape)} must match input rank {rank}")
    for axis, (index_extent, input_extent) in enumerate(zip(index_shape, input_shape)):
        if axis == normalized_dim:
            continue
        if int(index_extent) > int(input_extent):
            raise ValueError(
                f"gather index dim {axis} size {int(index_extent)} exceeds input dim {int(input_extent)}"
            )
    return normalized_dim


def normalize_batch_gather_attrs(input_shape: Sequence[int], index_shape: Sequence[int]) -> None:
    if len(input_shape) < 2:
        raise ValueError(f"batch_gather input rank {len(input_shape)} must be at least 2")
    if len(index_shape) != 2:
        raise ValueError(f"batch_gather indices rank {len(index_shape)} must be 2")
    if int(input_shape[0]) != int(index_shape[0]):
        raise ValueError(
            f"batch_gather batch size mismatch: input batch {int(input_shape[0])}, "
            f"indices batch {int(index_shape[0])}"
        )


def normalize_slice_scatter_attrs(
    start_indices: Any,
    input_shape: Sequence[int],
    update_shape: Sequence[int],
) -> list[int]:
    rank = len(input_shape)
    starts = _normalize_slice_scatter_int_sequence(start_indices, rank, "start_indices")
    if len(update_shape) != rank:
        raise ValueError(f"slice_scatter update rank {len(update_shape)} must match input rank {rank}")
    for axis, (start, size, extent) in enumerate(zip(starts, update_shape, input_shape)):
        if start < 0:
            raise ValueError(f"slice_scatter start_indices[{axis}] must be non-negative, got {start}")
        if start + int(size) > int(extent):
            raise ValueError(
                f"slice_scatter axis {axis} start {start} plus update dim {int(size)} exceeds input dim {int(extent)}"
            )
    return starts


def normalize_pad_widths(pad: Any, rank: int) -> tuple[list[int], list[int]]:
    if rank <= 0:
        raise ValueError("pad input must have rank >= 1")
    if not isinstance(pad, Sequence) or isinstance(pad, (str, bytes, bytearray)):
        raise ValueError(f"pad must be a non-empty even-length sequence of non-negative integers, got {pad!r}")
    values: list[int] = []
    for value in pad:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"pad must contain only non-bool integers, got {pad!r}")
        if int(value) < 0:
            raise ValueError(f"pad values must be non-negative, got {pad!r}")
        values.append(int(value))
    if not values:
        raise ValueError("pad must be non-empty")
    if len(values) % 2 != 0:
        raise ValueError(f"pad length must be even, got {len(values)}")
    if len(values) // 2 > rank:
        raise ValueError(f"pad length {len(values)} cannot describe more dimensions than input rank {rank}")
    left = [0] * rank
    right = [0] * rank
    for pair_index in range(len(values) // 2):
        axis = rank - 1 - pair_index
        left[axis] = values[2 * pair_index]
        right[axis] = values[2 * pair_index + 1]
    return left, right


def normalize_split_dim(dim: Any, rank: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"split dim must be an integer, got {dim!r}")
    if rank <= 0:
        raise ValueError("split input must have rank >= 1")
    normalized = int(dim)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"split dim {dim} is out of range for rank {rank}")
    return normalized


def normalize_split_sections(split_size_or_sections: Any, dim_extent: int) -> list[int]:
    if isinstance(split_size_or_sections, int) and not isinstance(split_size_or_sections, bool):
        split_size = int(split_size_or_sections)
        if split_size <= 0:
            raise ValueError(f"split size must be positive, got {split_size_or_sections!r}")
        sections = []
        remaining = int(dim_extent)
        while remaining > 0:
            size = min(split_size, remaining)
            sections.append(size)
            remaining -= size
        return sections
    if isinstance(split_size_or_sections, Sequence) and not isinstance(
        split_size_or_sections,
        (str, bytes, bytearray),
    ):
        sections = []
        for section in split_size_or_sections:
            if not isinstance(section, int) or isinstance(section, bool):
                raise ValueError(f"split sections must contain only positive integers, got {split_size_or_sections!r}")
            if int(section) <= 0:
                raise ValueError(f"split sections must be positive, got {split_size_or_sections!r}")
            sections.append(int(section))
        if not sections:
            raise ValueError("split sections must be non-empty")
        if sum(sections) != int(dim_extent):
            raise ValueError(f"split sections must sum to dim size {int(dim_extent)}, got {sum(sections)}")
        return sections
    raise ValueError(
        "split_size_or_sections must be a positive integer or non-empty sequence of positive integers, "
        f"got {split_size_or_sections!r}"
    )


def normalize_chunk_count(chunks: Any) -> int:
    if not isinstance(chunks, int) or isinstance(chunks, bool):
        raise ValueError(f"chunk chunks must be a positive integer, got {chunks!r}")
    normalized = int(chunks)
    if normalized <= 0:
        raise ValueError(f"chunk chunks must be positive, got {chunks!r}")
    return normalized


def chunk_sections(dim_extent: int, chunks: Any) -> list[int]:
    normalized_chunks = normalize_chunk_count(chunks)
    extent = int(dim_extent)
    chunk_size = (extent + normalized_chunks - 1) // normalized_chunks
    return normalize_split_sections(chunk_size, extent)


def _normalize_dynamic_slice_int_sequence(values: Any, rank: int, name: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"dynamic_slice {name} must be a sequence of integers, got {values!r}")
    normalized: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"dynamic_slice {name} must contain only integers, got {values!r}")
        normalized.append(int(value))
    if len(normalized) != rank:
        raise ValueError(f"dynamic_slice {name} length {len(normalized)} must match rank {rank}")
    return normalized


def _normalize_slice_scatter_int_sequence(values: Any, rank: int, name: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"slice_scatter {name} must be a sequence of integers, got {values!r}")
    normalized: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"slice_scatter {name} must contain only integers, got {values!r}")
        normalized.append(int(value))
    if len(normalized) != rank:
        raise ValueError(f"slice_scatter {name} length {len(normalized)} must match rank {rank}")
    return normalized


def resolve_concatenate_shape(input_shapes: Sequence[Sequence[int]], dim: int) -> list[int]:
    if not input_shapes:
        raise ValueError("concatenate expects a non-empty sequence of tensors")
    base_shape = [int(axis) for axis in input_shapes[0]]
    if not base_shape:
        raise ValueError("concatenate inputs must have rank >= 1")
    dim = normalize_concatenate_dim(dim, len(base_shape))
    output_shape = list(base_shape)
    output_shape[dim] = 0
    for index, shape in enumerate(input_shapes):
        current = [int(axis) for axis in shape]
        if len(current) != len(base_shape):
            raise ValueError(f"concatenate input {index} rank {len(current)} does not match rank {len(base_shape)}")
        for axis, (actual, expected) in enumerate(zip(current, base_shape)):
            if axis == dim:
                continue
            if actual != expected:
                raise ValueError(
                    f"concatenate input {index} axis {axis} has dim {actual}, expected {expected}"
                )
        output_shape[dim] += current[dim]
    return output_shape


def resolve_stack_shape(input_shapes: Sequence[Sequence[int]], dim: int) -> list[int]:
    if not input_shapes:
        raise ValueError("stack expects a non-empty sequence of tensors")
    base_shape = [int(axis) for axis in input_shapes[0]]
    dim = normalize_stack_dim(dim, len(base_shape))
    for index, shape in enumerate(input_shapes):
        current = [int(axis) for axis in shape]
        if current != base_shape:
            raise ValueError(f"stack input {index} shape {current} does not match {base_shape}")
    output_shape = list(base_shape)
    output_shape.insert(dim, len(input_shapes))
    return output_shape


def resolve_flip_shape(input_shape: Sequence[int], dims: Any) -> list[int]:
    output_shape = [int(axis) for axis in input_shape]
    normalize_flip_dims(dims, len(output_shape))
    return output_shape


def resolve_repeat_interleave_shape(input_shape: Sequence[int], repeats: Any, dim: Any) -> list[int]:
    output_shape = [int(axis) for axis in input_shape]
    dim = normalize_repeat_interleave_dim(dim, len(output_shape))
    repeats = normalize_repeat_interleave_repeats(repeats)
    output_shape[dim] *= repeats
    return output_shape


def resolve_permute_shape(input_shape: Sequence[int], dims: Any) -> list[int]:
    normalized_dims = normalize_permute_dims(dims, len(input_shape))
    shape = [int(axis) for axis in input_shape]
    return [shape[dim] for dim in normalized_dims]


def resolve_dynamic_slice_shape(input_shape: Sequence[int], start_indices: Any, slice_sizes: Any) -> list[int]:
    _, normalized_sizes = normalize_dynamic_slice_attrs(start_indices, slice_sizes, input_shape)
    return normalized_sizes


def resolve_index_select_shape(input_shape: Sequence[int], dim: Any, indices: Any) -> list[int]:
    normalized_dim, normalized_indices = normalize_index_select_attrs(dim, indices, input_shape)
    output_shape = [int(axis) for axis in input_shape]
    output_shape[normalized_dim] = len(normalized_indices)
    return output_shape


def resolve_gather_shape(input_shape: Sequence[int], index_shape: Sequence[int], dim: Any) -> list[int]:
    normalize_gather_attrs(dim, input_shape, index_shape)
    return [int(axis) for axis in index_shape]


def resolve_batch_gather_shape(input_shape: Sequence[int], index_shape: Sequence[int]) -> list[int]:
    normalize_batch_gather_attrs(input_shape, index_shape)
    return [int(index_shape[0]), int(index_shape[1]), *[int(axis) for axis in input_shape[2:]]]


def resolve_slice_scatter_shape(
    input_shape: Sequence[int],
    update_shape: Sequence[int],
    start_indices: Any,
) -> list[int]:
    normalize_slice_scatter_attrs(start_indices, input_shape, update_shape)
    return [int(axis) for axis in input_shape]


def resolve_pad_shape(input_shape: Sequence[int], pad: Any) -> list[int]:
    output_shape = [int(axis) for axis in input_shape]
    left, right = normalize_pad_widths(pad, len(output_shape))
    return [dim + left[axis] + right[axis] for axis, dim in enumerate(output_shape)]


@op_def
class Concatenate(OpDef):
    name = "concatenate"
    schema = OpSchema(
        inputs=("x0",),
        attrs=(AttrDef("dim", "int", default=0),),
    )
    infer_shape = infer_concatenate_shape
    infer_shape_with_attrs = infer_concatenate_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_concatenate", library="model", source_template="concatenate_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_concatenate", library="model", source_template="concatenate_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_concatenate", library="model", source_template="concatenate_gpu.j2"),
    }
    frontend = FrontendBinding("concatenate")
    variadic_inputs = True
    description = "Materialize a dense concatenation copy along a static dimension."

    @classmethod
    def forward(cls, inputs: Any, dim: int = 0) -> Tensor:
        return concatenate(inputs, dim)


@op_def
class Stack(OpDef):
    name = "stack"
    schema = OpSchema(
        inputs=("x0",),
        attrs=(AttrDef("dim", "int", default=0),),
    )
    infer_shape = infer_stack_shape
    infer_shape_with_attrs = infer_stack_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_stack", library="model", source_template="stack_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_stack", library="model", source_template="stack_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_stack", library="model", source_template="stack_gpu.j2"),
    }
    frontend = FrontendBinding("stack")
    variadic_inputs = True
    description = "Materialize a dense stack copy by inserting a static dimension."

    @classmethod
    def forward(cls, inputs: Any, dim: int = 0) -> Tensor:
        return stack(inputs, dim)


@op_def
class Flip(OpDef):
    name = "flip"
    schema = OpSchema(
        inputs=("x",),
        attrs=(AttrDef("dims", "ints", required=True),),
    )
    infer_shape = infer_flip_shape
    infer_shape_with_attrs = infer_flip_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_flip", library="model", source_template="flip_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_flip", library="model", source_template="flip_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_flip", library="model", source_template="flip_gpu.j2"),
    }
    frontend = FrontendBinding("flip")
    description = "Materialize a dense copy that reverses one or more static dimensions."

    @classmethod
    def forward(cls, x: Any, dims: Any) -> Tensor:
        return flip(x, dims)


@op_def
class RepeatInterleave(OpDef):
    name = "repeat_interleave"
    schema = OpSchema(
        inputs=("x",),
        attrs=(AttrDef("repeats", "int", required=True), AttrDef("dim", "int", required=True)),
    )
    infer_shape = infer_repeat_interleave_shape
    infer_shape_with_attrs = infer_repeat_interleave_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_repeat_interleave", library="model", source_template="repeat_interleave_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_repeat_interleave", library="model", source_template="repeat_interleave_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_repeat_interleave", library="model", source_template="repeat_interleave_gpu.j2"),
    }
    frontend = FrontendBinding("repeat_interleave")
    description = "Materialize a dense bounded repeat-interleave copy along a static dimension."

    @classmethod
    def forward(cls, x: Any, repeats: Any, dim: Any) -> Tensor:
        return repeat_interleave(x, repeats, dim)


@op_def
class Permute(OpDef):
    name = "permute"
    schema = OpSchema(
        inputs=("x",),
        attrs=(AttrDef("dims", "ints", required=True),),
    )
    infer_shape = infer_permute_shape
    infer_shape_with_attrs = infer_permute_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_permute", library="model", source_template="permute_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_permute", library="model", source_template="permute_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_permute", library="model", source_template="permute_gpu.j2"),
    }
    frontend = FrontendBinding("permute")
    description = "Materialize a dense bounded copy with permuted static dimensions."

    @classmethod
    def forward(cls, x: Any, dims: Any) -> Tensor:
        return permute(x, dims)


class _SpecializedPermute(OpDef):
    infer_shape = infer_permute_shape
    allowed_dtypes = COLLECTION_DTYPES


def _specialized_permute_shape_with_attrs(op_name: str):
    return lambda input_shapes, attrs: infer_specialized_permute_shape_with_attrs(input_shapes, attrs, op_name=op_name)


@op_def
class Permute021(_SpecializedPermute):
    _dims = SPECIALIZED_PERMUTE_DIMS["permute021"]
    name = "permute021"
    schema = OpSchema(inputs=("x",), attrs=(AttrDef("dims", "ints", default=tuple(_dims)),))
    infer_shape_with_attrs = _specialized_permute_shape_with_attrs(name)
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_permute021", library="model", source_template="permute_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_permute021", library="model", source_template="permute_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_permute021", library="model", source_template="permute_gpu.j2"),
    }
    frontend = FrontendBinding(name, default_attrs={"dims": list(_dims)})
    description = "Materialize a dense bounded copy for the fixed [0, 2, 1] permutation on rank-3 static tensors."

    @classmethod
    def forward(cls, x: Any) -> Tensor:
        return permute021(x)


@op_def
class Permute0213(_SpecializedPermute):
    _dims = SPECIALIZED_PERMUTE_DIMS["permute0213"]
    name = "permute0213"
    schema = OpSchema(inputs=("x",), attrs=(AttrDef("dims", "ints", default=tuple(_dims)),))
    infer_shape_with_attrs = _specialized_permute_shape_with_attrs(name)
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_permute0213", library="model", source_template="permute_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_permute0213", library="model", source_template="permute_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_permute0213", library="model", source_template="permute_gpu.j2"),
    }
    frontend = FrontendBinding(name, default_attrs={"dims": list(_dims)})
    description = "Materialize a dense bounded copy for the fixed [0, 2, 1, 3] permutation on rank-4 static tensors."

    @classmethod
    def forward(cls, x: Any) -> Tensor:
        return permute0213(x)


@op_def
class Permute102(_SpecializedPermute):
    _dims = SPECIALIZED_PERMUTE_DIMS["permute102"]
    name = "permute102"
    schema = OpSchema(inputs=("x",), attrs=(AttrDef("dims", "ints", default=tuple(_dims)),))
    infer_shape_with_attrs = _specialized_permute_shape_with_attrs(name)
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_permute102", library="model", source_template="permute_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_permute102", library="model", source_template="permute_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_permute102", library="model", source_template="permute_gpu.j2"),
    }
    frontend = FrontendBinding(name, default_attrs={"dims": list(_dims)})
    description = "Materialize a dense bounded copy for the fixed [1, 0, 2] permutation on rank-3 static tensors."

    @classmethod
    def forward(cls, x: Any) -> Tensor:
        return permute102(x)


@op_def
class Permute210(_SpecializedPermute):
    _dims = SPECIALIZED_PERMUTE_DIMS["permute210"]
    name = "permute210"
    schema = OpSchema(inputs=("x",), attrs=(AttrDef("dims", "ints", default=tuple(_dims)),))
    infer_shape_with_attrs = _specialized_permute_shape_with_attrs(name)
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_permute210", library="model", source_template="permute_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_permute210", library="model", source_template="permute_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_permute210", library="model", source_template="permute_gpu.j2"),
    }
    frontend = FrontendBinding(name, default_attrs={"dims": list(_dims)})
    description = "Materialize a dense bounded copy for the fixed [2, 1, 0] permutation on rank-3 static tensors."

    @classmethod
    def forward(cls, x: Any) -> Tensor:
        return permute210(x)


@op_def
class DynamicSlice(OpDef):
    name = "dynamic_slice"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("start_indices", "ints", required=True),
            AttrDef("slice_sizes", "ints", required=True),
        ),
    )
    infer_shape = infer_dynamic_slice_shape
    infer_shape_with_attrs = infer_dynamic_slice_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_dynamic_slice", library="model", source_template="dynamic_slice_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_dynamic_slice", library="model", source_template="dynamic_slice_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_dynamic_slice", library="model", source_template="dynamic_slice_gpu.j2"),
    }
    frontend = FrontendBinding("dynamic_slice")
    description = "Materialize a dense bounded slice copy with static starts and sizes."

    @classmethod
    def forward(cls, x: Any, start_indices: Any, slice_sizes: Any) -> Tensor:
        return dynamic_slice(x, start_indices, slice_sizes)


@op_def
class IndexSelect(OpDef):
    name = "index_select"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("dim", "int", required=True),
            AttrDef("indices", "ints", required=True),
        ),
    )
    infer_shape = infer_index_select_shape
    infer_shape_with_attrs = infer_index_select_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_index_select", library="model", source_template="index_select_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_index_select", library="model", source_template="index_select_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_index_select", library="model", source_template="index_select_gpu.j2"),
    }
    frontend = FrontendBinding("index_select")
    description = "Materialize a dense bounded select copy along one static dimension using static integer indices."

    @classmethod
    def forward(cls, x: Any, dim: Any, indices: Any) -> Tensor:
        return index_select(x, dim, indices)


@op_def
class Gather(OpDef):
    name = "gather"
    schema = OpSchema(
        inputs=("x", "index"),
        attrs=(AttrDef("dim", "int", required=True),),
    )
    infer_shape = infer_gather_shape
    infer_shape_with_attrs = infer_gather_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_gather", library="model", source_template="gather_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_gather", library="model", source_template="gather_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_gather", library="model", source_template="gather_gpu.j2"),
    }
    frontend = FrontendBinding("gather")
    description = "Materialize a dense bounded gather copy using a static-shape integer index tensor."

    @classmethod
    def forward(cls, x: Any, dim: Any, index: Any) -> Tensor:
        return gather(x, dim, index)


@op_def
class BatchGather(OpDef):
    name = "batch_gather"
    schema = OpSchema(inputs=("x", "indices"))
    infer_shape = infer_batch_gather_shape
    infer_shape_with_attrs = infer_batch_gather_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_batch_gather", library="model", source_template="gather_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_batch_gather", library="model", source_template="gather_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_batch_gather", library="model", source_template="gather_gpu.j2"),
    }
    frontend = FrontendBinding("batch_gather")
    description = "Materialize a dense bounded batch gather from axis 1 using static-shape integer indices."

    @classmethod
    def forward(cls, x: Any, indices: Any) -> Tensor:
        return batch_gather(x, indices)


@op_def
class SliceScatter(OpDef):
    name = "slice_scatter"
    schema = OpSchema(
        inputs=("x", "update"),
        attrs=(AttrDef("start_indices", "ints", required=True),),
    )
    infer_shape = infer_slice_scatter_shape
    infer_shape_with_attrs = infer_slice_scatter_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_slice_scatter", library="model", source_template="slice_scatter_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_slice_scatter", library="model", source_template="slice_scatter_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_slice_scatter", library="model", source_template="slice_scatter_gpu.j2"),
    }
    frontend = FrontendBinding("slice_scatter")
    description = "Materialize a dense bounded scatter-update copy with static starts."

    @classmethod
    def forward(cls, x: Any, update: Any, start_indices: Any) -> Tensor:
        return slice_scatter(x, update, start_indices)


@op_def
class Pad(OpDef):
    name = "pad"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("pad", "ints", required=True),
            AttrDef("value", "float", default=0.0),
        ),
    )
    infer_shape = infer_pad_shape
    infer_shape_with_attrs = infer_pad_shape_with_attrs
    allowed_dtypes = COLLECTION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_pad", library="model", source_template="pad_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_pad", library="model", source_template="pad_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_pad", library="model", source_template="pad_gpu.j2"),
    }
    frontend = FrontendBinding("pad")
    description = "Materialize a dense static constant-padding copy using PyTorch F.pad pair order."

    @classmethod
    def forward(cls, x: Any, pad_width: Any, value: Any = 0.0) -> Tensor:
        return pad(x, pad_width, value)


def concatenate(inputs: Any, dim: int = 0) -> Tensor:
    tensors = _as_tensor_sequence(inputs, "concatenate")
    dtype = _check_collection_tensors("concatenate", tensors)
    if any(tensor.dynamic for tensor in tensors):
        raise ValueError("concatenate currently supports only static input shapes")
    normalized_dim = normalize_concatenate_dim(dim, tensors[0].rank)
    out_shape = infer_concatenate_shape_with_attrs([tensor.shape for tensor in tensors], {"dim": normalized_dim})
    return tensors[0].builder.emit(
        "concatenate",
        tensors,
        out_shape,
        dtype,
        {"dim": normalized_dim},
        shape_spec=out_shape,
    )


def concatenate_fast(inputs: Any, dim: int = 0) -> Tensor:
    return concatenate(inputs, dim)


def concatenate_tanh(inputs: Any, dim: int = 0) -> Tensor:
    return tanh(concatenate(inputs, dim))


def stack(inputs: Any, dim: int = 0) -> Tensor:
    tensors = _as_tensor_sequence(inputs, "stack")
    dtype = _check_collection_tensors("stack", tensors)
    if any(tensor.dynamic for tensor in tensors):
        raise ValueError("stack currently supports only static input shapes")
    normalized_dim = normalize_stack_dim(dim, tensors[0].rank)
    out_shape = infer_stack_shape_with_attrs([tensor.shape for tensor in tensors], {"dim": normalized_dim})
    return tensors[0].builder.emit(
        "stack",
        tensors,
        out_shape,
        dtype,
        {"dim": normalized_dim},
        shape_spec=out_shape,
    )


def flip(x: Any, dims: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"flip does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("flip currently supports only static input shapes")
    normalized_dims = normalize_flip_dims(dims, tensor.rank)
    return tensor.builder.emit(
        "flip",
        [tensor],
        tensor.shape,
        tensor.dtype,
        {"dims": normalized_dims},
        shape_spec=tensor.shape_spec,
    )


def repeat_interleave(x: Any, repeats: Any, dim: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"repeat_interleave does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("repeat_interleave currently supports only static input shapes")
    normalized_dim = normalize_repeat_interleave_dim(dim, tensor.rank)
    normalized_repeats = normalize_repeat_interleave_repeats(repeats)
    out_shape = infer_repeat_interleave_shape_with_attrs(
        [tensor.shape],
        {"repeats": normalized_repeats, "dim": normalized_dim},
    )
    return tensor.builder.emit(
        "repeat_interleave",
        [tensor],
        out_shape,
        tensor.dtype,
        {"repeats": normalized_repeats, "dim": normalized_dim},
        shape_spec=out_shape,
    )


def permute(x: Any, dims: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"permute does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("permute currently supports only static input shapes")
    normalized_dims = normalize_permute_dims(dims, tensor.rank)
    out_shape = infer_permute_shape_with_attrs([tensor.shape], {"dims": normalized_dims})
    out_shape_spec = [_copy_shape_dim(tensor.shape_spec[axis]) for axis in normalized_dims]
    return tensor.builder.emit(
        "permute",
        [tensor],
        out_shape,
        tensor.dtype,
        {"dims": normalized_dims},
        shape_spec=out_shape_spec,
    )


def permute021(x: Any) -> Tensor:
    return _specialized_permute("permute021", x)


def permute0213(x: Any) -> Tensor:
    return _specialized_permute("permute0213", x)


def permute102(x: Any) -> Tensor:
    return _specialized_permute("permute102", x)


def permute210(x: Any) -> Tensor:
    return _specialized_permute("permute210", x)


def pixel_shuffle(x: Any, upscale_factor: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.rank != 4:
        raise ValueError(f"pixel_shuffle expects rank-4 input [N, C, H, W], got rank {tensor.rank}")
    if tensor.dynamic:
        raise ValueError("pixel_shuffle currently supports only static input shapes")
    factor = _normalize_pixel_factor(upscale_factor, "pixel_shuffle upscale_factor")
    batch, channels_in, height, width = tensor.shape
    channel_factor = factor * factor
    if channels_in % channel_factor != 0:
        raise ValueError(
            f"pixel_shuffle input channels {channels_in} must be divisible by upscale_factor^2 ({channel_factor})"
        )
    channels_out = channels_in // channel_factor
    reshaped = reshape(tensor, [batch, channels_out, factor, factor, height, width])
    shuffled = permute(reshaped, (0, 1, 4, 2, 5, 3))
    return reshape(shuffled, [batch, channels_out, height * factor, width * factor])


def pixel_unshuffle(x: Any, downscale_factor: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.rank != 4:
        raise ValueError(f"pixel_unshuffle expects rank-4 input [N, C, H, W], got rank {tensor.rank}")
    if tensor.dynamic:
        raise ValueError("pixel_unshuffle currently supports only static input shapes")
    factor = _normalize_pixel_factor(downscale_factor, "pixel_unshuffle downscale_factor")
    batch, channels, height_in, width_in = tensor.shape
    if height_in % factor != 0:
        raise ValueError(f"pixel_unshuffle input height {height_in} must be divisible by downscale_factor {factor}")
    if width_in % factor != 0:
        raise ValueError(f"pixel_unshuffle input width {width_in} must be divisible by downscale_factor {factor}")
    height_out = height_in // factor
    width_out = width_in // factor
    reshaped = reshape(tensor, [batch, channels, height_out, factor, width_out, factor])
    unshuffled = permute(reshaped, (0, 1, 3, 5, 2, 4))
    return reshape(unshuffled, [batch, channels * factor * factor, height_out, width_out])


def dynamic_slice(x: Any, start_indices: Any, slice_sizes: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"dynamic_slice does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("dynamic_slice currently supports only static input shapes")
    normalized_starts, normalized_sizes = normalize_dynamic_slice_attrs(start_indices, slice_sizes, tensor.shape)
    out_shape = infer_dynamic_slice_shape_with_attrs(
        [tensor.shape],
        {"start_indices": normalized_starts, "slice_sizes": normalized_sizes},
    )
    return tensor.builder.emit(
        "dynamic_slice",
        [tensor],
        out_shape,
        tensor.dtype,
        {"start_indices": normalized_starts, "slice_sizes": normalized_sizes},
        shape_spec=out_shape,
    )


def index_select(x: Any, dim: Any, indices: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"index_select does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("index_select currently supports only static input shapes")
    normalized_dim, normalized_indices = normalize_index_select_attrs(dim, indices, tensor.shape)
    out_shape = infer_index_select_shape_with_attrs(
        [tensor.shape],
        {"dim": normalized_dim, "indices": normalized_indices},
    )
    return tensor.builder.emit(
        "index_select",
        [tensor],
        out_shape,
        tensor.dtype,
        {"dim": normalized_dim, "indices": normalized_indices},
        shape_spec=out_shape,
    )


def gather(x: Any, dim: Any, index: Any) -> Tensor:
    tensor = as_tensor(x)
    index_tensor = as_tensor(index)
    if tensor.builder is not index_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"gather does not support dtype {tensor.dtype}")
    if index_tensor.dtype not in GATHER_INDEX_DTYPES:
        raise ValueError(f"gather index must have dtype int64 or int32, got {index_tensor.dtype}")
    if tensor.dynamic or index_tensor.dynamic:
        raise ValueError("gather currently supports only static input and index shapes")
    normalized_dim = normalize_gather_attrs(dim, tensor.shape, index_tensor.shape)
    out_shape = infer_gather_shape_with_attrs(
        [tensor.shape, index_tensor.shape],
        {"dim": normalized_dim},
    )
    return tensor.builder.emit(
        "gather",
        [tensor, index_tensor],
        out_shape,
        tensor.dtype,
        {"dim": normalized_dim},
        shape_spec=out_shape,
    )


def batch_gather(x: Any, indices: Any) -> Tensor:
    tensor = as_tensor(x)
    index_tensor = as_tensor(indices)
    if tensor.builder is not index_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"batch_gather does not support dtype {tensor.dtype}")
    if index_tensor.dtype not in GATHER_INDEX_DTYPES:
        raise ValueError(f"batch_gather indices must have dtype int64 or int32, got {index_tensor.dtype}")
    if tensor.dynamic or index_tensor.dynamic:
        raise ValueError("batch_gather currently supports only static input and index shapes")
    normalize_batch_gather_attrs(tensor.shape, index_tensor.shape)
    out_shape = infer_batch_gather_shape_with_attrs([tensor.shape, index_tensor.shape], {})
    return tensor.builder.emit(
        "batch_gather",
        [tensor, index_tensor],
        out_shape,
        tensor.dtype,
        {},
        shape_spec=out_shape,
    )


def slice_scatter(x: Any, update: Any, start_indices: Any) -> Tensor:
    tensor = as_tensor(x)
    update_tensor = as_tensor(update, dtype_hint=tensor.dtype)
    dtype = _check_collection_tensors("slice_scatter", [tensor, update_tensor])
    if tensor.dynamic or update_tensor.dynamic:
        raise ValueError("slice_scatter currently supports only static input shapes")
    normalized_starts = normalize_slice_scatter_attrs(start_indices, tensor.shape, update_tensor.shape)
    out_shape = infer_slice_scatter_shape_with_attrs(
        [tensor.shape, update_tensor.shape],
        {"start_indices": normalized_starts},
    )
    return tensor.builder.emit(
        "slice_scatter",
        [tensor, update_tensor],
        out_shape,
        dtype,
        {"start_indices": normalized_starts},
        shape_spec=tensor.shape_spec,
    )


def slice_reshape_scatter(x: Any, update: Any, start_indices: Any, slice_shape: Any) -> Tensor:
    tensor = as_tensor(x)
    update_tensor = as_tensor(update, dtype_hint=tensor.dtype)
    normalized_shape = _normalize_slice_reshape_scatter_shape(slice_shape, tensor.rank, update_tensor.numel)
    reshaped_update = reshape(update_tensor, normalized_shape)
    return slice_scatter(tensor, reshaped_update, start_indices)


def split(x: Any, split_size_or_sections: Any, dim: Any = 0) -> tuple[Tensor, ...]:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"split does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("split currently supports only static input shapes")
    normalized_dim = normalize_split_dim(dim, tensor.rank)
    sections = normalize_split_sections(split_size_or_sections, tensor.shape[normalized_dim])
    return _slice_sections(tensor, sections, normalized_dim)


def chunk(x: Any, chunks: Any, dim: Any = 0) -> tuple[Tensor, ...]:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"chunk does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("chunk currently supports only static input shapes")
    normalized_dim = normalize_split_dim(dim, tensor.rank)
    normalized_chunks = normalize_chunk_count(chunks)
    sections = chunk_sections(tensor.shape[normalized_dim], normalized_chunks)
    return _slice_sections(tensor, sections, normalized_dim)


def pad(x: Any, pad: Any, value: Any = 0.0) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"pad does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("pad currently supports only static input shapes")
    normalize_pad_widths(pad, tensor.rank)
    normalized_pad = [int(value) for value in pad]
    out_shape = infer_pad_shape_with_attrs([tensor.shape], {"pad": normalized_pad})
    if tensor.dtype == "bool":
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"pad value must be a constant scalar, got {value!r}")
        if isinstance(value, float) and not isfinite(float(value)):
            raise ValueError("pad value must be finite")
        normalized_value: bool | float = bool(value)
    else:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"pad value must be a constant numeric scalar, got {value!r}")
        if not isfinite(float(value)):
            raise ValueError("pad value must be finite")
        normalized_value = float(value)
    return tensor.builder.emit(
        "pad",
        [tensor],
        out_shape,
        tensor.dtype,
        {"pad": normalized_pad, "value": normalized_value},
        shape_spec=out_shape,
    )


def pad_last_dim(x: Any, left: Any, right: Any, value: Any = 0.0) -> Tensor:
    if not isinstance(left, int) or isinstance(left, bool):
        raise ValueError(f"pad_last_dim left must be a non-negative integer, got {left!r}")
    if not isinstance(right, int) or isinstance(right, bool):
        raise ValueError(f"pad_last_dim right must be a non-negative integer, got {right!r}")
    return pad(x, [int(left), int(right)], value=value)


def transpose(x: Any, dim0: Any, dim1: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"transpose does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("transpose currently supports only static input shapes")
    normalized_dim0, normalized_dim1 = normalize_transpose_dims(dim0, dim1, tensor.rank)
    dims = list(range(tensor.rank))
    dims[normalized_dim0], dims[normalized_dim1] = dims[normalized_dim1], dims[normalized_dim0]
    return permute(tensor, dims)


def _as_tensor_sequence(inputs: Any, op_name: str) -> list[Tensor]:
    if isinstance(inputs, (Tensor, Parameter)) or not isinstance(inputs, (list, tuple)):
        raise ValueError(f"{op_name} expects a non-empty sequence of tensors")
    if not inputs:
        raise ValueError(f"{op_name} expects a non-empty sequence of tensors")
    first = as_tensor(inputs[0])
    return [first, *(as_tensor(value, dtype_hint=first.dtype) for value in inputs[1:])]


def _check_collection_tensors(op_name: str, tensors: Sequence[Tensor]) -> str:
    first = tensors[0]
    for tensor in tensors[1:]:
        if tensor.builder is not first.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != first.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {first.dtype} vs {tensor.dtype}")
    if first.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {first.dtype}")
    return first.dtype


def _specialized_permute(op_name: str, x: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError(f"{op_name} currently supports only static input shapes")
    fixed_dims = SPECIALIZED_PERMUTE_DIMS[op_name]
    if tensor.rank != len(fixed_dims):
        raise ValueError(f"{op_name} expects rank-{len(fixed_dims)} input, got rank {tensor.rank}")
    normalized_dims = list(fixed_dims)
    out_shape = infer_permute_shape_with_attrs([tensor.shape], {"dims": normalized_dims})
    out_shape_spec = [_copy_shape_dim(tensor.shape_spec[axis]) for axis in normalized_dims]
    return tensor.builder.emit(
        op_name,
        [tensor],
        out_shape,
        tensor.dtype,
        {"dims": normalized_dims},
        shape_spec=out_shape_spec,
    )


def _normalize_pixel_factor(factor: Any, name: str) -> int:
    if not isinstance(factor, int) or isinstance(factor, bool):
        raise ValueError(f"{name} must be a positive integer")
    if factor <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(factor)


def _normalize_slice_reshape_scatter_shape(slice_shape: Any, rank: int, update_numel: int) -> list[int]:
    if not isinstance(slice_shape, (list, tuple)):
        raise ValueError(f"slice_reshape_scatter slice_shape must be a sequence of integers, got {slice_shape!r}")
    normalized: list[int] = []
    for dim in slice_shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"slice_reshape_scatter slice_shape must contain only integers, got {slice_shape!r}")
        if dim <= 0:
            raise ValueError(f"slice_reshape_scatter slice_shape dimensions must be positive, got {dim}")
        normalized.append(int(dim))
    if len(normalized) != rank:
        raise ValueError(f"slice_reshape_scatter slice_shape rank {len(normalized)} must match input rank {rank}")
    if int(prod(normalized)) != int(update_numel):
        raise ValueError(
            f"slice_reshape_scatter slice_shape {normalized} must preserve update element count {int(update_numel)}"
        )
    return normalized


def _slice_sections(tensor: Tensor, sections: Sequence[int], dim: int) -> tuple[Tensor, ...]:
    outputs = []
    start = 0
    for section in sections:
        starts = [0] * tensor.rank
        sizes = list(tensor.shape)
        starts[dim] = start
        sizes[dim] = section
        outputs.append(dynamic_slice(tensor, starts, sizes))
        start += section
    return tuple(outputs)


def _copy_shape_dim(dim: Any) -> Any:
    return dict(dim) if isinstance(dim, Mapping) else dim


__all__ = [
    "BatchGather",
    "COLLECTION_DTYPES",
    "Concatenate",
    "DynamicSlice",
    "Flip",
    "GATHER_INDEX_DTYPES",
    "Gather",
    "IndexSelect",
    "Pad",
    "Permute",
    "Permute021",
    "Permute0213",
    "Permute102",
    "Permute210",
    "RepeatInterleave",
    "SPECIALIZED_PERMUTE_DIMS",
    "SliceScatter",
    "Stack",
    "batch_gather",
    "chunk",
    "concatenate",
    "concatenate_fast",
    "concatenate_tanh",
    "dynamic_slice",
    "flip",
    "gather",
    "index_select",
    "infer_batch_gather_shape",
    "infer_batch_gather_shape_with_attrs",
    "infer_concatenate_shape",
    "infer_concatenate_shape_with_attrs",
    "infer_dynamic_slice_shape",
    "infer_dynamic_slice_shape_with_attrs",
    "infer_flip_shape",
    "infer_flip_shape_with_attrs",
    "infer_gather_shape",
    "infer_gather_shape_with_attrs",
    "infer_index_select_shape",
    "infer_index_select_shape_with_attrs",
    "infer_pad_shape",
    "infer_pad_shape_with_attrs",
    "infer_repeat_interleave_shape",
    "infer_repeat_interleave_shape_with_attrs",
    "infer_slice_scatter_shape",
    "infer_slice_scatter_shape_with_attrs",
    "infer_permute_shape",
    "infer_permute_shape_with_attrs",
    "infer_specialized_permute_shape_with_attrs",
    "infer_stack_shape",
    "infer_stack_shape_with_attrs",
    "pad",
    "pad_last_dim",
    "permute",
    "permute021",
    "permute0213",
    "permute102",
    "permute210",
    "pixel_shuffle",
    "pixel_unshuffle",
    "repeat_interleave",
    "slice_reshape_scatter",
    "slice_scatter",
    "split",
    "stack",
    "transpose",
    "chunk_sections",
    "normalize_batch_gather_attrs",
    "normalize_chunk_count",
    "normalize_concatenate_dim",
    "normalize_dynamic_slice_attrs",
    "normalize_flip_dims",
    "normalize_gather_attrs",
    "normalize_gather_dim",
    "normalize_index_select_attrs",
    "normalize_index_select_dim",
    "normalize_index_select_indices",
    "normalize_pad_widths",
    "normalize_repeat_interleave_dim",
    "normalize_repeat_interleave_repeats",
    "normalize_slice_scatter_attrs",
    "normalize_permute_dims",
    "normalize_split_dim",
    "normalize_split_sections",
    "normalize_stack_dim",
    "normalize_transpose_dims",
    "resolve_batch_gather_shape",
    "resolve_concatenate_shape",
    "resolve_dynamic_slice_shape",
    "resolve_flip_shape",
    "resolve_gather_shape",
    "resolve_index_select_shape",
    "resolve_pad_shape",
    "resolve_repeat_interleave_shape",
    "resolve_slice_scatter_shape",
    "resolve_permute_shape",
    "resolve_stack_shape",
]
