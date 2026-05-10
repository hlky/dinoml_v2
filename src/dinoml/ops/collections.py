from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


COLLECTION_DTYPES = ("float16", "float32", "bfloat16", "bool")


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


def infer_dynamic_slice_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("dynamic_slice expects one tensor input")
    return resolve_dynamic_slice_shape(input_shapes[0], attrs.get("start_indices"), attrs.get("slice_sizes"))


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


def register_collection_ops(registry: OpRegistry) -> None:
    registry.register(
        OpDef(
            name="concatenate",
            schema=OpSchema(
                inputs=("x0",),
                attrs=(AttrDef("dim", "int", default=0),),
            ),
            infer_shape=infer_concatenate_shape,
            infer_shape_with_attrs=infer_concatenate_shape_with_attrs,
            allowed_dtypes=COLLECTION_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_concatenate", library="model", source_template="concatenate_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_concatenate", library="model", source_template="concatenate_cuda.cu.j2"),
            },
            frontend=FrontendBinding("concatenate"),
            variadic_inputs=True,
            description="Materialize a dense concatenation copy along a static dimension.",
        )
    )
    registry.register(
        OpDef(
            name="stack",
            schema=OpSchema(
                inputs=("x0",),
                attrs=(AttrDef("dim", "int", default=0),),
            ),
            infer_shape=infer_stack_shape,
            infer_shape_with_attrs=infer_stack_shape_with_attrs,
            allowed_dtypes=COLLECTION_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_stack", library="model", source_template="stack_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_stack", library="model", source_template="stack_cuda.cu.j2"),
            },
            frontend=FrontendBinding("stack"),
            variadic_inputs=True,
            description="Materialize a dense stack copy by inserting a static dimension.",
        )
    )
    registry.register(
        OpDef(
            name="flip",
            schema=OpSchema(
                inputs=("x",),
                attrs=(AttrDef("dims", "ints", required=True),),
            ),
            infer_shape=infer_flip_shape,
            infer_shape_with_attrs=infer_flip_shape_with_attrs,
            allowed_dtypes=COLLECTION_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_flip", library="model", source_template="flip_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_flip", library="model", source_template="flip_cuda.cu.j2"),
            },
            frontend=FrontendBinding("flip"),
            description="Materialize a dense copy that reverses one or more static dimensions.",
        )
    )
    registry.register(
        OpDef(
            name="repeat_interleave",
            schema=OpSchema(
                inputs=("x",),
                attrs=(AttrDef("repeats", "int", required=True), AttrDef("dim", "int", required=True)),
            ),
            infer_shape=infer_repeat_interleave_shape,
            infer_shape_with_attrs=infer_repeat_interleave_shape_with_attrs,
            allowed_dtypes=COLLECTION_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_repeat_interleave", library="model", source_template="repeat_interleave_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_repeat_interleave", library="model", source_template="repeat_interleave_cuda.cu.j2"),
            },
            frontend=FrontendBinding("repeat_interleave"),
            description="Materialize a dense bounded repeat-interleave copy along a static dimension.",
        )
    )
    registry.register(
        OpDef(
            name="permute",
            schema=OpSchema(
                inputs=("x",),
                attrs=(AttrDef("dims", "ints", required=True),),
            ),
            infer_shape=infer_permute_shape,
            infer_shape_with_attrs=infer_permute_shape_with_attrs,
            allowed_dtypes=COLLECTION_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_permute", library="model", source_template="permute_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_permute", library="model", source_template="permute_cuda.cu.j2"),
            },
            frontend=FrontendBinding("permute"),
            description="Materialize a dense bounded copy with permuted static dimensions.",
        )
    )
    registry.register(
        OpDef(
            name="dynamic_slice",
            schema=OpSchema(
                inputs=("x",),
                attrs=(
                    AttrDef("start_indices", "ints", required=True),
                    AttrDef("slice_sizes", "ints", required=True),
                ),
            ),
            infer_shape=infer_dynamic_slice_shape,
            infer_shape_with_attrs=infer_dynamic_slice_shape_with_attrs,
            allowed_dtypes=COLLECTION_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_dynamic_slice", library="model", source_template="dynamic_slice_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_dynamic_slice", library="model", source_template="dynamic_slice_cuda.cu.j2"),
            },
            frontend=FrontendBinding("dynamic_slice"),
            description="Materialize a dense bounded slice copy with static starts and sizes.",
        )
    )


__all__ = [
    "COLLECTION_DTYPES",
    "infer_concatenate_shape",
    "infer_concatenate_shape_with_attrs",
    "infer_dynamic_slice_shape",
    "infer_dynamic_slice_shape_with_attrs",
    "infer_flip_shape",
    "infer_flip_shape_with_attrs",
    "infer_repeat_interleave_shape",
    "infer_repeat_interleave_shape_with_attrs",
    "infer_permute_shape",
    "infer_permute_shape_with_attrs",
    "infer_stack_shape",
    "infer_stack_shape_with_attrs",
    "normalize_concatenate_dim",
    "normalize_dynamic_slice_attrs",
    "normalize_flip_dims",
    "normalize_repeat_interleave_dim",
    "normalize_repeat_interleave_repeats",
    "normalize_permute_dims",
    "normalize_stack_dim",
    "normalize_transpose_dims",
    "register_collection_ops",
    "resolve_concatenate_shape",
    "resolve_dynamic_slice_shape",
    "resolve_flip_shape",
    "resolve_repeat_interleave_shape",
    "resolve_permute_shape",
    "resolve_stack_shape",
]
