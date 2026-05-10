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


__all__ = [
    "COLLECTION_DTYPES",
    "infer_concatenate_shape",
    "infer_concatenate_shape_with_attrs",
    "infer_flip_shape",
    "infer_flip_shape_with_attrs",
    "infer_stack_shape",
    "infer_stack_shape_with_attrs",
    "normalize_concatenate_dim",
    "normalize_flip_dims",
    "normalize_stack_dim",
    "register_collection_ops",
    "resolve_concatenate_shape",
    "resolve_flip_shape",
    "resolve_stack_shape",
]
