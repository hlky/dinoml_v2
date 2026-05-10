from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


COLLECTION_DTYPES = ("float16", "float32", "bfloat16", "bool")


def infer_concatenate_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_concatenate_shape_with_attrs(input_shapes, {"dim": 0})


def infer_concatenate_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if not input_shapes:
        raise ValueError("concatenate expects a non-empty sequence of tensors")
    dim = normalize_concatenate_dim(attrs.get("dim", 0), len(input_shapes[0]))
    return resolve_concatenate_shape(input_shapes, dim)


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


__all__ = [
    "COLLECTION_DTYPES",
    "infer_concatenate_shape",
    "infer_concatenate_shape_with_attrs",
    "normalize_concatenate_dim",
    "register_collection_ops",
    "resolve_concatenate_shape",
]
