from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


CREATION_DTYPES = ("float16", "float32", "bfloat16", "bool")
ARANGE_DTYPES = ("float16", "float32", "bfloat16")
RANDN_DTYPES = ("float16", "float32", "bfloat16")


def infer_full_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("full shape inference requires shape attrs")


def infer_full_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    return _infer_static_creation_shape(input_shapes, attrs, "full")


def infer_randn_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("randn shape inference requires shape attrs")


def infer_randn_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    _seed_attr(attrs)
    return _infer_static_creation_shape(input_shapes, attrs, "randn")


def _infer_static_creation_shape(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any], op_name: str) -> list[int]:
    if input_shapes:
        raise ValueError(f"{op_name} expects no tensor inputs")
    shape = attrs.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) == 0:
        raise ValueError(f"{op_name} requires a non-empty static shape")
    result = []
    for dim in shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or int(dim) <= 0:
            raise ValueError(f"{op_name} requires positive integer shape dimensions, got {shape!r}")
        result.append(int(dim))
    return result


def infer_arange_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("arange shape inference requires range attrs")


def infer_arange_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if input_shapes:
        raise ValueError("arange expects no tensor inputs")
    start = _finite_number_attr(attrs, "start")
    end = _finite_number_attr(attrs, "end")
    step = _finite_number_attr(attrs, "step")
    length = _arange_length(start, end, step)
    if length <= 0:
        raise ValueError("arange currently requires a non-empty range")
    return [length]


def register_creation_ops(registry: OpRegistry) -> None:
    registry.register(
        OpDef(
            name="full",
            schema=OpSchema(
                inputs=(),
                attrs=(
                    AttrDef("shape", "shape", required=True),
                    AttrDef("fill_value", "float", required=True),
                    AttrDef("dtype", "dtype", "float32"),
                ),
            ),
            infer_shape=infer_full_shape,
            infer_shape_with_attrs=infer_full_shape_with_attrs,
            allowed_dtypes=CREATION_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_full", library="model", source_template="full_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_full", library="model", source_template="full_cuda.cu.j2"),
            },
            frontend=FrontendBinding("full"),
            description="Create a dense tensor filled with a scalar value.",
        )
    )
    registry.register(
        OpDef(
            name="arange",
            schema=OpSchema(
                inputs=(),
                attrs=(
                    AttrDef("start", "float", required=True),
                    AttrDef("end", "float", required=True),
                    AttrDef("step", "float", 1.0),
                    AttrDef("dtype", "dtype", "float32"),
                ),
            ),
            infer_shape=infer_arange_shape,
            infer_shape_with_attrs=infer_arange_shape_with_attrs,
            allowed_dtypes=ARANGE_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_arange", library="model", source_template="arange_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_arange", library="model", source_template="arange_cuda.cu.j2"),
            },
            frontend=FrontendBinding("arange"),
            description="Create a dense 1D tensor from a static numeric range.",
        )
    )
    registry.register(
        OpDef(
            name="randn",
            schema=OpSchema(
                inputs=(),
                attrs=(
                    AttrDef("shape", "shape", required=True),
                    AttrDef("dtype", "dtype", "float32"),
                    AttrDef("seed", "int", 0),
                ),
            ),
            infer_shape=infer_randn_shape,
            infer_shape_with_attrs=infer_randn_shape_with_attrs,
            allowed_dtypes=RANDN_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_randn", library="model", source_template="randn_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_randn", library="model", source_template="randn_cuda.cu.j2"),
            },
            frontend=FrontendBinding("randn"),
            description="Create a dense tensor of deterministic standard normal samples.",
        )
    )


def _finite_number_attr(attrs: Mapping[str, Any], name: str) -> float:
    value = attrs.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"arange requires numeric {name}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"arange requires finite {name}")
    return value


def _arange_length(start: float, end: float, step: float) -> int:
    if step == 0.0:
        raise ValueError("arange step must not be zero")
    span = end - start
    if (step > 0.0 and span <= 0.0) or (step < 0.0 and span >= 0.0):
        return 0
    return max(0, int(math.ceil(span / step)))


def _seed_attr(attrs: Mapping[str, Any]) -> int:
    seed = attrs.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("randn requires integer seed")
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("randn seed must fit in uint64")
    return int(seed)


__all__ = [
    "ARANGE_DTYPES",
    "CREATION_DTYPES",
    "RANDN_DTYPES",
    "infer_arange_shape",
    "infer_arange_shape_with_attrs",
    "infer_full_shape",
    "infer_full_shape_with_attrs",
    "infer_randn_shape",
    "infer_randn_shape_with_attrs",
    "register_creation_ops",
]
