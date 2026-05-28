from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dinoml.frontend import GraphBuilder, Tensor
from dinoml.ir import normalize_dtype
from dinoml.shapes import Shape
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


CREATION_DTYPES = ("float16", "float32", "bfloat16", "bool")
ARANGE_DTYPES = ("float16", "float32", "bfloat16")
RANDN_DTYPES = ("float16", "float32", "bfloat16")
RANDN_RNGS = ("dinoml", "numpy", "torch")


def infer_full_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("full shape inference requires shape attrs")


def infer_full_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    return _infer_static_creation_shape(input_shapes, attrs, "full")


def infer_randn_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("randn shape inference requires shape attrs")


def infer_randn_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    _seed_attr(attrs)
    _randn_rng_attr(attrs)
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


@op_def
class Full(OpDef):
    name = "full"
    schema = OpSchema(
        inputs=(),
        attrs=(
            AttrDef("shape", "shape", required=True),
            AttrDef("fill_value", "float", required=True),
            AttrDef("dtype", "dtype", "float32"),
        ),
    )
    infer_shape = infer_full_shape
    infer_shape_with_attrs = infer_full_shape_with_attrs
    allowed_dtypes = CREATION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_full", library="model", source_template="full_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_full", library="model", source_template="full_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_full", library="model", source_template="full_gpu.j2"),
    }
    frontend = FrontendBinding("full")
    description = "Create a dense tensor filled with a scalar value."

    @classmethod
    def forward(cls, shape: Any, fill_value: Any, dtype: str = "float32") -> Tensor:
        dtype = normalize_dtype(dtype)
        if dtype not in CREATION_DTYPES:
            raise ValueError(f"full does not support dtype {dtype}")
        shape_obj = Shape(shape)
        if len(shape_obj) == 0:
            raise ValueError("full shape must not be empty")
        if shape_obj.dynamic:
            raise ValueError("full currently supports only static shapes")
        if dtype == "bool":
            normalized_fill: bool | float = bool(fill_value)
        else:
            normalized_fill = float(fill_value)
        attrs = {"shape": shape_obj.max_shape, "fill_value": normalized_fill, "dtype": dtype}
        return GraphBuilder.current().emit(
            "full",
            [],
            shape_obj.max_shape,
            dtype,
            attrs,
            shape_spec=shape_obj.to_json(),
        )


@op_def
class Arange(OpDef):
    name = "arange"
    schema = OpSchema(
        inputs=(),
        attrs=(
            AttrDef("start", "float", required=True),
            AttrDef("end", "float", required=True),
            AttrDef("step", "float", 1.0),
            AttrDef("dtype", "dtype", "float32"),
        ),
    )
    infer_shape = infer_arange_shape
    infer_shape_with_attrs = infer_arange_shape_with_attrs
    allowed_dtypes = ARANGE_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_arange", library="model", source_template="arange_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_arange", library="model", source_template="arange_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_arange", library="model", source_template="arange_gpu.j2"),
    }
    frontend = FrontendBinding("arange")
    description = "Create a dense 1D tensor from a static numeric range."

    @classmethod
    def forward(cls, start: Any, end: Any | None = None, step: Any = 1, dtype: str = "float32") -> Tensor:
        dtype = normalize_dtype(dtype)
        if dtype not in ARANGE_DTYPES:
            raise ValueError(f"arange does not support dtype {dtype}")
        if end is None:
            normalized_start = 0.0
            normalized_end = _creation_number(start, "end")
        else:
            normalized_start = _creation_number(start, "start")
            normalized_end = _creation_number(end, "end")
        normalized_step = _creation_number(step, "step")
        attrs = {"start": normalized_start, "end": normalized_end, "step": normalized_step, "dtype": dtype}
        out_shape = infer_arange_shape_with_attrs([], attrs)
        return GraphBuilder.current().emit(
            "arange",
            [],
            out_shape,
            dtype,
            attrs,
            shape_spec=out_shape,
        )


@op_def
class Randn(OpDef):
    name = "randn"
    schema = OpSchema(
        inputs=(),
        attrs=(
            AttrDef("shape", "shape", required=True),
            AttrDef("dtype", "dtype", "float32"),
            AttrDef("seed", "int", 0),
            AttrDef("rng", "str", "dinoml"),
        ),
    )
    infer_shape = infer_randn_shape
    infer_shape_with_attrs = infer_randn_shape_with_attrs
    allowed_dtypes = RANDN_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_randn", library="model", source_template="randn_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_randn", library="model", source_template="randn_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_randn", library="model", source_template="randn_gpu.j2"),
    }
    frontend = FrontendBinding("randn")
    description = "Create a dense tensor of deterministic standard normal samples."

    @classmethod
    def forward(cls, shape: Any, dtype: str = "float32", seed: int = 0, rng: str = "dinoml") -> Tensor:
        dtype = normalize_dtype(dtype)
        if dtype not in RANDN_DTYPES:
            raise ValueError(f"randn does not support dtype {dtype}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("randn requires integer seed")
        if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("randn seed must fit in uint64")
        rng = _normalize_randn_rng(rng)
        shape_obj = Shape(shape)
        if len(shape_obj) == 0:
            raise ValueError("randn shape must not be empty")
        if shape_obj.dynamic:
            raise ValueError("randn currently supports only static shapes")
        attrs = {"shape": shape_obj.max_shape, "dtype": dtype, "seed": int(seed), "rng": rng}
        return GraphBuilder.current().emit(
            "randn",
            [],
            shape_obj.max_shape,
            dtype,
            attrs,
            shape_spec=shape_obj.to_json(),
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


def _randn_rng_attr(attrs: Mapping[str, Any]) -> str:
    return _normalize_randn_rng(attrs.get("rng", "dinoml"))


def _normalize_randn_rng(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("randn rng must be a string")
    normalized = value.lower()
    if normalized not in RANDN_RNGS:
        supported = ", ".join(RANDN_RNGS)
        raise ValueError(f"randn rng must be one of: {supported}")
    return normalized


def _creation_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"arange requires numeric {name}")
    return float(value)


def full(shape: Any, fill_value: Any, dtype: str = "float32") -> Tensor:
    return Full.forward(shape, fill_value, dtype)


def arange(start: Any, end: Any | None = None, step: Any = 1, dtype: str = "float32") -> Tensor:
    return Arange.forward(start, end, step, dtype)


def randn(shape: Any, dtype: str = "float32", seed: int = 0, rng: str = "dinoml") -> Tensor:
    return Randn.forward(shape, dtype, seed, rng)


__all__ = [
    "ARANGE_DTYPES",
    "Arange",
    "CREATION_DTYPES",
    "Full",
    "RANDN_DTYPES",
    "RANDN_RNGS",
    "Randn",
    "arange",
    "full",
    "infer_arange_shape",
    "infer_arange_shape_with_attrs",
    "infer_full_shape",
    "infer_full_shape_with_attrs",
    "infer_randn_shape",
    "infer_randn_shape_with_attrs",
    "randn",
]
