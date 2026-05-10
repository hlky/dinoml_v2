from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


CREATION_DTYPES = ("float16", "float32", "bfloat16", "bool")


def infer_full_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("full shape inference requires shape attrs")


def infer_full_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if input_shapes:
        raise ValueError("full expects no tensor inputs")
    shape = attrs.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) == 0:
        raise ValueError("full requires a non-empty static shape")
    result = []
    for dim in shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or int(dim) <= 0:
            raise ValueError(f"full requires positive integer shape dimensions, got {shape!r}")
        result.append(int(dim))
    return result


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


__all__ = ["CREATION_DTYPES", "infer_full_shape", "infer_full_shape_with_attrs", "register_creation_ops"]
