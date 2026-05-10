from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


BROADCAST_DTYPES = ("float16", "float32", "bfloat16", "bool")


def infer_expand_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("expand expects one tensor input")
    return list(input_shapes[0])


def infer_expand_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("expand expects one tensor input")
    return resolve_expand_shape(input_shapes[0], attrs.get("shape"))


def resolve_expand_shape(input_shape: Sequence[int], requested_shape: Any) -> list[int]:
    if not isinstance(requested_shape, Sequence) or isinstance(requested_shape, (str, bytes)) or not requested_shape:
        raise ValueError("expand requires a non-empty static shape")
    requested = []
    for dim in requested_shape:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim == 0 or dim < -1:
            raise ValueError(f"expand shape dimensions must be positive or -1, got {requested_shape!r}")
        requested.append(int(dim))
    if len(requested) < len(input_shape):
        raise ValueError(f"expand shape rank {len(requested)} must be >= input rank {len(input_shape)}")
    prefix = len(requested) - len(input_shape)
    output_shape = []
    for axis, requested_dim in enumerate(requested):
        source_dim = 1 if axis < prefix else int(input_shape[axis - prefix])
        if requested_dim == -1:
            if axis < prefix:
                raise ValueError("expand cannot use -1 for a new leading dimension")
            output_dim = source_dim
        else:
            output_dim = requested_dim
        if source_dim != 1 and source_dim != output_dim:
            raise ValueError(f"expand input shape {list(input_shape)} is not broadcastable to {output_shape + [output_dim] + requested[axis + 1:]}")
        output_shape.append(output_dim)
    return output_shape


def register_broadcasting_ops(registry: OpRegistry) -> None:
    registry.register(
        OpDef(
            name="expand",
            schema=OpSchema(
                inputs=("x",),
                attrs=(AttrDef("shape", "shape", required=True),),
            ),
            infer_shape=infer_expand_shape,
            infer_shape_with_attrs=infer_expand_shape_with_attrs,
            allowed_dtypes=BROADCAST_DTYPES,
            backend_kernels={
                "cpu": KernelBinding(symbol="generated_expand", library="model", source_template="expand_cpu.cpp.j2"),
                "cuda": KernelBinding(symbol="generated_expand", library="model", source_template="expand_cuda.cu.j2"),
            },
            frontend=FrontendBinding("expand"),
            description="Materialize a dense broadcast of a tensor to a static shape.",
        )
    )


__all__ = ["BROADCAST_DTYPES", "infer_expand_shape", "infer_expand_shape_with_attrs", "register_broadcasting_ops", "resolve_expand_shape"]
