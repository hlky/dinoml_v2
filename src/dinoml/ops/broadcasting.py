from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.frontend import Parameter, Tensor, as_tensor
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def
from dinoml.ops.shape_views import reshape


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


@op_def
class Expand(OpDef):
    name = "expand"
    schema = OpSchema(
        inputs=("x",),
        attrs=(AttrDef("shape", "shape", required=True),),
    )
    infer_shape = infer_expand_shape
    infer_shape_with_attrs = infer_expand_shape_with_attrs
    allowed_dtypes = BROADCAST_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_expand", library="model", source_template="expand_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_expand", library="model", source_template="expand_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_expand", library="model", source_template="expand_gpu.j2"),
    }
    frontend = FrontendBinding("expand")
    description = "Materialize a dense broadcast of a tensor to a static shape."

    @classmethod
    def forward(cls, x: Any, shape: Any) -> Tensor:
        tensor = as_tensor(x)
        if tensor.dtype not in BROADCAST_DTYPES:
            raise ValueError(f"expand does not support dtype {tensor.dtype}")
        if tensor.dynamic:
            raise ValueError("expand currently supports only static input shapes")
        out_shape = resolve_expand_shape(tensor.shape, shape)
        return tensor.builder.emit(
            "expand",
            [tensor],
            out_shape,
            tensor.dtype,
            {"shape": list(shape)},
            shape_spec=out_shape,
        )


def expand(x: Any, shape: Any) -> Tensor:
    return Expand.forward(x, shape)


def expand_static_shape(x: Any, shape: Any) -> Tensor:
    return Expand.forward(x, shape)


def meshgrid(inputs: Any, indexing: str = "ij") -> tuple[Tensor, ...]:
    if isinstance(inputs, (Tensor, Parameter)) or not isinstance(inputs, (list, tuple)):
        raise ValueError("meshgrid expects a non-empty sequence of tensors")
    if not inputs:
        raise ValueError("meshgrid expects a non-empty sequence of tensors")
    if indexing != "ij":
        raise NotImplementedError('meshgrid currently supports indexing="ij" only')
    first = as_tensor(inputs[0])
    tensors = [first, *(as_tensor(value, dtype_hint=first.dtype) for value in inputs[1:])]
    for tensor in tensors[1:]:
        if tensor.builder is not first.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != first.dtype:
            raise ValueError(f"meshgrid dtype mismatch: {first.dtype} vs {tensor.dtype}")
    if first.dtype not in BROADCAST_DTYPES:
        raise ValueError(f"meshgrid does not support dtype {first.dtype}")
    for tensor in tensors:
        if tensor.rank != 1:
            raise ValueError(f"meshgrid expects rank-1 inputs, got rank {tensor.rank}")
        if tensor.dynamic:
            raise ValueError("meshgrid currently supports only static input shapes")
    grid_shape = [tensor.shape[0] for tensor in tensors]
    outputs = []
    for axis, tensor in enumerate(tensors):
        view_shape = [1] * len(tensors)
        view_shape[axis] = tensor.shape[0]
        outputs.append(expand(reshape(tensor, view_shape), grid_shape))
    return tuple(outputs)


__all__ = [
    "BROADCAST_DTYPES",
    "Expand",
    "expand",
    "expand_static_shape",
    "infer_expand_shape",
    "infer_expand_shape_with_attrs",
    "meshgrid",
    "resolve_expand_shape",
]
