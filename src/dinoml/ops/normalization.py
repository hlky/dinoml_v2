from __future__ import annotations

from typing import Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


NORMALIZATION_DTYPES = ("float16", "float32", "bfloat16")
T5_LAYER_NORM_DTYPES = NORMALIZATION_DTYPES
LAYER_NORM_DTYPES = NORMALIZATION_DTYPES


def infer_t5_layer_norm(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 2:
        raise ValueError("t5_layer_norm expects exactly two inputs")
    return list(shapes[0])


def infer_layer_norm(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 3:
        raise ValueError("layer_norm expects exactly three inputs")
    return list(shapes[0])


def register_normalization_ops(registry: OpRegistry) -> None:
    registry.register(
        OpDef(
            name="t5_layer_norm",
            schema=OpSchema(inputs=("x", "weight"), attrs=(AttrDef("eps", "float", 1e-6),)),
            infer_shape=infer_t5_layer_norm,
            backend_kernels={
                "cuda": KernelBinding("generated_t5_layer_norm", "model", source_template="t5_layer_norm_cuda"),
                "cpu": KernelBinding("generated_t5_layer_norm", "model", source_template="t5_layer_norm_cpu"),
            },
            frontend=FrontendBinding("t5_layer_norm"),
            allowed_dtypes=T5_LAYER_NORM_DTYPES,
            description=(
                "Bounded T5/RMS-style layer normalization over the last static dimension with fp32 accumulation, "
                "rank >= 1 inputs, and a required affine weight shaped [hidden]."
            ),
        )
    )
    registry.register(
        OpDef(
            name="layer_norm",
            schema=OpSchema(inputs=("x", "weight", "bias"), attrs=(AttrDef("eps", "float", 1e-5),)),
            infer_shape=infer_layer_norm,
            backend_kernels={
                "cuda": KernelBinding("generated_layer_norm", "model", source_template="layer_norm_cuda"),
                "cpu": KernelBinding("generated_layer_norm", "model", source_template="layer_norm_cpu"),
            },
            frontend=FrontendBinding("layer_norm"),
            allowed_dtypes=LAYER_NORM_DTYPES,
            description=(
                "Bounded affine LayerNorm over the last static dimension with fp32 accumulation, "
                "rank >= 1 inputs, and required rank-1 affine weight/bias tensors shaped [hidden]."
            ),
        )
    )


def _validate_affine_norm_inputs(
    op_name: str,
    x: object,
    weight: object,
    bias: object | None = None,
    *,
    allowed_dtypes: Sequence[str],
) -> tuple[Tensor, Tensor, Tensor | None]:
    x_tensor = as_tensor(x, dtype_hint="float32")
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    extra_tensor = None if bias is None else as_tensor(bias, dtype_hint=x_tensor.dtype)
    tensors = [x_tensor, weight_tensor] + ([] if extra_tensor is None else [extra_tensor])
    for tensor in tensors[1:]:
        if tensor.builder is not x_tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
    if x_tensor.dtype not in allowed_dtypes:
        raise ValueError(f"{op_name} does not support dtype {x_tensor.dtype}")
    if x_tensor.rank < 1:
        raise ValueError(f"{op_name} requires rank >= 1 input")
    if not isinstance(x_tensor.shape_spec[-1], int):
        raise ValueError(f"{op_name} currently requires a static last dimension")
    hidden = int(x_tensor.shape[-1])
    if hidden <= 0:
        raise ValueError(f"{op_name} last dimension must be positive")
    if weight_tensor.dtype != x_tensor.dtype:
        raise ValueError(f"{op_name} dtype mismatch: {x_tensor.dtype} vs {weight_tensor.dtype}")
    if weight_tensor.rank != 1:
        raise ValueError(f"{op_name} expects rank-1 weight [hidden], got rank {weight_tensor.rank}")
    if weight_tensor.dynamic or not isinstance(weight_tensor.shape_spec[0], int):
        raise ValueError(f"{op_name} currently requires a static weight shape")
    if int(weight_tensor.shape[0]) != hidden:
        raise ValueError(
            f"{op_name} weight length must match the input hidden size: "
            f"got hidden={x_tensor.shape[-1]}, weight={weight_tensor.shape[0]}"
        )
    if extra_tensor is not None:
        if extra_tensor.dtype != x_tensor.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {x_tensor.dtype} vs {extra_tensor.dtype}")
        if extra_tensor.rank != 1:
            raise ValueError(f"{op_name} expects rank-1 bias [hidden], got rank {extra_tensor.rank}")
        if extra_tensor.dynamic or not isinstance(extra_tensor.shape_spec[0], int):
            raise ValueError(f"{op_name} currently requires a static bias shape")
        if int(extra_tensor.shape[0]) != hidden:
            raise ValueError(
                f"{op_name} bias length must match the input hidden size: "
                f"got hidden={x_tensor.shape[-1]}, bias={extra_tensor.shape[0]}"
            )
    return x_tensor, weight_tensor, extra_tensor


def t5_layer_norm(x: object, weight: object, eps: float = 1e-6) -> Tensor:
    x_tensor, weight_tensor, _ = _validate_affine_norm_inputs(
        "t5_layer_norm",
        x,
        weight,
        allowed_dtypes=T5_LAYER_NORM_DTYPES,
    )
    return x_tensor.builder.emit(
        "t5_layer_norm",
        [x_tensor, weight_tensor],
        x_tensor.shape,
        x_tensor.dtype,
        {"eps": float(eps)},
        shape_spec=x_tensor.shape_spec,
    )


def layer_norm(x: object, weight: object, bias: object, eps: float = 1e-5) -> Tensor:
    x_tensor, weight_tensor, bias_tensor = _validate_affine_norm_inputs(
        "layer_norm",
        x,
        weight,
        bias,
        allowed_dtypes=LAYER_NORM_DTYPES,
    )
    assert bias_tensor is not None
    return x_tensor.builder.emit(
        "layer_norm",
        [x_tensor, weight_tensor, bias_tensor],
        x_tensor.shape,
        x_tensor.dtype,
        {"eps": float(eps)},
        shape_spec=x_tensor.shape_spec,
    )
