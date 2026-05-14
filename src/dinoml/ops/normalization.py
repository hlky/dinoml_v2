from __future__ import annotations

from typing import Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


T5_LAYER_NORM_DTYPES = ("float16", "float32", "bfloat16")


def infer_t5_layer_norm(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 2:
        raise ValueError("t5_layer_norm expects exactly two inputs")
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


def t5_layer_norm(x: object, weight: object, eps: float = 1e-6) -> Tensor:
    x_tensor = as_tensor(x, dtype_hint="float32")
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    if weight_tensor.builder is not x_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if x_tensor.dtype != weight_tensor.dtype:
        raise ValueError(f"t5_layer_norm dtype mismatch: {x_tensor.dtype} vs {weight_tensor.dtype}")
    if x_tensor.dtype not in T5_LAYER_NORM_DTYPES:
        raise ValueError(f"t5_layer_norm does not support dtype {x_tensor.dtype}")
    if x_tensor.rank < 1:
        raise ValueError("t5_layer_norm requires rank >= 1 input")
    if weight_tensor.rank != 1:
        raise ValueError(f"t5_layer_norm expects rank-1 weight [hidden], got rank {weight_tensor.rank}")
    if not isinstance(x_tensor.shape_spec[-1], int):
        raise ValueError("t5_layer_norm currently requires a static last dimension")
    if x_tensor.shape[-1] <= 0:
        raise ValueError("t5_layer_norm last dimension must be positive")
    if weight_tensor.dynamic or not isinstance(weight_tensor.shape_spec[0], int):
        raise ValueError("t5_layer_norm currently requires a static weight shape")
    if int(weight_tensor.shape[0]) != int(x_tensor.shape[-1]):
        raise ValueError(
            "t5_layer_norm weight length must match the input hidden size: "
            f"got hidden={x_tensor.shape[-1]}, weight={weight_tensor.shape[0]}"
        )
    return x_tensor.builder.emit(
        "t5_layer_norm",
        [x_tensor, weight_tensor],
        x_tensor.shape,
        x_tensor.dtype,
        {"eps": float(eps)},
        shape_spec=x_tensor.shape_spec,
    )
