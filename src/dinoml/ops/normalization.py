from __future__ import annotations

from typing import Sequence

import numpy as np

from dinoml.frontend import Parameter, Tensor, as_tensor
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


NORMALIZATION_DTYPES = ("float16", "float32", "bfloat16")
T5_LAYER_NORM_DTYPES = NORMALIZATION_DTYPES
LAYER_NORM_DTYPES = NORMALIZATION_DTYPES
ADD_LAYER_NORM_DTYPES = NORMALIZATION_DTYPES
GROUP_NORM_DTYPES = NORMALIZATION_DTYPES


def infer_t5_layer_norm(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 2:
        raise ValueError("t5_layer_norm expects exactly two inputs")
    return list(shapes[0])


def infer_layer_norm(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 3:
        raise ValueError("layer_norm expects exactly three inputs")
    return list(shapes[0])


def infer_add_layer_norm(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 4:
        raise ValueError("add_layer_norm expects exactly four inputs")
    return list(shapes[0])


def infer_group_norm(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 3:
        raise ValueError("group_norm expects exactly three inputs")
    return list(shapes[0])


@op_def
class T5LayerNorm(OpDef):
    name = "t5_layer_norm"
    schema = OpSchema(inputs=("x", "weight"), attrs=(AttrDef("eps", "float", 1e-6),))
    infer_shape = infer_t5_layer_norm
    backend_kernels = {
        "cuda": KernelBinding("generated_t5_layer_norm", "model", source_template="t5_layer_norm_gpu"),
        "rocm": KernelBinding("generated_t5_layer_norm", "model", source_template="t5_layer_norm_gpu"),
        "cpu": KernelBinding("generated_t5_layer_norm", "model", source_template="t5_layer_norm_cpu"),
    }
    frontend = FrontendBinding("t5_layer_norm")
    allowed_dtypes = T5_LAYER_NORM_DTYPES
    description = (
        "Bounded T5/RMS-style layer normalization over the last static dimension with fp32 accumulation, "
        "rank >= 1 inputs, and a required affine weight shaped [hidden]."
    )

    @classmethod
    def forward(cls, x: object, weight: object, eps: float = 1e-6) -> Tensor:
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


@op_def
class LayerNorm(OpDef):
    name = "layer_norm"
    schema = OpSchema(inputs=("x", "weight", "bias"), attrs=(AttrDef("eps", "float", 1e-5),))
    infer_shape = infer_layer_norm
    backend_kernels = {
        "cuda": KernelBinding("generated_layer_norm", "model", source_template="layer_norm_gpu"),
        "rocm": KernelBinding("generated_layer_norm", "model", source_template="layer_norm_gpu"),
        "cpu": KernelBinding("generated_layer_norm", "model", source_template="layer_norm_cpu"),
    }
    frontend = FrontendBinding("layer_norm")
    allowed_dtypes = LAYER_NORM_DTYPES
    description = (
        "Bounded affine LayerNorm over the last static dimension with fp32 accumulation, "
        "rank >= 1 inputs, and required rank-1 affine weight/bias tensors shaped [hidden]."
    )

    @classmethod
    def forward(cls, x: object, weight: object, bias: object, eps: float = 1e-5) -> Tensor:
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


@op_def
class AddLayerNorm(OpDef):
    name = "add_layer_norm"
    schema = OpSchema(inputs=("x", "residual", "weight", "bias"), attrs=(AttrDef("eps", "float", 1e-5),))
    infer_shape = infer_add_layer_norm
    backend_kernels = {
        "cuda": KernelBinding("generated_add_layer_norm", "model", source_template="add_layer_norm_gpu"),
        "rocm": KernelBinding("generated_add_layer_norm", "model", source_template="add_layer_norm_gpu"),
        "cpu": KernelBinding("generated_add_layer_norm", "model", source_template="add_layer_norm_cpu"),
    }
    frontend = FrontendBinding("add_layer_norm")
    allowed_dtypes = ADD_LAYER_NORM_DTYPES
    description = "Fused residual add plus affine LayerNorm; returns both the summed residual and normalized output."

    @classmethod
    def forward(cls, x: object, residual: object, weight: object, bias: object, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
        x_tensor, weight_tensor, bias_tensor = _validate_affine_norm_inputs(
            "add_layer_norm",
            x,
            weight,
            bias,
            allowed_dtypes=ADD_LAYER_NORM_DTYPES,
        )
        residual_tensor = as_tensor(residual, dtype_hint=x_tensor.dtype)
        if residual_tensor.builder is not x_tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if residual_tensor.dtype != x_tensor.dtype:
            raise ValueError(f"add_layer_norm dtype mismatch: {x_tensor.dtype} vs {residual_tensor.dtype}")
        if list(residual_tensor.shape) != list(x_tensor.shape):
            raise ValueError(f"add_layer_norm residual shape must match input: {residual_tensor.shape} vs {x_tensor.shape}")
        if list(residual_tensor.shape_spec) != list(x_tensor.shape_spec):
            raise ValueError("add_layer_norm residual shape_spec must match input")
        assert bias_tensor is not None
        summed, normalized = x_tensor.builder.emit_multi(
            "add_layer_norm",
            [x_tensor, residual_tensor, weight_tensor, bias_tensor],
            (
                (x_tensor.shape, x_tensor.dtype, x_tensor.shape_spec),
                (x_tensor.shape, x_tensor.dtype, x_tensor.shape_spec),
            ),
            {"eps": float(eps)},
        )
        return summed, normalized


@op_def
class GroupNorm(OpDef):
    name = "group_norm"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(AttrDef("num_groups", "int", required=True), AttrDef("eps", "float", 1e-5)),
    )
    infer_shape = infer_group_norm
    backend_kernels = {
        "cuda": KernelBinding("generated_group_norm", "model", source_template="group_norm_gpu"),
        "rocm": KernelBinding("generated_group_norm", "model", source_template="group_norm_gpu"),
        "cpu": KernelBinding("generated_group_norm", "model", source_template="group_norm_cpu"),
    }
    frontend = FrontendBinding("group_norm")
    allowed_dtypes = GROUP_NORM_DTYPES
    description = (
        "Affine GroupNorm over the last channel dimension with NHWC-style channel-last layout and fp32 accumulation, "
        "supported on CPU, CUDA, and ROCm for static non-batch dimensions."
    )

    @classmethod
    def forward(
        cls,
        x: object,
        num_groups: int,
        weight: object | None = None,
        bias: object | None = None,
        eps: float = 1e-5,
    ) -> Tensor:
        x_tensor, weight_tensor, bias_tensor, validated_groups = _validate_group_norm_inputs(
            "group_norm",
            x,
            num_groups,
            weight,
            bias,
            allowed_dtypes=GROUP_NORM_DTYPES,
        )
        return x_tensor.builder.emit(
            "group_norm",
            [x_tensor, weight_tensor, bias_tensor],
            x_tensor.shape,
            x_tensor.dtype,
            {"num_groups": validated_groups, "eps": float(eps)},
            shape_spec=x_tensor.shape_spec,
        )


@op_def
class GroupNormSwish(OpDef):
    name = "group_norm_swish"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(AttrDef("num_groups", "int", required=True), AttrDef("eps", "float", 1e-5)),
    )
    infer_shape = infer_group_norm
    backend_kernels = {
        "cuda": KernelBinding("generated_group_norm_swish", "model", source_template="group_norm_gpu"),
        "rocm": KernelBinding("generated_group_norm_swish", "model", source_template="group_norm_gpu"),
        "cpu": KernelBinding("generated_group_norm_swish", "model", source_template="group_norm_cpu"),
    }
    frontend = FrontendBinding("group_norm_swish")
    allowed_dtypes = GROUP_NORM_DTYPES
    description = (
        "Fused GroupNorm followed by swish/silu over the normalized output, supported on CPU, CUDA, and ROCm "
        "for static non-batch dimensions."
    )

    @classmethod
    def forward(
        cls,
        x: object,
        num_groups: int,
        weight: object | None = None,
        bias: object | None = None,
        eps: float = 1e-5,
    ) -> Tensor:
        x_tensor, weight_tensor, bias_tensor, validated_groups = _validate_group_norm_inputs(
            "group_norm_swish",
            x,
            num_groups,
            weight,
            bias,
            allowed_dtypes=GROUP_NORM_DTYPES,
        )
        return x_tensor.builder.emit(
            "group_norm_swish",
            [x_tensor, weight_tensor, bias_tensor],
            x_tensor.shape,
            x_tensor.dtype,
            {"num_groups": validated_groups, "eps": float(eps)},
            shape_spec=x_tensor.shape_spec,
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


def _validate_group_norm_inputs(
    op_name: str,
    x: object,
    num_groups: int,
    weight: object | None,
    bias: object | None,
    *,
    allowed_dtypes: Sequence[str],
) -> tuple[Tensor, Tensor, Tensor, int]:
    if not isinstance(num_groups, int) or isinstance(num_groups, bool):
        raise TypeError(f"{op_name} num_groups must be an integer, got {type(num_groups).__name__}")
    validated_groups = int(num_groups)
    if validated_groups <= 0:
        raise ValueError(f"{op_name} num_groups must be positive")

    x_tensor = as_tensor(x, dtype_hint="float32")
    if x_tensor.dtype not in allowed_dtypes:
        raise ValueError(f"{op_name} does not support dtype {x_tensor.dtype}")
    if x_tensor.rank < 2:
        raise ValueError(f"{op_name} requires rank >= 2 input")
    if not isinstance(x_tensor.shape_spec[-1], int):
        raise ValueError(f"{op_name} currently requires a static last dimension")
    channels = int(x_tensor.shape[-1])
    if channels <= 0:
        raise ValueError(f"{op_name} last dimension must be positive")
    if channels % validated_groups != 0:
        raise ValueError(f"{op_name} channel dimension {channels} must be divisible by num_groups {validated_groups}")

    if weight is None:
        weight = Parameter([channels], dtype=x_tensor.dtype, value=np.ones((channels,), dtype=np.float32))
    if bias is None:
        bias = Parameter([channels], dtype=x_tensor.dtype, value=np.zeros((channels,), dtype=np.float32))
    validated = _validate_affine_norm_inputs(op_name, x_tensor, weight, bias, allowed_dtypes=allowed_dtypes)
    x_validated, weight_validated, bias_validated = validated
    assert bias_validated is not None
    return x_validated, weight_validated, bias_validated, validated_groups


def t5_layer_norm(x: object, weight: object, eps: float = 1e-6) -> Tensor:
    return T5LayerNorm.forward(x, weight, eps)


def layer_norm(x: object, weight: object, bias: object, eps: float = 1e-5) -> Tensor:
    return LayerNorm.forward(x, weight, bias, eps)


def add_layer_norm(x: object, residual: object, weight: object, bias: object, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    return AddLayerNorm.forward(x, residual, weight, bias, eps)


def group_norm(
    x: object,
    num_groups: int,
    weight: object | None = None,
    bias: object | None = None,
    eps: float = 1e-5,
) -> Tensor:
    return GroupNorm.forward(x, num_groups, weight, bias, eps)


def group_norm_swish(
    x: object,
    num_groups: int,
    weight: object | None = None,
    bias: object | None = None,
    eps: float = 1e-5,
) -> Tensor:
    return GroupNormSwish.forward(x, num_groups, weight, bias, eps)


def rms_norm(x: object, weight: object | None = None, eps: float = 1e-6) -> Tensor:
    x_tensor = as_tensor(x, dtype_hint="float32")
    if x_tensor.dtype not in T5_LAYER_NORM_DTYPES:
        raise ValueError(f"rms_norm does not support dtype {x_tensor.dtype}")
    if x_tensor.rank < 1:
        raise ValueError("rms_norm requires rank >= 1 input")
    if not isinstance(x_tensor.shape_spec[-1], int):
        raise ValueError("rms_norm currently requires a static last dimension")
    hidden = int(x_tensor.shape[-1])
    if hidden <= 0:
        raise ValueError("rms_norm last dimension must be positive")
    if weight is None:
        weight = Parameter([hidden], dtype=x_tensor.dtype, value=np.ones((hidden,), dtype=np.float32))
    return T5LayerNorm.forward(x_tensor, weight, eps)


__all__ = [
    "ADD_LAYER_NORM_DTYPES",
    "AddLayerNorm",
    "GROUP_NORM_DTYPES",
    "GroupNorm",
    "GroupNormSwish",
    "LAYER_NORM_DTYPES",
    "LayerNorm",
    "NORMALIZATION_DTYPES",
    "T5_LAYER_NORM_DTYPES",
    "T5LayerNorm",
    "add_layer_norm",
    "group_norm",
    "group_norm_swish",
    "infer_group_norm",
    "infer_add_layer_norm",
    "infer_layer_norm",
    "infer_t5_layer_norm",
    "layer_norm",
    "rms_norm",
    "t5_layer_norm",
]
