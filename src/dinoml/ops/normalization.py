from __future__ import annotations

from typing import Any, Sequence

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


def infer_layernorm_sigmoid_mul(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 3:
        raise ValueError("layernorm_sigmoid_mul expects exactly three inputs")
    return list(shapes[0])


def infer_batch_layernorm_sigmoid_mul(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 3:
        raise ValueError("batch_layernorm_sigmoid_mul expects exactly three inputs")
    return list(shapes[0])


def infer_group_layernorm(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) < 3 or len(shapes) % 3 != 0:
        raise ValueError("group_layernorm expects flattened [inputs, weights, biases] triples")
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
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(AttrDef("normalized_shape", "list[int]"), AttrDef("eps", "float", 1e-5)),
    )
    infer_shape = infer_layer_norm
    backend_kernels = {
        "cuda": KernelBinding("generated_layer_norm", "model", source_template="layer_norm_gpu"),
        "rocm": KernelBinding("generated_layer_norm", "model", source_template="layer_norm_gpu"),
        "cpu": KernelBinding("generated_layer_norm", "model", source_template="layer_norm_cpu"),
    }
    frontend = FrontendBinding("layer_norm")
    allowed_dtypes = LAYER_NORM_DTYPES
    description = (
        "Bounded affine LayerNorm over a trailing normalized-shape suffix with fp32 accumulation, "
        "rank >= 1 inputs, and required affine weight/bias tensors matching normalized_shape."
    )

    @classmethod
    def forward(
        cls,
        x: object,
        weight: object,
        bias: object,
        eps: float = 1e-5,
        normalized_shape: Sequence[int] | None = None,
    ) -> Tensor:
        x_tensor, weight_tensor, bias_tensor, norm_shape = _validate_layer_norm_inputs(
            "layer_norm",
            x,
            weight,
            bias,
            normalized_shape,
            allowed_dtypes=LAYER_NORM_DTYPES,
        )
        return x_tensor.builder.emit(
            "layer_norm",
            [x_tensor, weight_tensor, bias_tensor],
            x_tensor.shape,
            x_tensor.dtype,
            {"normalized_shape": norm_shape, "eps": float(eps)},
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


@op_def
class LayernormSigmoidMul(OpDef):
    name = "layernorm_sigmoid_mul"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(AttrDef("normalized_shape", "list[int]", required=True), AttrDef("eps", "float", 1e-5)),
    )
    infer_shape = infer_layernorm_sigmoid_mul
    backend_kernels = {
        "cuda": KernelBinding("generated_layernorm_sigmoid_mul", "model", source_template="layernorm_sigmoid_mul_gpu"),
        "rocm": KernelBinding("generated_layernorm_sigmoid_mul", "model", source_template="layernorm_sigmoid_mul_gpu"),
        "cpu": KernelBinding("generated_layernorm_sigmoid_mul", "model", source_template="layernorm_sigmoid_mul_cpu"),
    }
    frontend = FrontendBinding("layernorm_sigmoid_mul")
    allowed_dtypes = LAYER_NORM_DTYPES
    description = (
        "Fused affine LayerNorm over a trailing normalized-shape suffix, followed by sigmoid and multiply by the "
        "original input."
    )


@op_def
class BatchLayernormSigmoidMul(OpDef):
    name = "batch_layernorm_sigmoid_mul"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(AttrDef("normalized_shape", "list[int]", required=True), AttrDef("eps", "float", 1e-5)),
    )
    infer_shape = infer_batch_layernorm_sigmoid_mul
    backend_kernels = {
        "cuda": KernelBinding(
            "generated_batch_layernorm_sigmoid_mul",
            "model",
            source_template="batch_layernorm_sigmoid_mul_gpu",
        ),
        "rocm": KernelBinding(
            "generated_batch_layernorm_sigmoid_mul",
            "model",
            source_template="batch_layernorm_sigmoid_mul_gpu",
        ),
        "cpu": KernelBinding(
            "generated_batch_layernorm_sigmoid_mul",
            "model",
            source_template="batch_layernorm_sigmoid_mul_cpu",
        ),
    }
    frontend = FrontendBinding("batch_layernorm_sigmoid_mul")
    allowed_dtypes = LAYER_NORM_DTYPES
    description = (
        "Rank-3 fused LayerNorm + sigmoid * input with per-batch affine tensors shaped [batch, hidden]."
    )


@op_def
class GroupLayernorm(OpDef):
    name = "group_layernorm"
    schema = OpSchema(
        inputs=("x0", "weight0", "bias0"),
        attrs=(AttrDef("group_count", "int", required=True), AttrDef("normalized_shapes", "list[list[int]]", required=True), AttrDef("eps", "float", 1e-5)),
    )
    infer_shape = infer_group_layernorm
    backend_kernels = {
        "cuda": KernelBinding("generated_group_layernorm", "model", source_template="group_layernorm_gpu"),
        "rocm": KernelBinding("generated_group_layernorm", "model", source_template="group_layernorm_gpu"),
        "cpu": KernelBinding("generated_group_layernorm", "model", source_template="group_layernorm_cpu"),
    }
    frontend = FrontendBinding("group_layernorm")
    variadic_inputs = True
    allowed_dtypes = LAYER_NORM_DTYPES
    description = (
        "Grouped affine LayerNorm over per-input trailing normalized-shape suffixes; each group shares the same "
        "leading batch dimensions."
    )


@op_def
class GroupLayernormSigmoidMul(OpDef):
    name = "group_layernorm_sigmoid_mul"
    schema = OpSchema(
        inputs=("x0", "weight0", "bias0"),
        attrs=(AttrDef("group_count", "int", required=True), AttrDef("normalized_shapes", "list[list[int]]", required=True), AttrDef("eps", "float", 1e-5)),
    )
    infer_shape = infer_group_layernorm
    backend_kernels = {
        "cuda": KernelBinding(
            "generated_group_layernorm_sigmoid_mul",
            "model",
            source_template="group_layernorm_gpu",
        ),
        "rocm": KernelBinding(
            "generated_group_layernorm_sigmoid_mul",
            "model",
            source_template="group_layernorm_gpu",
        ),
        "cpu": KernelBinding(
            "generated_group_layernorm_sigmoid_mul",
            "model",
            source_template="group_layernorm_cpu",
        ),
    }
    frontend = FrontendBinding("group_layernorm_sigmoid_mul")
    variadic_inputs = True
    allowed_dtypes = LAYER_NORM_DTYPES
    description = "Grouped affine LayerNorm followed by sigmoid and multiply by each original group input."


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


def _validate_layer_norm_inputs(
    op_name: str,
    x: object,
    weight: object,
    bias: object,
    normalized_shape: object | None,
    *,
    allowed_dtypes: Sequence[str],
) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    x_tensor = as_tensor(x, dtype_hint="float32")
    if x_tensor.dtype not in allowed_dtypes:
        raise ValueError(f"{op_name} does not support dtype {x_tensor.dtype}")
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    bias_tensor = as_tensor(bias, dtype_hint=x_tensor.dtype)
    for tensor in (weight_tensor, bias_tensor):
        if tensor.builder is not x_tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != x_tensor.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {x_tensor.dtype} vs {tensor.dtype}")
        if tensor.dynamic:
            raise ValueError(f"{op_name} currently requires static affine tensor shapes")
    norm_shape = _resolve_normalized_shape(op_name, x_tensor, weight_tensor, bias_tensor, normalized_shape)
    norm_rank = len(norm_shape)
    if x_tensor.rank < norm_rank:
        raise ValueError(f"{op_name} input rank {x_tensor.rank} must be at least len(normalized_shape)={norm_rank}")
    if list(x_tensor.shape[-norm_rank:]) != norm_shape:
        raise ValueError(
            f"{op_name} normalized_shape must match the input trailing dimensions: "
            f"got normalized_shape={norm_shape}, input_suffix={list(x_tensor.shape[-norm_rank:])}"
        )
    if any(not isinstance(dim, int) for dim in x_tensor.shape_spec[-norm_rank:]):
        raise ValueError(f"{op_name} currently requires a static normalized_shape suffix")
    if list(weight_tensor.shape) != norm_shape:
        raise ValueError(f"{op_name} weight shape must equal normalized_shape {norm_shape}, got {weight_tensor.shape}")
    if list(bias_tensor.shape) != norm_shape:
        raise ValueError(f"{op_name} bias shape must equal normalized_shape {norm_shape}, got {bias_tensor.shape}")
    return x_tensor, weight_tensor, bias_tensor, norm_shape


def _copy_shape_dim(dim: Any) -> Any:
    return dict(dim) if isinstance(dim, dict) else dim


def _normalize_static_shape_attr(op_name: str, normalized_shape: object) -> list[int]:
    if not isinstance(normalized_shape, Sequence) or isinstance(normalized_shape, (str, bytes, bytearray)):
        raise TypeError(f"{op_name} normalized_shape must be a sequence of positive integers")
    normalized: list[int] = []
    for dim in normalized_shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError(f"{op_name} normalized_shape must contain only positive integers, got {dim!r}")
        value = int(dim)
        if value <= 0:
            raise ValueError(f"{op_name} normalized_shape must contain only positive integers, got {value}")
        normalized.append(value)
    if not normalized:
        raise ValueError(f"{op_name} normalized_shape must be non-empty")
    return normalized


def _default_affine_parameter(shape: Sequence[int], dtype: str, fill: float) -> Parameter:
    value = np.full(tuple(int(dim) for dim in shape), fill, dtype=np.float32)
    return Parameter(list(shape), dtype=dtype, value=value)


def _resolve_normalized_shape(
    op_name: str,
    x_tensor: Tensor,
    weight: object | None,
    bias: object | None,
    normalized_shape: object | None,
) -> list[int]:
    if normalized_shape is not None:
        return _normalize_static_shape_attr(op_name, normalized_shape)
    for candidate in (weight, bias):
        if candidate is None:
            continue
        candidate_tensor = as_tensor(candidate, dtype_hint=x_tensor.dtype)
        return _normalize_static_shape_attr(op_name, candidate_tensor.shape)
    last_dim = x_tensor.shape_spec[-1]
    if not isinstance(last_dim, int):
        raise ValueError(f"{op_name} requires an explicit static normalized_shape when the input last dimension is dynamic")
    return [int(last_dim)]


def _validate_layernorm_sigmoid_mul_inputs(
    op_name: str,
    x: object,
    weight: object | None,
    bias: object | None,
    normalized_shape: object | None,
    *,
    allowed_dtypes: Sequence[str],
) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    x_tensor = as_tensor(x, dtype_hint="float32")
    if x_tensor.dtype not in allowed_dtypes:
        raise ValueError(f"{op_name} does not support dtype {x_tensor.dtype}")
    norm_shape = _resolve_normalized_shape(op_name, x_tensor, weight, bias, normalized_shape)
    norm_rank = len(norm_shape)
    if x_tensor.rank < norm_rank:
        raise ValueError(f"{op_name} input rank {x_tensor.rank} must be at least len(normalized_shape)={norm_rank}")
    suffix_shape = list(x_tensor.shape[-norm_rank:])
    if suffix_shape != norm_shape:
        raise ValueError(
            f"{op_name} normalized_shape must match the input trailing dimensions: "
            f"got normalized_shape={norm_shape}, input_suffix={suffix_shape}"
        )
    suffix_shape_spec = x_tensor.shape_spec[-norm_rank:]
    if any(not isinstance(dim, int) for dim in suffix_shape_spec):
        raise ValueError(f"{op_name} currently requires a static normalized_shape suffix")
    if weight is None:
        weight = _default_affine_parameter(norm_shape, x_tensor.dtype, 1.0)
    if bias is None:
        bias = _default_affine_parameter(norm_shape, x_tensor.dtype, 0.0)
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    bias_tensor = as_tensor(bias, dtype_hint=x_tensor.dtype)
    for tensor in (weight_tensor, bias_tensor):
        if tensor.builder is not x_tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != x_tensor.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {x_tensor.dtype} vs {tensor.dtype}")
        if tensor.dynamic:
            raise ValueError(f"{op_name} currently requires static affine tensor shapes")
    if list(weight_tensor.shape) != norm_shape:
        raise ValueError(f"{op_name} weight shape must equal normalized_shape {norm_shape}, got {weight_tensor.shape}")
    if list(bias_tensor.shape) != norm_shape:
        raise ValueError(f"{op_name} bias shape must equal normalized_shape {norm_shape}, got {bias_tensor.shape}")
    return x_tensor, weight_tensor, bias_tensor, norm_shape


def _validate_batch_layernorm_sigmoid_mul_inputs(
    x: object,
    weight: object | None,
    bias: object | None,
    normalized_shape: object | None,
    *,
    allowed_dtypes: Sequence[str],
) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    op_name = "batch_layernorm_sigmoid_mul"
    x_tensor = as_tensor(x, dtype_hint="float32")
    if x_tensor.dtype not in allowed_dtypes:
        raise ValueError(f"{op_name} does not support dtype {x_tensor.dtype}")
    if x_tensor.rank != 3:
        raise ValueError(f"{op_name} expects rank-3 input [batch, rows, hidden], got rank {x_tensor.rank}")
    norm_shape = _resolve_normalized_shape(op_name, x_tensor, weight, bias, normalized_shape)
    if len(norm_shape) != 1:
        raise ValueError(f"{op_name} normalized_shape must be rank-1, got {norm_shape}")
    hidden = int(norm_shape[0])
    if int(x_tensor.shape[-1]) != hidden:
        raise ValueError(
            f"{op_name} normalized_shape must match the input hidden size: "
            f"got normalized_shape={hidden}, input_hidden={x_tensor.shape[-1]}"
        )
    batch_dim = x_tensor.shape_spec[0]
    if not isinstance(x_tensor.shape_spec[-1], int):
        raise ValueError(f"{op_name} currently requires a static hidden dimension")
    if weight is None:
        if not isinstance(batch_dim, int):
            raise ValueError(f"{op_name} requires an explicit weight when the batch dimension is dynamic")
        weight = _default_affine_parameter([int(batch_dim), hidden], x_tensor.dtype, 1.0)
    if bias is None:
        if not isinstance(batch_dim, int):
            raise ValueError(f"{op_name} requires an explicit bias when the batch dimension is dynamic")
        bias = _default_affine_parameter([int(batch_dim), hidden], x_tensor.dtype, 0.0)
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    bias_tensor = as_tensor(bias, dtype_hint=x_tensor.dtype)
    for tensor in (weight_tensor, bias_tensor):
        if tensor.builder is not x_tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != x_tensor.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {x_tensor.dtype} vs {tensor.dtype}")
        if tensor.rank != 2:
            raise ValueError(f"{op_name} affine tensors must have shape [batch, hidden], got rank {tensor.rank}")
        if tensor.dynamic:
            raise ValueError(f"{op_name} currently requires static affine tensor shapes")
    if int(weight_tensor.shape[1]) != hidden or int(bias_tensor.shape[1]) != hidden:
        raise ValueError(f"{op_name} affine tensor hidden dimension must equal {hidden}")
    if weight_tensor.shape[0] != x_tensor.shape[0]:
        raise ValueError(
            f"{op_name} weight batch dimension must match the input batch size: "
            f"got {weight_tensor.shape[0]} vs {x_tensor.shape[0]}"
        )
    if bias_tensor.shape[0] != x_tensor.shape[0]:
        raise ValueError(
            f"{op_name} bias batch dimension must match the input batch size: "
            f"got {bias_tensor.shape[0]} vs {x_tensor.shape[0]}"
        )
    return x_tensor, weight_tensor, bias_tensor, norm_shape


def _normalize_group_affine_sequence(
    op_name: str,
    name: str,
    values: Sequence[object | None] | None,
    group_count: int,
) -> list[object | None]:
    if values is None:
        return [None] * group_count
    if isinstance(values, (Tensor, Parameter)) or not isinstance(values, (list, tuple)):
        raise TypeError(f"{op_name} {name} must be a sequence matching the number of inputs")
    if len(values) != group_count:
        raise ValueError(f"{op_name} {name} length must match the number of inputs: {len(values)} vs {group_count}")
    return list(values)


def _normalize_group_normalized_shapes(
    op_name: str,
    tensors: Sequence[Tensor],
    weights: Sequence[object | None],
    biases: Sequence[object | None],
    normalized_shapes: Sequence[Sequence[int]] | None,
) -> list[list[int]]:
    if normalized_shapes is not None:
        if len(normalized_shapes) != len(tensors):
            raise ValueError(
                f"{op_name} normalized_shapes length must match the number of inputs: "
                f"{len(normalized_shapes)} vs {len(tensors)}"
            )
        return [_normalize_static_shape_attr(op_name, shape) for shape in normalized_shapes]
    resolved: list[list[int]] = []
    for tensor, weight, bias in zip(tensors, weights, biases):
        resolved.append(_resolve_normalized_shape(op_name, tensor, weight, bias, None))
    return resolved


def _validate_group_layernorm_inputs(
    op_name: str,
    inputs: Sequence[object],
    weights: Sequence[object | None] | None,
    biases: Sequence[object | None] | None,
    normalized_shapes: Sequence[Sequence[int]] | None,
    *,
    allowed_dtypes: Sequence[str],
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[list[int]]]:
    if isinstance(inputs, (Tensor, Parameter)) or not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError(f"{op_name} expects a non-empty sequence of input tensors")
    tensors = [as_tensor(inputs[0], dtype_hint="float32")]
    tensors.extend(as_tensor(value, dtype_hint=tensors[0].dtype) for value in inputs[1:])
    first = tensors[0]
    if first.dtype not in allowed_dtypes:
        raise ValueError(f"{op_name} does not support dtype {first.dtype}")
    for tensor in tensors[1:]:
        if tensor.builder is not first.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != first.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {first.dtype} vs {tensor.dtype}")
    weight_values = _normalize_group_affine_sequence(op_name, "weights", weights, len(tensors))
    bias_values = _normalize_group_affine_sequence(op_name, "biases", biases, len(tensors))
    norm_shapes = _normalize_group_normalized_shapes(op_name, tensors, weight_values, bias_values, normalized_shapes)
    batch_prefix_spec: list[Any] | None = None
    validated_weights: list[Tensor] = []
    validated_biases: list[Tensor] = []
    for tensor, weight, bias, norm_shape in zip(tensors, weight_values, bias_values, norm_shapes):
        if tensor.rank < len(norm_shape):
            raise ValueError(
                f"{op_name} input rank {tensor.rank} must be at least len(normalized_shape)={len(norm_shape)}"
            )
        if list(tensor.shape[-len(norm_shape) :]) != norm_shape:
            raise ValueError(
                f"{op_name} input trailing dimensions must match normalized_shape: "
                f"got input={tensor.shape}, normalized_shape={norm_shape}"
            )
        if any(not isinstance(dim, int) for dim in tensor.shape_spec[-len(norm_shape) :]):
            raise ValueError(f"{op_name} currently requires static normalized_shape suffixes")
        batch_prefix = tensor.shape_spec[: tensor.rank - len(norm_shape)]
        if batch_prefix_spec is None:
            batch_prefix_spec = [_copy_shape_dim(dim) for dim in batch_prefix]
        elif list(batch_prefix) != list(batch_prefix_spec):
            raise ValueError(f"{op_name} inputs must share the same leading batch dimensions")
        if weight is None:
            weight = _default_affine_parameter(norm_shape, tensor.dtype, 1.0)
        if bias is None:
            bias = _default_affine_parameter(norm_shape, tensor.dtype, 0.0)
        weight_tensor = as_tensor(weight, dtype_hint=tensor.dtype)
        bias_tensor = as_tensor(bias, dtype_hint=tensor.dtype)
        for affine_tensor in (weight_tensor, bias_tensor):
            if affine_tensor.builder is not first.builder:
                raise ValueError("Cannot combine tensors from different DinoML traces")
            if affine_tensor.dtype != tensor.dtype:
                raise ValueError(f"{op_name} dtype mismatch: {tensor.dtype} vs {affine_tensor.dtype}")
            if affine_tensor.dynamic:
                raise ValueError(f"{op_name} currently requires static affine tensor shapes")
        if list(weight_tensor.shape) != norm_shape:
            raise ValueError(f"{op_name} weight shape must equal normalized_shape {norm_shape}, got {weight_tensor.shape}")
        if list(bias_tensor.shape) != norm_shape:
            raise ValueError(f"{op_name} bias shape must equal normalized_shape {norm_shape}, got {bias_tensor.shape}")
        validated_weights.append(weight_tensor)
        validated_biases.append(bias_tensor)
    return tensors, validated_weights, validated_biases, norm_shapes


def t5_layer_norm(x: object, weight: object, eps: float = 1e-6) -> Tensor:
    return T5LayerNorm.forward(x, weight, eps)


def layer_norm(
    x: object,
    weight: object,
    bias: object,
    eps: float = 1e-5,
    normalized_shape: Sequence[int] | None = None,
) -> Tensor:
    return LayerNorm.forward(x, weight, bias, eps, normalized_shape)


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


def layernorm_sigmoid_mul(
    x: object,
    weight: object | None = None,
    bias: object | None = None,
    normalized_shape: Sequence[int] | None = None,
    eps: float = 1e-5,
) -> Tensor:
    x_tensor, weight_tensor, bias_tensor, norm_shape = _validate_layernorm_sigmoid_mul_inputs(
        "layernorm_sigmoid_mul",
        x,
        weight,
        bias,
        normalized_shape,
        allowed_dtypes=LAYER_NORM_DTYPES,
    )
    return x_tensor.builder.emit(
        "layernorm_sigmoid_mul",
        [x_tensor, weight_tensor, bias_tensor],
        x_tensor.shape,
        x_tensor.dtype,
        {"normalized_shape": norm_shape, "eps": float(eps)},
        shape_spec=x_tensor.shape_spec,
    )


def batch_layernorm_sigmoid_mul(
    x: object,
    weight: object | None = None,
    bias: object | None = None,
    normalized_shape: Sequence[int] | None = None,
    eps: float = 1e-5,
) -> Tensor:
    x_tensor, weight_tensor, bias_tensor, norm_shape = _validate_batch_layernorm_sigmoid_mul_inputs(
        x,
        weight,
        bias,
        normalized_shape,
        allowed_dtypes=LAYER_NORM_DTYPES,
    )
    return x_tensor.builder.emit(
        "batch_layernorm_sigmoid_mul",
        [x_tensor, weight_tensor, bias_tensor],
        x_tensor.shape,
        x_tensor.dtype,
        {"normalized_shape": norm_shape, "eps": float(eps)},
        shape_spec=x_tensor.shape_spec,
    )


def group_layernorm(
    inputs: Sequence[object],
    weights: Sequence[object | None] | None = None,
    biases: Sequence[object | None] | None = None,
    normalized_shapes: Sequence[Sequence[int]] | None = None,
    eps: float = 1e-5,
) -> tuple[Tensor, ...]:
    tensors, weight_tensors, bias_tensors, norm_shapes = _validate_group_layernorm_inputs(
        "group_layernorm",
        inputs,
        weights,
        biases,
        normalized_shapes,
        allowed_dtypes=LAYER_NORM_DTYPES,
    )
    outputs = tuple((tensor.shape, tensor.dtype, tensor.shape_spec) for tensor in tensors)
    return tensors[0].builder.emit_multi(
        "group_layernorm",
        [*tensors, *weight_tensors, *bias_tensors],
        outputs,
        {"group_count": len(tensors), "normalized_shapes": norm_shapes, "eps": float(eps)},
    )


def group_layernorm_sigmoid_mul(
    inputs: Sequence[object],
    weights: Sequence[object | None] | None = None,
    biases: Sequence[object | None] | None = None,
    normalized_shapes: Sequence[Sequence[int]] | None = None,
    eps: float = 1e-5,
) -> tuple[Tensor, ...]:
    tensors, weight_tensors, bias_tensors, norm_shapes = _validate_group_layernorm_inputs(
        "group_layernorm_sigmoid_mul",
        inputs,
        weights,
        biases,
        normalized_shapes,
        allowed_dtypes=LAYER_NORM_DTYPES,
    )
    outputs = tuple((tensor.shape, tensor.dtype, tensor.shape_spec) for tensor in tensors)
    return tensors[0].builder.emit_multi(
        "group_layernorm_sigmoid_mul",
        [*tensors, *weight_tensors, *bias_tensors],
        outputs,
        {"group_count": len(tensors), "normalized_shapes": norm_shapes, "eps": float(eps)},
    )


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
    "GroupLayernorm",
    "GroupLayernormSigmoidMul",
    "GroupNorm",
    "GroupNormSwish",
    "LAYER_NORM_DTYPES",
    "BatchLayernormSigmoidMul",
    "LayerNorm",
    "LayernormSigmoidMul",
    "NORMALIZATION_DTYPES",
    "T5_LAYER_NORM_DTYPES",
    "T5LayerNorm",
    "add_layer_norm",
    "batch_layernorm_sigmoid_mul",
    "group_layernorm",
    "group_layernorm_sigmoid_mul",
    "group_norm",
    "group_norm_swish",
    "infer_batch_layernorm_sigmoid_mul",
    "infer_group_norm",
    "infer_group_layernorm",
    "infer_add_layer_norm",
    "infer_layer_norm",
    "infer_layernorm_sigmoid_mul",
    "infer_t5_layer_norm",
    "layernorm_sigmoid_mul",
    "layer_norm",
    "rms_norm",
    "t5_layer_norm",
]
