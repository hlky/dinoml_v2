from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dinoml.frontend import Parameter, Tensor, as_tensor
from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_candidate_set,
    cutlass_conv_candidates,
    cutlass_conv_profiler_symbol,
    cutlass_conv_symbol,
)
from dinoml.kernels.providers.ck.conv import (
    CK_CONV_OPS,
    ck_conv_candidate_set,
    ck_conv_candidates,
    ck_conv_profiler_symbol,
    ck_conv_symbol,
)
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, KernelVariant, OpDef, OpSchema, op_def


CONV2D_BIAS_DTYPES = ("float16", "float32")
CONV2D_BIAS_FAMILY_OPS = ("conv2d_bias", "conv2d_bias_relu", "conv2d_bias_add", "conv2d_bias_add_relu")


def infer_conv2d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv2d_shape_with_attrs(
        input_shapes,
        {"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
    )


def infer_conv2d_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 2:
        raise ValueError("conv2d expects activation and weight inputs")
    stride, padding, dilation, groups = normalize_conv2d_bias_attrs(
        attrs.get("stride", (1, 1)),
        attrs.get("padding", (0, 0)),
        attrs.get("dilation", (1, 1)),
        attrs.get("groups", 1),
    )
    return resolve_conv2d_shape(
        input_shapes[0],
        input_shapes[1],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def infer_conv2d_bias_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv2d_bias_shape_with_attrs(
        input_shapes,
        {"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
    )


def infer_conv2d_bias_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    return _infer_conv2d_bias_family_shape_with_attrs("conv2d_bias", input_shapes, attrs)


def infer_conv2d_bias_relu_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv2d_bias_relu_shape_with_attrs(
        input_shapes,
        {"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
    )


def infer_conv2d_bias_relu_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    return _infer_conv2d_bias_family_shape_with_attrs("conv2d_bias_relu", input_shapes, attrs)


def infer_conv2d_bias_add_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv2d_bias_add_shape_with_attrs(
        input_shapes,
        {"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
    )


def infer_conv2d_bias_add_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    return _infer_conv2d_bias_family_shape_with_attrs("conv2d_bias_add", input_shapes, attrs)


def infer_conv2d_bias_add_relu_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv2d_bias_add_relu_shape_with_attrs(
        input_shapes,
        {"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
    )


def infer_conv2d_bias_add_relu_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    return _infer_conv2d_bias_family_shape_with_attrs("conv2d_bias_add_relu", input_shapes, attrs)


def _infer_conv2d_bias_family_shape_with_attrs(
    op_name: str,
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    _validate_conv2d_bias_family_op_name(op_name)
    expected_inputs = 4 if _conv2d_bias_family_has_residual(op_name) else 3
    if len(input_shapes) != expected_inputs:
        extra = ", and residual" if expected_inputs == 4 else ""
        raise ValueError(f"{op_name} expects activation, weight, bias{extra} inputs")
    stride, padding, dilation, groups = normalize_conv2d_bias_attrs(
        attrs.get("stride", (1, 1)),
        attrs.get("padding", (0, 0)),
        attrs.get("dilation", (1, 1)),
        attrs.get("groups", 1),
    )
    return _resolve_conv2d_bias_family_shape(
        op_name,
        input_shapes[0],
        input_shapes[1],
        input_shapes[2],
        None if expected_inputs == 3 else input_shapes[3],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def normalize_conv2d_bias_attrs(
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> tuple[list[int], list[int], list[int], int]:
    normalized_stride = _normalize_positive_pair(stride, "conv2d_bias stride")
    normalized_padding = _normalize_non_negative_pair(padding, "conv2d_bias padding")
    normalized_dilation = _normalize_positive_pair(dilation, "conv2d_bias dilation")
    if not isinstance(groups, int) or isinstance(groups, bool):
        raise ValueError(f"conv2d_bias groups must be a non-bool integer, got {groups!r}")
    normalized_groups = int(groups)
    if normalized_groups <= 0:
        raise ValueError(f"conv2d_bias groups must be positive, got {groups!r}")
    if normalized_groups != 1:
        raise NotImplementedError(f"conv2d_bias currently supports groups=1 only, got {normalized_groups}")
    return list(normalized_stride), list(normalized_padding), list(normalized_dilation), normalized_groups


def resolve_conv2d_bias_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_conv2d_bias_family_shape(
        "conv2d_bias",
        input_shape,
        weight_shape,
        bias_shape,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def resolve_conv2d_bias_relu_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_conv2d_bias_family_shape(
        "conv2d_bias_relu",
        input_shape,
        weight_shape,
        bias_shape,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def resolve_conv2d_bias_add_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    residual_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_conv2d_bias_family_shape(
        "conv2d_bias_add",
        input_shape,
        weight_shape,
        bias_shape,
        residual_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def resolve_conv2d_bias_add_relu_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    residual_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_conv2d_bias_family_shape(
        "conv2d_bias_add_relu",
        input_shape,
        weight_shape,
        bias_shape,
        residual_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _resolve_conv2d_bias_family_shape(
    op_name: str,
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    residual_shape: Sequence[int] | None,
    *,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    _validate_conv2d_bias_family_op_name(op_name)
    if len(input_shape) != 4:
        raise ValueError(f"{op_name} expects rank-4 NCHW activation, got rank {len(input_shape)}")
    if len(weight_shape) != 4:
        raise ValueError(f"{op_name} expects rank-4 OIHW weight, got rank {len(weight_shape)}")
    if len(bias_shape) != 1:
        raise ValueError(f"{op_name} expects rank-1 bias, got rank {len(bias_shape)}")
    normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv2d_bias_attrs(
        stride,
        padding,
        dilation,
        groups,
    )
    if normalized_groups != 1:
        raise NotImplementedError(f"{op_name} currently supports groups=1 only, got {normalized_groups}")

    batch, in_channels, in_height, in_width = [int(dim) for dim in input_shape]
    out_channels, weight_in_channels, kernel_h, kernel_w = [int(dim) for dim in weight_shape]
    bias_channels = int(bias_shape[0])

    if weight_in_channels != in_channels:
        raise ValueError(
            f"{op_name} weight input channels must match activation channels for groups=1: "
            f"got activation C={in_channels}, weight C={weight_in_channels}"
        )
    if bias_channels != out_channels:
        raise ValueError(
            f"{op_name} bias length must match weight output channels, got bias {bias_channels} and weight O={out_channels}"
        )
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError(f"{op_name} kernel dimensions must be positive, got {weight_shape!r}")

    out_height = _conv_output_dim(
        op_name,
        in_height,
        kernel_h,
        normalized_stride[0],
        normalized_padding[0],
        normalized_dilation[0],
        "height",
    )
    out_width = _conv_output_dim(
        op_name,
        in_width,
        kernel_w,
        normalized_stride[1],
        normalized_padding[1],
        normalized_dilation[1],
        "width",
    )
    output_shape = [batch, out_channels, out_height, out_width]
    if _conv2d_bias_family_has_residual(op_name):
        if residual_shape is None:
            raise ValueError(f"{op_name} expects a residual input shape")
        if len(residual_shape) != 4:
            raise ValueError(f"{op_name} expects rank-4 residual, got rank {len(residual_shape)}")
        residual = [int(dim) for dim in residual_shape]
        if residual != output_shape:
            raise ValueError(
                f"{op_name} residual shape must match the conv output shape {output_shape}, got {residual}"
            )
    return output_shape


def resolve_conv2d_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    if len(input_shape) != 4:
        raise ValueError(f"conv2d expects rank-4 NCHW activation, got rank {len(input_shape)}")
    if len(weight_shape) != 4:
        raise ValueError(f"conv2d expects rank-4 OIHW weight, got rank {len(weight_shape)}")
    out_channels = int(weight_shape[0]) if weight_shape else 0
    return resolve_conv2d_bias_shape(
        input_shape,
        weight_shape,
        [out_channels],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _conv_output_dim(
    op_name: str,
    dim: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    axis_name: str,
) -> int:
    output = (int(dim) + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if output <= 0:
        raise ValueError(
            f"{op_name} output {axis_name} must be positive; got input={dim}, kernel={kernel}, "
            f"stride={stride}, padding={padding}, dilation={dilation}"
        )
    return output


def _normalize_positive_pair(value: Any, name: str) -> tuple[int, int]:
    pair = _normalize_pair(value, name)
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} must contain positive integers, got {value!r}")
    return pair


def _normalize_non_negative_pair(value: Any, name: str) -> tuple[int, int]:
    pair = _normalize_pair(value, name)
    if pair[0] < 0 or pair[1] < 0:
        raise ValueError(f"{name} must contain non-negative integers, got {value!r}")
    return pair


def _normalize_pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value), int(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"{name} must be an integer or pair of integers, got {value!r}")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
            raise ValueError(f"{name} must contain non-bool integers, got {value!r}")
        return int(values[0]), int(values[1])
    raise ValueError(f"{name} must be an integer or pair of integers, got {value!r}")


def _cutlass_conv_backend_kernels(op_name: str) -> dict[str, KernelBinding]:
    kernels = {
        "cpu": KernelBinding(
            symbol=f"generated_{op_name}",
            library="model",
            source_template="conv_cpu.cpp.j2",
        ),
        "cuda": KernelBinding(
            cutlass_conv_symbol(op_name, "float32"),
            "cutlass_conv",
            profiler_symbol=cutlass_conv_profiler_symbol(op_name, "float32"),
            dtype_variants={
                dtype: KernelVariant(
                    cutlass_conv_symbol(op_name, dtype),
                    profiler_symbol=cutlass_conv_profiler_symbol(op_name, dtype),
                    candidates=cutlass_conv_candidates(op_name, dtype),
                    candidate_set=cutlass_conv_candidate_set(op_name, dtype),
                )
                for dtype in CONV2D_BIAS_DTYPES
            },
        )
    }
    if op_name in CK_CONV_OPS:
        kernels["rocm"] = KernelBinding(
            ck_conv_symbol(op_name, "float32"),
            "ck_conv",
            profiler_symbol=ck_conv_profiler_symbol(op_name, "float32"),
            dtype_variants={
                dtype: KernelVariant(
                    ck_conv_symbol(op_name, dtype),
                    profiler_symbol=ck_conv_profiler_symbol(op_name, dtype),
                    candidates=ck_conv_candidates(op_name, dtype),
                    candidate_set=ck_conv_candidate_set(op_name, dtype),
                )
                for dtype in CONV2D_BIAS_DTYPES
            },
        )
    return kernels


def _conv2d_bias_family_forward(
    op_name: str,
    *,
    resolve_shape: Callable[..., list[int]],
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any | None = None,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
) -> Tensor:
    x_tensor = as_tensor(x)
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    bias_tensor = as_tensor(bias, dtype_hint=x_tensor.dtype)
    tensors = [x_tensor, weight_tensor, bias_tensor]
    residual_tensor = None if residual is None else as_tensor(residual, dtype_hint=x_tensor.dtype)
    if residual_tensor is not None:
        tensors.append(residual_tensor)
    for tensor in tensors[1:]:
        if tensor.builder is not x_tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != x_tensor.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {x_tensor.dtype} vs {tensor.dtype}")
    if x_tensor.dtype not in CONV2D_BIAS_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {x_tensor.dtype}")
    if x_tensor.rank != 4:
        raise ValueError(f"{op_name} expects rank-4 NCHW activation, got rank {x_tensor.rank}")
    if weight_tensor.rank != 4:
        raise ValueError(f"{op_name} expects rank-4 OIHW weight, got rank {weight_tensor.rank}")
    if bias_tensor.rank != 1:
        raise ValueError(f"{op_name} expects rank-1 bias, got rank {bias_tensor.rank}")
    if residual_tensor is not None and residual_tensor.rank != 4:
        raise ValueError(f"{op_name} expects rank-4 residual, got rank {residual_tensor.rank}")
    if any(tensor.dynamic for tensor in tensors):
        expected = "activation, weight, bias, and residual" if residual_tensor is not None else "activation, weight, and bias"
        raise ValueError(f"{op_name} currently supports only static {expected} shapes")
    normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv2d_bias_attrs(
        stride,
        padding,
        dilation,
        groups,
    )
    if residual_tensor is None:
        out_shape = resolve_shape(
            x_tensor.shape,
            weight_tensor.shape,
            bias_tensor.shape,
            stride=normalized_stride,
            padding=normalized_padding,
            dilation=normalized_dilation,
            groups=normalized_groups,
        )
    else:
        out_shape = resolve_shape(
            x_tensor.shape,
            weight_tensor.shape,
            bias_tensor.shape,
            residual_tensor.shape,
            stride=normalized_stride,
            padding=normalized_padding,
            dilation=normalized_dilation,
            groups=normalized_groups,
        )
    return x_tensor.builder.emit(
        op_name,
        tensors,
        out_shape,
        x_tensor.dtype,
        {
            "stride": normalized_stride,
            "padding": normalized_padding,
            "dilation": normalized_dilation,
            "groups": normalized_groups,
        },
        shape_spec=out_shape,
    )


@op_def
class Conv2dBias(OpDef):
    name = "conv2d_bias"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
            AttrDef("dilation", "ints", default=(1, 1)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_conv2d_bias_shape
    infer_shape_with_attrs = infer_conv2d_bias_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _cutlass_conv_backend_kernels("conv2d_bias")
    frontend = FrontendBinding("conv2d_bias")
    description = (
        "Bounded conv2d_bias frontend with public NCHW/OIHW semantics, "
        "groups=1 only, static rank-4 shapes, and CPU reference execution. "
        "Compiled CPU artifacts now also have a bounded generated naive "
        "runtime for the admitted float16/float32 contract. CUDA compile "
        "emits artifact-visible CUTLASS Conv pack/launch/unpack metadata, "
        "materializes the support boundary when possible, and runs "
        "correctness-first static groups=1 CUTLASS launchers for "
        "float32 SIMT and the bounded float16 SIMT/TensorOp candidates."
    )

    @classmethod
    def forward(
        cls,
        x: Any,
        weight: Any,
        bias: Any,
        stride: Any = 1,
        padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
    ) -> Tensor:
        return _conv2d_bias_family_forward(
            "conv2d_bias",
            resolve_shape=resolve_conv2d_bias_shape,
            x=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )


@op_def
class Conv2dBiasRelu(OpDef):
    name = "conv2d_bias_relu"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
            AttrDef("dilation", "ints", default=(1, 1)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_conv2d_bias_relu_shape
    infer_shape_with_attrs = infer_conv2d_bias_relu_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _cutlass_conv_backend_kernels("conv2d_bias_relu")
    frontend = FrontendBinding("conv2d_bias_relu")
    description = (
        "Bounded fused conv2d_bias_relu frontend sharing the public NCHW/OIHW, "
        "groups=1, static rank-4 Conv contract. CPU reference and compiled CPU "
        "artifacts apply the ReLU epilogue in the same generated Conv loop, while "
        "CUDA compile/profile/runtime keep the fused bias+ReLU CUTLASS Conv choice "
        "artifact-visible through the same manifest/profile/execution-plan path as "
        "`conv2d_bias`."
    )

    @classmethod
    def forward(
        cls,
        x: Any,
        weight: Any,
        bias: Any,
        stride: Any = 1,
        padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
    ) -> Tensor:
        return _conv2d_bias_family_forward(
            "conv2d_bias_relu",
            resolve_shape=resolve_conv2d_bias_relu_shape,
            x=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )


@op_def
class Conv2dBiasAdd(OpDef):
    name = "conv2d_bias_add"
    schema = OpSchema(
        inputs=("x", "weight", "bias", "residual"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
            AttrDef("dilation", "ints", default=(1, 1)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_conv2d_bias_add_shape
    infer_shape_with_attrs = infer_conv2d_bias_add_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _cutlass_conv_backend_kernels("conv2d_bias_add")
    frontend = FrontendBinding("conv2d_bias_add")
    description = (
        "Bounded fused conv2d_bias_add frontend for v1-style residual Conv: "
        "public tensors stay NCHW/OIHW, groups remain 1, all tensors are static rank-4, "
        "and the residual input must match the Conv output shape exactly. CPU reference "
        "and compiled CPU artifacts apply bias and residual add in one generated loop, "
        "while CUDA keeps the fused residual epilogue artifact-visible "
        "through the CUTLASS Conv manifest/profile/execution-plan path."
    )

    @classmethod
    def forward(
        cls,
        x: Any,
        weight: Any,
        bias: Any,
        residual: Any,
        stride: Any = 1,
        padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
    ) -> Tensor:
        return _conv2d_bias_family_forward(
            "conv2d_bias_add",
            resolve_shape=resolve_conv2d_bias_add_shape,
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )


@op_def
class Conv2dBiasAddRelu(OpDef):
    name = "conv2d_bias_add_relu"
    schema = OpSchema(
        inputs=("x", "weight", "bias", "residual"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
            AttrDef("dilation", "ints", default=(1, 1)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_conv2d_bias_add_relu_shape
    infer_shape_with_attrs = infer_conv2d_bias_add_relu_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _cutlass_conv_backend_kernels("conv2d_bias_add_relu")
    frontend = FrontendBinding("conv2d_bias_add_relu")
    description = (
        "Bounded fused conv2d_bias_add_relu frontend for v1-style residual Conv: "
        "public tensors stay NCHW/OIHW, groups remain 1, all tensors are static rank-4, "
        "and the residual input must match the Conv output shape exactly. CPU reference "
        "and compiled CPU artifacts apply bias, residual add, and trailing ReLU in one "
        "generated loop, while CUDA keeps the fused residual+ReLU epilogue artifact-visible "
        "through the CUTLASS Conv manifest/profile/execution-plan path."
    )

    @classmethod
    def forward(
        cls,
        x: Any,
        weight: Any,
        bias: Any,
        residual: Any,
        stride: Any = 1,
        padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
    ) -> Tensor:
        return _conv2d_bias_family_forward(
            "conv2d_bias_add_relu",
            resolve_shape=resolve_conv2d_bias_add_relu_shape,
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )


def conv2d(
    x: Any,
    weight: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    x_tensor = as_tensor(x)
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    if weight_tensor.builder is not x_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if weight_tensor.dtype != x_tensor.dtype:
        raise ValueError(f"conv2d dtype mismatch: {x_tensor.dtype} vs {weight_tensor.dtype}")
    if x_tensor.dtype not in CONV2D_BIAS_DTYPES:
        raise ValueError(f"conv2d does not support dtype {x_tensor.dtype}")
    if x_tensor.rank != 4:
        raise ValueError(f"conv2d expects rank-4 NCHW activation, got rank {x_tensor.rank}")
    if weight_tensor.rank != 4:
        raise ValueError(f"conv2d expects rank-4 OIHW weight, got rank {weight_tensor.rank}")
    if x_tensor.dynamic or weight_tensor.dynamic:
        raise ValueError("conv2d currently supports only static activation and weight shapes")
    normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv2d_bias_attrs(
        stride,
        padding,
        dilation,
        groups,
    )
    out_shape = resolve_conv2d_shape(
        x_tensor.shape,
        weight_tensor.shape,
        stride=normalized_stride,
        padding=normalized_padding,
        dilation=normalized_dilation,
        groups=normalized_groups,
    )
    zero_bias = as_tensor(
        Parameter(
            [int(weight_tensor.shape[0])],
            dtype=x_tensor.dtype,
            name="conv2d_zero_bias",
            value=np.zeros((int(weight_tensor.shape[0]),), dtype=np.float32),
        ),
        dtype_hint=x_tensor.dtype,
    )
    return x_tensor.builder.emit(
        "conv2d_bias",
        [x_tensor, weight_tensor, zero_bias],
        out_shape,
        x_tensor.dtype,
        {
            "stride": normalized_stride,
            "padding": normalized_padding,
            "dilation": normalized_dilation,
            "groups": normalized_groups,
            "bias_mode": "explicit_zero_constant",
            "source_op": "conv2d",
        },
        shape_spec=out_shape,
    )


def conv2d_bias(
    x: Any,
    weight: Any,
    bias: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return Conv2dBias.forward(x, weight, bias, stride, padding, dilation, groups)


def conv2d_bias_relu(
    x: Any,
    weight: Any,
    bias: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return Conv2dBiasRelu.forward(x, weight, bias, stride, padding, dilation, groups)


def conv2d_bias_add(
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return Conv2dBiasAdd.forward(x, weight, bias, residual, stride, padding, dilation, groups)


def conv2d_bias_add_relu(
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return Conv2dBiasAddRelu.forward(x, weight, bias, residual, stride, padding, dilation, groups)


def _validate_conv2d_bias_family_op_name(op_name: str) -> None:
    if op_name not in CONV2D_BIAS_FAMILY_OPS:
        raise ValueError(f"Unsupported conv2d bias family op {op_name!r}")


def _conv2d_bias_family_has_residual(op_name: str) -> bool:
    _validate_conv2d_bias_family_op_name(op_name)
    return op_name in {"conv2d_bias_add", "conv2d_bias_add_relu"}


__all__ = [
    "Conv2dBias",
    "Conv2dBiasAdd",
    "Conv2dBiasAddRelu",
    "Conv2dBiasRelu",
    "CONV2D_BIAS_FAMILY_OPS",
    "CONV2D_BIAS_DTYPES",
    "conv2d",
    "conv2d_bias",
    "conv2d_bias_add",
    "conv2d_bias_add_relu",
    "conv2d_bias_relu",
    "infer_conv2d_shape",
    "infer_conv2d_shape_with_attrs",
    "infer_conv2d_bias_shape",
    "infer_conv2d_bias_shape_with_attrs",
    "infer_conv2d_bias_relu_shape",
    "infer_conv2d_bias_relu_shape_with_attrs",
    "infer_conv2d_bias_add_shape",
    "infer_conv2d_bias_add_shape_with_attrs",
    "infer_conv2d_bias_add_relu_shape",
    "infer_conv2d_bias_add_relu_shape_with_attrs",
    "normalize_conv2d_bias_attrs",
    "resolve_conv2d_shape",
    "resolve_conv2d_bias_shape",
    "resolve_conv2d_bias_relu_shape",
    "resolve_conv2d_bias_add_shape",
    "resolve_conv2d_bias_add_relu_shape",
]
