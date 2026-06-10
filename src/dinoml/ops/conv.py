from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dinoml.frontend import Parameter, Tensor, as_tensor
from dinoml.kernels.providers.ck.conv import (
    CK_CONV_OPS,
    CK_CONV_SUPPORTED_DTYPES,
    CK_TRANSPOSED_CONV_OPS,
    ck_conv_candidate_set,
    ck_conv_candidates,
    ck_conv_profiler_symbol,
    ck_conv_symbol,
    ck_transposed_conv_candidate_set,
    ck_transposed_conv_candidates,
    ck_transposed_conv_profiler_symbol,
    ck_transposed_conv_symbol,
)
from dinoml.kernels.providers.cutlass.conv import (
    CUTLASS_CONV_OPS,
    CUTLASS_TRANSPOSED_CONV_OPS,
    cutlass_conv_candidate_set,
    cutlass_conv_candidates,
    cutlass_conv_profiler_symbol,
    cutlass_conv_symbol,
    cutlass_transposed_conv_candidate_set,
    cutlass_transposed_conv_candidates,
    cutlass_transposed_conv_profiler_symbol,
    cutlass_transposed_conv_symbol,
)
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, KernelVariant, OpDef, OpSchema, op_def


CONV2D_BIAS_DTYPES = ("float16", "float32", "bfloat16")
CUTLASS_CONV2D_BIAS_DTYPES = ("float16", "float32", "bfloat16")
CONV1D_BIAS_FAMILY_OPS = ("conv1d_bias", "conv1d_bias_relu", "conv1d_bias_add", "conv1d_bias_add_relu")
CONV2D_BIAS_FAMILY_OPS = ("conv2d_bias", "conv2d_bias_relu", "conv2d_bias_add", "conv2d_bias_add_relu")
TRANSPOSED_CONV2D_FAMILY_OPS = (
    "transposed_conv2d",
    "transposed_conv2d_bias",
    "transposed_conv2d_bias_relu",
    "transposed_conv2d_bias_add",
    "transposed_conv2d_bias_add_relu",
)
TRANSPOSED_CONV2D_BIAS_FAMILY_OPS = (
    "transposed_conv2d_bias",
    "transposed_conv2d_bias_relu",
    "transposed_conv2d_bias_add",
    "transposed_conv2d_bias_add_relu",
)


def infer_conv1d_bias_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv1d_bias_shape_with_attrs(
        input_shapes,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )


def infer_conv1d_bias_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    return _infer_conv1d_bias_family_shape_with_attrs("conv1d_bias", input_shapes, attrs)


def infer_conv1d_bias_relu_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv1d_bias_relu_shape_with_attrs(
        input_shapes,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )


def infer_conv1d_bias_relu_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    return _infer_conv1d_bias_family_shape_with_attrs("conv1d_bias_relu", input_shapes, attrs)


def infer_conv1d_bias_add_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv1d_bias_add_shape_with_attrs(
        input_shapes,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )


def infer_conv1d_bias_add_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    return _infer_conv1d_bias_family_shape_with_attrs("conv1d_bias_add", input_shapes, attrs)


def infer_conv1d_bias_add_relu_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv1d_bias_add_relu_shape_with_attrs(
        input_shapes,
        {"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
    )


def infer_conv1d_bias_add_relu_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_conv1d_bias_family_shape_with_attrs("conv1d_bias_add_relu", input_shapes, attrs)


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


def infer_conv2d_bias_add_relu_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_conv2d_bias_family_shape_with_attrs("conv2d_bias_add_relu", input_shapes, attrs)


def infer_transposed_conv2d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_transposed_conv2d_shape_with_attrs(
        input_shapes,
        {
            "stride": (1, 1),
            "padding": (0, 0),
            "output_padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
        },
    )


def infer_transposed_conv2d_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 2:
        raise ValueError("transposed_conv2d expects activation and weight inputs")
    stride, padding, output_padding, dilation, groups = normalize_transposed_conv2d_attrs(
        attrs.get("stride", (1, 1)),
        attrs.get("padding", (0, 0)),
        attrs.get("output_padding", (0, 0)),
        attrs.get("dilation", (1, 1)),
        attrs.get("groups", 1),
    )
    return resolve_transposed_conv2d_shape(
        input_shapes[0],
        input_shapes[1],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def infer_transposed_conv2d_bias_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_transposed_conv2d_bias_shape_with_attrs(
        input_shapes,
        {
            "stride": (1, 1),
            "padding": (0, 0),
            "output_padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
        },
    )


def infer_transposed_conv2d_bias_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_transposed_conv2d_bias_family_shape_with_attrs("transposed_conv2d_bias", input_shapes, attrs)


def infer_transposed_conv2d_bias_relu_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_transposed_conv2d_bias_relu_shape_with_attrs(
        input_shapes,
        {
            "stride": (1, 1),
            "padding": (0, 0),
            "output_padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
        },
    )


def infer_transposed_conv2d_bias_relu_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_transposed_conv2d_bias_family_shape_with_attrs("transposed_conv2d_bias_relu", input_shapes, attrs)


def infer_transposed_conv2d_bias_add_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_transposed_conv2d_bias_add_shape_with_attrs(
        input_shapes,
        {
            "stride": (1, 1),
            "padding": (0, 0),
            "output_padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
        },
    )


def infer_transposed_conv2d_bias_add_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_transposed_conv2d_bias_family_shape_with_attrs("transposed_conv2d_bias_add", input_shapes, attrs)


def infer_transposed_conv2d_bias_add_relu_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_transposed_conv2d_bias_add_relu_shape_with_attrs(
        input_shapes,
        {
            "stride": (1, 1),
            "padding": (0, 0),
            "output_padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
        },
    )


def infer_transposed_conv2d_bias_add_relu_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_transposed_conv2d_bias_family_shape_with_attrs("transposed_conv2d_bias_add_relu", input_shapes, attrs)


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


def _infer_conv1d_bias_family_shape_with_attrs(
    op_name: str,
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    _validate_conv1d_bias_family_op_name(op_name)
    expected_inputs = 4 if _conv1d_bias_family_has_residual(op_name) else 3
    if len(input_shapes) != expected_inputs:
        extra = ", and residual" if expected_inputs == 4 else ""
        raise ValueError(f"{op_name} expects activation, weight, bias{extra} inputs")
    stride, padding, dilation, groups = normalize_conv1d_bias_attrs(
        attrs.get("stride", 1),
        attrs.get("padding", 0),
        attrs.get("dilation", 1),
        attrs.get("groups", 1),
    )
    return _resolve_conv1d_bias_family_shape(
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


def _infer_transposed_conv2d_bias_family_shape_with_attrs(
    op_name: str,
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    _validate_transposed_conv2d_bias_family_op_name(op_name)
    expected_inputs = 4 if _transposed_conv2d_bias_family_has_residual(op_name) else 3
    if len(input_shapes) != expected_inputs:
        extra = ", and residual" if expected_inputs == 4 else ""
        raise ValueError(f"{op_name} expects activation, weight, bias{extra} inputs")
    stride, padding, output_padding, dilation, groups = normalize_transposed_conv2d_attrs(
        attrs.get("stride", (1, 1)),
        attrs.get("padding", (0, 0)),
        attrs.get("output_padding", (0, 0)),
        attrs.get("dilation", (1, 1)),
        attrs.get("groups", 1),
    )
    return _resolve_transposed_conv2d_bias_family_shape(
        op_name,
        input_shapes[0],
        input_shapes[1],
        input_shapes[2],
        None if expected_inputs == 3 else input_shapes[3],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def normalize_conv2d_bias_attrs(
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> tuple[list[int], list[int], list[int], int]:
    normalized_stride = _normalize_positive_pair(stride, "conv2d stride")
    normalized_padding = _normalize_non_negative_pair(padding, "conv2d padding")
    normalized_dilation = _normalize_positive_pair(dilation, "conv2d dilation")
    normalized_groups = _normalize_groups(groups, "conv2d")
    return list(normalized_stride), list(normalized_padding), list(normalized_dilation), normalized_groups


def normalize_conv1d_bias_attrs(
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> tuple[list[int], list[int], list[int], int]:
    normalized_stride = _normalize_positive_single(stride, "conv1d stride")
    normalized_padding = _normalize_non_negative_single(padding, "conv1d padding")
    normalized_dilation = _normalize_positive_single(dilation, "conv1d dilation")
    normalized_groups = _normalize_groups(groups, "conv1d")
    return [normalized_stride], [normalized_padding], [normalized_dilation], normalized_groups


def normalize_transposed_conv2d_attrs(
    stride: Any,
    padding: Any,
    output_padding: Any,
    dilation: Any,
    groups: Any,
) -> tuple[list[int], list[int], list[int], list[int], int]:
    normalized_stride = _normalize_positive_pair(stride, "transposed_conv2d stride")
    normalized_padding = _normalize_non_negative_pair(padding, "transposed_conv2d padding")
    normalized_output_padding = _normalize_non_negative_pair(output_padding, "transposed_conv2d output_padding")
    normalized_dilation = _normalize_positive_pair(dilation, "transposed_conv2d dilation")
    normalized_groups = _normalize_groups(groups, "transposed_conv2d")
    for axis, (value, limit) in enumerate(zip(normalized_output_padding, normalized_stride, strict=True)):
        if value >= limit:
            axis_name = "height" if axis == 0 else "width"
            raise ValueError(
                "transposed_conv2d output_padding must be smaller than stride "
                f"for {axis_name}, got output_padding={value} and stride={limit}"
            )
    return (
        list(normalized_stride),
        list(normalized_padding),
        list(normalized_output_padding),
        list(normalized_dilation),
        normalized_groups,
    )


def resolve_conv1d_bias_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_conv1d_bias_family_shape(
        "conv1d_bias",
        input_shape,
        weight_shape,
        bias_shape,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def resolve_conv1d_bias_relu_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_conv1d_bias_family_shape(
        "conv1d_bias_relu",
        input_shape,
        weight_shape,
        bias_shape,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def resolve_conv1d_bias_add_shape(
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
    return _resolve_conv1d_bias_family_shape(
        "conv1d_bias_add",
        input_shape,
        weight_shape,
        bias_shape,
        residual_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def resolve_conv1d_bias_add_relu_shape(
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
    return _resolve_conv1d_bias_family_shape(
        "conv1d_bias_add_relu",
        input_shape,
        weight_shape,
        bias_shape,
        residual_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


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


def resolve_transposed_conv2d_bias_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    output_padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_transposed_conv2d_bias_family_shape(
        "transposed_conv2d_bias",
        input_shape,
        weight_shape,
        bias_shape,
        None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def resolve_transposed_conv2d_bias_relu_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    output_padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_transposed_conv2d_bias_family_shape(
        "transposed_conv2d_bias_relu",
        input_shape,
        weight_shape,
        bias_shape,
        None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def resolve_transposed_conv2d_bias_add_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    residual_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    output_padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_transposed_conv2d_bias_family_shape(
        "transposed_conv2d_bias_add",
        input_shape,
        weight_shape,
        bias_shape,
        residual_shape,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def resolve_transposed_conv2d_bias_add_relu_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    residual_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    output_padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    return _resolve_transposed_conv2d_bias_family_shape(
        "transposed_conv2d_bias_add_relu",
        input_shape,
        weight_shape,
        bias_shape,
        residual_shape,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
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
    _validate_rank(input_shape, 4, f"{op_name} expects rank-4 NCHW activation")
    _validate_rank(weight_shape, 4, f"{op_name} expects rank-4 OIHW weight")
    _validate_rank(bias_shape, 1, f"{op_name} expects rank-1 bias")
    normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv2d_bias_attrs(
        stride,
        padding,
        dilation,
        groups,
    )
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
    _validate_kernel_shape(op_name, kernel_h, kernel_w)
    output_shape = [
        batch,
        out_channels,
        _conv_output_dim(op_name, in_height, kernel_h, normalized_stride[0], normalized_padding[0], normalized_dilation[0], "height"),
        _conv_output_dim(op_name, in_width, kernel_w, normalized_stride[1], normalized_padding[1], normalized_dilation[1], "width"),
    ]
    if _conv2d_bias_family_has_residual(op_name):
        _validate_residual_shape(op_name, residual_shape, output_shape)
    if normalized_groups != 1:
        raise NotImplementedError(f"{op_name} currently supports groups=1 only, got {normalized_groups}")
    return output_shape


def _resolve_conv1d_bias_family_shape(
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
    _validate_conv1d_bias_family_op_name(op_name)
    _validate_rank(input_shape, 3, f"{op_name} expects rank-3 NCW activation")
    _validate_rank(weight_shape, 3, f"{op_name} expects rank-3 OIW weight")
    _validate_rank(bias_shape, 1, f"{op_name} expects rank-1 bias")
    normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv1d_bias_attrs(
        stride,
        padding,
        dilation,
        groups,
    )
    batch, in_channels, in_width = [int(dim) for dim in input_shape]
    out_channels, weight_in_channels, kernel_w = [int(dim) for dim in weight_shape]
    bias_channels = int(bias_shape[0])
    if weight_in_channels != in_channels:
        raise ValueError(
            f"{op_name} weight input channels must match activation channels for groups=1: "
            f"got activation C={in_channels}, weight I={weight_in_channels}"
        )
    if bias_channels != out_channels:
        raise ValueError(
            f"{op_name} bias length must match weight output channels, got bias {bias_channels} and weight O={out_channels}"
        )
    _validate_kernel_extent(op_name, kernel_w, "width")
    output_shape = [
        batch,
        out_channels,
        _conv_output_dim(
            op_name,
            in_width,
            kernel_w,
            normalized_stride[0],
            normalized_padding[0],
            normalized_dilation[0],
            "width",
        ),
    ]
    if _conv1d_bias_family_has_residual(op_name):
        _validate_residual_shape_1d(op_name, residual_shape, output_shape)
    if normalized_groups != 1:
        raise NotImplementedError(f"{op_name} currently supports groups=1 only, got {normalized_groups}")
    return output_shape


def _resolve_transposed_conv2d_bias_family_shape(
    op_name: str,
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    bias_shape: Sequence[int],
    residual_shape: Sequence[int] | None,
    *,
    stride: Any,
    padding: Any,
    output_padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    _validate_transposed_conv2d_bias_family_op_name(op_name)
    _validate_rank(input_shape, 4, f"{op_name} expects rank-4 NCHW activation")
    _validate_rank(weight_shape, 4, f"{op_name} expects rank-4 IOHW weight")
    _validate_rank(bias_shape, 1, f"{op_name} expects rank-1 bias")
    normalized_stride, normalized_padding, normalized_output_padding, normalized_dilation, normalized_groups = (
        normalize_transposed_conv2d_attrs(stride, padding, output_padding, dilation, groups)
    )
    batch, in_channels, in_height, in_width = [int(dim) for dim in input_shape]
    weight_in_channels, out_channels, kernel_h, kernel_w = [int(dim) for dim in weight_shape]
    bias_channels = int(bias_shape[0])
    if weight_in_channels != in_channels:
        raise ValueError(
            f"{op_name} weight input channels must match activation channels for groups=1: "
            f"got activation C={in_channels}, weight I={weight_in_channels}"
        )
    if bias_channels != out_channels:
        raise ValueError(
            f"{op_name} bias length must match weight output channels, got bias {bias_channels} and weight O={out_channels}"
        )
    _validate_kernel_shape(op_name, kernel_h, kernel_w)
    output_shape = [
        batch,
        out_channels,
        _transposed_conv_output_dim(
            op_name,
            in_height,
            kernel_h,
            normalized_stride[0],
            normalized_padding[0],
            normalized_output_padding[0],
            normalized_dilation[0],
            "height",
        ),
        _transposed_conv_output_dim(
            op_name,
            in_width,
            kernel_w,
            normalized_stride[1],
            normalized_padding[1],
            normalized_output_padding[1],
            normalized_dilation[1],
            "width",
        ),
    ]
    if _transposed_conv2d_bias_family_has_residual(op_name):
        _validate_residual_shape(op_name, residual_shape, output_shape)
    if normalized_groups != 1:
        raise NotImplementedError(f"{op_name} currently supports groups=1 only, got {normalized_groups}")
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
    _validate_rank(input_shape, 4, f"conv2d expects rank-4 NCHW activation")
    _validate_rank(weight_shape, 4, f"conv2d expects rank-4 OIHW weight")
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


def resolve_transposed_conv2d_shape(
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    *,
    stride: Any,
    padding: Any,
    output_padding: Any,
    dilation: Any,
    groups: Any,
) -> list[int]:
    _validate_rank(input_shape, 4, f"transposed_conv2d expects rank-4 NCHW activation")
    _validate_rank(weight_shape, 4, f"transposed_conv2d expects rank-4 IOHW weight")
    out_channels = int(weight_shape[1]) if weight_shape else 0
    return resolve_transposed_conv2d_bias_shape(
        input_shape,
        weight_shape,
        [out_channels],
        stride=stride,
        padding=padding,
        output_padding=output_padding,
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


def _transposed_conv_output_dim(
    op_name: str,
    dim: int,
    kernel: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
    axis_name: str,
) -> int:
    output = (int(dim) - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    if output <= 0:
        raise ValueError(
            f"{op_name} output {axis_name} must be positive; got input={dim}, kernel={kernel}, stride={stride}, "
            f"padding={padding}, output_padding={output_padding}, dilation={dilation}"
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


def _normalize_positive_single(value: Any, name: str) -> int:
    single = _normalize_single(value, name)
    if single <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return single


def _normalize_non_negative_single(value: Any, name: str) -> int:
    single = _normalize_single(value, name)
    if single < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return single


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


def _normalize_single(value: Any, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if len(values) != 1:
            raise ValueError(f"{name} must be an integer or length-1 sequence of integers, got {value!r}")
        if not isinstance(values[0], int) or isinstance(values[0], bool):
            raise ValueError(f"{name} must contain non-bool integers, got {value!r}")
        return int(values[0])
    raise ValueError(f"{name} must be an integer or length-1 sequence of integers, got {value!r}")


def _normalize_groups(groups: Any, op_name: str) -> int:
    if not isinstance(groups, int) or isinstance(groups, bool):
        raise ValueError(f"{op_name} groups must be a non-bool integer, got {groups!r}")
    normalized_groups = int(groups)
    if normalized_groups <= 0:
        raise ValueError(f"{op_name} groups must be positive, got {groups!r}")
    if normalized_groups != 1:
        raise NotImplementedError(f"{op_name} currently supports groups=1 only, got {normalized_groups}")
    return normalized_groups


def _conv1d_backend_kernels(op_name: str) -> dict[str, KernelBinding]:
    kernels = {
        "cpu": KernelBinding(
            symbol=f"generated_{op_name}",
            library="model",
            source_template="conv1d_cpu.cpp.j2",
        )
    }
    if op_name in CUTLASS_CONV_OPS:
        kernels["cuda"] = KernelBinding(
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
                for dtype in CUTLASS_CONV2D_BIAS_DTYPES
            },
        )
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
                for dtype in CK_CONV_SUPPORTED_DTYPES
            },
        )
    return kernels


def _cutlass_conv_backend_kernels(op_name: str) -> dict[str, KernelBinding]:
    kernels = {
        "cpu": KernelBinding(
            symbol=f"generated_{op_name}",
            library="model",
            source_template="conv_cpu.cpp.j2",
        )
    }
    if op_name in CUTLASS_CONV_OPS:
        kernels["cuda"] = KernelBinding(
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
                for dtype in CUTLASS_CONV2D_BIAS_DTYPES
            },
        )
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
                for dtype in CK_CONV_SUPPORTED_DTYPES
            },
        )
    return kernels


def _transposed_conv_backend_kernels(op_name: str) -> dict[str, KernelBinding]:
    kernels = {
        "cpu": KernelBinding(
            symbol=f"generated_{op_name}",
            library="model",
            source_template="conv_cpu.cpp.j2",
        )
    }
    if op_name in CUTLASS_TRANSPOSED_CONV_OPS:
        kernels["cuda"] = KernelBinding(
            cutlass_transposed_conv_symbol(op_name, "float32"),
            "cutlass_conv",
            profiler_symbol=cutlass_transposed_conv_profiler_symbol(op_name, "float32"),
            dtype_variants={
                dtype: KernelVariant(
                    cutlass_transposed_conv_symbol(op_name, dtype),
                    profiler_symbol=cutlass_transposed_conv_profiler_symbol(op_name, dtype),
                    candidates=cutlass_transposed_conv_candidates(op_name, dtype),
                    candidate_set=cutlass_transposed_conv_candidate_set(op_name, dtype),
                )
                for dtype in CUTLASS_CONV2D_BIAS_DTYPES
            },
        )
    if op_name in CK_TRANSPOSED_CONV_OPS:
        kernels["rocm"] = KernelBinding(
            ck_transposed_conv_symbol(op_name, "float32"),
            "ck_conv",
            profiler_symbol=ck_transposed_conv_profiler_symbol(op_name, "float32"),
            dtype_variants={
                dtype: KernelVariant(
                    ck_transposed_conv_symbol(op_name, dtype),
                    profiler_symbol=ck_transposed_conv_profiler_symbol(op_name, dtype),
                    candidates=ck_transposed_conv_candidates(op_name, dtype),
                    candidate_set=ck_transposed_conv_candidate_set(op_name, dtype),
                )
                for dtype in CK_CONV_SUPPORTED_DTYPES
            },
        )
    return kernels


def _conv1d_bias_family_forward(
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
    _validate_tensor_rank(x_tensor, 3, f"{op_name} expects rank-3 NCW activation")
    _validate_tensor_rank(weight_tensor, 3, f"{op_name} expects rank-3 OIW weight")
    _validate_tensor_rank(bias_tensor, 1, f"{op_name} expects rank-1 bias")
    if residual_tensor is not None:
        _validate_tensor_rank(residual_tensor, 3, f"{op_name} expects rank-3 residual")
    if any(tensor.dynamic for tensor in tensors):
        expected = "activation, weight, bias, and residual" if residual_tensor is not None else "activation, weight, and bias"
        raise ValueError(f"{op_name} currently supports only static {expected} shapes")
    normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv1d_bias_attrs(
        stride,
        padding,
        dilation,
        groups,
    )
    attrs = {
        "stride": normalized_stride,
        "padding": normalized_padding,
        "dilation": normalized_dilation,
        "groups": normalized_groups,
    }
    if residual_tensor is None:
        out_shape = resolve_shape(x_tensor.shape, weight_tensor.shape, bias_tensor.shape, **attrs)
    else:
        out_shape = resolve_shape(x_tensor.shape, weight_tensor.shape, bias_tensor.shape, residual_tensor.shape, **attrs)
    inputs = [x_tensor, weight_tensor, bias_tensor]
    if residual_tensor is not None:
        inputs.append(residual_tensor)
    return x_tensor.builder.emit(
        op_name,
        inputs,
        out_shape,
        x_tensor.dtype,
        attrs,
        shape_spec=out_shape,
    )


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
    return _conv_family_forward(
        op_name,
        resolve_shape=resolve_shape,
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        stride=stride,
        padding=padding,
        output_padding=None,
        dilation=dilation,
        groups=groups,
        transposed=False,
    )


def _transposed_conv2d_bias_family_forward(
    op_name: str,
    *,
    resolve_shape: Callable[..., list[int]],
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any | None = None,
    stride: Any,
    padding: Any,
    output_padding: Any,
    dilation: Any,
    groups: int,
) -> Tensor:
    return _conv_family_forward(
        op_name,
        resolve_shape=resolve_shape,
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        transposed=True,
    )


def _conv_family_forward(
    op_name: str,
    *,
    resolve_shape: Callable[..., list[int]],
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any | None,
    stride: Any,
    padding: Any,
    output_padding: Any | None,
    dilation: Any,
    groups: int,
    transposed: bool,
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
    _validate_tensor_rank(x_tensor, 4, f"{op_name} expects rank-4 NCHW activation")
    _validate_tensor_rank(
        weight_tensor,
        4,
        f"{op_name} expects rank-4 {'IOHW' if transposed else 'OIHW'} weight",
    )
    _validate_tensor_rank(bias_tensor, 1, f"{op_name} expects rank-1 bias")
    if residual_tensor is not None:
        _validate_tensor_rank(residual_tensor, 4, f"{op_name} expects rank-4 residual")
    if any(tensor.dynamic for tensor in tensors):
        expected = "activation, weight, bias, and residual" if residual_tensor is not None else "activation, weight, and bias"
        raise ValueError(f"{op_name} currently supports only static {expected} shapes")
    attrs = _normalized_conv_attrs(
        transposed=transposed,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    if residual_tensor is None:
        out_shape = resolve_shape(
            x_tensor.shape,
            weight_tensor.shape,
            bias_tensor.shape,
            **attrs,
        )
    else:
        out_shape = resolve_shape(
            x_tensor.shape,
            weight_tensor.shape,
            bias_tensor.shape,
            residual_tensor.shape,
            **attrs,
        )
    return x_tensor.builder.emit(
        op_name,
        tensors,
        out_shape,
        x_tensor.dtype,
        attrs,
        shape_spec=out_shape,
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
    _validate_tensor_rank(x_tensor, 4, "conv2d expects rank-4 NCHW activation")
    _validate_tensor_rank(weight_tensor, 4, "conv2d expects rank-4 OIHW weight")
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


def transposed_conv2d(
    x: Any,
    weight: Any,
    stride: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    x_tensor = as_tensor(x)
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    if weight_tensor.builder is not x_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if weight_tensor.dtype != x_tensor.dtype:
        raise ValueError(f"transposed_conv2d dtype mismatch: {x_tensor.dtype} vs {weight_tensor.dtype}")
    if x_tensor.dtype not in CONV2D_BIAS_DTYPES:
        raise ValueError(f"transposed_conv2d does not support dtype {x_tensor.dtype}")
    _validate_tensor_rank(x_tensor, 4, "transposed_conv2d expects rank-4 NCHW activation")
    _validate_tensor_rank(weight_tensor, 4, "transposed_conv2d expects rank-4 IOHW weight")
    if x_tensor.dynamic or weight_tensor.dynamic:
        raise ValueError("transposed_conv2d currently supports only static activation and weight shapes")
    normalized_stride, normalized_padding, normalized_output_padding, normalized_dilation, normalized_groups = (
        normalize_transposed_conv2d_attrs(
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        )
    )
    out_shape = resolve_transposed_conv2d_shape(
        x_tensor.shape,
        weight_tensor.shape,
        stride=normalized_stride,
        padding=normalized_padding,
        output_padding=normalized_output_padding,
        dilation=normalized_dilation,
        groups=normalized_groups,
    )
    return x_tensor.builder.emit(
        "transposed_conv2d",
        [x_tensor, weight_tensor],
        out_shape,
        x_tensor.dtype,
        {
            "stride": normalized_stride,
            "padding": normalized_padding,
            "output_padding": normalized_output_padding,
            "dilation": normalized_dilation,
            "groups": normalized_groups,
        },
        shape_spec=out_shape,
    )


def conv1d_bias(
    x: Any,
    weight: Any,
    bias: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return Conv1dBias.forward(x, weight, bias, stride, padding, dilation, groups)


def conv1d_bias_relu(
    x: Any,
    weight: Any,
    bias: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return Conv1dBiasRelu.forward(x, weight, bias, stride, padding, dilation, groups)


def conv1d_bias_add(
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return Conv1dBiasAdd.forward(x, weight, bias, residual, stride, padding, dilation, groups)


def conv1d_bias_add_relu(
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return Conv1dBiasAddRelu.forward(x, weight, bias, residual, stride, padding, dilation, groups)


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


def transposed_conv2d_bias(
    x: Any,
    weight: Any,
    bias: Any,
    stride: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return TransposedConv2dBias.forward(x, weight, bias, stride, padding, output_padding, dilation, groups)


def transposed_conv2d_bias_relu(
    x: Any,
    weight: Any,
    bias: Any,
    stride: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return TransposedConv2dBiasRelu.forward(x, weight, bias, stride, padding, output_padding, dilation, groups)


def transposed_conv2d_bias_add(
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any,
    stride: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return TransposedConv2dBiasAdd.forward(x, weight, bias, residual, stride, padding, output_padding, dilation, groups)


def transposed_conv2d_bias_add_relu(
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any,
    stride: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return TransposedConv2dBiasAddRelu.forward(
        x,
        weight,
        bias,
        residual,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )


@op_def
class Conv1dBias(OpDef):
    name = "conv1d_bias"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0,)),
            AttrDef("dilation", "ints", default=(1,)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_conv1d_bias_shape
    infer_shape_with_attrs = infer_conv1d_bias_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _conv1d_backend_kernels("conv1d_bias")
    frontend = FrontendBinding("conv1d_bias")

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
        return _conv1d_bias_family_forward(
            "conv1d_bias",
            resolve_shape=resolve_conv1d_bias_shape,
            x=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )


@op_def
class Conv1dBiasRelu(OpDef):
    name = "conv1d_bias_relu"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0,)),
            AttrDef("dilation", "ints", default=(1,)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_conv1d_bias_relu_shape
    infer_shape_with_attrs = infer_conv1d_bias_relu_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _conv1d_backend_kernels("conv1d_bias_relu")
    frontend = FrontendBinding("conv1d_bias_relu")

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
        return _conv1d_bias_family_forward(
            "conv1d_bias_relu",
            resolve_shape=resolve_conv1d_bias_relu_shape,
            x=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )


@op_def
class Conv1dBiasAdd(OpDef):
    name = "conv1d_bias_add"
    schema = OpSchema(
        inputs=("x", "weight", "bias", "residual"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0,)),
            AttrDef("dilation", "ints", default=(1,)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_conv1d_bias_add_shape
    infer_shape_with_attrs = infer_conv1d_bias_add_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _conv1d_backend_kernels("conv1d_bias_add")
    frontend = FrontendBinding("conv1d_bias_add")

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
        return _conv1d_bias_family_forward(
            "conv1d_bias_add",
            resolve_shape=resolve_conv1d_bias_add_shape,
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
class Conv1dBiasAddRelu(OpDef):
    name = "conv1d_bias_add_relu"
    schema = OpSchema(
        inputs=("x", "weight", "bias", "residual"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0,)),
            AttrDef("dilation", "ints", default=(1,)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_conv1d_bias_add_relu_shape
    infer_shape_with_attrs = infer_conv1d_bias_add_relu_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _conv1d_backend_kernels("conv1d_bias_add_relu")
    frontend = FrontendBinding("conv1d_bias_add_relu")

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
        return _conv1d_bias_family_forward(
            "conv1d_bias_add_relu",
            resolve_shape=resolve_conv1d_bias_add_relu_shape,
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


@op_def
class TransposedConv2d(OpDef):
    name = "transposed_conv2d"
    schema = OpSchema(
        inputs=("x", "weight"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
            AttrDef("output_padding", "ints", default=(0, 0)),
            AttrDef("dilation", "ints", default=(1, 1)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_transposed_conv2d_shape
    infer_shape_with_attrs = infer_transposed_conv2d_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _transposed_conv_backend_kernels("transposed_conv2d")
    frontend = FrontendBinding("transposed_conv2d")


@op_def
class TransposedConv2dBias(OpDef):
    name = "transposed_conv2d_bias"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
            AttrDef("output_padding", "ints", default=(0, 0)),
            AttrDef("dilation", "ints", default=(1, 1)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_transposed_conv2d_bias_shape
    infer_shape_with_attrs = infer_transposed_conv2d_bias_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _transposed_conv_backend_kernels("transposed_conv2d_bias")
    frontend = FrontendBinding("transposed_conv2d_bias")

    @classmethod
    def forward(
        cls,
        x: Any,
        weight: Any,
        bias: Any,
        stride: Any = 1,
        padding: Any = 0,
        output_padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
    ) -> Tensor:
        return _transposed_conv2d_bias_family_forward(
            "transposed_conv2d_bias",
            resolve_shape=resolve_transposed_conv2d_bias_shape,
            x=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )


@op_def
class TransposedConv2dBiasRelu(OpDef):
    name = "transposed_conv2d_bias_relu"
    schema = OpSchema(
        inputs=("x", "weight", "bias"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
            AttrDef("output_padding", "ints", default=(0, 0)),
            AttrDef("dilation", "ints", default=(1, 1)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_transposed_conv2d_bias_relu_shape
    infer_shape_with_attrs = infer_transposed_conv2d_bias_relu_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _transposed_conv_backend_kernels("transposed_conv2d_bias_relu")
    frontend = FrontendBinding("transposed_conv2d_bias_relu")

    @classmethod
    def forward(
        cls,
        x: Any,
        weight: Any,
        bias: Any,
        stride: Any = 1,
        padding: Any = 0,
        output_padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
    ) -> Tensor:
        return _transposed_conv2d_bias_family_forward(
            "transposed_conv2d_bias_relu",
            resolve_shape=resolve_transposed_conv2d_bias_relu_shape,
            x=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )


@op_def
class TransposedConv2dBiasAdd(OpDef):
    name = "transposed_conv2d_bias_add"
    schema = OpSchema(
        inputs=("x", "weight", "bias", "residual"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
            AttrDef("output_padding", "ints", default=(0, 0)),
            AttrDef("dilation", "ints", default=(1, 1)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_transposed_conv2d_bias_add_shape
    infer_shape_with_attrs = infer_transposed_conv2d_bias_add_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _transposed_conv_backend_kernels("transposed_conv2d_bias_add")
    frontend = FrontendBinding("transposed_conv2d_bias_add")

    @classmethod
    def forward(
        cls,
        x: Any,
        weight: Any,
        bias: Any,
        residual: Any,
        stride: Any = 1,
        padding: Any = 0,
        output_padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
    ) -> Tensor:
        return _transposed_conv2d_bias_family_forward(
            "transposed_conv2d_bias_add",
            resolve_shape=resolve_transposed_conv2d_bias_add_shape,
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )


@op_def
class TransposedConv2dBiasAddRelu(OpDef):
    name = "transposed_conv2d_bias_add_relu"
    schema = OpSchema(
        inputs=("x", "weight", "bias", "residual"),
        attrs=(
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
            AttrDef("output_padding", "ints", default=(0, 0)),
            AttrDef("dilation", "ints", default=(1, 1)),
            AttrDef("groups", "int", default=1),
        ),
    )
    infer_shape = infer_transposed_conv2d_bias_add_relu_shape
    infer_shape_with_attrs = infer_transposed_conv2d_bias_add_relu_shape_with_attrs
    allowed_dtypes = CONV2D_BIAS_DTYPES
    backend_kernels = _transposed_conv_backend_kernels("transposed_conv2d_bias_add_relu")
    frontend = FrontendBinding("transposed_conv2d_bias_add_relu")

    @classmethod
    def forward(
        cls,
        x: Any,
        weight: Any,
        bias: Any,
        residual: Any,
        stride: Any = 1,
        padding: Any = 0,
        output_padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
    ) -> Tensor:
        return _transposed_conv2d_bias_family_forward(
            "transposed_conv2d_bias_add_relu",
            resolve_shape=resolve_transposed_conv2d_bias_add_relu_shape,
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )


def _normalized_conv_attrs(
    *,
    transposed: bool,
    stride: Any,
    padding: Any,
    output_padding: Any | None,
    dilation: Any,
    groups: int,
) -> dict[str, Any]:
    if not transposed:
        normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv2d_bias_attrs(
            stride,
            padding,
            dilation,
            groups,
        )
        return {
            "stride": normalized_stride,
            "padding": normalized_padding,
            "dilation": normalized_dilation,
            "groups": normalized_groups,
        }
    normalized_stride, normalized_padding, normalized_output_padding, normalized_dilation, normalized_groups = (
        normalize_transposed_conv2d_attrs(
            stride,
            padding,
            (0, 0) if output_padding is None else output_padding,
            dilation,
            groups,
        )
    )
    return {
        "stride": normalized_stride,
        "padding": normalized_padding,
        "output_padding": normalized_output_padding,
        "dilation": normalized_dilation,
        "groups": normalized_groups,
    }


def _validate_rank(shape: Sequence[int], expected_rank: int, message: str) -> None:
    if len(shape) != expected_rank:
        raise ValueError(f"{message}, got rank {len(shape)}")


def _validate_tensor_rank(tensor: Tensor, expected_rank: int, message: str) -> None:
    if tensor.rank != expected_rank:
        raise ValueError(f"{message}, got rank {tensor.rank}")


def _validate_kernel_shape(op_name: str, kernel_h: int, kernel_w: int) -> None:
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError(f"{op_name} kernel dimensions must be positive, got ({kernel_h}, {kernel_w})")


def _validate_kernel_extent(op_name: str, kernel: int, axis_name: str) -> None:
    if kernel <= 0:
        raise ValueError(f"{op_name} kernel {axis_name} must be positive, got {kernel}")


def _validate_residual_shape(op_name: str, residual_shape: Sequence[int] | None, output_shape: Sequence[int]) -> None:
    if residual_shape is None:
        raise ValueError(f"{op_name} expects a residual input shape")
    _validate_rank(residual_shape, 4, f"{op_name} expects rank-4 residual")
    residual = [int(dim) for dim in residual_shape]
    if residual != [int(dim) for dim in output_shape]:
        raise ValueError(f"{op_name} residual shape must match the conv output shape {list(output_shape)}, got {residual}")


def _validate_residual_shape_1d(op_name: str, residual_shape: Sequence[int] | None, output_shape: Sequence[int]) -> None:
    if residual_shape is None:
        raise ValueError(f"{op_name} expects a residual input shape")
    _validate_rank(residual_shape, 3, f"{op_name} expects rank-3 residual")
    residual = [int(dim) for dim in residual_shape]
    if residual != [int(dim) for dim in output_shape]:
        raise ValueError(f"{op_name} residual shape must match the conv output shape {list(output_shape)}, got {residual}")


def _validate_conv1d_bias_family_op_name(op_name: str) -> None:
    if op_name not in CONV1D_BIAS_FAMILY_OPS:
        raise ValueError(f"Unsupported conv1d bias family op {op_name!r}")


def _conv1d_bias_family_has_residual(op_name: str) -> bool:
    _validate_conv1d_bias_family_op_name(op_name)
    return op_name in {"conv1d_bias_add", "conv1d_bias_add_relu"}


def _validate_conv2d_bias_family_op_name(op_name: str) -> None:
    if op_name not in CONV2D_BIAS_FAMILY_OPS:
        raise ValueError(f"Unsupported conv2d bias family op {op_name!r}")


def _conv2d_bias_family_has_residual(op_name: str) -> bool:
    _validate_conv2d_bias_family_op_name(op_name)
    return op_name in {"conv2d_bias_add", "conv2d_bias_add_relu"}


def _validate_transposed_conv2d_bias_family_op_name(op_name: str) -> None:
    if op_name not in TRANSPOSED_CONV2D_BIAS_FAMILY_OPS:
        raise ValueError(f"Unsupported transposed_conv2d bias family op {op_name!r}")


def _transposed_conv2d_bias_family_has_residual(op_name: str) -> bool:
    _validate_transposed_conv2d_bias_family_op_name(op_name)
    return op_name in {"transposed_conv2d_bias_add", "transposed_conv2d_bias_add_relu"}


__all__ = [
    "CONV2D_BIAS_DTYPES",
    "CONV1D_BIAS_FAMILY_OPS",
    "CONV2D_BIAS_FAMILY_OPS",
    "Conv1dBias",
    "Conv1dBiasAdd",
    "Conv1dBiasAddRelu",
    "Conv1dBiasRelu",
    "Conv2dBias",
    "Conv2dBiasAdd",
    "Conv2dBiasAddRelu",
    "Conv2dBiasRelu",
    "TRANSPOSED_CONV2D_BIAS_FAMILY_OPS",
    "TRANSPOSED_CONV2D_FAMILY_OPS",
    "TransposedConv2d",
    "TransposedConv2dBias",
    "TransposedConv2dBiasAdd",
    "TransposedConv2dBiasAddRelu",
    "TransposedConv2dBiasRelu",
    "conv1d_bias",
    "conv1d_bias_add",
    "conv1d_bias_add_relu",
    "conv1d_bias_relu",
    "conv2d",
    "conv2d_bias",
    "conv2d_bias_add",
    "conv2d_bias_add_relu",
    "conv2d_bias_relu",
    "infer_conv1d_bias_add_relu_shape",
    "infer_conv1d_bias_add_relu_shape_with_attrs",
    "infer_conv1d_bias_add_shape",
    "infer_conv1d_bias_add_shape_with_attrs",
    "infer_conv1d_bias_relu_shape",
    "infer_conv1d_bias_relu_shape_with_attrs",
    "infer_conv1d_bias_shape",
    "infer_conv1d_bias_shape_with_attrs",
    "infer_conv2d_bias_add_relu_shape",
    "infer_conv2d_bias_add_relu_shape_with_attrs",
    "infer_conv2d_bias_add_shape",
    "infer_conv2d_bias_add_shape_with_attrs",
    "infer_conv2d_bias_relu_shape",
    "infer_conv2d_bias_relu_shape_with_attrs",
    "infer_conv2d_bias_shape",
    "infer_conv2d_bias_shape_with_attrs",
    "infer_conv2d_shape",
    "infer_conv2d_shape_with_attrs",
    "infer_transposed_conv2d_bias_add_relu_shape",
    "infer_transposed_conv2d_bias_add_relu_shape_with_attrs",
    "infer_transposed_conv2d_bias_add_shape",
    "infer_transposed_conv2d_bias_add_shape_with_attrs",
    "infer_transposed_conv2d_bias_relu_shape",
    "infer_transposed_conv2d_bias_relu_shape_with_attrs",
    "infer_transposed_conv2d_bias_shape",
    "infer_transposed_conv2d_bias_shape_with_attrs",
    "infer_transposed_conv2d_shape",
    "infer_transposed_conv2d_shape_with_attrs",
    "normalize_conv1d_bias_attrs",
    "normalize_conv2d_bias_attrs",
    "normalize_transposed_conv2d_attrs",
    "resolve_conv1d_bias_add_relu_shape",
    "resolve_conv1d_bias_add_shape",
    "resolve_conv1d_bias_relu_shape",
    "resolve_conv1d_bias_shape",
    "resolve_conv2d_bias_add_relu_shape",
    "resolve_conv2d_bias_add_shape",
    "resolve_conv2d_bias_relu_shape",
    "resolve_conv2d_bias_shape",
    "resolve_conv2d_shape",
    "resolve_transposed_conv2d_bias_add_relu_shape",
    "resolve_transposed_conv2d_bias_add_shape",
    "resolve_transposed_conv2d_bias_relu_shape",
    "resolve_transposed_conv2d_bias_shape",
    "resolve_transposed_conv2d_shape",
    "transposed_conv2d",
    "transposed_conv2d_bias",
    "transposed_conv2d_bias_add",
    "transposed_conv2d_bias_add_relu",
    "transposed_conv2d_bias_relu",
]
