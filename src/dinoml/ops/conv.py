from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_candidate_set,
    cutlass_conv_candidates,
    cutlass_conv_profiler_symbol,
    cutlass_conv_symbol,
)
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, KernelVariant, OpDef, OpRegistry, OpSchema


CONV2D_BIAS_DTYPES = ("float16", "float32")


def infer_conv2d_bias_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_conv2d_bias_shape_with_attrs(
        input_shapes,
        {"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
    )


def infer_conv2d_bias_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 3:
        raise ValueError("conv2d_bias expects activation, weight, and bias inputs")
    stride, padding, dilation, groups = normalize_conv2d_bias_attrs(
        attrs.get("stride", (1, 1)),
        attrs.get("padding", (0, 0)),
        attrs.get("dilation", (1, 1)),
        attrs.get("groups", 1),
    )
    return resolve_conv2d_bias_shape(
        input_shapes[0],
        input_shapes[1],
        input_shapes[2],
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
    if len(input_shape) != 4:
        raise ValueError(f"conv2d_bias expects rank-4 NCHW activation, got rank {len(input_shape)}")
    if len(weight_shape) != 4:
        raise ValueError(f"conv2d_bias expects rank-4 OIHW weight, got rank {len(weight_shape)}")
    if len(bias_shape) != 1:
        raise ValueError(f"conv2d_bias expects rank-1 bias, got rank {len(bias_shape)}")
    normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv2d_bias_attrs(
        stride,
        padding,
        dilation,
        groups,
    )
    if normalized_groups != 1:
        raise NotImplementedError(f"conv2d_bias currently supports groups=1 only, got {normalized_groups}")

    batch, in_channels, in_height, in_width = [int(dim) for dim in input_shape]
    out_channels, weight_in_channels, kernel_h, kernel_w = [int(dim) for dim in weight_shape]
    bias_channels = int(bias_shape[0])

    if weight_in_channels != in_channels:
        raise ValueError(
            "conv2d_bias weight input channels must match activation channels for groups=1: "
            f"got activation C={in_channels}, weight C={weight_in_channels}"
        )
    if bias_channels != out_channels:
        raise ValueError(
            f"conv2d_bias bias length must match weight output channels, got bias {bias_channels} and weight O={out_channels}"
        )
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError(f"conv2d_bias kernel dimensions must be positive, got {weight_shape!r}")

    out_height = _conv_output_dim(
        "conv2d_bias",
        in_height,
        kernel_h,
        normalized_stride[0],
        normalized_padding[0],
        normalized_dilation[0],
        "height",
    )
    out_width = _conv_output_dim(
        "conv2d_bias",
        in_width,
        kernel_w,
        normalized_stride[1],
        normalized_padding[1],
        normalized_dilation[1],
        "width",
    )
    return [batch, out_channels, out_height, out_width]


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


def register_conv_ops(registry: OpRegistry) -> None:
    registry.register(
        OpDef(
            name="conv2d_bias",
            schema=OpSchema(
                inputs=("x", "weight", "bias"),
                attrs=(
                    AttrDef("stride", "ints", required=True),
                    AttrDef("padding", "ints", default=(0, 0)),
                    AttrDef("dilation", "ints", default=(1, 1)),
                    AttrDef("groups", "int", default=1),
                ),
            ),
            infer_shape=infer_conv2d_bias_shape,
            infer_shape_with_attrs=infer_conv2d_bias_shape_with_attrs,
            allowed_dtypes=CONV2D_BIAS_DTYPES,
            backend_kernels=_cutlass_conv_backend_kernels("conv2d_bias"),
            frontend=FrontendBinding("conv2d_bias"),
            description=(
                "Bounded conv2d_bias frontend with public NCHW/OIHW semantics, "
                "groups=1 only, static rank-4 shapes, and CPU reference execution. "
                "CUDA compile emits manifest/codegen scaffold metadata and then rejects "
                "before module build until a provider-backed launcher is real."
            ),
        )
    )


def _cutlass_conv_backend_kernels(op_name: str) -> dict[str, KernelBinding]:
    return {
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


__all__ = [
    "CONV2D_BIAS_DTYPES",
    "infer_conv2d_bias_shape",
    "infer_conv2d_bias_shape_with_attrs",
    "normalize_conv2d_bias_attrs",
    "register_conv_ops",
    "resolve_conv2d_bias_shape",
]
