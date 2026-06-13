from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dinoml import ops
from dinoml.frontend import Tensor


def one_hot(input: Any, num_classes: int) -> Tensor:
    return ops.one_hot(input, num_classes)


def pad(input: Any, pad: Any, mode: str = "constant", value: Any | None = None) -> Tensor:
    if mode != "constant":
        raise NotImplementedError(f"pad currently supports only mode='constant', got {mode!r}")
    return ops.pad(input, pad, value=0.0 if value is None else value)


def softmax(input: Any, dim: int | None = None, _stacklevel: int = 3, dtype: Any | None = None) -> Tensor:
    del _stacklevel
    if dtype is not None:
        raise NotImplementedError("softmax currently does not support dtype=")
    return ops.softmax(input, dim=-1 if dim is None else dim)


def log_softmax(input: Any, dim: int | None = None, _stacklevel: int = 3, dtype: Any | None = None) -> Tensor:
    del _stacklevel
    if dtype is not None:
        raise NotImplementedError("log_softmax currently does not support dtype=")
    return ops.log(ops.softmax(input, dim=-1 if dim is None else dim))


def silu(input: Any, inplace: bool = False) -> Tensor:
    if inplace:
        raise NotImplementedError("silu currently does not support inplace=True")
    return ops.silu(input)


def layer_norm(
    input: Any,
    normalized_shape: int | Sequence[int],
    weight: Any | None = None,
    bias: Any | None = None,
    eps: float = 1e-5,
) -> Tensor:
    if weight is None or bias is None:
        raise NotImplementedError("layer_norm currently requires weight and bias tensors")
    if isinstance(normalized_shape, int) and not isinstance(normalized_shape, bool):
        normalized_shape = [int(normalized_shape)]
    return ops.layer_norm(input, weight, bias, eps=eps, normalized_shape=normalized_shape)


def group_norm(
    input: Any,
    num_groups: int,
    weight: Any | None = None,
    bias: Any | None = None,
    eps: float = 1e-5,
) -> Tensor:
    return ops.group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)


def conv1d(
    input: Any,
    weight: Any,
    bias: Any | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> Tensor:
    if bias is None:
        raise NotImplementedError("conv1d currently requires bias tensor input")
    return ops.conv1d_bias(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv3d(
    input: Any,
    weight: Any,
    bias: Any | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> Tensor:
    if bias is None:
        return ops.conv3d(
            input,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    return ops.conv3d_bias(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv_transpose1d(
    input: Any,
    weight: Any,
    bias: Any | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    output_padding: int | Sequence[int] = 0,
    groups: int = 1,
    dilation: int | Sequence[int] = 1,
) -> Tensor:
    if bias is not None:
        raise NotImplementedError("conv_transpose1d currently requires bias=None")
    return ops.transposed_conv1d(
        input,
        weight,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def conv_transpose2d(
    input: Any,
    weight: Any,
    bias: Any | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    output_padding: int | Sequence[int] = 0,
    groups: int = 1,
    dilation: int | Sequence[int] = 1,
) -> Tensor:
    if bias is not None:
        raise NotImplementedError("conv_transpose2d currently requires bias=None")
    return ops.transposed_conv2d(
        input,
        weight,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def pixel_shuffle(input: Any, upscale_factor: int) -> Tensor:
    return ops.pixel_shuffle(input, upscale_factor)


def pixel_unshuffle(input: Any, downscale_factor: int) -> Tensor:
    return ops.pixel_unshuffle(input, downscale_factor)


def normalize(input: Any, p: float = 2.0, dim: int = -1, eps: float = 1e-12, out: Any | None = None) -> Tensor:
    return ops.normalize(input, p=p, dim=dim, eps=eps, out=out)


__all__ = [
    "one_hot",
    "pad",
    "softmax",
    "log_softmax",
    "silu",
    "layer_norm",
    "group_norm",
    "conv1d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "pixel_shuffle",
    "pixel_unshuffle",
    "normalize",
]
