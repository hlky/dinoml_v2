from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


POOLING_DTYPES = ("float16", "float32", "bfloat16")


def infer_avg_pool1d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_avg_pool1d_shape_with_attrs(
        input_shapes,
        {"kernel_size": (1,), "stride": (1,), "padding": (0,)},
    )


def infer_avg_pool2d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_avg_pool2d_shape_with_attrs(
        input_shapes,
        {"kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0)},
    )


def infer_max_pool2d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_max_pool2d_shape_with_attrs(
        input_shapes,
        {"kernel_size": (1, 1), "stride": (1, 1), "padding": (0, 0)},
    )


def infer_avg_pool1d_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("avg_pool1d expects one tensor input")
    return _resolve_pool1d_shape(
        "avg_pool1d",
        input_shapes[0],
        attrs.get("kernel_size"),
        attrs.get("stride"),
        attrs.get("padding", (0,)),
    )


def infer_avg_pool2d_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("avg_pool2d expects one tensor input")
    return _resolve_pool2d_shape(
        "avg_pool2d",
        input_shapes[0],
        attrs.get("kernel_size"),
        attrs.get("stride"),
        attrs.get("padding", (0, 0)),
    )


def infer_max_pool2d_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("max_pool2d expects one tensor input")
    return _resolve_pool2d_shape(
        "max_pool2d",
        input_shapes[0],
        attrs.get("kernel_size"),
        attrs.get("stride"),
        attrs.get("padding", (0, 0)),
    )


def normalize_avg_pool1d_attrs(
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> tuple[list[int], list[int], list[int]]:
    return _normalize_pool1d_attrs("avg_pool1d", kernel_size, stride, padding)


def normalize_avg_pool2d_attrs(
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> tuple[list[int], list[int], list[int]]:
    return _normalize_pool2d_attrs("avg_pool2d", kernel_size, stride, padding)


def normalize_max_pool2d_attrs(
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> tuple[list[int], list[int], list[int]]:
    return _normalize_pool2d_attrs("max_pool2d", kernel_size, stride, padding)


def resolve_avg_pool1d_shape(
    input_shape: Sequence[int],
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> list[int]:
    return _resolve_pool1d_shape("avg_pool1d", input_shape, kernel_size, stride, padding)


def resolve_avg_pool2d_shape(
    input_shape: Sequence[int],
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> list[int]:
    return _resolve_pool2d_shape("avg_pool2d", input_shape, kernel_size, stride, padding)


def resolve_max_pool2d_shape(
    input_shape: Sequence[int],
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> list[int]:
    return _resolve_pool2d_shape("max_pool2d", input_shape, kernel_size, stride, padding)


def _normalize_pool1d_attrs(
    op_name: str,
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> tuple[list[int], list[int], list[int]]:
    kernel = _normalize_positive_single(kernel_size, f"{op_name} kernel_size")
    normalized_stride = kernel if stride is None else _normalize_positive_single(stride, f"{op_name} stride")
    normalized_padding = _normalize_non_negative_single(padding, f"{op_name} padding")
    return list(kernel), list(normalized_stride), list(normalized_padding)


def _normalize_pool2d_attrs(
    op_name: str,
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> tuple[list[int], list[int], list[int]]:
    kernel = _normalize_positive_pair(kernel_size, f"{op_name} kernel_size")
    normalized_stride = kernel if stride is None else _normalize_positive_pair(stride, f"{op_name} stride")
    normalized_padding = _normalize_non_negative_pair(padding, f"{op_name} padding")
    return list(kernel), list(normalized_stride), list(normalized_padding)


def _resolve_pool1d_shape(
    op_name: str,
    input_shape: Sequence[int],
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> list[int]:
    if len(input_shape) != 3:
        raise ValueError(f"{op_name} expects rank-3 NCL input, got rank {len(input_shape)}")
    kernel, normalized_stride, normalized_padding = _normalize_pool1d_attrs(op_name, kernel_size, stride, padding)
    n, c, length = [int(dim) for dim in input_shape]
    out_length = _pool_output_dim(op_name, length, kernel[0], normalized_stride[0], normalized_padding[0], "length")
    return [n, c, out_length]


def _resolve_pool2d_shape(
    op_name: str,
    input_shape: Sequence[int],
    kernel_size: Any,
    stride: Any | None,
    padding: Any,
) -> list[int]:
    if len(input_shape) != 4:
        raise ValueError(f"{op_name} expects rank-4 NCHW input, got rank {len(input_shape)}")
    kernel, normalized_stride, normalized_padding = _normalize_pool2d_attrs(op_name, kernel_size, stride, padding)
    n, c, height, width = [int(dim) for dim in input_shape]
    out_height = _pool_output_dim(op_name, height, kernel[0], normalized_stride[0], normalized_padding[0], "height")
    out_width = _pool_output_dim(op_name, width, kernel[1], normalized_stride[1], normalized_padding[1], "width")
    return [n, c, out_height, out_width]


def _pool_output_dim(op_name: str, dim: int, kernel: int, stride: int, padding: int, axis_name: str) -> int:
    output = (int(dim) + 2 * padding - kernel) // stride + 1
    if output <= 0:
        raise ValueError(
            f"{op_name} output {axis_name} must be positive; got input={dim}, "
            f"kernel={kernel}, stride={stride}, padding={padding}"
        )
    return output


def _normalize_positive_single(value: Any, name: str) -> tuple[int]:
    single = _normalize_single(value, name)
    if single[0] <= 0:
        raise ValueError(f"{name} must contain positive integers, got {value!r}")
    return single


def _normalize_positive_pair(value: Any, name: str) -> tuple[int, int]:
    pair = _normalize_pair(value, name)
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} must contain positive integers, got {value!r}")
    return pair


def _normalize_non_negative_single(value: Any, name: str) -> tuple[int]:
    single = _normalize_single(value, name)
    if single[0] < 0:
        raise ValueError(f"{name} must contain non-negative integers, got {value!r}")
    return single


def _normalize_non_negative_pair(value: Any, name: str) -> tuple[int, int]:
    pair = _normalize_pair(value, name)
    if pair[0] < 0 or pair[1] < 0:
        raise ValueError(f"{name} must contain non-negative integers, got {value!r}")
    return pair


def _normalize_single(value: Any, name: str) -> tuple[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (int(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if len(values) != 1:
            raise ValueError(f"{name} must be an integer or length-1 sequence of integers, got {value!r}")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
            raise ValueError(f"{name} must contain non-bool integers, got {value!r}")
        return (int(values[0]),)
    raise ValueError(f"{name} must be an integer or length-1 sequence of integers, got {value!r}")


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


@op_def
class AvgPool1d(OpDef):
    name = "avg_pool1d"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("kernel_size", "ints", required=True),
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0,)),
        ),
    )
    infer_shape = infer_avg_pool1d_shape
    infer_shape_with_attrs = infer_avg_pool1d_shape_with_attrs
    allowed_dtypes = POOLING_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_avg_pool1d", library="model", source_template="avg_pool1d_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_avg_pool1d", library="model", source_template="avg_pool1d_cuda.cu.j2"),
    }
    frontend = FrontendBinding("avg_pool1d")
    description = (
        "Dense rank-3 NCL avg_pool1d with static shapes, floor output shape, "
        "zero padding included in the divisor, and fp32 accumulation."
    )

    @classmethod
    def forward(cls, x: Any, kernel_size: Any, stride: Any | None = None, padding: Any = 0) -> Tensor:
        tensor = as_tensor(x)
        if tensor.dtype not in POOLING_DTYPES:
            raise ValueError(f"avg_pool1d does not support dtype {tensor.dtype}")
        if tensor.rank != 3:
            raise ValueError(f"avg_pool1d expects rank-3 NCL input, got rank {tensor.rank}")
        if tensor.dynamic:
            raise ValueError("avg_pool1d currently supports only static input shapes")
        kernel, normalized_stride, normalized_padding = normalize_avg_pool1d_attrs(kernel_size, stride, padding)
        out_shape = resolve_avg_pool1d_shape(tensor.shape, kernel, normalized_stride, normalized_padding)
        return tensor.builder.emit(
            "avg_pool1d",
            [tensor],
            out_shape,
            tensor.dtype,
            {"kernel_size": kernel, "stride": normalized_stride, "padding": normalized_padding},
            shape_spec=out_shape,
        )


@op_def
class AvgPool2d(OpDef):
    name = "avg_pool2d"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("kernel_size", "ints", required=True),
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
        ),
    )
    infer_shape = infer_avg_pool2d_shape
    infer_shape_with_attrs = infer_avg_pool2d_shape_with_attrs
    allowed_dtypes = POOLING_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_avg_pool2d", library="model", source_template="avg_pool2d_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_avg_pool2d", library="model", source_template="avg_pool2d_cuda.cu.j2"),
    }
    frontend = FrontendBinding("avg_pool2d")
    description = (
        "Dense rank-4 NCHW avg_pool2d with static shapes, floor output shape, "
        "zero padding included in the divisor, and fp32 accumulation."
    )

    @classmethod
    def forward(cls, x: Any, kernel_size: Any, stride: Any | None = None, padding: Any = 0) -> Tensor:
        tensor = as_tensor(x)
        if tensor.dtype not in POOLING_DTYPES:
            raise ValueError(f"avg_pool2d does not support dtype {tensor.dtype}")
        if tensor.rank != 4:
            raise ValueError(f"avg_pool2d expects rank-4 NCHW input, got rank {tensor.rank}")
        if tensor.dynamic:
            raise ValueError("avg_pool2d currently supports only static input shapes")
        kernel, normalized_stride, normalized_padding = normalize_avg_pool2d_attrs(kernel_size, stride, padding)
        out_shape = resolve_avg_pool2d_shape(tensor.shape, kernel, normalized_stride, normalized_padding)
        return tensor.builder.emit(
            "avg_pool2d",
            [tensor],
            out_shape,
            tensor.dtype,
            {"kernel_size": kernel, "stride": normalized_stride, "padding": normalized_padding},
            shape_spec=out_shape,
        )


@op_def
class MaxPool2d(OpDef):
    name = "max_pool2d"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("kernel_size", "ints", required=True),
            AttrDef("stride", "ints", required=True),
            AttrDef("padding", "ints", default=(0, 0)),
        ),
    )
    infer_shape = infer_max_pool2d_shape
    infer_shape_with_attrs = infer_max_pool2d_shape_with_attrs
    allowed_dtypes = POOLING_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_max_pool2d", library="model", source_template="max_pool2d_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_max_pool2d", library="model", source_template="max_pool2d_cuda.cu.j2"),
    }
    frontend = FrontendBinding("max_pool2d")
    description = (
        "Dense rank-4 NCHW max_pool2d with static shapes, floor output shape, "
        "implicit negative-infinity padding, and fp32 comparisons."
    )

    @classmethod
    def forward(cls, x: Any, kernel_size: Any, stride: Any | None = None, padding: Any = 0) -> Tensor:
        tensor = as_tensor(x)
        if tensor.dtype not in POOLING_DTYPES:
            raise ValueError(f"max_pool2d does not support dtype {tensor.dtype}")
        if tensor.rank != 4:
            raise ValueError(f"max_pool2d expects rank-4 NCHW input, got rank {tensor.rank}")
        if tensor.dynamic:
            raise ValueError("max_pool2d currently supports only static input shapes")
        kernel, normalized_stride, normalized_padding = normalize_max_pool2d_attrs(kernel_size, stride, padding)
        out_shape = resolve_max_pool2d_shape(tensor.shape, kernel, normalized_stride, normalized_padding)
        return tensor.builder.emit(
            "max_pool2d",
            [tensor],
            out_shape,
            tensor.dtype,
            {"kernel_size": kernel, "stride": normalized_stride, "padding": normalized_padding},
            shape_spec=out_shape,
        )


def avg_pool1d(x: Any, kernel_size: Any, stride: Any | None = None, padding: Any = 0) -> Tensor:
    return AvgPool1d.forward(x, kernel_size, stride, padding)


def avg_pool2d(x: Any, kernel_size: Any, stride: Any | None = None, padding: Any = 0) -> Tensor:
    return AvgPool2d.forward(x, kernel_size, stride, padding)


def max_pool2d(x: Any, kernel_size: Any, stride: Any | None = None, padding: Any = 0) -> Tensor:
    return MaxPool2d.forward(x, kernel_size, stride, padding)


__all__ = [
    "AvgPool1d",
    "AvgPool2d",
    "MaxPool2d",
    "POOLING_DTYPES",
    "avg_pool1d",
    "avg_pool2d",
    "infer_avg_pool1d_shape",
    "infer_avg_pool1d_shape_with_attrs",
    "infer_avg_pool2d_shape",
    "infer_avg_pool2d_shape_with_attrs",
    "infer_max_pool2d_shape",
    "infer_max_pool2d_shape_with_attrs",
    "normalize_avg_pool1d_attrs",
    "normalize_avg_pool2d_attrs",
    "normalize_max_pool2d_attrs",
    "resolve_avg_pool1d_shape",
    "resolve_avg_pool2d_shape",
    "resolve_max_pool2d_shape",
    "max_pool2d",
]
