from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.frontend import GraphBuilder, Tensor, as_tensor
from dinoml.ir import normalize_dtype
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def
from dinoml.shapes import symbolic_int_expr


TENSOR_FILTER_HELPER_DTYPES = ("float16", "float32")
TENSOR_FILTER_HELPER_OPS = (
    "fir_downsample2d",
    "fir_filter_pad2",
    "fir_upsample2d",
    "kdownsample2d_weight",
    "kupsample2d_weight",
)


def infer_fir_downsample2d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("fir_downsample2d expects one tensor input")
    return resolve_fir_downsample2d_shape(input_shapes[0])


def infer_fir_filter_pad2_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("fir_filter_pad2 expects one tensor input")
    return resolve_fir_filter_pad2_shape(input_shapes[0])


def infer_fir_upsample2d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_fir_upsample2d_shape_with_attrs(
        input_shapes,
        {"up": 2, "pad0": 2, "pad1": 1},
    )


def infer_fir_upsample2d_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("fir_upsample2d expects one tensor input")
    normalized = normalize_fir_upsample2d_attrs(
        up=attrs.get("up", 2),
        pad0=attrs.get("pad0", 2),
        pad1=attrs.get("pad1", 1),
    )
    return resolve_fir_upsample2d_shape(
        input_shapes[0],
        up=int(normalized["up"]),
        pad0=int(normalized["pad0"]),
        pad1=int(normalized["pad1"]),
    )


def infer_kdownsample2d_weight_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("kdownsample2d_weight shape inference requires attrs")


def infer_kdownsample2d_weight_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    if input_shapes:
        raise ValueError("kdownsample2d_weight expects no tensor inputs")
    channels = normalize_tensor_filter_channels(attrs.get("channels"), "kdownsample2d_weight channels")
    return resolve_tensor_filter_weight_shape(channels)


def infer_kupsample2d_weight_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    raise ValueError("kupsample2d_weight shape inference requires attrs")


def infer_kupsample2d_weight_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    if input_shapes:
        raise ValueError("kupsample2d_weight expects no tensor inputs")
    channels = normalize_tensor_filter_channels(attrs.get("channels"), "kupsample2d_weight channels")
    return resolve_tensor_filter_weight_shape(channels)


def normalize_fir_upsample2d_attrs(*, up: Any = 2, pad0: Any = 2, pad1: Any = 1) -> dict[str, int]:
    normalized_up = _positive_int_attr(up, "fir_upsample2d up")
    normalized_pad0 = _int_attr(pad0, "fir_upsample2d pad0")
    normalized_pad1 = _int_attr(pad1, "fir_upsample2d pad1")
    return {
        "up": normalized_up,
        "pad0": normalized_pad0,
        "pad1": normalized_pad1,
    }


def normalize_tensor_filter_channels(value: Any, name: str) -> int:
    return _positive_int_attr(value, name)


def resolve_fir_downsample2d_shape(input_shape: Sequence[int]) -> list[int]:
    if len(input_shape) != 4:
        raise ValueError(f"fir_downsample2d expects rank-4 NHWC input, got rank {len(input_shape)}")
    n, height, width, channels = [int(dim) for dim in input_shape]
    out_height = height // 2
    out_width = width // 2
    if out_height <= 0 or out_width <= 0:
        raise ValueError(
            "fir_downsample2d output spatial dims must be positive; "
            f"got input height={height}, width={width}"
        )
    return [n, out_height, out_width, channels]


def resolve_fir_filter_pad2_shape(input_shape: Sequence[int]) -> list[int]:
    if len(input_shape) != 4:
        raise ValueError(f"fir_filter_pad2 expects rank-4 NHWC input, got rank {len(input_shape)}")
    n, height, width, channels = [int(dim) for dim in input_shape]
    return [n, height + 1, width + 1, channels]


def resolve_fir_upsample2d_shape(
    input_shape: Sequence[int],
    *,
    up: int,
    pad0: int,
    pad1: int,
) -> list[int]:
    if len(input_shape) != 4:
        raise ValueError(f"fir_upsample2d expects rank-4 NHWC input, got rank {len(input_shape)}")
    n, height, width, channels = [int(dim) for dim in input_shape]
    out_height = int(height) * int(up) + int(pad0) + int(pad1) - 3
    out_width = int(width) * int(up) + int(pad0) + int(pad1) - 3
    if out_height <= 0 or out_width <= 0:
        raise ValueError(
            "fir_upsample2d output spatial dims must be positive; "
            f"got input height={height}, width={width}, up={up}, pad0={pad0}, pad1={pad1}"
        )
    return [n, out_height, out_width, channels]


def resolve_tensor_filter_weight_shape(channels: int) -> list[int]:
    normalized_channels = normalize_tensor_filter_channels(channels, "tensor filter helper channels")
    return [normalized_channels, normalized_channels, 4, 4]


@op_def
class FirDownsample2d(OpDef):
    name = "fir_downsample2d"
    schema = OpSchema(inputs=("x",))
    infer_shape = infer_fir_downsample2d_shape
    allowed_dtypes = TENSOR_FILTER_HELPER_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_fir_downsample2d", library="model", source_template="fir_downsample2d_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_fir_downsample2d", library="model", source_template="fir_downsample2d_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_fir_downsample2d", library="model", source_template="fir_downsample2d_gpu.j2"),
    }
    frontend = FrontendBinding("fir_downsample2d")
    description = "Channels-last FIR downsampling with a fixed [1, 3, 3, 1] filter and stride 2."

    @classmethod
    def forward(cls, x: Any) -> Tensor:
        tensor = as_tensor(x)
        _validate_tensor_filter_input("fir_downsample2d", tensor)
        out_shape = resolve_fir_downsample2d_shape(tensor.shape)
        out_shape_spec = [
            _copy_shape_dim(tensor.shape_spec[0]),
            symbolic_int_expr("div", _copy_shape_dim(tensor.shape_spec[1]), 2),
            symbolic_int_expr("div", _copy_shape_dim(tensor.shape_spec[2]), 2),
            _copy_shape_dim(tensor.shape_spec[3]),
        ]
        return tensor.builder.emit("fir_downsample2d", [tensor], out_shape, tensor.dtype, {}, shape_spec=out_shape_spec)


@op_def
class FirFilterPad2(OpDef):
    name = "fir_filter_pad2"
    schema = OpSchema(inputs=("x",))
    infer_shape = infer_fir_filter_pad2_shape
    allowed_dtypes = TENSOR_FILTER_HELPER_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_fir_filter_pad2", library="model", source_template="fir_filter_pad2_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_fir_filter_pad2", library="model", source_template="fir_filter_pad2_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_fir_filter_pad2", library="model", source_template="fir_filter_pad2_gpu.j2"),
    }
    frontend = FrontendBinding("fir_filter_pad2")
    description = "Channels-last FIR filter stage with pad=2 and a fixed normalized [1, 3, 3, 1] filter."

    @classmethod
    def forward(cls, x: Any) -> Tensor:
        tensor = as_tensor(x)
        _validate_tensor_filter_input("fir_filter_pad2", tensor)
        out_shape = resolve_fir_filter_pad2_shape(tensor.shape)
        out_shape_spec = [
            _copy_shape_dim(tensor.shape_spec[0]),
            symbolic_int_expr("add", _copy_shape_dim(tensor.shape_spec[1]), 1),
            symbolic_int_expr("add", _copy_shape_dim(tensor.shape_spec[2]), 1),
            _copy_shape_dim(tensor.shape_spec[3]),
        ]
        return tensor.builder.emit("fir_filter_pad2", [tensor], out_shape, tensor.dtype, {}, shape_spec=out_shape_spec)


@op_def
class FirUpsample2d(OpDef):
    name = "fir_upsample2d"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("up", "int", 2),
            AttrDef("pad0", "int", 2),
            AttrDef("pad1", "int", 1),
        ),
    )
    infer_shape = infer_fir_upsample2d_shape
    infer_shape_with_attrs = infer_fir_upsample2d_shape_with_attrs
    allowed_dtypes = TENSOR_FILTER_HELPER_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_fir_upsample2d", library="model", source_template="fir_upsample2d_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_fir_upsample2d", library="model", source_template="fir_upsample2d_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_fir_upsample2d", library="model", source_template="fir_upsample2d_gpu.j2"),
    }
    frontend = FrontendBinding("fir_upsample2d")
    description = "Channels-last FIR upsampling with a fixed [1, 3, 3, 1] filter, configurable up factor, and pads."

    @classmethod
    def forward(cls, x: Any, up: int = 2, pad0: int = 2, pad1: int = 1) -> Tensor:
        tensor = as_tensor(x)
        _validate_tensor_filter_input("fir_upsample2d", tensor)
        normalized = normalize_fir_upsample2d_attrs(up=up, pad0=pad0, pad1=pad1)
        out_shape = resolve_fir_upsample2d_shape(
            tensor.shape,
            up=int(normalized["up"]),
            pad0=int(normalized["pad0"]),
            pad1=int(normalized["pad1"]),
        )
        out_shape_spec = [
            _copy_shape_dim(tensor.shape_spec[0]),
            symbolic_int_expr(
                "sub",
                symbolic_int_expr(
                    "add",
                    symbolic_int_expr("mul", _copy_shape_dim(tensor.shape_spec[1]), int(normalized["up"])),
                    int(normalized["pad0"]) + int(normalized["pad1"]),
                ),
                3,
            ),
            symbolic_int_expr(
                "sub",
                symbolic_int_expr(
                    "add",
                    symbolic_int_expr("mul", _copy_shape_dim(tensor.shape_spec[2]), int(normalized["up"])),
                    int(normalized["pad0"]) + int(normalized["pad1"]),
                ),
                3,
            ),
            _copy_shape_dim(tensor.shape_spec[3]),
        ]
        return tensor.builder.emit(
            "fir_upsample2d",
            [tensor],
            out_shape,
            tensor.dtype,
            normalized,
            shape_spec=out_shape_spec,
        )


@op_def
class Kdownsample2dWeight(OpDef):
    name = "kdownsample2d_weight"
    schema = OpSchema(
        inputs=(),
        attrs=(
            AttrDef("channels", "int", required=True),
            AttrDef("dtype", "dtype", "float32"),
        ),
    )
    infer_shape = infer_kdownsample2d_weight_shape
    infer_shape_with_attrs = infer_kdownsample2d_weight_shape_with_attrs
    allowed_dtypes = TENSOR_FILTER_HELPER_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_kdownsample2d_weight", library="model", source_template="kdownsample2d_weight_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_kdownsample2d_weight", library="model", source_template="kdownsample2d_weight_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_kdownsample2d_weight", library="model", source_template="kdownsample2d_weight_gpu.j2"),
    }
    frontend = FrontendBinding("kdownsample2d_weight")
    description = "Create the fixed depthwise downsampling FIR kernel tensor used by Diffusers-style helper paths."

    @classmethod
    def forward(cls, channels: int, dtype: str = "float32") -> Tensor:
        normalized_dtype = normalize_dtype(dtype)
        if normalized_dtype not in TENSOR_FILTER_HELPER_DTYPES:
            raise ValueError(f"kdownsample2d_weight does not support dtype {normalized_dtype}")
        normalized_channels = normalize_tensor_filter_channels(channels, "kdownsample2d_weight channels")
        out_shape = resolve_tensor_filter_weight_shape(normalized_channels)
        attrs = {"channels": normalized_channels, "dtype": normalized_dtype}
        return GraphBuilder.current().emit(
            "kdownsample2d_weight",
            [],
            out_shape,
            normalized_dtype,
            attrs,
            shape_spec=out_shape,
        )


@op_def
class Kupsample2dWeight(OpDef):
    name = "kupsample2d_weight"
    schema = OpSchema(
        inputs=(),
        attrs=(
            AttrDef("channels", "int", required=True),
            AttrDef("dtype", "dtype", "float32"),
        ),
    )
    infer_shape = infer_kupsample2d_weight_shape
    infer_shape_with_attrs = infer_kupsample2d_weight_shape_with_attrs
    allowed_dtypes = TENSOR_FILTER_HELPER_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_kupsample2d_weight", library="model", source_template="kupsample2d_weight_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_kupsample2d_weight", library="model", source_template="kupsample2d_weight_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_kupsample2d_weight", library="model", source_template="kupsample2d_weight_gpu.j2"),
    }
    frontend = FrontendBinding("kupsample2d_weight")
    description = "Create the fixed depthwise upsampling FIR kernel tensor used by Diffusers-style helper paths."

    @classmethod
    def forward(cls, channels: int, dtype: str = "float32") -> Tensor:
        normalized_dtype = normalize_dtype(dtype)
        if normalized_dtype not in TENSOR_FILTER_HELPER_DTYPES:
            raise ValueError(f"kupsample2d_weight does not support dtype {normalized_dtype}")
        normalized_channels = normalize_tensor_filter_channels(channels, "kupsample2d_weight channels")
        out_shape = resolve_tensor_filter_weight_shape(normalized_channels)
        attrs = {"channels": normalized_channels, "dtype": normalized_dtype}
        return GraphBuilder.current().emit(
            "kupsample2d_weight",
            [],
            out_shape,
            normalized_dtype,
            attrs,
            shape_spec=out_shape,
        )


def fir_downsample2d(x: Any) -> Tensor:
    return FirDownsample2d.forward(x)


def fir_filter_pad2(x: Any) -> Tensor:
    return FirFilterPad2.forward(x)


def fir_upsample2d(x: Any, up: int = 2, pad0: int = 2, pad1: int = 1) -> Tensor:
    return FirUpsample2d.forward(x, up, pad0, pad1)


def kdownsample2d_weight(channels: int, dtype: str = "float32") -> Tensor:
    return Kdownsample2dWeight.forward(channels, dtype)


def kupsample2d_weight(channels: int, dtype: str = "float32") -> Tensor:
    return Kupsample2dWeight.forward(channels, dtype)


def _validate_tensor_filter_input(op_name: str, tensor: Tensor) -> None:
    if tensor.dtype not in TENSOR_FILTER_HELPER_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {tensor.dtype}")
    if tensor.rank != 4:
        raise ValueError(f"{op_name} expects rank-4 NHWC input, got rank {tensor.rank}")


def _positive_int_attr(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _int_attr(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _copy_shape_dim(dim: int | Mapping[str, Any]) -> int | dict[str, Any]:
    return dict(dim) if isinstance(dim, Mapping) else int(dim)


__all__ = [
    "FirDownsample2d",
    "FirFilterPad2",
    "FirUpsample2d",
    "Kdownsample2dWeight",
    "Kupsample2dWeight",
    "TENSOR_FILTER_HELPER_DTYPES",
    "TENSOR_FILTER_HELPER_OPS",
    "fir_downsample2d",
    "fir_filter_pad2",
    "fir_upsample2d",
    "infer_fir_downsample2d_shape",
    "infer_fir_filter_pad2_shape",
    "infer_fir_upsample2d_shape",
    "infer_fir_upsample2d_shape_with_attrs",
    "infer_kdownsample2d_weight_shape",
    "infer_kdownsample2d_weight_shape_with_attrs",
    "infer_kupsample2d_weight_shape",
    "infer_kupsample2d_weight_shape_with_attrs",
    "kdownsample2d_weight",
    "kupsample2d_weight",
    "normalize_fir_upsample2d_attrs",
    "normalize_tensor_filter_channels",
    "resolve_fir_downsample2d_shape",
    "resolve_fir_filter_pad2_shape",
    "resolve_fir_upsample2d_shape",
    "resolve_tensor_filter_weight_shape",
]
