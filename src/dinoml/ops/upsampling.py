from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.collections import permute
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def
from dinoml.shapes import symbolic_int_expr


UPSAMPLING_DTYPES = ("float16", "float32", "bfloat16")

_UPSAMPLING_LINEAR_MODES = {
    "upsampling1d": "linear",
    "upsampling1d_add": "linear",
    "upsampling2d": "bilinear",
    "upsampling2d_add": "bilinear",
    "upsampling3d": "trilinear",
    "upsampling3d_add": "trilinear",
}

_UPSAMPLING_ALLOWED_MODES = {
    "upsampling1d": ("linear", "nearest", "nearest-exact"),
    "upsampling1d_add": ("linear", "nearest", "nearest-exact"),
    "upsampling2d": ("bilinear", "nearest", "nearest-exact"),
    "upsampling2d_add": ("bilinear", "nearest", "nearest-exact"),
    "upsampling3d": ("trilinear", "nearest", "nearest-exact"),
    "upsampling3d_add": ("trilinear", "nearest", "nearest-exact"),
}

_UPSAMPLING_RANKS = {
    "upsampling1d": 3,
    "upsampling1d_add": 3,
    "upsampling2d": 4,
    "upsampling2d_add": 4,
    "upsampling3d": 5,
    "upsampling3d_add": 5,
    "upsampling3d_compress_time": 5,
}


def infer_upsampling1d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_upsampling1d_shape_with_attrs(
        input_shapes,
        {"scale_factor": 1.0, "mode": "nearest", "align_corners": None},
    )


def infer_upsampling1d_add_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_upsampling1d_add_shape_with_attrs(
        input_shapes,
        {"scale_factor": 1.0, "mode": "nearest", "align_corners": None},
    )


def infer_upsampling2d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_upsampling2d_shape_with_attrs(
        input_shapes,
        {"scale_factor": 1.0, "mode": "nearest", "align_corners": None},
    )


def infer_upsampling2d_add_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_upsampling2d_add_shape_with_attrs(
        input_shapes,
        {"scale_factor": 1.0, "mode": "nearest", "align_corners": None},
    )


def infer_upsampling3d_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_upsampling3d_shape_with_attrs(
        input_shapes,
        {"scale_factor": 1.0, "mode": "nearest", "align_corners": None},
    )


def infer_upsampling3d_add_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_upsampling3d_add_shape_with_attrs(
        input_shapes,
        {"scale_factor": 1.0, "mode": "nearest", "align_corners": None},
    )


def infer_upsampling3d_compress_time_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(input_shapes) != 1:
        raise ValueError("upsampling3d_compress_time expects one tensor input")
    return resolve_upsampling3d_compress_time_shape(input_shapes[0])


def infer_upsampling1d_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_upsampling_shape_with_attrs("upsampling1d", input_shapes, attrs)


def infer_upsampling1d_add_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_upsampling_shape_with_attrs("upsampling1d_add", input_shapes, attrs)


def infer_upsampling2d_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_upsampling_shape_with_attrs("upsampling2d", input_shapes, attrs)


def infer_upsampling2d_add_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_upsampling_shape_with_attrs("upsampling2d_add", input_shapes, attrs)


def infer_upsampling3d_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_upsampling_shape_with_attrs("upsampling3d", input_shapes, attrs)


def infer_upsampling3d_add_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    return _infer_upsampling_shape_with_attrs("upsampling3d_add", input_shapes, attrs)


def normalize_upsampling_attrs(
    op_name: str,
    *,
    scale_factor: Any,
    mode: Any,
    align_corners: Any = False,
) -> dict[str, Any]:
    allowed_modes = _UPSAMPLING_ALLOWED_MODES[op_name]
    if not isinstance(mode, str):
        raise ValueError(f"{op_name} mode must be a string, got {mode!r}")
    normalized_mode = str(mode)
    if normalized_mode not in allowed_modes:
        raise ValueError(f"{op_name} mode must be one of {allowed_modes}, got {mode!r}")
    normalized_scale = _positive_finite_float(scale_factor, f"{op_name} scale_factor")
    linear_mode = _UPSAMPLING_LINEAR_MODES[op_name]
    if normalized_mode == linear_mode:
        if not isinstance(align_corners, bool):
            raise ValueError(f"{op_name} align_corners must be bool for mode {normalized_mode}, got {align_corners!r}")
        normalized_align = bool(align_corners)
    else:
        if align_corners not in (None, False):
            raise ValueError(
                f"{op_name} align_corners must be None or False for mode {normalized_mode}, got {align_corners!r}"
            )
        normalized_align = None
    return {
        "scale_factor": normalized_scale,
        "mode": normalized_mode,
        "align_corners": normalized_align,
    }


def resolve_upsampling1d_shape(input_shape: Sequence[int], scale_factor: float) -> list[int]:
    if len(input_shape) != 3:
        raise ValueError(f"upsampling1d expects rank-3 NWC input, got rank {len(input_shape)}")
    n, width, channels = [int(dim) for dim in input_shape]
    out_width = _scaled_extent("upsampling1d", width, scale_factor, "width")
    return [n, out_width, channels]


def resolve_upsampling2d_shape(input_shape: Sequence[int], scale_factor: float) -> list[int]:
    if len(input_shape) != 4:
        raise ValueError(f"upsampling2d expects rank-4 NHWC input, got rank {len(input_shape)}")
    n, height, width, channels = [int(dim) for dim in input_shape]
    out_height = _scaled_extent("upsampling2d", height, scale_factor, "height")
    out_width = _scaled_extent("upsampling2d", width, scale_factor, "width")
    return [n, out_height, out_width, channels]


def resolve_upsampling3d_shape(input_shape: Sequence[int], scale_factor: float) -> list[int]:
    if len(input_shape) != 5:
        raise ValueError(f"upsampling3d expects rank-5 NFHWC input, got rank {len(input_shape)}")
    n, frames, height, width, channels = [int(dim) for dim in input_shape]
    out_frames = _scaled_extent("upsampling3d", frames, scale_factor, "frames")
    out_height = _scaled_extent("upsampling3d", height, scale_factor, "height")
    out_width = _scaled_extent("upsampling3d", width, scale_factor, "width")
    return [n, out_frames, out_height, out_width, channels]


def resolve_upsampling3d_compress_time_shape(input_shape: Sequence[int]) -> list[int]:
    if len(input_shape) != 5:
        raise ValueError(f"upsampling3d_compress_time expects rank-5 NFHWC input, got rank {len(input_shape)}")
    n, frames, height, width, channels = [int(dim) for dim in input_shape]
    out_frames = _compress_time_frames(frames)
    out_height = _scaled_extent("upsampling3d_compress_time", height, 2.0, "height")
    out_width = _scaled_extent("upsampling3d_compress_time", width, 2.0, "width")
    return [n, out_frames, out_height, out_width, channels]


def upsampling1d(
    x: Any,
    scale_factor: float,
    mode: str,
    align_corners: bool | None = False,
) -> Tensor:
    return _upsampling_forward("upsampling1d", x, None, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def upsampling1d_add(
    x: Any,
    residual: Any,
    scale_factor: float,
    mode: str,
    align_corners: bool | None = False,
) -> Tensor:
    return _upsampling_forward(
        "upsampling1d_add",
        x,
        residual,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


def upsampling2d(
    x: Any,
    scale_factor: float,
    mode: str,
    align_corners: bool | None = False,
) -> Tensor:
    return _upsampling_forward("upsampling2d", x, None, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def upsampling2d_add(
    x: Any,
    residual: Any,
    scale_factor: float,
    mode: str,
    align_corners: bool | None = False,
) -> Tensor:
    return _upsampling_forward(
        "upsampling2d_add",
        x,
        residual,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


def upsampling3d(
    x: Any,
    scale_factor: float,
    mode: str,
    align_corners: bool | None = False,
) -> Tensor:
    return _upsampling_forward("upsampling3d", x, None, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def upsampling3d_add(
    x: Any,
    residual: Any,
    scale_factor: float,
    mode: str,
    align_corners: bool | None = False,
) -> Tensor:
    return _upsampling_forward(
        "upsampling3d_add",
        x,
        residual,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


def upsampling3d_compress_time(x: Any) -> Tensor:
    tensor = as_tensor(x)
    _validate_upsampling_tensor("upsampling3d_compress_time", tensor)
    if tensor.rank != 5:
        raise ValueError(f"upsampling3d_compress_time expects rank-5 NFHWC input, got rank {tensor.rank}")
    if not isinstance(tensor.shape_spec[1], int):
        raise ValueError("upsampling3d_compress_time requires a static frame dimension")
    out_shape = resolve_upsampling3d_compress_time_shape(tensor.shape)
    out_shape_spec = _compress_time_shape_spec(tensor.shape_spec)
    return tensor.builder.emit(
        "upsampling3d_compress_time",
        [tensor],
        out_shape,
        tensor.dtype,
        {},
        shape_spec=out_shape_spec,
    )


def interpolate(
    x: Any,
    size: Any = None,
    scale_factor: Any = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: Any = None,
    antialias: bool = False,
) -> Tensor:
    tensor = as_tensor(x)
    if size is not None:
        raise NotImplementedError("interpolate currently supports only scale_factor=; size= is not supported")
    if scale_factor is None:
        raise NotImplementedError("interpolate currently requires scale_factor=")
    if recompute_scale_factor not in (None, False):
        raise NotImplementedError("interpolate does not support recompute_scale_factor")
    if antialias:
        raise NotImplementedError("interpolate does not support antialias=True")
    op_name = _interpolate_upsampling_op_name(tensor.rank, mode)
    normalized_scale = _normalize_interpolate_scale_factor(scale_factor, spatial_rank=tensor.rank - 2)
    normalized_align = _normalize_interpolate_align_corners(mode, align_corners)
    layout_adapted = permute(tensor, _interpolate_to_channels_last_dims(tensor.rank))
    upsampled = _upsampling_forward(
        op_name,
        layout_adapted,
        None,
        scale_factor=normalized_scale,
        mode=mode,
        align_corners=normalized_align,
    )
    return permute(upsampled, _interpolate_to_channels_first_dims(tensor.rank))


def _infer_upsampling_shape_with_attrs(
    op_name: str,
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    expected_inputs = 2 if op_name.endswith("_add") else 1
    if len(input_shapes) != expected_inputs:
        raise ValueError(f"{op_name} expects {expected_inputs} tensor inputs")
    normalized = normalize_upsampling_attrs(
        op_name,
        scale_factor=attrs.get("scale_factor"),
        mode=attrs.get("mode"),
        align_corners=attrs.get("align_corners", False),
    )
    input_shape = input_shapes[0]
    if op_name.startswith("upsampling1d"):
        output_shape = resolve_upsampling1d_shape(input_shape, float(normalized["scale_factor"]))
    elif op_name.startswith("upsampling2d"):
        output_shape = resolve_upsampling2d_shape(input_shape, float(normalized["scale_factor"]))
    else:
        output_shape = resolve_upsampling3d_shape(input_shape, float(normalized["scale_factor"]))
    if op_name.endswith("_add"):
        _validate_residual_shape(op_name, input_shapes[1], output_shape)
    return output_shape


def _upsampling_forward(
    op_name: str,
    x: Any,
    residual: Any | None,
    *,
    scale_factor: float,
    mode: str,
    align_corners: bool | None,
) -> Tensor:
    tensor = as_tensor(x)
    _validate_upsampling_tensor(op_name, tensor)
    expected_rank = _UPSAMPLING_RANKS[op_name]
    if tensor.rank != expected_rank:
        layout = {3: "NWC", 4: "NHWC", 5: "NFHWC"}[expected_rank]
        raise ValueError(f"{op_name} expects rank-{expected_rank} {layout} input, got rank {tensor.rank}")
    normalized = normalize_upsampling_attrs(
        op_name,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )
    if tensor.rank == 3:
        out_shape = resolve_upsampling1d_shape(tensor.shape, float(normalized["scale_factor"]))
    elif tensor.rank == 4:
        out_shape = resolve_upsampling2d_shape(tensor.shape, float(normalized["scale_factor"]))
    else:
        out_shape = resolve_upsampling3d_shape(tensor.shape, float(normalized["scale_factor"]))
    out_shape_spec = _upsampled_shape_spec(op_name, tensor.shape_spec, float(normalized["scale_factor"]))
    inputs = [tensor]
    if residual is not None:
        residual_tensor = as_tensor(residual, dtype_hint=tensor.dtype)
        if residual_tensor.builder is not tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if residual_tensor.dtype != tensor.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {tensor.dtype} vs {residual_tensor.dtype}")
        if residual_tensor.rank != tensor.rank:
            raise ValueError(f"{op_name} residual rank must match the input rank {tensor.rank}, got {residual_tensor.rank}")
        _validate_residual_shape(op_name, residual_tensor.shape, out_shape)
        inputs.append(residual_tensor)
    return tensor.builder.emit(op_name, inputs, out_shape, tensor.dtype, normalized, shape_spec=out_shape_spec)


def _validate_upsampling_tensor(op_name: str, tensor: Tensor) -> None:
    if tensor.dtype not in UPSAMPLING_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {tensor.dtype}")


def _upsampled_shape_spec(
    op_name: str,
    input_shape_spec: Sequence[int | Mapping[str, Any]],
    scale_factor: float,
) -> list[int | dict[str, Any]]:
    spatial_rank = _UPSAMPLING_RANKS[op_name] - 2
    output_shape_spec = [_copy_shape_dim(input_shape_spec[0])]
    for axis in range(1, spatial_rank + 1):
        output_shape_spec.append(_scaled_shape_spec_dim(op_name, input_shape_spec[axis], scale_factor, axis))
    output_shape_spec.append(_copy_shape_dim(input_shape_spec[-1]))
    return output_shape_spec


def _compress_time_shape_spec(input_shape_spec: Sequence[int | Mapping[str, Any]]) -> list[int | dict[str, Any]]:
    frames = input_shape_spec[1]
    if not isinstance(frames, int):
        raise ValueError("upsampling3d_compress_time requires a static frame dimension")
    return [
        _copy_shape_dim(input_shape_spec[0]),
        _compress_time_frames(int(frames)),
        _scaled_shape_spec_dim("upsampling3d_compress_time", input_shape_spec[2], 2.0, 2),
        _scaled_shape_spec_dim("upsampling3d_compress_time", input_shape_spec[3], 2.0, 3),
        _copy_shape_dim(input_shape_spec[4]),
    ]


def _scaled_shape_spec_dim(
    op_name: str,
    dim: int | Mapping[str, Any],
    scale_factor: float,
    axis: int,
) -> int | dict[str, Any]:
    if isinstance(dim, int):
        return _scaled_extent(op_name, int(dim), scale_factor, f"axis {axis}")
    integral_scale = _integral_scale_factor(scale_factor)
    if integral_scale is None:
        raise ValueError(
            f"{op_name} with non-integer scale_factor={scale_factor} requires a static input size for axis {axis}"
        )
    return symbolic_int_expr("mul", _copy_shape_dim(dim), integral_scale)


def _scaled_extent(op_name: str, extent: int, scale_factor: float, axis_name: str) -> int:
    output = int(int(extent) * float(scale_factor))
    if output <= 0:
        raise ValueError(
            f"{op_name} output {axis_name} must be positive; got input={extent}, scale_factor={scale_factor}"
        )
    return output


def _integral_scale_factor(scale_factor: float) -> int | None:
    rounded = round(float(scale_factor))
    if math.isclose(float(scale_factor), float(rounded)):
        return int(rounded)
    return None


def _interpolate_upsampling_op_name(rank: int, mode: Any) -> str:
    if not isinstance(mode, str):
        raise ValueError(f"interpolate mode must be a string, got {mode!r}")
    if rank == 3:
        allowed = ("linear", "nearest", "nearest-exact")
        if mode not in allowed:
            raise ValueError(f"interpolate rank-3 mode must be one of {allowed}, got {mode!r}")
        return "upsampling1d"
    if rank == 4:
        allowed = ("bilinear", "nearest", "nearest-exact")
        if mode not in allowed:
            raise ValueError(f"interpolate rank-4 mode must be one of {allowed}, got {mode!r}")
        return "upsampling2d"
    if rank == 5:
        allowed = ("trilinear", "nearest", "nearest-exact")
        if mode not in allowed:
            raise ValueError(f"interpolate rank-5 mode must be one of {allowed}, got {mode!r}")
        return "upsampling3d"
    raise ValueError(f"interpolate expects rank-3, rank-4, or rank-5 dense tensors, got rank {rank}")


def _normalize_interpolate_scale_factor(scale_factor: Any, *, spatial_rank: int) -> float:
    if isinstance(scale_factor, Sequence) and not isinstance(scale_factor, (str, bytes, bytearray)):
        values = tuple(scale_factor)
        if len(values) != spatial_rank:
            raise ValueError(
                f"interpolate scale_factor sequence length {len(values)} must match spatial rank {spatial_rank}"
            )
        normalized_values = tuple(_positive_finite_float(value, "interpolate scale_factor") for value in values)
        first = normalized_values[0]
        if any(not math.isclose(value, first) for value in normalized_values[1:]):
            raise NotImplementedError("interpolate currently requires a uniform scale_factor across spatial dims")
        return float(first)
    return _positive_finite_float(scale_factor, "interpolate scale_factor")


def _normalize_interpolate_align_corners(mode: str, align_corners: Any) -> bool | None:
    if mode in {"linear", "bilinear", "trilinear"}:
        if align_corners is None:
            return False
        if not isinstance(align_corners, bool):
            raise ValueError(f"interpolate align_corners must be bool for mode {mode}, got {align_corners!r}")
        return bool(align_corners)
    if align_corners not in (None, False):
        raise ValueError(f"interpolate align_corners must be None or False for mode {mode}, got {align_corners!r}")
    return None


def _interpolate_to_channels_last_dims(rank: int) -> list[int]:
    if rank == 3:
        return [0, 2, 1]
    if rank == 4:
        return [0, 2, 3, 1]
    if rank == 5:
        return [0, 2, 3, 4, 1]
    raise ValueError(f"interpolate expects rank-3, rank-4, or rank-5 dense tensors, got rank {rank}")


def _interpolate_to_channels_first_dims(rank: int) -> list[int]:
    if rank == 3:
        return [0, 2, 1]
    if rank == 4:
        return [0, 3, 1, 2]
    if rank == 5:
        return [0, 4, 1, 2, 3]
    raise ValueError(f"interpolate expects rank-3, rank-4, or rank-5 dense tensors, got rank {rank}")


def _compress_time_frames(frames: int) -> int:
    if int(frames) <= 0:
        raise ValueError(f"upsampling3d_compress_time expects a positive frame dimension, got {frames}")
    if int(frames) == 1:
        return 1
    if int(frames) % 2 == 1:
        return 2 * int(frames) - 1
    return 2 * int(frames)


def _validate_residual_shape(op_name: str, residual_shape: Sequence[int], output_shape: Sequence[int]) -> None:
    residual = [int(dim) for dim in residual_shape]
    if residual != [int(dim) for dim in output_shape]:
        raise ValueError(f"{op_name} residual shape must match the upsampled output shape {list(output_shape)}, got {residual}")


def _positive_finite_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return normalized


def _copy_shape_dim(dim: int | Mapping[str, Any]) -> int | dict[str, Any]:
    return dict(dim) if isinstance(dim, Mapping) else int(dim)


def _backend_kernels(op_name: str) -> dict[str, KernelBinding]:
    return {
        "cpu": KernelBinding(symbol=f"generated_{op_name}", library="model", source_template="upsampling_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol=f"generated_{op_name}", library="model", source_template="upsampling_gpu.j2"),
        "rocm": KernelBinding(symbol=f"generated_{op_name}", library="model", source_template="upsampling_gpu.j2"),
    }


@op_def
class Upsampling1d(OpDef):
    name = "upsampling1d"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("scale_factor", "float", required=True),
            AttrDef("mode", "str", required=True),
            AttrDef("align_corners", "bool", default=False),
        ),
    )
    infer_shape = infer_upsampling1d_shape
    infer_shape_with_attrs = infer_upsampling1d_shape_with_attrs
    allowed_dtypes = UPSAMPLING_DTYPES
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding("upsampling1d")
    description = "Channels-last rank-3 upsampling with linear, nearest, or nearest-exact interpolation."


@op_def
class Upsampling1dAdd(OpDef):
    name = "upsampling1d_add"
    schema = OpSchema(
        inputs=("x", "residual"),
        attrs=(
            AttrDef("scale_factor", "float", required=True),
            AttrDef("mode", "str", required=True),
            AttrDef("align_corners", "bool", default=False),
        ),
    )
    infer_shape = infer_upsampling1d_add_shape
    infer_shape_with_attrs = infer_upsampling1d_add_shape_with_attrs
    allowed_dtypes = UPSAMPLING_DTYPES
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding("upsampling1d_add")
    description = "Channels-last rank-3 upsampling fused with a residual add."


@op_def
class Upsampling2d(OpDef):
    name = "upsampling2d"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("scale_factor", "float", required=True),
            AttrDef("mode", "str", required=True),
            AttrDef("align_corners", "bool", default=False),
        ),
    )
    infer_shape = infer_upsampling2d_shape
    infer_shape_with_attrs = infer_upsampling2d_shape_with_attrs
    allowed_dtypes = UPSAMPLING_DTYPES
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding("upsampling2d")
    description = "Channels-last rank-4 upsampling with bilinear, nearest, or nearest-exact interpolation."


@op_def
class Upsampling2dAdd(OpDef):
    name = "upsampling2d_add"
    schema = OpSchema(
        inputs=("x", "residual"),
        attrs=(
            AttrDef("scale_factor", "float", required=True),
            AttrDef("mode", "str", required=True),
            AttrDef("align_corners", "bool", default=False),
        ),
    )
    infer_shape = infer_upsampling2d_add_shape
    infer_shape_with_attrs = infer_upsampling2d_add_shape_with_attrs
    allowed_dtypes = UPSAMPLING_DTYPES
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding("upsampling2d_add")
    description = "Channels-last rank-4 upsampling fused with a residual add."


@op_def
class Upsampling3d(OpDef):
    name = "upsampling3d"
    schema = OpSchema(
        inputs=("x",),
        attrs=(
            AttrDef("scale_factor", "float", required=True),
            AttrDef("mode", "str", required=True),
            AttrDef("align_corners", "bool", default=False),
        ),
    )
    infer_shape = infer_upsampling3d_shape
    infer_shape_with_attrs = infer_upsampling3d_shape_with_attrs
    allowed_dtypes = UPSAMPLING_DTYPES
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding("upsampling3d")
    description = "Channels-last rank-5 upsampling with trilinear, nearest, or nearest-exact interpolation."


@op_def
class Upsampling3dAdd(OpDef):
    name = "upsampling3d_add"
    schema = OpSchema(
        inputs=("x", "residual"),
        attrs=(
            AttrDef("scale_factor", "float", required=True),
            AttrDef("mode", "str", required=True),
            AttrDef("align_corners", "bool", default=False),
        ),
    )
    infer_shape = infer_upsampling3d_add_shape
    infer_shape_with_attrs = infer_upsampling3d_add_shape_with_attrs
    allowed_dtypes = UPSAMPLING_DTYPES
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding("upsampling3d_add")
    description = "Channels-last rank-5 upsampling fused with a residual add."


@op_def
class Upsampling3dCompressTime(OpDef):
    name = "upsampling3d_compress_time"
    schema = OpSchema(inputs=("x",))
    infer_shape = infer_upsampling3d_compress_time_shape
    allowed_dtypes = UPSAMPLING_DTYPES
    backend_kernels = _backend_kernels(name)
    frontend = FrontendBinding("upsampling3d_compress_time")
    description = "Channels-last rank-5 nearest spatial upsampling with compress-time frame handling."


__all__ = [
    "UPSAMPLING_DTYPES",
    "Upsampling1d",
    "Upsampling1dAdd",
    "Upsampling2d",
    "Upsampling2dAdd",
    "Upsampling3d",
    "Upsampling3dAdd",
    "Upsampling3dCompressTime",
    "infer_upsampling1d_add_shape",
    "infer_upsampling1d_add_shape_with_attrs",
    "infer_upsampling1d_shape",
    "infer_upsampling1d_shape_with_attrs",
    "infer_upsampling2d_add_shape",
    "infer_upsampling2d_add_shape_with_attrs",
    "infer_upsampling2d_shape",
    "infer_upsampling2d_shape_with_attrs",
    "infer_upsampling3d_add_shape",
    "infer_upsampling3d_add_shape_with_attrs",
    "infer_upsampling3d_compress_time_shape",
    "infer_upsampling3d_shape",
    "infer_upsampling3d_shape_with_attrs",
    "interpolate",
    "normalize_upsampling_attrs",
    "resolve_upsampling1d_shape",
    "resolve_upsampling2d_shape",
    "resolve_upsampling3d_compress_time_shape",
    "resolve_upsampling3d_shape",
    "upsampling1d",
    "upsampling1d_add",
    "upsampling2d",
    "upsampling2d_add",
    "upsampling3d",
    "upsampling3d_add",
    "upsampling3d_compress_time",
]
