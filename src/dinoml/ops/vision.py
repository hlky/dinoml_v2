from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


VISION_DTYPES = ("float16", "float32", "bfloat16")


def infer_roi_align_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_roi_align_shape_with_attrs(
        input_shapes,
        {
            "pooled_size": (1, 1),
            "sampling_ratio": 0,
            "spatial_scale": 1.0,
            "position_sensitive": False,
            "continuous_coordinate": False,
        },
    )


def infer_multi_level_roi_align_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_multi_level_roi_align_shape_with_attrs(
        input_shapes,
        {
            "pooled_size": (1, 1),
            "sampling_ratio": 0,
            "spatial_scale": 1.0,
            "position_sensitive": False,
            "continuous_coordinate": False,
            "im_shape": (1, 1),
        },
    )


def infer_roi_align_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    feature_shape, rois_shape = normalize_roi_align_shapes(input_shapes)
    pooled_h, pooled_w = normalize_roi_align_pooled_size(attrs.get("pooled_size", (1, 1)))
    return [int(rois_shape[0]), int(feature_shape[1]), pooled_h, pooled_w]


def infer_multi_level_roi_align_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    feature_shape, rois_shape = normalize_multi_level_roi_align_shapes(input_shapes)
    pooled_h, pooled_w = normalize_roi_align_pooled_size(attrs.get("pooled_size", (1, 1)))
    normalize_multi_level_roi_align_im_shape(attrs.get("im_shape", (1, 1)))
    return [int(rois_shape[0]), int(feature_shape[1]), pooled_h, pooled_w]


def normalize_roi_align_pooled_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        pooled_h = int(value)
        pooled_w = int(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) != 2:
            raise ValueError(f"roi_align pooled_size must be an integer or pair of integers, got {value!r}")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in items):
            raise ValueError(f"roi_align pooled_size must contain non-bool integers, got {value!r}")
        pooled_h, pooled_w = int(items[0]), int(items[1])
    else:
        raise ValueError(f"roi_align pooled_size must be an integer or pair of integers, got {value!r}")
    if pooled_h <= 0 or pooled_w <= 0:
        raise ValueError(f"roi_align pooled_size must contain positive integers, got {value!r}")
    return pooled_h, pooled_w


def normalize_roi_align_sampling_ratio(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"roi_align sampling_ratio must be an integer, got {value!r}")
    return int(value)


def normalize_roi_align_spatial_scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"roi_align spatial_scale must be a finite number, got {value!r}")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"roi_align spatial_scale must be a positive finite number, got {value!r}")
    return scale


def normalize_roi_align_bool_attr(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"roi_align {name} must be a bool, got {value!r}")
    return bool(value)


def normalize_multi_level_roi_align_im_shape(value: Any) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if len(items) != 2:
            raise ValueError(f"multi_level_roi_align im_shape must be a pair of integers, got {value!r}")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in items):
            raise ValueError(f"multi_level_roi_align im_shape must contain non-bool integers, got {value!r}")
        im_h, im_w = int(items[0]), int(items[1])
    else:
        raise ValueError(f"multi_level_roi_align im_shape must be a pair of integers, got {value!r}")
    if im_h <= 0 or im_w <= 0:
        raise ValueError(f"multi_level_roi_align im_shape must contain positive integers, got {value!r}")
    return im_h, im_w


def normalize_roi_align_attrs(
    *,
    pooled_size: Any,
    sampling_ratio: Any,
    spatial_scale: Any,
    position_sensitive: Any,
    continuous_coordinate: Any,
) -> dict[str, Any]:
    pooled_h, pooled_w = normalize_roi_align_pooled_size(pooled_size)
    normalized_position_sensitive = normalize_roi_align_bool_attr("position_sensitive", position_sensitive)
    if normalized_position_sensitive:
        raise NotImplementedError("roi_align position_sensitive=True is not supported in DinoML v2")
    return {
        "pooled_size": [pooled_h, pooled_w],
        "sampling_ratio": normalize_roi_align_sampling_ratio(sampling_ratio),
        "spatial_scale": normalize_roi_align_spatial_scale(spatial_scale),
        "position_sensitive": normalized_position_sensitive,
        "continuous_coordinate": normalize_roi_align_bool_attr("continuous_coordinate", continuous_coordinate),
    }


def normalize_multi_level_roi_align_attrs(
    *,
    pooled_size: Any,
    sampling_ratio: Any,
    spatial_scale: Any,
    position_sensitive: Any,
    continuous_coordinate: Any,
    im_shape: Any,
) -> dict[str, Any]:
    attrs = normalize_roi_align_attrs(
        pooled_size=pooled_size,
        sampling_ratio=sampling_ratio,
        spatial_scale=spatial_scale,
        position_sensitive=position_sensitive,
        continuous_coordinate=continuous_coordinate,
    )
    im_h, im_w = normalize_multi_level_roi_align_im_shape(im_shape)
    attrs["im_shape"] = [im_h, im_w]
    return attrs


def normalize_roi_align_shapes(input_shapes: Sequence[Sequence[int]]) -> tuple[list[int], list[int]]:
    if len(input_shapes) != 2:
        raise ValueError("roi_align expects exactly two inputs")
    feature_shape = [int(dim) for dim in input_shapes[0]]
    rois_shape = [int(dim) for dim in input_shapes[1]]
    if len(feature_shape) != 4:
        raise ValueError("roi_align expects x with rank-4 NCHW shape")
    if len(rois_shape) != 2 or rois_shape[1] != 5:
        raise ValueError("roi_align expects rois with shape [num_rois, 5]")
    return feature_shape, rois_shape


def normalize_multi_level_roi_align_shapes(input_shapes: Sequence[Sequence[int]]) -> tuple[list[int], list[int]]:
    if len(input_shapes) != 5:
        raise ValueError("multi_level_roi_align expects exactly five inputs")
    p2_shape, p3_shape, p4_shape, p5_shape, rois_shape = [[int(dim) for dim in shape] for shape in input_shapes]
    if any(len(shape) != 4 for shape in (p2_shape, p3_shape, p4_shape, p5_shape)):
        raise ValueError("multi_level_roi_align expects all pyramid inputs with rank-4 NCHW shape")
    if len(rois_shape) != 2 or rois_shape[1] != 5:
        raise ValueError("multi_level_roi_align expects rois with shape [num_rois, 5]")
    base_batch = p2_shape[0]
    base_channels = p2_shape[1]
    for level_name, shape in (("p3", p3_shape), ("p4", p4_shape), ("p5", p5_shape)):
        if shape[0] != base_batch:
            raise ValueError(f"multi_level_roi_align {level_name} batch size must match p2")
        if shape[1] != base_channels:
            raise ValueError(f"multi_level_roi_align {level_name} channels must match p2")
    return p2_shape, rois_shape


@op_def
class RoiAlign(OpDef):
    name = "roi_align"
    schema = OpSchema(
        inputs=("x", "rois"),
        attrs=(
            AttrDef("pooled_size", "ints", required=True),
            AttrDef("sampling_ratio", "int", default=0),
            AttrDef("spatial_scale", "float", default=1.0),
            AttrDef("position_sensitive", "bool", default=False),
            AttrDef("continuous_coordinate", "bool", default=False),
        ),
    )
    infer_shape = infer_roi_align_shape
    infer_shape_with_attrs = infer_roi_align_shape_with_attrs
    allowed_dtypes = VISION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_roi_align", library="model", source_template="roi_align_cpu.cpp.j2"),
        "cuda": KernelBinding(symbol="generated_roi_align", library="model", source_template="roi_align_gpu.j2"),
        "rocm": KernelBinding(symbol="generated_roi_align", library="model", source_template="roi_align_gpu.j2"),
    }
    frontend = FrontendBinding("roi_align")
    description = (
        "Dense NCHW roi_align over [num_rois, 5] image-space boxes with bilinear sampling, "
        "static shapes, and v1-compatible non-position-sensitive ROI geometry semantics."
    )

    @classmethod
    def forward(
        cls,
        x: Any,
        rois: Any,
        *,
        pooled_size: Any,
        sampling_ratio: int = 0,
        spatial_scale: float = 1.0,
        position_sensitive: bool = False,
        continuous_coordinate: bool = False,
    ) -> Tensor:
        x_tensor = as_tensor(x)
        rois_tensor = as_tensor(rois, dtype_hint=x_tensor.dtype)
        if rois_tensor.builder is not x_tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if x_tensor.dtype not in VISION_DTYPES:
            raise ValueError(f"roi_align does not support dtype {x_tensor.dtype}")
        if rois_tensor.dtype != x_tensor.dtype:
            raise ValueError("roi_align rois dtype must match x dtype")
        if x_tensor.dynamic or rois_tensor.dynamic:
            raise ValueError("roi_align currently supports only static input shapes")
        attrs = normalize_roi_align_attrs(
            pooled_size=pooled_size,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
            position_sensitive=position_sensitive,
            continuous_coordinate=continuous_coordinate,
        )
        out_shape = infer_roi_align_shape_with_attrs([x_tensor.shape, rois_tensor.shape], attrs)
        return x_tensor.builder.emit(
            "roi_align",
            [x_tensor, rois_tensor],
            out_shape,
            x_tensor.dtype,
            attrs,
            shape_spec=out_shape,
        )


@op_def
class MultiLevelRoiAlign(OpDef):
    name = "multi_level_roi_align"
    schema = OpSchema(
        inputs=("p2", "p3", "p4", "p5", "rois"),
        attrs=(
            AttrDef("pooled_size", "ints", required=True),
            AttrDef("sampling_ratio", "int", default=0),
            AttrDef("spatial_scale", "float", default=1.0),
            AttrDef("position_sensitive", "bool", default=False),
            AttrDef("continuous_coordinate", "bool", default=False),
            AttrDef("im_shape", "ints", required=True),
        ),
    )
    infer_shape = infer_multi_level_roi_align_shape
    infer_shape_with_attrs = infer_multi_level_roi_align_shape_with_attrs
    allowed_dtypes = VISION_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(
            symbol="generated_multi_level_roi_align",
            library="model",
            source_template="multi_level_roi_align_cpu.cpp.j2",
        ),
        "cuda": KernelBinding(
            symbol="generated_multi_level_roi_align",
            library="model",
            source_template="multi_level_roi_align_gpu.j2",
        ),
        "rocm": KernelBinding(
            symbol="generated_multi_level_roi_align",
            library="model",
            source_template="multi_level_roi_align_gpu.j2",
        ),
    }
    frontend = FrontendBinding("multi_level_roi_align")
    description = (
        "Dense NCHW multi_level_roi_align across four pyramid feature maps with v1-style FPN level selection, "
        "image-space [num_rois, 5] boxes, and static shapes."
    )

    @classmethod
    def forward(
        cls,
        p2: Any,
        p3: Any,
        p4: Any,
        p5: Any,
        rois: Any,
        *,
        pooled_size: Any,
        sampling_ratio: int = 0,
        spatial_scale: float = 1.0,
        position_sensitive: bool = False,
        continuous_coordinate: bool = False,
        im_shape: Any,
    ) -> Tensor:
        p2_tensor = as_tensor(p2)
        p3_tensor = as_tensor(p3, dtype_hint=p2_tensor.dtype)
        p4_tensor = as_tensor(p4, dtype_hint=p2_tensor.dtype)
        p5_tensor = as_tensor(p5, dtype_hint=p2_tensor.dtype)
        rois_tensor = as_tensor(rois, dtype_hint=p2_tensor.dtype)
        if any(tensor.builder is not p2_tensor.builder for tensor in (p3_tensor, p4_tensor, p5_tensor, rois_tensor)):
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if p2_tensor.dtype not in VISION_DTYPES:
            raise ValueError(f"multi_level_roi_align does not support dtype {p2_tensor.dtype}")
        if any(tensor.dtype != p2_tensor.dtype for tensor in (p3_tensor, p4_tensor, p5_tensor, rois_tensor)):
            raise ValueError("multi_level_roi_align inputs and rois must share dtype")
        if any(tensor.dynamic for tensor in (p2_tensor, p3_tensor, p4_tensor, p5_tensor, rois_tensor)):
            raise ValueError("multi_level_roi_align currently supports only static input shapes")
        attrs = normalize_multi_level_roi_align_attrs(
            pooled_size=pooled_size,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
            position_sensitive=position_sensitive,
            continuous_coordinate=continuous_coordinate,
            im_shape=im_shape,
        )
        out_shape = infer_multi_level_roi_align_shape_with_attrs(
            [p2_tensor.shape, p3_tensor.shape, p4_tensor.shape, p5_tensor.shape, rois_tensor.shape],
            attrs,
        )
        return p2_tensor.builder.emit(
            "multi_level_roi_align",
            [p2_tensor, p3_tensor, p4_tensor, p5_tensor, rois_tensor],
            out_shape,
            p2_tensor.dtype,
            attrs,
            shape_spec=out_shape,
        )


def roi_align(
    x: Any,
    rois: Any,
    *,
    pooled_size: Any,
    sampling_ratio: int = 0,
    spatial_scale: float = 1.0,
    position_sensitive: bool = False,
    continuous_coordinate: bool = False,
) -> Tensor:
    return RoiAlign.forward(
        x,
        rois,
        pooled_size=pooled_size,
        sampling_ratio=sampling_ratio,
        spatial_scale=spatial_scale,
        position_sensitive=position_sensitive,
        continuous_coordinate=continuous_coordinate,
    )


def multi_level_roi_align(
    p2: Any,
    p3: Any,
    p4: Any,
    p5: Any,
    rois: Any,
    *,
    pooled_size: Any,
    sampling_ratio: int = 0,
    spatial_scale: float = 1.0,
    position_sensitive: bool = False,
    continuous_coordinate: bool = False,
    im_shape: Any,
) -> Tensor:
    return MultiLevelRoiAlign.forward(
        p2,
        p3,
        p4,
        p5,
        rois,
        pooled_size=pooled_size,
        sampling_ratio=sampling_ratio,
        spatial_scale=spatial_scale,
        position_sensitive=position_sensitive,
        continuous_coordinate=continuous_coordinate,
        im_shape=im_shape,
    )


__all__ = [
    "MultiLevelRoiAlign",
    "RoiAlign",
    "VISION_DTYPES",
    "infer_multi_level_roi_align_shape",
    "infer_multi_level_roi_align_shape_with_attrs",
    "infer_roi_align_shape",
    "infer_roi_align_shape_with_attrs",
    "multi_level_roi_align",
    "normalize_multi_level_roi_align_attrs",
    "normalize_multi_level_roi_align_im_shape",
    "normalize_multi_level_roi_align_shapes",
    "normalize_roi_align_attrs",
    "normalize_roi_align_bool_attr",
    "normalize_roi_align_pooled_size",
    "normalize_roi_align_sampling_ratio",
    "normalize_roi_align_shapes",
    "normalize_roi_align_spatial_scale",
    "roi_align",
]
