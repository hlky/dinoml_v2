from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.vision import (
    VISION_DTYPES,
    infer_multi_level_roi_align_shape_with_attrs,
    infer_roi_align_shape_with_attrs,
    normalize_multi_level_roi_align_im_shape,
    normalize_multi_level_roi_align_shapes,
    normalize_roi_align_attrs,
    normalize_roi_align_shapes,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, str(node["op"]))
    context = _context(target, node, tensor_map)
    if spec.is_gpu:
        context.update(spec.gpu_template_context())
    template_name = {
        ("roi_align", False): "roi_align_cpu.cpp.j2",
        ("roi_align", True): "roi_align_gpu.j2",
        ("multi_level_roi_align", False): "multi_level_roi_align_cpu.cpp.j2",
        ("multi_level_roi_align", True): "multi_level_roi_align_gpu.j2",
    }[(str(node["op"]), spec.is_gpu)]
    return render_op_template(template_name, context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, str(node["op"]))
    func = _function_name(node, tensor_map)
    output = _c_ident(node["outputs"][0])
    if node["op"] == "roi_align":
        x = _c_ident(node["inputs"][0])
        rois = _c_ident(node["inputs"][1])
        args = f"ptr_{x}, ptr_{rois}, ptr_{output}, runtime_numel_{output}"
    else:
        p2 = _c_ident(node["inputs"][0])
        p3 = _c_ident(node["inputs"][1])
        p4 = _c_ident(node["inputs"][2])
        p5 = _c_ident(node["inputs"][3])
        rois = _c_ident(node["inputs"][4])
        args = f"ptr_{p2}, ptr_{p3}, ptr_{p4}, ptr_{p5}, ptr_{rois}, ptr_{output}, runtime_numel_{output}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, str(node["op"]))
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    op_name = str(node["op"])
    output_tensor = tensor_map[str(node["outputs"][0])]
    dtype = str(output_tensor["dtype"])
    if op_name == "roi_align":
        x_tensor = tensor_map[str(node["inputs"][0])]
        rois_tensor = tensor_map[str(node["inputs"][1])]
        attrs = _validate_roi_align_node(node, x_tensor, rois_tensor, output_tensor)
        return {
            "func": _function_name(node, tensor_map),
            "kernel": f"{_function_name(node, tensor_map)}_kernel",
            "storage_type": target_storage_type(dtype, target),
            "batch": int(x_tensor["shape"][0]),
            "num_rois": int(output_tensor["shape"][0]),
            "channels": int(output_tensor["shape"][1]),
            "height": int(x_tensor["shape"][2]),
            "width": int(x_tensor["shape"][3]),
            "pooled_h": int(output_tensor["shape"][2]),
            "pooled_w": int(output_tensor["shape"][3]),
            "sampling_ratio": int(attrs["sampling_ratio"]),
            "spatial_scale": float(attrs["spatial_scale"]),
            "continuous_coordinate": bool(attrs["continuous_coordinate"]),
            "block_size": 256,
        }
    p2_tensor = tensor_map[str(node["inputs"][0])]
    p3_tensor = tensor_map[str(node["inputs"][1])]
    p4_tensor = tensor_map[str(node["inputs"][2])]
    p5_tensor = tensor_map[str(node["inputs"][3])]
    rois_tensor = tensor_map[str(node["inputs"][4])]
    attrs = _validate_multi_level_roi_align_node(node, p2_tensor, p3_tensor, p4_tensor, p5_tensor, rois_tensor, output_tensor)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(dtype, target),
        "batch": int(p2_tensor["shape"][0]),
        "num_rois": int(output_tensor["shape"][0]),
        "channels": int(output_tensor["shape"][1]),
        "p2_h": int(p2_tensor["shape"][2]),
        "p2_w": int(p2_tensor["shape"][3]),
        "p3_h": int(p3_tensor["shape"][2]),
        "p3_w": int(p3_tensor["shape"][3]),
        "p4_h": int(p4_tensor["shape"][2]),
        "p4_w": int(p4_tensor["shape"][3]),
        "p5_h": int(p5_tensor["shape"][2]),
        "p5_w": int(p5_tensor["shape"][3]),
        "pooled_h": int(output_tensor["shape"][2]),
        "pooled_w": int(output_tensor["shape"][3]),
        "sampling_ratio": int(attrs["sampling_ratio"]),
        "spatial_scale": float(attrs["spatial_scale"]),
        "continuous_coordinate": bool(attrs["continuous_coordinate"]),
        "im_h": int(attrs["im_shape"][0]),
        "im_w": int(attrs["im_shape"][1]),
        "first_threshold": (224.0 * 224.0) / (float(attrs["im_shape"][0]) * float(attrs["im_shape"][1]) * 4.0),
        "block_size": 256,
    }


def _validate_roi_align_node(
    node: Mapping[str, Any],
    x_tensor: Mapping[str, Any],
    rois_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> dict[str, Any]:
    if node["op"] != "roi_align":
        raise ValueError(f"Unsupported ROI align op: {node['op']}")
    if len(node.get("inputs", [])) != 2:
        raise ValueError("roi_align expects two inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("roi_align expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in VISION_DTYPES:
        raise NotImplementedError(f"roi_align lowering does not support dtype {dtype}")
    if str(x_tensor["dtype"]) != dtype or str(rois_tensor["dtype"]) != dtype:
        raise ValueError("roi_align x, rois, and output must share dtype")
    attrs = normalize_roi_align_attrs(**node.get("attrs", {}))
    normalize_roi_align_shapes([x_tensor["shape"], rois_tensor["shape"]])
    expected_shape = infer_roi_align_shape_with_attrs([x_tensor["shape"], rois_tensor["shape"]], attrs)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("roi_align output shape does not match attrs")
    return attrs


def _validate_multi_level_roi_align_node(
    node: Mapping[str, Any],
    p2_tensor: Mapping[str, Any],
    p3_tensor: Mapping[str, Any],
    p4_tensor: Mapping[str, Any],
    p5_tensor: Mapping[str, Any],
    rois_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> dict[str, Any]:
    if node["op"] != "multi_level_roi_align":
        raise ValueError(f"Unsupported multi-level ROI align op: {node['op']}")
    if len(node.get("inputs", [])) != 5:
        raise ValueError("multi_level_roi_align expects five inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("multi_level_roi_align expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in VISION_DTYPES:
        raise NotImplementedError(f"multi_level_roi_align lowering does not support dtype {dtype}")
    if any(str(tensor["dtype"]) != dtype for tensor in (p2_tensor, p3_tensor, p4_tensor, p5_tensor, rois_tensor)):
        raise ValueError("multi_level_roi_align inputs, rois, and output must share dtype")
    attrs = dict(node.get("attrs", {}))
    attrs = {
        **normalize_roi_align_attrs(
            pooled_size=attrs.get("pooled_size"),
            sampling_ratio=attrs.get("sampling_ratio", 0),
            spatial_scale=attrs.get("spatial_scale", 1.0),
            position_sensitive=attrs.get("position_sensitive", False),
            continuous_coordinate=attrs.get("continuous_coordinate", False),
        ),
        "im_shape": list(normalize_multi_level_roi_align_im_shape(attrs.get("im_shape"))),
    }
    normalize_multi_level_roi_align_shapes(
        [p2_tensor["shape"], p3_tensor["shape"], p4_tensor["shape"], p5_tensor["shape"], rois_tensor["shape"]]
    )
    expected_shape = infer_multi_level_roi_align_shape_with_attrs(
        [p2_tensor["shape"], p3_tensor["shape"], p4_tensor["shape"], p5_tensor["shape"], rois_tensor["shape"]],
        attrs,
    )
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("multi_level_roi_align output shape does not match attrs")
    return attrs


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    op_name = str(node["op"])
    if op_name == "roi_align":
        x_tensor = tensor_map[str(node["inputs"][0])]
        rois_tensor = tensor_map[str(node["inputs"][1])]
        output_tensor = tensor_map[str(node["outputs"][0])]
        attrs = normalize_roi_align_attrs(**node.get("attrs", {}))
        signature = {
            "op": op_name,
            "x_shape": list(x_tensor["shape"]),
            "rois_shape": list(rois_tensor["shape"]),
            "output_shape": list(output_tensor["shape"]),
            "dtype": str(output_tensor["dtype"]),
            "attrs": attrs,
        }
    else:
        tensors = [tensor_map[str(name)] for name in node["inputs"]]
        output_tensor = tensor_map[str(node["outputs"][0])]
        attrs = dict(node.get("attrs", {}))
        signature = {
            "op": op_name,
            "input_shapes": [list(tensor["shape"]) for tensor in tensors],
            "output_shape": list(output_tensor["shape"]),
            "dtype": str(output_tensor["dtype"]),
            "attrs": {
                **normalize_roi_align_attrs(
                    pooled_size=attrs.get("pooled_size"),
                    sampling_ratio=attrs.get("sampling_ratio", 0),
                    spatial_scale=attrs.get("spatial_scale", 1.0),
                    position_sensitive=attrs.get("position_sensitive", False),
                    continuous_coordinate=attrs.get("continuous_coordinate", False),
                ),
                "im_shape": list(normalize_multi_level_roi_align_im_shape(attrs.get("im_shape"))),
            },
        }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{op_name}_{digest}"


ROI_ALIGN_LOWERING = OpLowering(
    op_name="roi_align",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


MULTI_LEVEL_ROI_ALIGN_LOWERING = OpLowering(
    op_name="multi_level_roi_align",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
