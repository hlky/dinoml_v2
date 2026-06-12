from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.vision import (
    VISION_DTYPES,
    infer_batched_nms_shape_with_attrs,
    infer_efficient_nms_output_shapes,
    infer_nms_shape_with_attrs,
    normalize_batched_nms_attrs,
    normalize_batched_nms_shapes,
    normalize_efficient_nms_attrs,
    normalize_efficient_nms_shapes,
    normalize_nms_attrs,
    normalize_nms_shapes,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, str(node["op"]))
    context = _context(target, node, tensor_map)
    if spec.is_gpu:
        context.update(spec.gpu_template_context())
    return render_op_template(
        {
            ("nms", False): "nms_cpu.cpp.j2",
            ("nms", True): "nms_gpu.j2",
            ("batched_nms", False): "batched_nms_cpu.cpp.j2",
            ("batched_nms", True): "batched_nms_gpu.j2",
            ("efficient_nms", False): "efficient_nms_cpu.cpp.j2",
            ("efficient_nms", True): "efficient_nms_gpu.j2",
        }[(str(node["op"]), spec.is_gpu)],
        context,
    )


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest, tensor_map
    spec = supported_target_spec(target, str(node["op"]))
    func = _function_name(node)
    if node["op"] == "nms":
        boxes = _c_ident(node["inputs"][0])
        scores = _c_ident(node["inputs"][1])
        out = _c_ident(node["outputs"][0])
        args = (
            f"ptr_{boxes}, ptr_{scores}, ptr_{out}, "
            f"runtime_numel_{boxes}, runtime_numel_{scores}, runtime_numel_{out}"
        )
    elif node["op"] == "batched_nms":
        boxes = _c_ident(node["inputs"][0])
        out = _c_ident(node["outputs"][0])
        args = f"ptr_{boxes}, ptr_{out}, runtime_numel_{boxes}, runtime_numel_{out}"
    else:
        boxes = _c_ident(node["inputs"][0])
        scores = _c_ident(node["inputs"][1])
        num_det = _c_ident(node["outputs"][0])
        det_boxes = _c_ident(node["outputs"][1])
        det_scores = _c_ident(node["outputs"][2])
        det_classes = _c_ident(node["outputs"][3])
        args = (
            f"ptr_{boxes}, ptr_{scores}, ptr_{num_det}, ptr_{det_boxes}, ptr_{det_scores}, ptr_{det_classes}, "
            f"runtime_numel_{boxes}, runtime_numel_{scores}, runtime_numel_{num_det}, "
            f"runtime_numel_{det_boxes}, runtime_numel_{det_scores}, runtime_numel_{det_classes}"
        )
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del tensor_map
    supported_target_spec(target, str(node["op"]))
    return f"{target}:{_function_name(node)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target, tensor_map
    return _function_name(node)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    op_name = str(node["op"])
    if op_name == "nms":
        boxes_tensor = tensor_map[str(node["inputs"][0])]
        scores_tensor = tensor_map[str(node["inputs"][1])]
        output_tensor = tensor_map[str(node["outputs"][0])]
        attrs = _validate_nms_node(node, boxes_tensor, scores_tensor, output_tensor)
        return {
            "func": _function_name(node),
            "storage_type": target_storage_type(str(output_tensor["dtype"]), target),
            "int_storage_type": target_storage_type("int64", target),
            "batch": int(boxes_tensor["shape"][0]),
            "num_boxes": int(boxes_tensor["shape"][1]),
            "max_output": int(output_tensor["shape"][1]),
            "pre_nms_top": int(attrs["pre_nms_top"]),
            "iou_threshold": _float_literal(float(attrs["iou_threshold"])),
            "min_box_size": _float_literal(float(attrs["min_box_size"])),
            "block_size": 1,
        }
    if op_name == "batched_nms":
        boxes_tensor = tensor_map[str(node["inputs"][0])]
        output_tensor = tensor_map[str(node["outputs"][0])]
        attrs = _validate_batched_nms_node(node, boxes_tensor, output_tensor)
        return {
            "func": _function_name(node),
            "storage_type": target_storage_type(str(boxes_tensor["dtype"]), target),
            "int_storage_type": target_storage_type("int64", target),
            "num_boxes": int(boxes_tensor["shape"][0]),
            "keep_n": int(attrs["keep_n"]),
            "iou_threshold": _float_literal(float(attrs["iou_threshold"])),
            "block_size": 1,
        }
    boxes_tensor = tensor_map[str(node["inputs"][0])]
    scores_tensor = tensor_map[str(node["inputs"][1])]
    output_tensors = [tensor_map[str(name)] for name in node["outputs"]]
    attrs = _validate_efficient_nms_node(node, boxes_tensor, scores_tensor, output_tensors)
    return {
        "func": _function_name(node),
        "storage_type": target_storage_type(str(boxes_tensor["dtype"]), target),
        "int_storage_type": target_storage_type("int64", target),
        "batch": int(boxes_tensor["shape"][0]),
        "num_boxes": int(boxes_tensor["shape"][1]),
        "num_classes": int(boxes_tensor["shape"][2]),
        "max_output": int(output_tensors[1]["shape"][1]),
        "pre_nms_top": int(attrs["pre_nms_top"]),
        "iou_threshold": _float_literal(float(attrs["iou_threshold"])),
        "min_box_size": _float_literal(float(attrs["min_box_size"])),
        "block_size": 1,
    }


def _validate_nms_node(
    node: Mapping[str, Any],
    boxes_tensor: Mapping[str, Any],
    scores_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> dict[str, Any]:
    if len(node.get("inputs", [])) != 2 or len(node.get("outputs", [])) != 1:
        raise ValueError("nms expects two inputs and one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in VISION_DTYPES:
        raise NotImplementedError(f"nms lowering does not support dtype {dtype}")
    if str(boxes_tensor["dtype"]) != dtype or str(scores_tensor["dtype"]) != dtype:
        raise ValueError("nms boxes, scores, and output must share dtype")
    attrs = normalize_nms_attrs(**node.get("attrs", {}))
    normalize_nms_shapes([boxes_tensor["shape"], scores_tensor["shape"]])
    expected = infer_nms_shape_with_attrs([boxes_tensor["shape"], scores_tensor["shape"]], attrs)
    if list(output_tensor["shape"]) != list(expected):
        raise ValueError("nms output shape does not match attrs")
    return attrs


def _validate_batched_nms_node(
    node: Mapping[str, Any],
    boxes_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> dict[str, Any]:
    if len(node.get("inputs", [])) != 1 or len(node.get("outputs", [])) != 1:
        raise ValueError("batched_nms expects one input and one output")
    dtype = str(boxes_tensor["dtype"])
    if dtype not in VISION_DTYPES:
        raise NotImplementedError(f"batched_nms lowering does not support dtype {dtype}")
    if str(output_tensor["dtype"]) != "int64":
        raise ValueError("batched_nms output dtype must be int64")
    attrs = normalize_batched_nms_attrs(**node.get("attrs", {}))
    normalize_batched_nms_shapes([boxes_tensor["shape"]])
    expected = infer_batched_nms_shape_with_attrs([boxes_tensor["shape"]], attrs)
    if list(output_tensor["shape"]) != list(expected):
        raise ValueError("batched_nms output shape does not match attrs")
    return attrs


def _validate_efficient_nms_node(
    node: Mapping[str, Any],
    boxes_tensor: Mapping[str, Any],
    scores_tensor: Mapping[str, Any],
    output_tensors: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(node.get("inputs", [])) != 2 or len(output_tensors) != 4:
        raise ValueError("efficient_nms expects two inputs and four outputs")
    dtype = str(boxes_tensor["dtype"])
    if dtype not in VISION_DTYPES:
        raise NotImplementedError(f"efficient_nms lowering does not support dtype {dtype}")
    if str(scores_tensor["dtype"]) != dtype:
        raise ValueError("efficient_nms boxes and scores must share dtype")
    attrs = normalize_efficient_nms_attrs(**node.get("attrs", {}))
    normalize_efficient_nms_shapes([boxes_tensor["shape"], scores_tensor["shape"]])
    expected = infer_efficient_nms_output_shapes([boxes_tensor["shape"], scores_tensor["shape"]], attrs)
    expected_dtypes = ("int64", dtype, dtype, "int64")
    for output_tensor, expected_shape, expected_dtype in zip(output_tensors, expected, expected_dtypes):
        if list(output_tensor["shape"]) != list(expected_shape):
            raise ValueError("efficient_nms output shapes do not match attrs")
        if str(output_tensor["dtype"]) != expected_dtype:
            raise ValueError("efficient_nms output dtypes do not match contract")
    return attrs


def _function_name(node: Mapping[str, Any]) -> str:
    signature = {
        "op": str(node["op"]),
        "attrs": dict(node.get("attrs", {})),
        "outputs": list(node.get("outputs", [])),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{node['op']}_{digest}"


def _float_literal(value: float) -> str:
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal = f"{literal}.0"
    return f"{literal}f"


NMS_LOWERING = OpLowering(
    op_name="nms",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


BATCHED_NMS_LOWERING = OpLowering(
    op_name="batched_nms",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


EFFICIENT_NMS_LOWERING = OpLowering(
    op_name="efficient_nms",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
