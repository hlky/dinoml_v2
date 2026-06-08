from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.gather import _index_storage_type
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.qwen2_5_vl import (
    QWEN2_5_VL_STITCH_IMAGE_FEATURES_DTYPES,
    normalize_qwen2_5_vl_stitch_image_features_shapes,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "qwen2_5_vl_stitch_image_features")
    context = _context(target, node, tensor_map)
    if spec.is_gpu:
        context.update(spec.gpu_template_context())
    template_name = (
        "qwen2_5_vl_stitch_image_features_gpu.j2"
        if spec.is_gpu
        else "qwen2_5_vl_stitch_image_features_cpu.cpp.j2"
    )
    return render_op_template(template_name, context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "qwen2_5_vl_stitch_image_features")
    func = _function_name(node, tensor_map)
    input_ids = _c_ident(node["inputs"][0])
    inputs_embeds = _c_ident(node["inputs"][1])
    image_features = _c_ident(node["inputs"][2])
    out = _c_ident(node["outputs"][0])
    args = (
        f"ptr_{input_ids}, ptr_{inputs_embeds}, ptr_{image_features}, ptr_{out}, "
        f"runtime_numel_{input_ids}, runtime_numel_{image_features}, runtime_numel_{out}"
    )
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "qwen2_5_vl_stitch_image_features")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_ids_tensor = tensor_map[str(node["inputs"][0])]
    inputs_embeds_tensor = tensor_map[str(node["inputs"][1])]
    image_features_tensor = tensor_map[str(node["inputs"][2])]
    output_tensor = tensor_map[str(node["outputs"][0])]
    image_token_id = _validate_node_contract(
        node,
        input_ids_tensor,
        inputs_embeds_tensor,
        image_features_tensor,
        output_tensor,
    )
    dtype = str(output_tensor["dtype"])
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(dtype, target),
        "index_storage_type": _index_storage_type(str(input_ids_tensor["dtype"])),
        "hidden_size": int(inputs_embeds_tensor["shape"][2]),
        "image_token_id": image_token_id,
        "block_size": 128,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_ids_tensor: Mapping[str, Any],
    inputs_embeds_tensor: Mapping[str, Any],
    image_features_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> int:
    if node["op"] != "qwen2_5_vl_stitch_image_features":
        raise ValueError(f"Unsupported Qwen2.5-VL op: {node['op']}")
    if len(node.get("inputs", [])) != 3:
        raise ValueError("qwen2_5_vl_stitch_image_features expects three inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("qwen2_5_vl_stitch_image_features expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in QWEN2_5_VL_STITCH_IMAGE_FEATURES_DTYPES:
        raise NotImplementedError(f"qwen2_5_vl_stitch_image_features lowering does not support dtype {dtype}")
    if str(inputs_embeds_tensor["dtype"]) != dtype or str(image_features_tensor["dtype"]) != dtype:
        raise ValueError("qwen2_5_vl_stitch_image_features data inputs and output must share dtype")
    if str(input_ids_tensor["dtype"]) not in {"int64", "int32"}:
        raise ValueError("qwen2_5_vl_stitch_image_features input_ids must have dtype int64 or int32")
    normalize_qwen2_5_vl_stitch_image_features_shapes(
        [input_ids_tensor["shape"], inputs_embeds_tensor["shape"], image_features_tensor["shape"]]
    )
    if list(output_tensor["shape"]) != list(inputs_embeds_tensor["shape"]):
        raise ValueError("qwen2_5_vl_stitch_image_features output shape must match inputs_embeds")
    attrs = node.get("attrs", {})
    image_token_id = attrs.get("image_token_id")
    if not isinstance(image_token_id, int) or isinstance(image_token_id, bool):
        raise ValueError("qwen2_5_vl_stitch_image_features image_token_id must be an integer attr")
    return int(image_token_id)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_ids_tensor = tensor_map[str(node["inputs"][0])]
    inputs_embeds_tensor = tensor_map[str(node["inputs"][1])]
    image_features_tensor = tensor_map[str(node["inputs"][2])]
    output_tensor = tensor_map[str(node["outputs"][0])]
    signature = {
        "op": "qwen2_5_vl_stitch_image_features",
        "input_ids_shape": list(input_ids_tensor["shape"]),
        "inputs_embeds_shape": list(inputs_embeds_tensor["shape"]),
        "image_features_shape": list(image_features_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "dtype": str(output_tensor["dtype"]),
        "index_dtype": str(input_ids_tensor["dtype"]),
        "image_token_id": int(node.get("attrs", {}).get("image_token_id", 0)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"qwen2_5_vl_stitch_image_features_{digest}"


QWEN2_5_VL_STITCH_IMAGE_FEATURES_LOWERING = OpLowering(
    op_name="qwen2_5_vl_stitch_image_features",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
