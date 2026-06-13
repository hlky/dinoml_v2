from __future__ import annotations

import math
from math import prod
from typing import Any, Mapping, Sequence

from dinoml.ir import (
    IR_SCHEMA_VERSION,
    OUTPUT_SHAPE_REPORT_METADATA_VERSION,
    VIEW_METADATA_VERSION,
    VIEW_ONLY_TRANSFORMS,
    normalize_dtype,
)
from dinoml.layout import validate_layout
from dinoml.ops.collections import INDEX_ADD_DTYPES, broadcast_shape_spec, normalize_index_add_attrs, normalize_one_hot_num_classes
from dinoml.ops.collections import (
    SCATTER_DTYPES,
    SCATTER_REDUCE_DTYPES,
    normalize_scatter_attrs,
    normalize_scatter_reduce_include_self,
    normalize_scatter_reduce_name,
)
from dinoml.ops.definitions import get_op_def
from dinoml.ops.elementwise import (
    CAST_ELEMENTWISE_DTYPES,
    ELEMENTWISE_BY_NAME,
    ELEMENTWISE_OUTPUT_DTYPES,
    EQ_ELEMENTWISE_DTYPES,
    FLOAT_ELEMENTWISE_DTYPES,
    elementwise_output_dtype,
)
from dinoml.ops.positional import (
    GET_3D_ROTARY_POS_EMBED_ALLEGRO_DTYPES,
    GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS,
    GET_1D_ROTARY_POS_EMBED_DTYPES,
    ROTARY_POSITIONAL_FUSION_DTYPES,
    ROTARY_POSITIONAL_FUSION_OPS,
    normalize_get_2d_rotary_pos_embed_attrs,
    normalize_get_2d_rotary_pos_embed_lumina_attrs,
    normalize_get_3d_rotary_pos_embed_allegro_attrs,
    normalize_get_3d_rotary_pos_embed_attrs,
    normalize_get_1d_rotary_pos_embed_attrs,
    rotary_output_cols,
)
from dinoml.ops.glm_ocr import GLM_OCR_STITCH_IMAGE_FEATURES_DTYPES
from dinoml.ops.qwen2_5_vl import QWEN2_5_VL_STITCH_IMAGE_FEATURES_DTYPES
from dinoml.ops.vision import (
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
from dinoml.passes.utils import tensor_map
from dinoml.shapes import is_dynamic_shape, normalize_symbolic_int, symbolic_int_expr, validate_shape_spec


class ValidationError(ValueError):
    pass


def validate_ir(ir: Mapping[str, Any]) -> None:
    if ir.get("schema_version") != IR_SCHEMA_VERSION:
        raise ValidationError(f"Unsupported IR schema: {ir.get('schema_version')}")
    tensors = tensor_map(ir)
    for section in ("inputs", "constants", "outputs", "nodes", "tensors"):
        if section not in ir:
            raise ValidationError(f"IR is missing section: {section}")
    for tensor in tensors.values():
        normalize_dtype(tensor["dtype"])
        if any(not isinstance(dim, int) or dim <= 0 for dim in tensor["shape"]):
            raise ValidationError(f"Tensor {tensor['name']} has invalid static shape {tensor['shape']}")
        if "shape_spec" in tensor:
            try:
                validate_shape_spec(tensor["shape_spec"], tensor["shape"])
            except ValueError as exc:
                raise ValidationError(f"Tensor {tensor['name']} has invalid shape_spec: {exc}") from exc
        if "layout" in tensor:
            try:
                validate_layout(tensor["layout"], tensor["shape"])
            except ValueError as exc:
                raise ValidationError(f"Tensor {tensor['name']} has invalid layout: {exc}") from exc
    for node in ir["nodes"]:
        for name in node["inputs"]:
            if name not in tensors:
                raise ValidationError(f"Node {node['id']} references missing input tensor {name}")
        for name in node["outputs"]:
            if name not in tensors:
                raise ValidationError(f"Node {node['id']} references missing output tensor {name}")
        _validate_node(node, tensors)
    for output in ir["outputs"]:
        if output["tensor"] not in tensors:
            raise ValidationError(f"Output {output['name']} references missing tensor {output['tensor']}")
    for state in ir.get("states", []):
        tensor_name = state.get("tensor")
        if tensor_name not in tensors:
            raise ValidationError(f"State {state.get('name')} references missing tensor {tensor_name}")
        tensor = tensors[tensor_name]
        if list(state.get("shape", [])) != list(tensor["shape"]):
            raise ValidationError(f"State {state.get('name')} shape metadata must match tensor table")
        if str(state.get("dtype")) != str(tensor["dtype"]):
            raise ValidationError(f"State {state.get('name')} dtype metadata must match tensor table")
    validate_view_metadata(ir.get("metadata", {}).get("views"), tensors)
    validate_view_metadata(ir.get("metadata", {}).get("memory_plan", {}).get("views"), tensors)
    validate_output_shape_report_metadata(ir.get("metadata", {}).get("output_shape_reports"), ir["outputs"], tensors)
    _validate_masked_select_output_contract(ir, tensors)


def validate_view_metadata(view_metadata: Any, tensors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    if view_metadata is None:
        return []
    if not isinstance(view_metadata, Mapping):
        raise ValidationError("View metadata must be a mapping")
    if view_metadata.get("version") != VIEW_METADATA_VERSION:
        raise ValidationError(f"Unsupported view metadata version: {view_metadata.get('version')}")
    views = view_metadata.get("views", [])
    if not isinstance(views, list):
        raise ValidationError("View metadata views must be a list")

    normalized = []
    seen_tensors: set[str] = set()
    for idx, view in enumerate(views):
        if not isinstance(view, Mapping):
            raise ValidationError(f"View metadata entry {idx} must be a mapping")
        tensor_name = view.get("tensor")
        source_name = view.get("source")
        if not isinstance(tensor_name, str) or not isinstance(source_name, str):
            raise ValidationError(f"View metadata entry {idx} must name tensor and source")
        if tensor_name in seen_tensors:
            raise ValidationError(f"Tensor {tensor_name} has duplicate view metadata")
        seen_tensors.add(tensor_name)
        if tensor_name == source_name:
            raise ValidationError(f"View tensor {tensor_name} cannot alias itself")
        if tensor_name not in tensors:
            raise ValidationError(f"View tensor {tensor_name} is not present in tensor table")
        if source_name not in tensors:
            raise ValidationError(f"View source {source_name} is not present in tensor table")

        tensor = tensors[tensor_name]
        source = tensors[source_name]
        if tensor["dtype"] != source["dtype"]:
            raise ValidationError(f"View tensor {tensor_name} dtype must match source {source_name}")
        offset_elements = view.get("offset_elements", 0)
        if not isinstance(offset_elements, int) or offset_elements < 0:
            raise ValidationError(f"View tensor {tensor_name} must use a non-negative integer offset")
        tensor_numel = prod(tensor["shape"])
        source_numel = prod(source["shape"])
        if offset_elements + tensor_numel > source_numel:
            raise ValidationError(f"View tensor {tensor_name} exceeds source {source_name} storage")
        if view.get("kind") != "shape_view":
            raise ValidationError(f"View tensor {tensor_name} must use kind shape_view")
        transform = view.get("transform")
        if transform not in VIEW_ONLY_TRANSFORMS:
            raise ValidationError(f"View tensor {tensor_name} has unsupported transform {transform}")
        if "shape" in view and list(view["shape"]) != list(tensor["shape"]):
            raise ValidationError(f"View tensor {tensor_name} shape metadata must match tensor table")
        if "shape_spec" in view:
            try:
                validate_shape_spec(view["shape_spec"], tensor["shape"])
            except ValueError as exc:
                raise ValidationError(f"View tensor {tensor_name} has invalid shape_spec: {exc}") from exc

        normalized.append(
            {
                "tensor": tensor_name,
                "source": source_name,
                "kind": "shape_view",
                "transform": transform,
                "offset_elements": offset_elements,
                "shape": list(tensor["shape"]),
                "shape_spec": list(view.get("shape_spec", tensor.get("shape_spec", tensor["shape"]))),
            }
        )
    return normalized


def validate_output_shape_report_metadata(
    report_metadata: Any,
    outputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    if report_metadata is None:
        return []
    if not isinstance(report_metadata, Mapping):
        raise ValidationError("Output shape report metadata must be a mapping")
    if report_metadata.get("version") != OUTPUT_SHAPE_REPORT_METADATA_VERSION:
        raise ValidationError(f"Unsupported output shape report metadata version: {report_metadata.get('version')}")
    reports = report_metadata.get("reports", [])
    if not isinstance(reports, list):
        raise ValidationError("Output shape report metadata reports must be a list")

    output_by_name = {str(output["name"]): output for output in outputs}
    normalized = []
    seen_outputs: set[str] = set()
    for idx, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ValidationError(f"Output shape report entry {idx} must be a mapping")
        output_name = report.get("output")
        if not isinstance(output_name, str) or not output_name:
            raise ValidationError(f"Output shape report entry {idx} must name an output")
        if output_name in seen_outputs:
            raise ValidationError(f"Output {output_name} has duplicate shape report metadata")
        seen_outputs.add(output_name)
        output = output_by_name.get(output_name)
        if output is None:
            raise ValidationError(f"Output shape report references unknown output {output_name}")
        tensor_name = str(output["tensor"])
        if tensor_name not in tensors:
            raise ValidationError(f"Output shape report tensor {tensor_name} is not present in tensor table")
        kind = report.get("kind")
        if kind != "shape_buffer":
            raise ValidationError(f"Output shape report {output_name} has unsupported kind {kind}")
        normalized.append({"output": output_name, "kind": "shape_buffer"})
    return normalized


def _validate_node(node: Mapping[str, Any], tensors: Mapping[str, Mapping[str, Any]]) -> None:
    op_def = get_op_def(node["op"])
    inputs = [tensors[name] for name in node["inputs"]]
    if node["op"] == "fused_elementwise":
        _validate_fused_elementwise_node(node, inputs, tensors)
        return
    if node["op"] == "where":
        _validate_where_node(node, inputs, tensors)
        return
    if node["op"] == "masked_select":
        _validate_masked_select_node(node, inputs, tensors)
        return
    if node["op"] == "argmax":
        _validate_argmax_node(node, inputs, tensors)
        return
    if node["op"] in {"topk_values", "topk_indices"}:
        _validate_topk_node(node, inputs, tensors)
        return
    if node["op"] in {"mode_values", "mode_indices"}:
        _validate_mode_node(node, inputs, tensors)
        return
    if node["op"] in GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS:
        _validate_get_1d_rotary_pos_embed_node(node, inputs, tensors)
        return
    if node["op"] in ROTARY_POSITIONAL_FUSION_OPS:
        _validate_rotary_positional_fusion_node(node, inputs, tensors)
        return
    if node["op"] == "t5_layer_norm":
        _validate_t5_layer_norm_node(node, inputs, tensors)
        return
    if node["op"] == "layer_norm":
        _validate_layer_norm_node(node, inputs, tensors)
        return
    if node["op"] == "layernorm_sigmoid_mul":
        _validate_layernorm_sigmoid_mul_node(node, inputs, tensors)
        return
    if node["op"] == "batch_layernorm_sigmoid_mul":
        _validate_batch_layernorm_sigmoid_mul_node(node, inputs, tensors)
        return
    if node["op"] in {"group_layernorm", "group_layernorm_sigmoid_mul"}:
        _validate_group_layernorm_node(node, inputs, tensors)
        return
    if node["op"] == "add_layer_norm":
        _validate_add_layer_norm_node(node, inputs, tensors)
        return
    if node["op"] == "qkv_split":
        _validate_qkv_split_node(node, inputs, tensors)
        return
    if node["op"] in {"glm_ocr_text_rope", "glm_ocr_vision_rope"}:
        _validate_glm_ocr_rope_node(node, inputs, tensors)
        return
    if node["op"] == "glm_ocr_stitch_image_features":
        _validate_glm_ocr_stitch_image_features_node(node, inputs, tensors)
        return
    if node["op"] == "qwen2_5_vl_stitch_image_features":
        _validate_qwen2_5_vl_stitch_image_features_node(node, inputs, tensors)
        return
    if node["op"] in {"flash_attention_static_kv_cache", "flash_attention_static_kv_cache_bias"}:
        _validate_flash_attention_static_kv_cache_node(node, inputs, tensors)
        return
    if node["op"] == "flash_attention_varlen":
        _validate_flash_attention_varlen_node(node, inputs, tensors)
        return
    if node["op"] == "embedding":
        _validate_embedding_node(node, inputs, tensors)
        return
    if node["op"] == "nms":
        _validate_nms_node(node, inputs, tensors)
        return
    if node["op"] == "batched_nms":
        _validate_batched_nms_node(node, inputs, tensors)
        return
    if node["op"] == "efficient_nms":
        _validate_efficient_nms_node(node, inputs, tensors)
        return
    if node["op"] in {
        "avg_pool1d",
        "avg_pool2d",
        "conv1d_bias",
        "conv1d_bias_relu",
        "conv1d_bias_add",
        "conv1d_bias_add_relu",
        "conv2d_bias",
        "conv2d_bias_relu",
        "conv2d_bias_add",
        "conv2d_bias_add_relu",
        "max_pool2d",
        "concatenate",
        "stack",
        "flip",
        "repeat_interleave",
        "permute",
        "permute021",
        "permute0213",
        "permute102",
        "permute210",
        "dynamic_slice",
            "index_select",
            "runtime_index_select",
            "index_add",
            "scatter",
            "scatter_add",
            "scatter_reduce",
            "one_hot",
        "gather",
        "batch_gather",
        "slice_scatter",
        "pad",
    }:
        _validate_collection_node(node, inputs, tensors)
        return
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError(f"{op_def.name} expects {op_def.input_count_description()}")
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if node["outputs"]:
        output_name = node["outputs"][0]
        if list(tensors[output_name]["shape"]) != list(expected_shape):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has shape {tensors[output_name]['shape']}, "
                f"expected {expected_shape}"
            )
    if inputs:
        if any(input_info["dtype"] != inputs[0]["dtype"] for input_info in inputs):
            raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
        if inputs[0]["dtype"] not in op_def.allowed_dtypes:
            raise ValidationError(f"{op_def.name} does not support dtype {inputs[0]['dtype']}")
        try:
            expected_output_dtype = (
                elementwise_output_dtype(str(node["op"]), str(inputs[0]["dtype"]), node.get("attrs", {}))
                if str(node["op"]) in ELEMENTWISE_BY_NAME
                else str(inputs[0]["dtype"])
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        for output_name in node["outputs"]:
            if str(tensors[output_name]["dtype"]) != expected_output_dtype:
                raise ValidationError(
                    f"Node {node['id']} output {output_name} has dtype {tensors[output_name]['dtype']}, "
                    f"expected {expected_output_dtype}"
                )
    else:
        expected_dtype = str(node.get("attrs", {}).get("dtype", tensors[node["outputs"][0]]["dtype"]))
        try:
            expected_dtype = normalize_dtype(expected_dtype)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if expected_dtype not in op_def.allowed_dtypes:
            raise ValidationError(f"{op_def.name} does not support dtype {expected_dtype}")
        for output_name in node["outputs"]:
            if str(tensors[output_name]["dtype"]) != expected_dtype:
                raise ValidationError(
                    f"Node {node['id']} output {output_name} has dtype {tensors[output_name]['dtype']}, "
                    f"expected {expected_dtype}"
                )


def _validate_fused_elementwise_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    if not node["outputs"]:
        raise ValidationError(f"Node {node['id']} must have at least one output")
    if not node["inputs"]:
        raise ValidationError(f"Node {node['id']} must have at least one input")
    sub_ops = node.get("attrs", {}).get("sub_ops")
    if not isinstance(sub_ops, list) or not sub_ops:
        raise ValidationError(f"Node {node['id']} must carry fused sub_ops metadata")
    output_shapes = {tuple(tensors[name]["shape"]) for name in node["outputs"]}
    if len(output_shapes) != 1:
        raise ValidationError(f"Node {node['id']} fused outputs must have the same shape")
    dtype_env = {name: str(tensor["dtype"]) for name, tensor in tensors.items()}
    for sub_op in sub_ops:
        elementwise_spec = ELEMENTWISE_BY_NAME.get(sub_op.get("op"))
        if elementwise_spec is None:
            raise ValidationError(f"Unsupported fused sub-op: {sub_op.get('op')}")
        sub_inputs = list(sub_op.get("inputs", []))
        if len(sub_inputs) != elementwise_spec.arity:
            raise ValidationError(
                f"Fused sub-op {sub_op.get('op')} expects {elementwise_spec.arity} inputs, got {len(sub_inputs)}"
            )
        if len(sub_op.get("outputs", [])) != 1:
            raise ValidationError(f"Fused sub-op {sub_op.get('op')} must have exactly one output")
        try:
            input_dtypes = [dtype_env[name] for name in sub_inputs]
        except KeyError as exc:
            raise ValidationError(f"Fused sub-op {sub_op.get('op')} references missing input {exc.args[0]}") from exc
        output_name = sub_op["outputs"][0]
        output_dtype = _fused_elementwise_output_dtype(sub_op, input_dtypes)
        dtype_env[output_name] = output_dtype
        if output_name in tensors and str(tensors[output_name]["dtype"]) != output_dtype:
            raise ValidationError(
                f"Fused sub-op {sub_op.get('op')} output {output_name} has dtype {tensors[output_name]['dtype']}, "
                f"expected {output_dtype}"
            )


def _validate_where_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if len(inputs) != 3:
        raise ValidationError("where expects 3 inputs")
    try:
        get_op_def("where").infer_shape([input_info["shape"] for input_info in inputs])
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if str(inputs[0]["dtype"]) != "bool":
        raise ValidationError(f"where condition must have dtype bool, got {inputs[0]['dtype']}")
    if str(inputs[1]["dtype"]) != str(inputs[2]["dtype"]):
        raise ValidationError(f"where x/y dtype mismatch: {inputs[1]['dtype']} vs {inputs[2]['dtype']}")
    if str(inputs[1]["dtype"]) not in ELEMENTWISE_OUTPUT_DTYPES:
        raise ValidationError(f"where does not support dtype {inputs[1]['dtype']}")
    output_name = node["outputs"][0]
    if str(tensors[output_name]["dtype"]) != str(inputs[1]["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {tensors[output_name]['dtype']}, "
            f"expected {inputs[1]['dtype']}"
        )


def _validate_masked_select_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if len(inputs) != 2:
        raise ValidationError("masked_select expects 2 inputs")
    try:
        expected_shape = get_op_def("masked_select").infer_shape([input_info["shape"] for input_info in inputs])
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if str(inputs[1]["dtype"]) != "bool":
        raise ValidationError(f"masked_select mask must have dtype bool, got {inputs[1]['dtype']}")
    if str(inputs[0]["dtype"]) not in get_op_def("masked_select").allowed_dtypes:
        raise ValidationError(f"masked_select does not support dtype {inputs[0]['dtype']}")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, expected {expected_shape}"
        )
    if str(output["dtype"]) != str(inputs[0]["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, expected {inputs[0]['dtype']}"
        )
    expected_shape_spec = [_masked_select_capacity_spec(inputs[0], inputs[1])]
    actual_shape_spec = list(output.get("shape_spec", output["shape"]))
    if actual_shape_spec != expected_shape_spec:
        raise ValidationError(
            f"masked_select output shape_spec must equal the broadcast capacity expression {expected_shape_spec}, "
            f"got {actual_shape_spec}"
        )


def _validate_argmax_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_def = get_op_def("argmax")
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError("argmax expects exactly one input")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    dynamic_tensors = [
        str(tensor["name"])
        for tensor in [*inputs, output]
        if is_dynamic_shape(tensor.get("shape_spec", tensor["shape"]))
    ]
    if dynamic_tensors:
        raise ValidationError(f"argmax currently supports only static shapes: {dynamic_tensors}")
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {expected_shape}"
        )
    input_dtype = str(inputs[0]["dtype"])
    if input_dtype not in op_def.allowed_dtypes:
        raise ValidationError(f"argmax does not support dtype {input_dtype}")
    if str(output["dtype"]) != "int64":
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, expected int64"
        )


def _validate_topk_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_name = str(node["op"])
    op_def = get_op_def(op_name)
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError(f"{op_name} expects exactly one input")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    dynamic_tensors = [
        str(tensor["name"])
        for tensor in [*inputs, output]
        if is_dynamic_shape(tensor.get("shape_spec", tensor["shape"]))
    ]
    if dynamic_tensors:
        raise ValidationError(f"topk currently supports only static shapes: {dynamic_tensors}")
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {expected_shape}"
        )
    input_dtype = str(inputs[0]["dtype"])
    if input_dtype not in op_def.allowed_dtypes:
        raise ValidationError(f"topk does not support dtype {input_dtype}")
    expected_dtype = input_dtype if op_name == "topk_values" else "int64"
    if str(output["dtype"]) != expected_dtype:
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, expected {expected_dtype}"
        )


def _validate_mode_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_name = str(node["op"])
    op_def = get_op_def(op_name)
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError(f"{op_name} expects exactly one input")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    dynamic_tensors = [
        str(tensor["name"])
        for tensor in [*inputs, output]
        if is_dynamic_shape(tensor.get("shape_spec", tensor["shape"]))
    ]
    if dynamic_tensors:
        raise ValidationError(f"mode currently supports only static shapes: {dynamic_tensors}")
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {expected_shape}"
        )
    input_dtype = str(inputs[0]["dtype"])
    if input_dtype not in op_def.allowed_dtypes:
        raise ValidationError(f"mode does not support dtype {input_dtype}")
    expected_dtype = input_dtype if op_name == "mode_values" else "int64"
    if str(output["dtype"]) != expected_dtype:
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, expected {expected_dtype}"
        )


def _validate_get_1d_rotary_pos_embed_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_name = str(node["op"])
    op_def = get_op_def(op_name)
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if len(inputs) not in {0, 1}:
        raise ValidationError(f"{op_name} expects zero or one input")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    if inputs and len(inputs[0]["shape"]) != 1:
        raise ValidationError(f"{op_name} expects rank-1 pos input, got rank {len(inputs[0]['shape'])}")
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, expected {expected_shape}"
        )
    input_dtype = None if not inputs else str(inputs[0]["dtype"])
    if input_dtype is not None and input_dtype != "float32":
        raise ValidationError(f"{op_name} requires float32 pos input, got {input_dtype}")
    if str(output["dtype"]) not in GET_1D_ROTARY_POS_EMBED_DTYPES:
        raise ValidationError(f"{op_name} does not support output dtype {output['dtype']}")
    normalized_attrs = normalize_get_1d_rotary_pos_embed_attrs(
        dim=node.get("attrs", {}).get("dim"),
        theta=node.get("attrs", {}).get("theta", 10000.0),
        use_real=node.get("attrs", {}).get("use_real", True),
        linear_factor=node.get("attrs", {}).get("linear_factor", 1.0),
        ntk_factor=node.get("attrs", {}).get("ntk_factor", 1.0),
        repeat_interleave_real=node.get("attrs", {}).get("repeat_interleave_real", True),
        output_kind=node.get("attrs", {}).get("output_kind"),
    )
    expected_kind = "cos" if op_name.endswith("_cos") else "sin"
    if str(normalized_attrs["output_kind"]) != expected_kind:
        raise ValidationError(f"{op_name} must use output_kind={expected_kind}")
    expected_cols = rotary_output_cols(normalized_attrs)
    if int(output["shape"][1]) != expected_cols:
        raise ValidationError(
            f"Node {node['id']} output {output_name} has width {output['shape'][1]}, expected {expected_cols}"
        )


def _validate_rotary_positional_fusion_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_name = str(node["op"])
    if inputs:
        raise ValidationError(f"{op_name} expects no tensor inputs")
    output_tensors = [tensors[name] for name in node["outputs"]]
    output_dtype = str(node.get("attrs", {}).get("dtype", output_tensors[0]["dtype"]))
    if op_name == "get_2d_rotary_pos_embed":
        if len(output_tensors) != 2:
            raise ValidationError("get_2d_rotary_pos_embed expects exactly two outputs")
        normalized = normalize_get_2d_rotary_pos_embed_attrs(
            embed_dim=node.get("attrs", {}).get("embed_dim"),
            crop_start_h=node.get("attrs", {}).get("crop_start_h"),
            crop_start_w=node.get("attrs", {}).get("crop_start_w"),
            crop_stop_h=node.get("attrs", {}).get("crop_stop_h"),
            crop_stop_w=node.get("attrs", {}).get("crop_stop_w"),
            grid_h=node.get("attrs", {}).get("grid_h"),
            grid_w=node.get("attrs", {}).get("grid_w"),
            theta=node.get("attrs", {}).get("theta", 10000.0),
            use_real=node.get("attrs", {}).get("use_real", True),
        )
        expected_shape = [int(normalized["grid_h"]) * int(normalized["grid_w"]), int(normalized["embed_dim"])]
        expected_dtypes = [output_dtype, output_dtype]
    elif op_name == "get_2d_rotary_pos_embed_lumina":
        if len(output_tensors) != 2:
            raise ValidationError("get_2d_rotary_pos_embed_lumina expects exactly two outputs")
        normalized = normalize_get_2d_rotary_pos_embed_lumina_attrs(
            embed_dim=node.get("attrs", {}).get("embed_dim"),
            len_h=node.get("attrs", {}).get("len_h"),
            len_w=node.get("attrs", {}).get("len_w"),
            linear_factor=node.get("attrs", {}).get("linear_factor", 1.0),
            ntk_factor=node.get("attrs", {}).get("ntk_factor", 1.0),
        )
        expected_shape = [int(normalized["len_h"]), int(normalized["len_w"]), int(normalized["embed_dim"]) // 2]
        expected_dtypes = [output_dtype, output_dtype]
    elif op_name == "get_3d_rotary_pos_embed":
        if len(output_tensors) != 2:
            raise ValidationError("get_3d_rotary_pos_embed expects exactly two outputs")
        normalized = normalize_get_3d_rotary_pos_embed_attrs(
            embed_dim=node.get("attrs", {}).get("embed_dim"),
            crop_start_h=node.get("attrs", {}).get("crop_start_h"),
            crop_start_w=node.get("attrs", {}).get("crop_start_w"),
            crop_stop_h=node.get("attrs", {}).get("crop_stop_h"),
            crop_stop_w=node.get("attrs", {}).get("crop_stop_w"),
            grid_h=node.get("attrs", {}).get("grid_h"),
            grid_w=node.get("attrs", {}).get("grid_w"),
            temporal_size=node.get("attrs", {}).get("temporal_size"),
            theta=node.get("attrs", {}).get("theta", 10000.0),
            use_real=node.get("attrs", {}).get("use_real", True),
            grid_type=node.get("attrs", {}).get("grid_type", "linspace"),
            max_h=node.get("attrs", {}).get("max_h", 0),
            max_w=node.get("attrs", {}).get("max_w", 0),
        )
        expected_shape = [
            int(normalized["temporal_size"]) * int(normalized["grid_h"]) * int(normalized["grid_w"]),
            int(normalized["embed_dim"]),
        ]
        expected_dtypes = [output_dtype, output_dtype]
    elif op_name == "get_3d_rotary_pos_embed_allegro":
        if len(output_tensors) != 9:
            raise ValidationError("get_3d_rotary_pos_embed_allegro expects exactly nine outputs")
        normalized = normalize_get_3d_rotary_pos_embed_allegro_attrs(
            height=node.get("attrs", {}).get("height"),
            width=node.get("attrs", {}).get("width"),
            num_frames=node.get("attrs", {}).get("num_frames"),
            vae_scale_factor_spatial=node.get("attrs", {}).get("vae_scale_factor_spatial", 8),
            patch_size=node.get("attrs", {}).get("patch_size", 2),
            interpolation_scale_h=node.get("attrs", {}).get("interpolation_scale_h", 2.0),
            interpolation_scale_t=node.get("attrs", {}).get("interpolation_scale_t", 2.2),
            interpolation_scale_w=node.get("attrs", {}).get("interpolation_scale_w", 2.0),
            attention_head_dim=node.get("attrs", {}).get("attention_head_dim", 96),
        )
        dim_axis = int(normalized["attention_head_dim"]) // 3
        grid_shape = [1, int(normalized["num_frames"]) * int(normalized["grid_h"]) * int(normalized["grid_w"])]
        expected_shapes = [
            [int(normalized["num_frames"]), dim_axis],
            [int(normalized["num_frames"]), dim_axis],
            [int(normalized["grid_h"]), dim_axis],
            [int(normalized["grid_h"]), dim_axis],
            [int(normalized["grid_w"]), dim_axis],
            [int(normalized["grid_w"]), dim_axis],
            grid_shape,
            grid_shape,
            grid_shape,
        ]
        expected_dtypes = [output_dtype] * 6 + ["int64", "int64", "int64"]
        if output_dtype not in ROTARY_POSITIONAL_FUSION_DTYPES:
            raise ValidationError(f"{op_name} does not support dtype {output_dtype}")
        if any(str(output_tensors[idx]["dtype"]) != expected_dtypes[idx] for idx in range(len(output_tensors))):
            raise ValidationError(f"{op_name} output dtypes do not match the fused contract")
        if any(list(output_tensors[idx]["shape"]) != expected_shapes[idx] for idx in range(len(output_tensors))):
            raise ValidationError(f"{op_name} output shapes do not match the fused contract")
        return
    else:
        raise ValidationError(f"Unsupported rotary positional fusion op: {op_name}")

    if output_dtype not in ROTARY_POSITIONAL_FUSION_DTYPES:
        raise ValidationError(f"{op_name} does not support dtype {output_dtype}")
    for output_tensor, expected_dtype in zip(output_tensors, expected_dtypes):
        if list(output_tensor["shape"]) != expected_shape:
            raise ValidationError(f"{op_name} output shape must match the fused contract")
        if str(output_tensor["dtype"]) != expected_dtype:
            raise ValidationError(f"{op_name} output dtype must match the fused contract")


def _validate_t5_layer_norm_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_def = get_op_def("t5_layer_norm")
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError("t5_layer_norm expects exactly two inputs")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    x_tensor, weight_tensor = inputs
    x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
    weight_shape_spec = weight_tensor.get("shape_spec", weight_tensor["shape"])
    if not x_tensor["shape"]:
        raise ValidationError("t5_layer_norm requires rank >= 1 input")
    if len(weight_tensor["shape"]) != 1:
        raise ValidationError("t5_layer_norm requires rank-1 weight")
    if not isinstance(x_shape_spec[-1], int):
        raise ValidationError("t5_layer_norm currently requires a static last dimension")
    if not isinstance(weight_shape_spec[0], int):
        raise ValidationError("t5_layer_norm currently requires a static weight shape")
    if int(weight_tensor["shape"][0]) != int(x_tensor["shape"][-1]):
        raise ValidationError(
            "t5_layer_norm weight length must match the input hidden size: "
            f"got hidden={x_tensor['shape'][-1]}, weight={weight_tensor['shape'][0]}"
        )
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {expected_shape}"
        )
    if list(output["shape"]) != list(x_tensor["shape"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {x_tensor['shape']}"
        )
    if any(input_info["dtype"] != x_tensor["dtype"] for input_info in inputs):
        raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
    if x_tensor["dtype"] not in op_def.allowed_dtypes:
        raise ValidationError(f"t5_layer_norm does not support dtype {x_tensor['dtype']}")
    if str(output["dtype"]) != str(x_tensor["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
            f"expected {x_tensor['dtype']}"
        )


def _validate_layer_norm_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_def = get_op_def("layer_norm")
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError("layer_norm expects exactly three inputs")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    x_tensor, weight_tensor, bias_tensor = inputs
    x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
    if not x_tensor["shape"]:
        raise ValidationError("layer_norm requires rank >= 1 input")
    normalized_shape = node.get("attrs", {}).get("normalized_shape")
    if normalized_shape is None:
        normalized_shape = list(weight_tensor["shape"])
    if not isinstance(normalized_shape, list) or not normalized_shape:
        raise ValidationError("layer_norm requires a non-empty normalized_shape attr or affine suffix shape")
    if any(not isinstance(dim, int) or int(dim) <= 0 for dim in normalized_shape):
        raise ValidationError("layer_norm normalized_shape must contain only positive integers")
    norm_rank = len(normalized_shape)
    if len(x_tensor["shape"]) < norm_rank:
        raise ValidationError("layer_norm input rank must be at least len(normalized_shape)")
    if list(x_tensor["shape"][-norm_rank:]) != [int(dim) for dim in normalized_shape]:
        raise ValidationError("layer_norm input suffix must match normalized_shape")
    if any(not isinstance(dim, int) for dim in x_shape_spec[-norm_rank:]):
        raise ValidationError("layer_norm currently requires a static normalized_shape suffix")
    if list(weight_tensor["shape"]) != [int(dim) for dim in normalized_shape]:
        raise ValidationError("layer_norm weight shape must match normalized_shape")
    if list(bias_tensor["shape"]) != [int(dim) for dim in normalized_shape]:
        raise ValidationError("layer_norm bias shape must match normalized_shape")
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {expected_shape}"
        )
    if list(output["shape"]) != list(x_tensor["shape"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {x_tensor['shape']}"
        )
    if any(input_info["dtype"] != x_tensor["dtype"] for input_info in inputs):
        raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
    if x_tensor["dtype"] not in op_def.allowed_dtypes:
        raise ValidationError(f"layer_norm does not support dtype {x_tensor['dtype']}")
    if str(output["dtype"]) != str(x_tensor["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
            f"expected {x_tensor['dtype']}"
        )


def _validate_layernorm_sigmoid_mul_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_def = get_op_def("layernorm_sigmoid_mul")
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if len(inputs) != 3:
        raise ValidationError("layernorm_sigmoid_mul expects exactly three inputs")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    x_tensor, weight_tensor, bias_tensor = inputs
    x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
    normalized_shape = node.get("attrs", {}).get("normalized_shape")
    if not isinstance(normalized_shape, list) or not normalized_shape:
        raise ValidationError("layernorm_sigmoid_mul requires a non-empty normalized_shape attr")
    if any(not isinstance(dim, int) or int(dim) <= 0 for dim in normalized_shape):
        raise ValidationError("layernorm_sigmoid_mul normalized_shape must contain only positive integers")
    norm_rank = len(normalized_shape)
    if len(x_tensor["shape"]) < norm_rank:
        raise ValidationError("layernorm_sigmoid_mul input rank must be at least len(normalized_shape)")
    if list(x_tensor["shape"][-norm_rank:]) != [int(dim) for dim in normalized_shape]:
        raise ValidationError("layernorm_sigmoid_mul input suffix must match normalized_shape")
    if any(not isinstance(dim, int) for dim in x_shape_spec[-norm_rank:]):
        raise ValidationError("layernorm_sigmoid_mul currently requires a static normalized_shape suffix")
    if list(weight_tensor["shape"]) != [int(dim) for dim in normalized_shape]:
        raise ValidationError("layernorm_sigmoid_mul weight shape must match normalized_shape")
    if list(bias_tensor["shape"]) != [int(dim) for dim in normalized_shape]:
        raise ValidationError("layernorm_sigmoid_mul bias shape must match normalized_shape")
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape) or list(output["shape"]) != list(x_tensor["shape"]):
        raise ValidationError("layernorm_sigmoid_mul output shape must match the input shape")
    if any(input_info["dtype"] != x_tensor["dtype"] for input_info in inputs):
        raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
    if x_tensor["dtype"] not in op_def.allowed_dtypes:
        raise ValidationError(f"layernorm_sigmoid_mul does not support dtype {x_tensor['dtype']}")
    if str(output["dtype"]) != str(x_tensor["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, expected {x_tensor['dtype']}"
        )


def _validate_batch_layernorm_sigmoid_mul_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_def = get_op_def("batch_layernorm_sigmoid_mul")
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if len(inputs) != 3:
        raise ValidationError("batch_layernorm_sigmoid_mul expects exactly three inputs")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    x_tensor, weight_tensor, bias_tensor = inputs
    x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
    if len(x_tensor["shape"]) != 3:
        raise ValidationError("batch_layernorm_sigmoid_mul expects rank-3 input")
    normalized_shape = node.get("attrs", {}).get("normalized_shape")
    if not isinstance(normalized_shape, list) or len(normalized_shape) != 1:
        raise ValidationError("batch_layernorm_sigmoid_mul requires normalized_shape=[hidden]")
    hidden = normalized_shape[0]
    if not isinstance(hidden, int) or int(hidden) <= 0:
        raise ValidationError("batch_layernorm_sigmoid_mul normalized_shape must contain one positive integer")
    if not isinstance(x_shape_spec[-1], int):
        raise ValidationError("batch_layernorm_sigmoid_mul currently requires a static hidden dimension")
    if int(x_tensor["shape"][-1]) != int(hidden):
        raise ValidationError("batch_layernorm_sigmoid_mul input hidden size must match normalized_shape")
    if len(weight_tensor["shape"]) != 2 or list(weight_tensor["shape"])[1:] != [int(hidden)]:
        raise ValidationError("batch_layernorm_sigmoid_mul weight shape must be [batch, hidden]")
    if len(bias_tensor["shape"]) != 2 or list(bias_tensor["shape"])[1:] != [int(hidden)]:
        raise ValidationError("batch_layernorm_sigmoid_mul bias shape must be [batch, hidden]")
    if weight_tensor["shape"][0] != x_tensor["shape"][0]:
        raise ValidationError("batch_layernorm_sigmoid_mul weight batch dimension must match the input batch size")
    if bias_tensor["shape"][0] != x_tensor["shape"][0]:
        raise ValidationError("batch_layernorm_sigmoid_mul bias batch dimension must match the input batch size")
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape) or list(output["shape"]) != list(x_tensor["shape"]):
        raise ValidationError("batch_layernorm_sigmoid_mul output shape must match the input shape")
    if any(input_info["dtype"] != x_tensor["dtype"] for input_info in inputs):
        raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
    if x_tensor["dtype"] not in op_def.allowed_dtypes:
        raise ValidationError(f"batch_layernorm_sigmoid_mul does not support dtype {x_tensor['dtype']}")
    if str(output["dtype"]) != str(x_tensor["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, expected {x_tensor['dtype']}"
        )


def _validate_group_layernorm_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_name = str(node["op"])
    op_def = get_op_def(op_name)
    group_count = node.get("attrs", {}).get("group_count")
    if not isinstance(group_count, int) or isinstance(group_count, bool) or int(group_count) <= 0:
        raise ValidationError(f"{op_name} requires a positive integer group_count")
    if len(inputs) != 3 * int(group_count):
        raise ValidationError(f"{op_name} expects flattened [inputs, weights, biases] triples")
    if len(node["outputs"]) != int(group_count):
        raise ValidationError(f"{op_name} expects exactly one output per group")
    normalized_shapes = node.get("attrs", {}).get("normalized_shapes")
    if not isinstance(normalized_shapes, list) or len(normalized_shapes) != int(group_count):
        raise ValidationError(f"{op_name} requires normalized_shapes matching group_count")
    x_tensors = inputs[:group_count]
    weight_tensors = inputs[group_count : 2 * group_count]
    bias_tensors = inputs[2 * group_count :]
    batch_prefix_shape = None
    batch_prefix_spec = None
    x_dtype = str(x_tensors[0]["dtype"])
    if x_dtype not in op_def.allowed_dtypes:
        raise ValidationError(f"{op_name} does not support dtype {x_dtype}")
    for index, (x_tensor, weight_tensor, bias_tensor, output_name, norm_shape) in enumerate(
        zip(x_tensors, weight_tensors, bias_tensors, node["outputs"], normalized_shapes)
    ):
        output_tensor = tensors[output_name]
        if not isinstance(norm_shape, list) or not norm_shape:
            raise ValidationError(f"{op_name} requires non-empty normalized_shapes[{index}]")
        if any(not isinstance(dim, int) or int(dim) <= 0 for dim in norm_shape):
            raise ValidationError(f"{op_name} normalized_shapes[{index}] must contain only positive integers")
        if any(str(tensor["dtype"]) != x_dtype for tensor in (x_tensor, weight_tensor, bias_tensor, output_tensor)):
            raise ValidationError(f"{op_name} group {index} requires matching input/output dtypes")
        norm_rank = len(norm_shape)
        x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
        if len(x_tensor["shape"]) < norm_rank:
            raise ValidationError(f"{op_name} group {index} input rank must be at least len(normalized_shape)")
        if list(x_tensor["shape"][-norm_rank:]) != [int(dim) for dim in norm_shape]:
            raise ValidationError(f"{op_name} group {index} input suffix must match normalized_shape")
        if any(not isinstance(dim, int) for dim in x_shape_spec[-norm_rank:]):
            raise ValidationError(f"{op_name} group {index} currently requires a static normalized_shape suffix")
        if list(weight_tensor["shape"]) != [int(dim) for dim in norm_shape]:
            raise ValidationError(f"{op_name} group {index} weight shape must match normalized_shape")
        if list(bias_tensor["shape"]) != [int(dim) for dim in norm_shape]:
            raise ValidationError(f"{op_name} group {index} bias shape must match normalized_shape")
        prefix_shape = list(x_tensor["shape"][: len(x_tensor["shape"]) - norm_rank])
        prefix_spec = list(x_shape_spec[: len(x_shape_spec) - norm_rank])
        if batch_prefix_shape is None:
            batch_prefix_shape = prefix_shape
            batch_prefix_spec = prefix_spec
        elif prefix_shape != batch_prefix_shape or prefix_spec != batch_prefix_spec:
            raise ValidationError(f"{op_name} inputs must share the same leading batch dimensions")
        if list(output_tensor["shape"]) != list(x_tensor["shape"]):
            raise ValidationError(f"{op_name} group {index} output shape must match the input shape")
        if str(output_tensor["dtype"]) != x_dtype:
            raise ValidationError(f"{op_name} group {index} output dtype must match the input dtype")


def _validate_add_layer_norm_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_def = get_op_def("add_layer_norm")
    if len(node["outputs"]) != 2:
        raise ValidationError(f"Node {node['id']} must have exactly two outputs")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError("add_layer_norm expects exactly four inputs")
    x_tensor, residual_tensor, weight_tensor, bias_tensor = inputs
    x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
    weight_shape_spec = weight_tensor.get("shape_spec", weight_tensor["shape"])
    bias_shape_spec = bias_tensor.get("shape_spec", bias_tensor["shape"])
    if not x_tensor["shape"]:
        raise ValidationError("add_layer_norm requires rank >= 1 input")
    if list(residual_tensor["shape"]) != list(x_tensor["shape"]):
        raise ValidationError("add_layer_norm residual shape must match input shape")
    if len(weight_tensor["shape"]) != 1:
        raise ValidationError("add_layer_norm requires rank-1 weight")
    if len(bias_tensor["shape"]) != 1:
        raise ValidationError("add_layer_norm requires rank-1 bias")
    if not isinstance(x_shape_spec[-1], int):
        raise ValidationError("add_layer_norm currently requires a static last dimension")
    if not isinstance(weight_shape_spec[0], int):
        raise ValidationError("add_layer_norm currently requires a static weight shape")
    if not isinstance(bias_shape_spec[0], int):
        raise ValidationError("add_layer_norm currently requires a static bias shape")
    if int(weight_tensor["shape"][0]) != int(x_tensor["shape"][-1]):
        raise ValidationError(
            "add_layer_norm weight length must match the input hidden size: "
            f"got hidden={x_tensor['shape'][-1]}, weight={weight_tensor['shape'][0]}"
        )
    if int(bias_tensor["shape"][0]) != int(x_tensor["shape"][-1]):
        raise ValidationError(
            "add_layer_norm bias length must match the input hidden size: "
            f"got hidden={x_tensor['shape'][-1]}, bias={bias_tensor['shape'][0]}"
        )
    if any(input_info["dtype"] != x_tensor["dtype"] for input_info in inputs):
        raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
    if x_tensor["dtype"] not in op_def.allowed_dtypes:
        raise ValidationError(f"add_layer_norm does not support dtype {x_tensor['dtype']}")
    for output_name in node["outputs"]:
        output = tensors[output_name]
        if list(output["shape"]) != list(x_tensor["shape"]):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has shape {output['shape']}, "
                f"expected {x_tensor['shape']}"
            )
        if str(output["dtype"]) != str(x_tensor["dtype"]):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
                f"expected {x_tensor['dtype']}"
            )


def _validate_qkv_split_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_def = get_op_def("qkv_split")
    if len(inputs) != 1:
        raise ValidationError("qkv_split expects exactly one input")
    if len(node["outputs"]) != 3:
        raise ValidationError("qkv_split expects exactly three outputs")
    qkv_tensor = inputs[0]
    if not qkv_tensor["shape"]:
        raise ValidationError("qkv_split requires rank >= 1 input")
    if not isinstance(qkv_tensor["shape"][-1], int) or int(qkv_tensor["shape"][-1]) % 3 != 0:
        raise ValidationError("qkv_split requires a static input last dimension divisible by 3")
    if qkv_tensor["dtype"] not in op_def.allowed_dtypes:
        raise ValidationError(f"qkv_split does not support dtype {qkv_tensor['dtype']}")
    expected_shape = list(qkv_tensor["shape"])
    expected_shape[-1] = int(expected_shape[-1]) // 3
    for output_name in node["outputs"]:
        output_tensor = tensors[output_name]
        if list(output_tensor["shape"]) != expected_shape:
            raise ValidationError("qkv_split output shape must equal input shape with last dim / 3")
        if output_tensor["dtype"] != qkv_tensor["dtype"]:
            raise ValidationError("qkv_split output dtype must match input dtype")


def _validate_glm_ocr_rope_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_name = str(node["op"])
    op_def = get_op_def(op_name)
    if len(inputs) != 4:
        raise ValidationError(f"{op_name} expects exactly four inputs")
    if len(node["outputs"]) != 2:
        raise ValidationError(f"{op_name} expects exactly two outputs")
    q_tensor, k_tensor, cos_tensor, sin_tensor = inputs
    try:
        expected_q_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    q_output = tensors[node["outputs"][0]]
    k_output = tensors[node["outputs"][1]]
    if list(q_output["shape"]) != list(expected_q_shape):
        raise ValidationError(f"{op_name} q output shape must match q input")
    if list(k_output["shape"]) != list(k_tensor["shape"]):
        raise ValidationError(f"{op_name} k output shape must match k input")
    data_dtype = str(q_tensor["dtype"])
    if data_dtype not in op_def.allowed_dtypes:
        raise ValidationError(f"{op_name} does not support dtype {data_dtype}")
    if str(k_tensor["dtype"]) != data_dtype:
        raise ValidationError(f"{op_name} q/k dtype mismatch")
    trig_dtype = str(cos_tensor["dtype"])
    if trig_dtype != str(sin_tensor["dtype"]):
        raise ValidationError(f"{op_name} cos/sin dtype mismatch")
    if trig_dtype not in op_def.allowed_dtypes:
        raise ValidationError(f"{op_name} does not support cos/sin dtype {trig_dtype}")
    if str(q_output["dtype"]) != data_dtype or str(k_output["dtype"]) != data_dtype:
        raise ValidationError(f"{op_name} output dtype must match q/k dtype")


def _validate_glm_ocr_stitch_image_features_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(node["outputs"]) != 1:
        raise ValidationError("glm_ocr_stitch_image_features expects exactly one output")
    if len(inputs) != 3:
        raise ValidationError("glm_ocr_stitch_image_features expects exactly three inputs")
    input_ids_tensor, inputs_embeds_tensor, image_features_tensor = inputs
    output_name = node["outputs"][0]
    output_tensor = tensors[output_name]
    if len(input_ids_tensor["shape"]) != 2:
        raise ValidationError("glm_ocr_stitch_image_features expects input_ids with shape [batch, seq]")
    if len(inputs_embeds_tensor["shape"]) != 3:
        raise ValidationError("glm_ocr_stitch_image_features expects inputs_embeds with shape [batch, seq, hidden]")
    if len(image_features_tensor["shape"]) != 2:
        raise ValidationError("glm_ocr_stitch_image_features expects image_features with shape [image_seq, hidden]")
    if int(input_ids_tensor["shape"][0]) != 1 or int(inputs_embeds_tensor["shape"][0]) != 1:
        raise ValidationError("glm_ocr_stitch_image_features currently supports only batch=1")
    if int(input_ids_tensor["shape"][1]) != int(inputs_embeds_tensor["shape"][1]):
        raise ValidationError(
            "glm_ocr_stitch_image_features input_ids and inputs_embeds sequence lengths must match"
        )
    if int(inputs_embeds_tensor["shape"][2]) != int(image_features_tensor["shape"][1]):
        raise ValidationError(
            "glm_ocr_stitch_image_features image_features hidden size must match inputs_embeds"
        )
    if str(input_ids_tensor["dtype"]) not in {"int64", "int32"}:
        raise ValidationError(
            f"glm_ocr_stitch_image_features input_ids must have dtype int64 or int32, got {input_ids_tensor['dtype']}"
        )
    data_dtype = str(inputs_embeds_tensor["dtype"])
    if data_dtype not in GLM_OCR_STITCH_IMAGE_FEATURES_DTYPES:
        raise ValidationError(f"glm_ocr_stitch_image_features does not support dtype {data_dtype}")
    if str(image_features_tensor["dtype"]) != data_dtype:
        raise ValidationError("glm_ocr_stitch_image_features image_features dtype must match inputs_embeds")
    if str(output_tensor["dtype"]) != data_dtype:
        raise ValidationError("glm_ocr_stitch_image_features output dtype must match inputs_embeds")
    if list(output_tensor["shape"]) != list(inputs_embeds_tensor["shape"]):
        raise ValidationError("glm_ocr_stitch_image_features output shape must match inputs_embeds")
    image_token_id = node.get("attrs", {}).get("image_token_id")
    if not isinstance(image_token_id, int) or isinstance(image_token_id, bool):
        raise ValidationError("glm_ocr_stitch_image_features requires integer image_token_id attr")


def _validate_qwen2_5_vl_stitch_image_features_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(node["outputs"]) != 1:
        raise ValidationError("qwen2_5_vl_stitch_image_features expects exactly one output")
    if len(inputs) != 3:
        raise ValidationError("qwen2_5_vl_stitch_image_features expects exactly three inputs")
    input_ids_tensor, inputs_embeds_tensor, image_features_tensor = inputs
    output_name = node["outputs"][0]
    output_tensor = tensors[output_name]
    if len(input_ids_tensor["shape"]) != 2:
        raise ValidationError("qwen2_5_vl_stitch_image_features expects input_ids with shape [batch, seq]")
    if len(inputs_embeds_tensor["shape"]) != 3:
        raise ValidationError("qwen2_5_vl_stitch_image_features expects inputs_embeds with shape [batch, seq, hidden]")
    if len(image_features_tensor["shape"]) != 2:
        raise ValidationError("qwen2_5_vl_stitch_image_features expects image_features with shape [image_seq, hidden]")
    if input_ids_tensor["shape"][0] != inputs_embeds_tensor["shape"][0]:
        raise ValidationError("qwen2_5_vl_stitch_image_features input_ids and inputs_embeds batch sizes must match")
    if input_ids_tensor["shape"][1] != inputs_embeds_tensor["shape"][1]:
        raise ValidationError(
            "qwen2_5_vl_stitch_image_features input_ids and inputs_embeds sequence lengths must match"
        )
    if inputs_embeds_tensor["shape"][2] != image_features_tensor["shape"][1]:
        raise ValidationError(
            "qwen2_5_vl_stitch_image_features image_features hidden size must match inputs_embeds"
        )
    if str(input_ids_tensor["dtype"]) not in {"int64", "int32"}:
        raise ValidationError(
            "qwen2_5_vl_stitch_image_features input_ids must have dtype int64 or int32, "
            f"got {input_ids_tensor['dtype']}"
        )
    data_dtype = str(inputs_embeds_tensor["dtype"])
    if data_dtype not in QWEN2_5_VL_STITCH_IMAGE_FEATURES_DTYPES:
        raise ValidationError(f"qwen2_5_vl_stitch_image_features does not support dtype {data_dtype}")
    if str(image_features_tensor["dtype"]) != data_dtype:
        raise ValidationError("qwen2_5_vl_stitch_image_features image_features dtype must match inputs_embeds")
    if str(output_tensor["dtype"]) != data_dtype:
        raise ValidationError("qwen2_5_vl_stitch_image_features output dtype must match inputs_embeds")
    if list(output_tensor["shape"]) != list(inputs_embeds_tensor["shape"]):
        raise ValidationError("qwen2_5_vl_stitch_image_features output shape must match inputs_embeds")
    image_token_id = node.get("attrs", {}).get("image_token_id")
    if not isinstance(image_token_id, int) or isinstance(image_token_id, bool):
        raise ValidationError("qwen2_5_vl_stitch_image_features requires integer image_token_id attr")


def _validate_flash_attention_varlen_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_def = get_op_def("flash_attention_varlen")
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if len(inputs) != 4:
        raise ValidationError("flash_attention_varlen expects exactly four inputs")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError, TypeError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, expected {expected_shape}"
        )
    data_dtype = str(inputs[0]["dtype"])
    if data_dtype not in op_def.allowed_dtypes:
        raise ValidationError(f"flash_attention_varlen does not support dtype {data_dtype}")
    for tensor in [*inputs[:3], output]:
        if str(tensor["dtype"]) != data_dtype:
            raise ValidationError(
                f"flash_attention_varlen data tensors must share dtype {data_dtype}, got {tensor['dtype']}"
            )
    if str(inputs[3]["dtype"]) != "int32":
        raise ValidationError(f"flash_attention_varlen cu_seqlens must be int32, got {inputs[3]['dtype']}")


def _validate_flash_attention_static_kv_cache_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_name = str(node["op"])
    op_def = get_op_def(op_name)
    expected_input_count = 7 if op_name == "flash_attention_static_kv_cache_bias" else 6
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if len(inputs) != expected_input_count:
        raise ValidationError(f"{op_name} expects exactly {expected_input_count} inputs")
    q_tensor, past_key_tensor, past_value_tensor, new_key_tensor, new_value_tensor, cache_seqlens_tensor = inputs[:6]
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    output_name = node["outputs"][0]
    output = tensors[output_name]
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, expected {expected_shape}"
        )
    data_dtype = str(q_tensor["dtype"])
    if data_dtype not in op_def.allowed_dtypes:
        raise ValidationError(f"{op_name} does not support dtype {data_dtype}")
    data_tensors = [past_key_tensor, past_value_tensor, new_key_tensor, new_value_tensor]
    if op_name == "flash_attention_static_kv_cache_bias":
        data_tensors.append(inputs[6])
    for input_tensor in data_tensors:
        if str(input_tensor["dtype"]) != data_dtype:
            raise ValidationError(f"Node {node['id']} has mismatched {op_name} data input dtype")
    if str(cache_seqlens_tensor["dtype"]) != "int32":
        raise ValidationError(
            f"{op_name} cache_seqlens must have dtype int32, got {cache_seqlens_tensor['dtype']}"
        )
    if str(output["dtype"]) != data_dtype:
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, expected {data_dtype}"
        )


def _validate_collection_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_name = str(node["op"])
    op_def = get_op_def(op_name)
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError(f"{op_name} expects a non-empty sequence of tensors")
    output_name = node["outputs"][0]
    output = tensors[output_name]
    dynamic_tensors = [
        str(tensor["name"])
        for tensor in [*inputs, output]
        if is_dynamic_shape(tensor.get("shape_spec", tensor["shape"]))
    ]
    dynamic_shape_allowed_ops = {
        "concatenate",
        "concatenate_fast",
        "concatenate_tanh",
        "dynamic_slice",
        "index_add",
        "index_select",
        "one_hot",
        "runtime_index_select",
        "permute",
        "permute021",
        "permute0213",
        "permute102",
        "permute210",
        "slice_scatter",
        "softmax",
        "stack",
    }
    if dynamic_tensors and op_name not in dynamic_shape_allowed_ops:
        raise ValidationError(f"{op_name} currently supports only static shapes: {dynamic_tensors}")
    try:
        expected_shape = op_def.infer_shape_for([input_info["shape"] for input_info in inputs], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {expected_shape}"
        )
    if op_name in {"avg_pool1d", "avg_pool2d", "max_pool2d"} and len(inputs) == 1:
        if str(inputs[0]["dtype"]) not in {"float16", "float32", "bfloat16"}:
            raise ValidationError(f"{op_name} does not support dtype {inputs[0]['dtype']}")
    if op_name == "pad":
        value = node.get("attrs", {}).get("value", 0.0)
        if inputs[0]["dtype"] == "bool":
            if not isinstance(value, (bool, int, float)):
                raise ValidationError(f"pad value must be a constant scalar, got {value!r}")
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValidationError(f"pad value must be a constant numeric scalar, got {value!r}")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
            raise ValidationError("pad value must be finite")
    if op_name in {"gather", "runtime_index_select"}:
        if str(inputs[0]["dtype"]) not in op_def.allowed_dtypes:
            raise ValidationError(f"{op_name} does not support dtype {inputs[0]['dtype']}")
        if str(inputs[1]["dtype"]) not in {"int64", "int32"}:
            raise ValidationError(f"{op_name} index must have dtype int64 or int32, got {inputs[1]['dtype']}")
        if str(output["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
                f"expected {inputs[0]['dtype']}"
            )
        return
    if op_name == "index_add":
        if str(inputs[0]["dtype"]) not in INDEX_ADD_DTYPES:
            raise ValidationError(f"index_add does not support dtype {inputs[0]['dtype']}")
        if str(inputs[1]["dtype"]) not in {"int64", "int32"}:
            raise ValidationError(f"index_add index must have dtype int64 or int32, got {inputs[1]['dtype']}")
        if str(inputs[2]["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} has mismatched input dtypes for index_add: "
                f"{inputs[0]['dtype']} vs {inputs[2]['dtype']}"
            )
        if str(output["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
                f"expected {inputs[0]['dtype']}"
            )
        try:
            normalize_index_add_attrs(
                node.get("attrs", {}).get("dim", 0),
                inputs[0].get("shape_spec", inputs[0]["shape"]),
                inputs[1].get("shape_spec", inputs[1]["shape"]),
                inputs[2].get("shape_spec", inputs[2]["shape"]),
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        expected_shape_spec = list(inputs[0].get("shape_spec", inputs[0]["shape"]))
        actual_shape_spec = list(output.get("shape_spec", output["shape"]))
        if actual_shape_spec != expected_shape_spec:
            raise ValidationError(
                f"index_add output shape_spec must match input shape_spec {expected_shape_spec}, "
                f"got {actual_shape_spec}"
            )
        return
    if op_name == "scatter":
        if str(inputs[0]["dtype"]) not in SCATTER_DTYPES:
            raise ValidationError(f"scatter does not support dtype {inputs[0]['dtype']}")
        if str(inputs[1]["dtype"]) not in {"int64", "int32"}:
            raise ValidationError(f"scatter index must have dtype int64 or int32, got {inputs[1]['dtype']}")
        if str(inputs[2]["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} has mismatched input dtypes for scatter: "
                f"{inputs[0]['dtype']} vs {inputs[2]['dtype']}"
            )
        if str(output["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
                f"expected {inputs[0]['dtype']}"
            )
        try:
            normalize_scatter_attrs(
                node.get("attrs", {}).get("dim", 0),
                inputs[0]["shape"],
                inputs[1]["shape"],
                inputs[2]["shape"],
                op_name="scatter",
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return
    if op_name == "scatter_add":
        if str(inputs[0]["dtype"]) not in SCATTER_REDUCE_DTYPES:
            raise ValidationError(f"scatter_add does not support dtype {inputs[0]['dtype']}")
        if str(inputs[1]["dtype"]) not in {"int64", "int32"}:
            raise ValidationError(f"scatter_add index must have dtype int64 or int32, got {inputs[1]['dtype']}")
        if str(inputs[2]["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} has mismatched input dtypes for scatter_add: "
                f"{inputs[0]['dtype']} vs {inputs[2]['dtype']}"
            )
        if str(output["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
                f"expected {inputs[0]['dtype']}"
            )
        try:
            normalize_scatter_attrs(
                node.get("attrs", {}).get("dim", 0),
                inputs[0]["shape"],
                inputs[1]["shape"],
                inputs[2]["shape"],
                op_name="scatter_add",
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return
    if op_name == "scatter_reduce":
        if str(inputs[0]["dtype"]) not in SCATTER_REDUCE_DTYPES:
            raise ValidationError(f"scatter_reduce does not support dtype {inputs[0]['dtype']}")
        if str(inputs[1]["dtype"]) not in {"int64", "int32"}:
            raise ValidationError(f"scatter_reduce index must have dtype int64 or int32, got {inputs[1]['dtype']}")
        if str(inputs[2]["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} has mismatched input dtypes for scatter_reduce: "
                f"{inputs[0]['dtype']} vs {inputs[2]['dtype']}"
            )
        if str(output["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
                f"expected {inputs[0]['dtype']}"
            )
        attrs = node.get("attrs", {})
        try:
            normalize_scatter_attrs(
                attrs.get("dim", 0),
                inputs[0]["shape"],
                inputs[1]["shape"],
                inputs[2]["shape"],
                op_name="scatter_reduce",
            )
            normalize_scatter_reduce_name(attrs.get("reduce", "sum"))
            normalize_scatter_reduce_include_self(attrs.get("include_self", True))
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return
    if op_name == "one_hot":
        if len(inputs) != 1:
            raise ValidationError("one_hot expects exactly one input")
        if str(inputs[0]["dtype"]) not in {"int64", "int32"}:
            raise ValidationError(f"one_hot input must have dtype int64 or int32, got {inputs[0]['dtype']}")
        num_classes = normalize_one_hot_num_classes(node.get("attrs", {}).get("num_classes"))
        expected_shape_spec = [
            *(dict(dim) if isinstance(dim, Mapping) else dim for dim in inputs[0].get("shape_spec", inputs[0]["shape"])),
            num_classes,
        ]
        actual_shape_spec = list(output.get("shape_spec", output["shape"]))
        if actual_shape_spec != expected_shape_spec:
            raise ValidationError(
                f"one_hot output shape_spec must equal input shape_spec plus num_classes {num_classes}, "
                f"got {actual_shape_spec}"
            )
        if str(output["dtype"]) != "int64":
            raise ValidationError(
                f"Node {node['id']} output {output_name} has dtype {output['dtype']}, expected int64"
            )
        return
    if op_name == "batch_gather":
        if str(inputs[0]["dtype"]) not in op_def.allowed_dtypes:
            raise ValidationError(f"batch_gather does not support dtype {inputs[0]['dtype']}")
        if str(inputs[1]["dtype"]) not in {"int64", "int32"}:
            raise ValidationError(f"batch_gather indices must have dtype int64 or int32, got {inputs[1]['dtype']}")
        if str(output["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
                f"expected {inputs[0]['dtype']}"
            )
        return
    if any(input_info["dtype"] != inputs[0]["dtype"] for input_info in inputs):
        raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
    if inputs[0]["dtype"] not in op_def.allowed_dtypes:
        raise ValidationError(f"{op_name} does not support dtype {inputs[0]['dtype']}")
    if str(output["dtype"]) != str(inputs[0]["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
            f"expected {inputs[0]['dtype']}"
        )


def _validate_masked_select_output_contract(
    ir: Mapping[str, Any],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    output_names_by_tensor = {
        str(output["tensor"]): str(output["name"])
        for output in ir.get("outputs", [])
    }
    reported_outputs = {
        str(report.get("output"))
        for report in ir.get("metadata", {}).get("output_shape_reports", {}).get("reports", [])
        if isinstance(report, Mapping) and report.get("kind") == "shape_buffer"
    }
    consumer_counts: dict[str, int] = {}
    for node in ir.get("nodes", []):
        for input_name in node.get("inputs", []):
            consumer_counts[str(input_name)] = consumer_counts.get(str(input_name), 0) + 1
    view_sources = {
        str(view.get("source"))
        for view in ir.get("metadata", {}).get("views", {}).get("views", [])
        if isinstance(view, Mapping)
    }
    for node in ir.get("nodes", []):
        if str(node.get("op")) != "masked_select":
            continue
        output_name = str(node["outputs"][0])
        public_output_name = output_names_by_tensor.get(output_name)
        if public_output_name is None:
            raise ValidationError(
                "masked_select currently requires a public output because its data-dependent result length "
                "is reported through the output shape buffer"
            )
        if consumer_counts.get(output_name, 0) != 0:
            raise ValidationError("masked_select output cannot feed downstream ops before jagged data flow is modeled")
        if output_name in view_sources:
            raise ValidationError("masked_select output cannot feed shape views before jagged data flow is modeled")
        if public_output_name not in reported_outputs:
            raise ValidationError(
                f"masked_select public output {public_output_name!r} must be listed in metadata.output_shape_reports"
            )
        output_tensor = tensors[output_name]
        if len(output_tensor.get("shape_spec", output_tensor["shape"])) != 1:
            raise ValidationError("masked_select output must be rank-1")


def _validate_embedding_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    op_def = get_op_def("embedding")
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if len(inputs) != 2:
        raise ValidationError("embedding expects exactly two inputs")
    table_tensor = inputs[0]
    index_tensor = inputs[1]
    output_name = node["outputs"][0]
    output = tensors[output_name]
    table_shape_spec = table_tensor.get("shape_spec", table_tensor["shape"])
    if is_dynamic_shape(table_shape_spec):
        raise ValidationError("embedding currently requires a static table shape [vocab, hidden]")
    try:
        expected_shape = op_def.infer_shape_for([table_tensor["shape"], index_tensor["shape"]], node.get("attrs", {}))
    except (ValueError, NotImplementedError) as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {expected_shape}"
        )
    if str(table_tensor["dtype"]) not in op_def.allowed_dtypes:
        raise ValidationError(f"embedding does not support dtype {table_tensor['dtype']}")
    if str(index_tensor["dtype"]) not in {"int64", "int32"}:
        raise ValidationError(f"embedding indices must have dtype int64 or int32, got {index_tensor['dtype']}")
    if str(output["dtype"]) != str(table_tensor["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
            f"expected {table_tensor['dtype']}"
        )


def _validate_nms_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(node["outputs"]) != 1:
        raise ValidationError("nms expects exactly one output")
    if len(inputs) != 2:
        raise ValidationError("nms expects exactly two inputs")
    output = tensors[str(node["outputs"][0])]
    try:
        attrs = normalize_nms_attrs(**node.get("attrs", {}))
        normalize_nms_shapes([inputs[0]["shape"], inputs[1]["shape"]])
        expected_shape = infer_nms_shape_with_attrs([inputs[0]["shape"], inputs[1]["shape"]], attrs)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(f"nms output shape must be {expected_shape}, got {output['shape']}")
    data_dtype = str(inputs[0]["dtype"])
    if data_dtype != str(inputs[1]["dtype"]) or str(output["dtype"]) != data_dtype:
        raise ValidationError("nms boxes, scores, and output must share dtype")


def _validate_batched_nms_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(node["outputs"]) != 1:
        raise ValidationError("batched_nms expects exactly one output")
    if len(inputs) != 1:
        raise ValidationError("batched_nms expects exactly one input")
    output = tensors[str(node["outputs"][0])]
    try:
        attrs = normalize_batched_nms_attrs(**node.get("attrs", {}))
        normalize_batched_nms_shapes([inputs[0]["shape"]])
        expected_shape = infer_batched_nms_shape_with_attrs([inputs[0]["shape"]], attrs)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(f"batched_nms output shape must be {expected_shape}, got {output['shape']}")
    if str(output["dtype"]) != "int64":
        raise ValidationError(f"batched_nms output dtype must be int64, got {output['dtype']}")


def _validate_efficient_nms_node(
    node: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(node["outputs"]) != 4:
        raise ValidationError("efficient_nms expects exactly four outputs")
    if len(inputs) != 2:
        raise ValidationError("efficient_nms expects exactly two inputs")
    outputs = [tensors[str(name)] for name in node["outputs"]]
    try:
        attrs = normalize_efficient_nms_attrs(**node.get("attrs", {}))
        normalize_efficient_nms_shapes([inputs[0]["shape"], inputs[1]["shape"]])
        expected_shapes = infer_efficient_nms_output_shapes([inputs[0]["shape"], inputs[1]["shape"]], attrs)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    data_dtype = str(inputs[0]["dtype"])
    if data_dtype != str(inputs[1]["dtype"]):
        raise ValidationError("efficient_nms boxes and scores must share dtype")
    expected_dtypes = ("int64", data_dtype, data_dtype, "int64")
    for output, expected_shape, expected_dtype in zip(outputs, expected_shapes, expected_dtypes):
        if list(output["shape"]) != list(expected_shape):
            raise ValidationError(f"efficient_nms output shape must be {expected_shape}, got {output['shape']}")
        if str(output["dtype"]) != expected_dtype:
            raise ValidationError(f"efficient_nms output dtype must be {expected_dtype}, got {output['dtype']}")


def _masked_select_capacity_spec(
    input_tensor: Mapping[str, Any],
    mask_tensor: Mapping[str, Any],
) -> int | dict[str, Any]:
    broadcast_shape = broadcast_shape_spec(
        input_tensor.get("shape_spec", input_tensor["shape"]),
        mask_tensor.get("shape_spec", mask_tensor["shape"]),
    )
    total: int | dict[str, Any] = 1
    for dim in broadcast_shape:
        total = symbolic_int_expr("mul", total, normalize_symbolic_int(dim))
    return total


def _fused_elementwise_output_dtype(sub_op: Mapping[str, Any], input_dtypes: Sequence[str]) -> str:
    op = str(sub_op.get("op"))
    if op == "cast":
        if len(input_dtypes) != 1:
            raise ValidationError(f"Fused sub-op cast expects 1 inputs, got {len(input_dtypes)}")
        if input_dtypes[0] not in CAST_ELEMENTWISE_DTYPES:
            raise ValidationError(f"Fused sub-op cast does not support input dtype {input_dtypes[0]}")
        output_dtype = str(sub_op.get("attrs", {}).get("dtype", ""))
        if output_dtype not in CAST_ELEMENTWISE_DTYPES:
            raise ValidationError(f"Fused sub-op cast does not support dtype {output_dtype}")
        return output_dtype
    if op == "where":
        if len(input_dtypes) != 3:
            raise ValidationError(f"Fused sub-op where expects 3 inputs, got {len(input_dtypes)}")
        if input_dtypes[0] != "bool":
            raise ValidationError(f"Fused sub-op where condition must have dtype bool, got {input_dtypes[0]}")
        if input_dtypes[1] != input_dtypes[2]:
            raise ValidationError(f"Fused sub-op where x/y dtype mismatch: {input_dtypes[1]} vs {input_dtypes[2]}")
        if input_dtypes[1] not in ELEMENTWISE_OUTPUT_DTYPES:
            raise ValidationError(f"Fused sub-op where does not support dtype {input_dtypes[1]}")
        return input_dtypes[1]
    if input_dtypes and any(dtype != input_dtypes[0] for dtype in input_dtypes):
        raise ValidationError(f"Fused sub-op {op} has mismatched input dtypes")
    supported_dtypes = EQ_ELEMENTWISE_DTYPES if op == "eq" else FLOAT_ELEMENTWISE_DTYPES
    if input_dtypes and input_dtypes[0] not in supported_dtypes:
        raise ValidationError(f"Fused sub-op {op} does not support dtype {input_dtypes[0]}")
    return elementwise_output_dtype(op, input_dtypes[0])
