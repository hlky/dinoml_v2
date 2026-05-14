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
from dinoml.ops.definitions import get_op_def
from dinoml.ops.elementwise import (
    CAST_ELEMENTWISE_DTYPES,
    ELEMENTWISE_BY_NAME,
    ELEMENTWISE_OUTPUT_DTYPES,
    FLOAT_ELEMENTWISE_DTYPES,
    elementwise_output_dtype,
)
from dinoml.ops.positional import (
    GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS,
    GET_1D_ROTARY_POS_EMBED_DTYPES,
    normalize_get_1d_rotary_pos_embed_attrs,
    rotary_output_cols,
)
from dinoml.passes.utils import tensor_map
from dinoml.shapes import is_dynamic_shape, validate_shape_spec


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
    validate_view_metadata(ir.get("metadata", {}).get("views"), tensors)
    validate_view_metadata(ir.get("metadata", {}).get("memory_plan", {}).get("views"), tensors)
    validate_output_shape_report_metadata(ir.get("metadata", {}).get("output_shape_reports"), ir["outputs"], tensors)


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
        if prod(tensor["shape"]) != prod(source["shape"]):
            raise ValidationError(f"View tensor {tensor_name} must preserve source element count")
        if view.get("kind") != "shape_view":
            raise ValidationError(f"View tensor {tensor_name} must use kind shape_view")
        transform = view.get("transform")
        if transform not in VIEW_ONLY_TRANSFORMS:
            raise ValidationError(f"View tensor {tensor_name} has unsupported transform {transform}")
        offset_elements = view.get("offset_elements", 0)
        if not isinstance(offset_elements, int) or offset_elements != 0:
            raise ValidationError(f"View tensor {tensor_name} must use zero offset for shape-only aliases")
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
                "offset_elements": 0,
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
    if node["op"] == "argmax":
        _validate_argmax_node(node, inputs, tensors)
        return
    if node["op"] in {"topk_values", "topk_indices"}:
        _validate_topk_node(node, inputs, tensors)
        return
    if node["op"] in GET_1D_ROTARY_POS_EMBED_COMPONENT_OPS:
        _validate_get_1d_rotary_pos_embed_node(node, inputs, tensors)
        return
    if node["op"] == "t5_layer_norm":
        _validate_t5_layer_norm_node(node, inputs, tensors)
        return
    if node["op"] == "layer_norm":
        _validate_layer_norm_node(node, inputs, tensors)
        return
    if node["op"] == "embedding":
        _validate_embedding_node(node, inputs, tensors)
        return
    if node["op"] in {
        "avg_pool1d",
        "avg_pool2d",
        "conv2d_bias",
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
    weight_shape_spec = weight_tensor.get("shape_spec", weight_tensor["shape"])
    bias_shape_spec = bias_tensor.get("shape_spec", bias_tensor["shape"])
    if not x_tensor["shape"]:
        raise ValidationError("layer_norm requires rank >= 1 input")
    if len(weight_tensor["shape"]) != 1:
        raise ValidationError("layer_norm requires rank-1 weight")
    if len(bias_tensor["shape"]) != 1:
        raise ValidationError("layer_norm requires rank-1 bias")
    if not isinstance(x_shape_spec[-1], int):
        raise ValidationError("layer_norm currently requires a static last dimension")
    if not isinstance(weight_shape_spec[0], int):
        raise ValidationError("layer_norm currently requires a static weight shape")
    if not isinstance(bias_shape_spec[0], int):
        raise ValidationError("layer_norm currently requires a static bias shape")
    if int(weight_tensor["shape"][0]) != int(x_tensor["shape"][-1]):
        raise ValidationError(
            "layer_norm weight length must match the input hidden size: "
            f"got hidden={x_tensor['shape'][-1]}, weight={weight_tensor['shape'][0]}"
        )
    if int(bias_tensor["shape"][0]) != int(x_tensor["shape"][-1]):
        raise ValidationError(
            "layer_norm bias length must match the input hidden size: "
            f"got hidden={x_tensor['shape'][-1]}, bias={bias_tensor['shape'][0]}"
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
        raise ValidationError(f"layer_norm does not support dtype {x_tensor['dtype']}")
    if str(output["dtype"]) != str(x_tensor["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
            f"expected {x_tensor['dtype']}"
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
    if dynamic_tensors:
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
    if op_name == "gather":
        if str(inputs[0]["dtype"]) not in op_def.allowed_dtypes:
            raise ValidationError(f"gather does not support dtype {inputs[0]['dtype']}")
        if str(inputs[1]["dtype"]) not in {"int64", "int32"}:
            raise ValidationError(f"gather index must have dtype int64 or int32, got {inputs[1]['dtype']}")
        if str(output["dtype"]) != str(inputs[0]["dtype"]):
            raise ValidationError(
                f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
                f"expected {inputs[0]['dtype']}"
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
    if input_dtypes and input_dtypes[0] not in FLOAT_ELEMENTWISE_DTYPES:
        raise ValidationError(f"Fused sub-op {op} does not support dtype {input_dtypes[0]}")
    return elementwise_output_dtype(op, input_dtypes[0])
