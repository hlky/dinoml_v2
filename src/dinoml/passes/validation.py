from __future__ import annotations

from math import prod
from typing import Any, Mapping, Sequence

from dinoml.ir import IR_SCHEMA_VERSION, VIEW_METADATA_VERSION, VIEW_ONLY_TRANSFORMS, normalize_dtype
from dinoml.layout import validate_layout
from dinoml.ops.definitions import get_op_def
from dinoml.ops.elementwise import (
    CAST_ELEMENTWISE_DTYPES,
    ELEMENTWISE_BY_NAME,
    ELEMENTWISE_OUTPUT_DTYPES,
    FLOAT_ELEMENTWISE_DTYPES,
    elementwise_output_dtype,
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


def _validate_node(node: Mapping[str, Any], tensors: Mapping[str, Mapping[str, Any]]) -> None:
    op_def = get_op_def(node["op"])
    inputs = [tensors[name] for name in node["inputs"]]
    if node["op"] == "fused_elementwise":
        _validate_fused_elementwise_node(node, inputs, tensors)
        return
    if node["op"] == "where":
        _validate_where_node(node, inputs, tensors)
        return
    if node["op"] in {
        "concatenate",
        "stack",
        "flip",
        "repeat_interleave",
        "permute",
        "dynamic_slice",
        "index_select",
        "slice_scatter",
    }:
        _validate_collection_node(node, inputs, tensors)
        return
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError(f"{op_def.name} expects {op_def.input_count} inputs")
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
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if list(output["shape"]) != list(expected_shape):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has shape {output['shape']}, "
            f"expected {expected_shape}"
        )
    if any(input_info["dtype"] != inputs[0]["dtype"] for input_info in inputs):
        raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
    if inputs[0]["dtype"] not in op_def.allowed_dtypes:
        raise ValidationError(f"{op_name} does not support dtype {inputs[0]['dtype']}")
    if str(output["dtype"]) != str(inputs[0]["dtype"]):
        raise ValidationError(
            f"Node {node['id']} output {output_name} has dtype {output['dtype']}, "
            f"expected {inputs[0]['dtype']}"
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
