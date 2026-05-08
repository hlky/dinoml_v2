from __future__ import annotations

from math import prod
from typing import Any, Mapping, Sequence

from dinoml.ir import IR_SCHEMA_VERSION, VIEW_METADATA_VERSION, VIEW_ONLY_TRANSFORMS, normalize_dtype
from dinoml.ops.definitions import get_op_def
from dinoml.ops.elementwise import ELEMENTWISE_BY_NAME, FLOAT_ELEMENTWISE_DTYPES
from dinoml.passes.utils import tensor_map
from dinoml.shapes import validate_shape_spec


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
    if len(node["outputs"]) != 1:
        raise ValidationError(f"Node {node['id']} must have exactly one output")
    if not op_def.accepts_input_count(len(inputs)):
        raise ValidationError(f"{op_def.name} expects {op_def.input_count} inputs")
    try:
        op_def.infer_shape([input_info["shape"] for input_info in inputs])
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if inputs:
        if any(input_info["dtype"] != inputs[0]["dtype"] for input_info in inputs):
            raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
        if inputs[0]["dtype"] not in op_def.allowed_dtypes:
            raise ValidationError(f"{op_def.name} does not support dtype {inputs[0]['dtype']}")


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
    if inputs:
        if any(input_info["dtype"] != inputs[0]["dtype"] for input_info in inputs):
            raise ValidationError(f"Node {node['id']} has mismatched input dtypes")
        if inputs[0]["dtype"] not in FLOAT_ELEMENTWISE_DTYPES:
            raise ValidationError(f"fused_elementwise does not support dtype {inputs[0]['dtype']}")
    for sub_op in sub_ops:
        elementwise_spec = ELEMENTWISE_BY_NAME.get(sub_op.get("op"))
        if elementwise_spec is None:
            raise ValidationError(f"Unsupported fused sub-op: {sub_op.get('op')}")
        if len(sub_op.get("inputs", [])) != elementwise_spec.arity:
            raise ValidationError(
                f"Fused sub-op {sub_op.get('op')} expects {elementwise_spec.arity} inputs, got {len(sub_op.get('inputs', []))}"
            )
        if len(sub_op.get("outputs", [])) != 1:
            raise ValidationError(f"Fused sub-op {sub_op.get('op')} must have exactly one output")
