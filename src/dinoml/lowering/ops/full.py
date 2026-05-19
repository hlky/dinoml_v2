from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping

from dinoml.lowering.ops.template_rendering import render_op_template
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.creation import CREATION_DTYPES
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import LoweringTargetSpec, lowering_target_spec, storage_type


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = _supported_target_spec(target)
    return render_op_template(f"full_{spec.op_template_flavor}.j2", _context(spec, node, tensor_map))


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = _supported_target_spec(target)
    func = _function_name(node, tensor_map)
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    _supported_target_spec(target)
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(spec: LoweringTargetSpec, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, output_tensor)
    dtype = str(output_tensor["dtype"])
    context = {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type(dtype, spec.name),
        "fill_literal": _float_literal(float(node.get("attrs", {}).get("fill_value", 0.0))),
        "block_size": 256,
    }
    if spec.is_gpu:
        context.update(
            {
                "gpu_stream_type": spec.stream_type,
                "gpu_check_macro": spec.check_macro,
                "gpu_last_error_call": spec.last_error_call,
            }
        )
    return context


def _validate_node_contract(node: Mapping[str, Any], output_tensor: Mapping[str, Any]) -> None:
    if node["op"] != "full":
        raise ValueError(f"Unsupported creation op: {node['op']}")
    if node.get("inputs"):
        raise ValueError("full expects no tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("full expects exactly one output")
    if str(output_tensor["dtype"]) not in CREATION_DTYPES:
        raise NotImplementedError(f"full lowering does not support dtype {output_tensor['dtype']}")
    if list(node.get("attrs", {}).get("shape", [])) != list(output_tensor["shape"]):
        raise ValueError("full output shape does not match shape attr")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "full",
        "shape": list(output_tensor["shape"]),
        "dtype": str(output_tensor["dtype"]),
        "fill_value": float(node.get("attrs", {}).get("fill_value", 0.0)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"full_{digest}"


def _float_literal(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("full lowering supports only finite fill_value for now")
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal = f"{literal}.0"
    return f"{literal}f"


def _supported_target_spec(target: str) -> LoweringTargetSpec:
    try:
        spec = lowering_target_spec(target)
    except ValueError as exc:
        raise ValueError(f"Unsupported full target: {target}") from exc
    if not spec.generated_module_admitted:
        raise ValueError(f"Unsupported full target: {target}")
    return spec


FULL_LOWERING = OpLowering(
    op_name="full",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
