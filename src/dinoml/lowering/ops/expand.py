from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.broadcasting import BROADCAST_DTYPES, resolve_expand_shape
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "expand")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("expand_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("expand_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "expand")
    func = _function_name(node, tensor_map)
    inp = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{inp}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "expand")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, input_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = target_storage_type(dtype, target)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "index_body": _index_body(input_tensor["shape"], output_tensor["shape"]),
        "block_size": 256,
    }


def _validate_node_contract(node: Mapping[str, Any], input_tensor: Mapping[str, Any], output_tensor: Mapping[str, Any]) -> None:
    if node["op"] != "expand":
        raise ValueError(f"Unsupported broadcasting op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("expand expects one tensor input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("expand expects exactly one output")
    if str(input_tensor["dtype"]) != str(output_tensor["dtype"]):
        raise ValueError("expand input and output dtype must match")
    if str(output_tensor["dtype"]) not in BROADCAST_DTYPES:
        raise NotImplementedError(f"expand lowering does not support dtype {output_tensor['dtype']}")
    expected_shape = resolve_expand_shape(input_tensor["shape"], node.get("attrs", {}).get("shape"))
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("expand output shape does not match shape attr")


def _index_body(input_shape: Any, output_shape: Any) -> str:
    aligned_input = [1] * (len(output_shape) - len(input_shape)) + [int(dim) for dim in input_shape]
    lines = [
        "  int64_t remaining = idx;",
        "  int64_t input_idx = 0;",
        "  int64_t input_stride = 1;",
        "  int64_t coord = 0;",
    ]
    for axis in reversed(range(len(output_shape))):
        output_dim = int(output_shape[axis])
        input_dim = int(aligned_input[axis])
        lines.append(f"  coord = remaining % {output_dim};")
        lines.append(f"  remaining /= {output_dim};")
        if input_dim != 1:
            lines.append("  input_idx += coord * input_stride;")
        lines.append(f"  input_stride *= {input_dim};")
    lines.append("  return input_idx;")
    return "\n".join(lines)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "expand",
        "input_shape": list(input_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"expand_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


EXPAND_LOWERING = OpLowering(
    op_name="expand",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
